"""Render trained policies to mp4 without a viewer.

One environment, deterministic (mean) actions, same start frame for every
checkpoint, so videos of different checkpoints are directly comparable.
Recording stops at the end of the first episode (clip end, time-out or failure).

Run from the project root on the GPU host:

  MYOSUITE_MIMIC_CLIP=$PWD/clips/core10/NAME.npz MUJOCO_GL=egl uv run python scripts/render_policy.py \
      --task myoMimicFullbody-v0 --experiment logs/rsl_rl/EXP --iters 2000 5000 -1

--iters picks, over all runs of the experiment, the checkpoint closest to each
iteration (-1 = the last one). Checkpoint files can also be given directly.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ENTITY, VARIANT = "mimic_fullbody_robot", "fullbody"
# Reward per step when the state equals the reference (used to scale the plot).
MAX_REWARD = {"myoMimicFullbody-v0": 5.05, "myoMuscleMimicFullbody-v0": 5.05, "fbMuscleMimic-S1-v0": 1.21}


def annotate(frame: np.ndarray, title: str, k: int, rewards: np.ndarray, dt: float, top: float) -> np.ndarray:
    """Write the reward of step *k* on the frame and plot the reward of the whole episode."""
    img = Image.fromarray(np.asarray(frame)).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default(size=max(12, h // 32))
    now = rewards[k - 1] if k > 0 else float("nan")
    mean = rewards[:k].mean() if k > 0 else float("nan")
    lines = [title, f"t = {k * dt:5.2f} s", f"reward {now:5.2f} / {top:.2f}   mean so far {mean:5.2f}"]
    line_h = int(font.size * 1.35)
    draw.rectangle([0, 0, w, 8 + line_h * len(lines)], fill=(0, 0, 0, 140))
    for i, text in enumerate(lines):
        draw.text((10, 4 + i * line_h), text, font=font, fill=(255, 255, 255, 255))
    # Reward over the episode along the bottom, with a cursor at the current step.
    strip, pad = h // 6, 10
    y0, y1 = h - strip, h - pad
    draw.rectangle([0, y0 - pad, w, h], fill=(0, 0, 0, 140))
    n = len(rewards)
    if n > 1:
        xs = pad + (w - 2 * pad) * np.arange(n) / (n - 1)
        ys = y1 - (y1 - y0) * np.clip(rewards / top, 0.0, 1.0)
        draw.line(list(zip(xs.tolist(), ys.tolist())), fill=(120, 220, 120, 255), width=2)
        draw.line([(pad, y0), (w - pad, y0)], fill=(255, 255, 255, 90), width=1)  # maximum
        x = pad + (w - 2 * pad) * max(k - 1, 0) / (n - 1)
        draw.line([(x, y0 - pad), (x, h)], fill=(255, 255, 255, 255), width=2)
    return np.asarray(img)


@contextlib.contextmanager
def fixed_start_frame(frame: int):
    """While active, every torch.randint call returns *frame*.

    The MyoSuite reset event draws the clip start frame with torch.randint and
    offers no way to choose it; it is the only randint call during a reset.
    """
    original = torch.randint

    def constant(*a, **k):
        return torch.full_like(original(*a, **k), frame)

    torch.randint = constant
    try:
        yield
    finally:
        torch.randint = original


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="*", type=Path, help="model_*.pt files of the same task")
    p.add_argument("--experiment", type=Path, default=None, help="logs/rsl_rl/<experiment>: search all its runs")
    p.add_argument("--iters", type=int, nargs="*", default=[], help="iterations to render with --experiment; -1 = last")
    p.add_argument("--task", default="myoMimicFullbody-v0")
    p.add_argument("--out", type=Path, default=Path("videos/policy"))
    p.add_argument("--start-frame", type=int, default=0, help="clip frame the episode starts from")
    p.add_argument("--steps", type=int, default=None, help="maximum control steps (default: until the episode ends)")
    p.add_argument("--fps", type=int, default=50, help="video frame rate; playback is real time")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-terminations", action="store_true", help="keep going after a tracking failure")
    p.add_argument("--no-overlay", action="store_true", help="do not draw the reward on the video")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    checkpoints = list(args.checkpoints)
    if args.experiment is not None:
        found = {int(f.stem.split("_")[1]): f for f in sorted(args.experiment.glob("*/model_*.pt"))}  # later run wins
        if not found:
            raise SystemExit(f"no model_*.pt under {args.experiment}/*/")
        for it in args.iters or [-1]:
            checkpoints.append(found[max(found) if it < 0 else min(found, key=lambda k: abs(k - it))])
    if not checkpoints:
        raise SystemExit("give checkpoint files or --experiment")

    import mjlab.tasks  # noqa: F401  registers the tasks (needs MYOSUITE_MIMIC_CLIP)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import _resolve_mimic_mjlab_ids

    if args.task not in list_tasks():
        raise SystemExit(f"task {args.task!r} is not registered; is MYOSUITE_MIMIC_CLIP set to an existing clip?")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width, env_cfg.viewer.height = args.width, args.height
    if args.no_terminations:
        env_cfg.terminations = {k: v for k, v in env_cfg.terminations.items() if v.time_out}

    base = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)

    source = _resolve_mimic_mjlab_ids(base, ENTITY, VARIANT)["clip_source"]
    n_frames = int(source.n_frames)
    if not 0 <= args.start_frame < n_frames:
        raise SystemExit(f"--start-frame must be in [0, {n_frames})")
    every = max(1, round(1.0 / (args.fps * base.step_dt)))
    max_steps = args.steps or 10**9
    args.out.mkdir(parents=True, exist_ok=True)

    for ckpt in checkpoints:
        runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)

        with fixed_start_frame(args.start_frame):  # same clip frame for every checkpoint
            obs, _ = env.reset()
        started = int(source.frame_indices(base.episode_length_buf)[0])

        name = f"{ckpt.resolve().parent.parent.name}_it{int(ckpt.stem.split('_')[1]):05d}.mp4"
        frames, frame_steps, rewards, step, done = [base.render()], [0], [], 0, False
        with torch.no_grad():
            while not done and step < max_steps:
                obs, rew, dones, extras = env.step(policy(obs))
                step += 1
                rewards.append(float(rew[0]))
                done = bool(dones[0])
                if not done and step % every == 0:
                    frames.append(base.render())
                    frame_steps.append(step)
        if done:
            log = extras.get("log", {})
            ended = [k.split("/")[-1] for k, v in log.items() if k.startswith("Episode_Termination/") and float(v) > 0]
        else:
            ended = ["step limit"]
        rewards = np.asarray(rewards)
        top = MAX_REWARD.get(args.task) or float(max(rewards.max(), 1e-6))
        title = f"{ckpt.resolve().parent.parent.name}   iteration {int(ckpt.stem.split('_')[1])}   start frame {started}"
        if not args.no_overlay:
            frames = [annotate(f, title, k, rewards, base.step_dt, top) for f, k in zip(frames, frame_steps)]
        imageio.mimsave(args.out / name, [np.asarray(f) for f in frames], fps=round(1.0 / (every * base.step_dt)), macro_block_size=1)
        print(
            f"[render] {args.out / name}: start frame {started}, {step} steps ({step * base.step_dt:.2f} s), "
            f"ended by {', '.join(ended) or 'unknown'}, reward per step {np.mean(rewards):.3f}"
        )
    env.close()


if __name__ == "__main__":
    main()
