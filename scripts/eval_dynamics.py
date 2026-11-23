"""Compare the lower-body joint torques and ground reaction forces produced by a
trained policy with the ForceBody labels of the trial the clip was made from.

One environment, mean actions, episode started at clip frame 0, so simulation
step k corresponds to trial frame k.

Simulation side
  joint torque  generalized force of muscles and passive elements
                (qfrc_actuator + qfrc_passive), projected onto the independent
                coordinate: tau_j + sum_i (d q_i / d q_j) tau_i over the joints i
                coupled to j (knee: tibia translation/rotation and patella).
                Divided by the model mass (N m / kg).
  GRF           net contact force on each foot (calcaneus subtree), averaged over
                the physics substeps of a control step, in body weights.
  Both are low-pass filtered (default 10 Hz, zero phase) before the comparison.

ForceBody side
  tau_selfconsistent / mass_kg_dynamics   (N m / kg), sigma_tau as a band
  grf / (mass_kg_dynamics * 9.81)         (body weights), rotated into the model frame

Run from the project root on the GPU host:

  MYOSUITE_MIMIC_CLIP=$PWD/clips/core10/NAME.npz MUJOCO_GL=egl uv run python scripts/eval_dynamics.py \
      --task myoMimicFullbody-v0 --experiment logs/rsl_rl/EXP --iter -1
"""

from __future__ import annotations

import argparse
import contextlib
import os
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, filtfilt

ENTITY = "mimic_fullbody_robot"
G = 9.81
# ForceBody coordinate (nimble name) -> MyoFullBody joint, per side
LEG = (("hip_{s}_0", "hip_flexion_{s}"), ("hip_{s}_1", "hip_adduction_{s}"), ("hip_{s}_2", "hip_rotation_{s}"),
       ("walker_knee_{s}_0", "knee_angle_{s}"), ("ankle_{s}_0", "ankle_angle_{s}"),
       ("subtalar_{s}_0", "subtalar_angle_{s}"), ("mtp_{s}_0", "mtp_angle_{s}"))
# OpenSim axes (x forward, y up, z right) -> model world (faces -y, z up, right = -x); same as fb2myo.py
C = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
GRF_AXES = (("vertical", np.array([0.0, 0.0, 1.0])), ("forward", np.array([0.0, -1.0, 0.0])),
            ("to the right", np.array([-1.0, 0.0, 0.0])))


@contextlib.contextmanager
def fixed_start_frame(frame: int):
    """While active every torch.randint call returns *frame* (see render_policy.py)."""
    original = torch.randint

    def constant(*a, **k):
        return torch.full_like(original(*a, **k), frame)

    torch.randint = constant
    try:
        yield
    finally:
        torch.randint = original


def lowpass(x: np.ndarray, cutoff: float, rate: float) -> np.ndarray:
    if cutoff <= 0 or len(x) < 30:
        return x
    b, a = butter(4, cutoff / (0.5 * rate))
    return filtfilt(b, a, x, axis=0)


def find_trial(clip_path: Path, given: Path | None) -> Path:
    if given is not None:
        return given
    name = str(np.load(clip_path, allow_pickle=True)["source_trial"])
    hits = sorted(Path("data").rglob(name))
    if not hits:
        raise SystemExit(f"trial {name} (from the clip) not found under data/; pass --trial")
    return hits[0]


class Coupling:
    """Projection of generalized forces onto independent coordinates (joint equalities)."""

    def __init__(self, model) -> None:
        import mujoco

        self.model = model
        self.dependents: dict[int, list[tuple[int, np.ndarray]]] = {}
        for e in range(model.neq):
            if model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT or model.eq_obj2id[e] < 0:
                continue
            dep, master = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            self.dependents.setdefault(master, []).append((dep, np.array(model.eq_data[e, :5])))

    def joint(self, name: str) -> int:
        for j in range(self.model.njnt):
            if self.model.joint(j).name.split("/")[-1] == name:
                return j
        raise KeyError(name)

    def project(self, name: str, qpos: np.ndarray, qfrc: np.ndarray) -> np.ndarray:
        """Generalized force along the independent coordinate *name*, (T,)."""
        m, j = self.model, self.joint(name)
        x = qpos[:, m.jnt_qposadr[j]] - m.qpos0[m.jnt_qposadr[j]]
        tau = qfrc[:, m.jnt_dofadr[j]].copy()
        for dep, a in self.dependents.get(j, []):
            slope = a[1] + 2 * a[2] * x + 3 * a[3] * x**2 + 4 * a[4] * x**3
            tau += slope * qfrc[:, m.jnt_dofadr[dep]]
        return tau


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="myoMimicFullbody-v0")
    p.add_argument("--experiment", type=Path, default=None, help="logs/rsl_rl/<experiment>")
    p.add_argument("--iter", type=int, default=-1, help="iteration to load with --experiment; -1 = last")
    p.add_argument("--checkpoint", type=Path, default=None, help="a model_*.pt file (instead of --experiment)")
    p.add_argument("--trial", type=Path, default=None, help="ForceBody trial npz (default: looked up from the clip)")
    p.add_argument("--out", type=Path, default=Path("eval/dynamics"))
    p.add_argument("--cutoff", type=float, default=10.0, help="low-pass cutoff for simulated signals, Hz; 0 = off")
    p.add_argument("--no-terminations", action="store_true", help="keep going after a tracking failure")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    clip_path = Path(os.environ.get("MYOSUITE_MIMIC_CLIP", ""))
    if not clip_path.is_file():
        raise SystemExit("set MYOSUITE_MIMIC_CLIP to the clip the policy was trained on")
    if args.checkpoint is None:
        if args.experiment is None:
            raise SystemExit("give --experiment or --checkpoint")
        found = {int(f.stem.split("_")[1]): f for f in sorted(args.experiment.glob("*/model_*.pt"))}
        if not found:
            raise SystemExit(f"no model_*.pt under {args.experiment}/*/")
        args.checkpoint = found[max(found) if args.iter < 0 else min(found, key=lambda k: abs(k - args.iter))]
    trial_path = find_trial(clip_path, args.trial)
    trial = np.load(trial_path, allow_pickle=True)

    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import _resolve_mimic_mjlab_ids

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg, agent_cfg = load_env_cfg(args.task, play=True), load_rl_cfg(args.task)
    env_cfg.scene.num_envs = 1
    if args.no_terminations:
        env_cfg.terminations = {k: v for k, v in env_cfg.terminations.items() if v.time_out}
    env_cfg.scene.sensors = tuple(env_cfg.scene.sensors) + (
        ContactSensorCfg(
            name="feet_grf",
            primary=ContactMatch(mode="subtree", pattern=("calcn_r", "calcn_l"), entity=ENTITY),
            fields=("found", "force"), reduce="netforce", num_slots=1, history_length=env_cfg.decimation,
        ),
    )
    base = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    model = base.sim.mj_model
    data = base.scene[ENTITY].data.data
    sensor = base.scene["feet_grf"]
    order = [n.split("/")[-1] for n in sensor.primary_names]
    assert order == ["calcn_r", "calcn_l"], order
    source = _resolve_mimic_mjlab_ids(base, ENTITY, "fullbody")["clip_source"]
    mass = float(model.body_mass.sum())
    dt = base.step_dt

    with fixed_start_frame(0):
        obs, _ = env.reset()
    qpos, qfrc, grf = [], [], []
    done, step = False, 0
    with torch.no_grad():
        while not done:
            obs, _, dones, extras = env.step(policy(obs))
            step += 1
            done = bool(dones[0])
            if done:  # the environment has already been reset
                break
            qpos.append(data.qpos[0].cpu().numpy().astype(np.float64))
            qfrc.append((data.qfrc_actuator[0] + data.qfrc_passive[0]).cpu().numpy().astype(np.float64))
            # force of the foot on the ground, mean over the substeps -> ground on the foot
            grf.append(-sensor.data.force_history[0].mean(dim=-2).cpu().numpy().astype(np.float64))
    ended = [k.split("/")[-1] for k, v in extras.get("log", {}).items()
             if k.startswith("Episode_Termination/") and float(v) > 0]
    qpos, qfrc, grf = np.asarray(qpos), np.asarray(qfrc), np.asarray(grf)  # (T, nq), (T, nv), (T, 2, 3)
    n = len(qpos)
    frames = np.arange(1, n + 1)  # state after step k is compared with clip frame k
    if n < 50:
        raise SystemExit(f"the episode lasted only {n} steps; nothing to compare")

    # ---------------- ForceBody labels ----------------
    names = [str(x) for x in trial["dof_names"]]
    m_dyn = float(trial["mass_kg_dynamics"])
    n_trial = trial["q_gt"].shape[0]
    if n_trial != int(source.n_frames):
        print(f"[warn] trial has {n_trial} frames, clip has {int(source.n_frames)}: frame alignment may be off")
    keep = frames < n_trial
    frames, qpos, qfrc, grf, n = frames[keep], qpos[keep], qfrc[keep], grf[keep], int(keep.sum())
    has_tau = "tau_selfconsistent" in trial.files
    tau_fb = np.asarray(trial["tau_selfconsistent"], float)[frames] / m_dyn if has_tau else None
    sig_fb = np.asarray(trial["sigma_tau"], float)[frames] / m_dyn if "sigma_tau" in trial.files else None
    tau_ok = np.asarray(trial["tau_frame_ok"], bool)[frames] if "tau_frame_ok" in trial.files else np.ones(n, bool)
    grf_ok = np.asarray(trial["grf_frame_ok"], bool)[frames] if "grf_frame_ok" in trial.files else np.ones(n, bool)
    mc_ok = np.asarray(trial["mc_frame_ok"], bool)[frames] if "mc_frame_ok" in trial.files else tau_ok  # mask for sigma_tau
    label_usable = bool(trial["torque_label_usable"]) if "torque_label_usable" in trial.files else True
    grf_fb = np.asarray(trial["grf"], float)[frames][:, :2] @ C.T / (m_dyn * G)  # (T, 2, 3), right then left
    q_fb = np.asarray(trial["q_gt"], float)[frames]

    # ---------------- simulation ----------------
    coupling = Coupling(model)
    rate = 1.0 / dt
    grf_sim = lowpass(grf / (mass * G), args.cutoff, rate)
    t = frames * dt

    rows = []
    fig, axes = plt.subplots(len(LEG), 2, figsize=(13, 2.2 * len(LEG)), sharex=True)
    angle_fig, angle_axes = plt.subplots(len(LEG), 2, figsize=(13, 2.2 * len(LEG)), sharex=True)
    for c, side in enumerate("rl"):
        for r, (fb_name, myo_name) in enumerate(LEG):
            fb_name, myo_name = fb_name.format(s=side), myo_name.format(s=side)
            raw = coupling.project(myo_name, qpos, qfrc) / mass
            sim = lowpass(raw, args.cutoff, rate)
            ax = axes[r, c]
            ax.plot(t, raw, color="tab:blue", lw=0.5, alpha=0.3)
            ax.plot(t, sim, color="tab:blue", lw=1.5, label="policy (muscles + passive)")
            j = coupling.joint(myo_name)
            q_sim = qpos[:, model.jnt_qposadr[j]]
            q_ref = q_fb[:, names.index(fb_name)]
            lo, hi = model.jnt_range[j]
            near = np.radians(2.0)
            row = {"coordinate": myo_name, "angle_rmse_deg": float(np.degrees(np.sqrt(np.mean((q_sim - q_ref) ** 2)))),
                   "angle_bias_deg": float(np.degrees(np.mean(q_sim - q_ref))),
                   "at_limit": float(np.mean((q_sim < lo + near) | (q_sim > hi - near))),
                   "ref_locked": bool(np.ptp(q_ref) < np.radians(0.5))}
            bx = angle_axes[r, c]
            bx.plot(t, np.degrees(q_ref), color="k", lw=1.2, label="ForceBody q_gt")
            bx.plot(t, np.degrees(q_sim), color="tab:blue", lw=1.5, label="policy")
            for bound in (lo, hi):
                bx.axhline(np.degrees(bound), color="tab:red", lw=0.8, ls="--", alpha=0.6)
            span = np.degrees([min(q_ref.min(), q_sim.min()), max(q_ref.max(), q_sim.max())])
            bx.set_ylim(span[0] - 8, span[1] + 8)
            bx.set_ylabel("deg")
            bx.set_title(myo_name + ("  (locked in ForceBody)" if row["ref_locked"] else ""), fontsize=9, loc="left")
            bx.grid(alpha=0.3)
            if has_tau:
                ref_raw = tau_fb[:, names.index(fb_name)]
                ref = lowpass(ref_raw, args.cutoff, rate)
                ax.plot(t, ref_raw, color="k", lw=0.5, alpha=0.3)
                ax.plot(t, ref, color="k", lw=1.2, label="ForceBody inverse dynamics")
                if sig_fb is not None:
                    s = sig_fb[:, names.index(fb_name)]
                    ax.fill_between(t, ref - 2 * s, ref + 2 * s, color="k", alpha=0.15, lw=0, label="± 2 sigma")
                    row["mean_sigma"] = float(s[mc_ok].mean())
                    row["within_2sigma"] = float(np.mean(np.abs(sim - ref)[mc_ok] <= 2 * s[mc_ok]))
                row["rmse"] = float(np.sqrt(np.mean((sim - ref)[tau_ok] ** 2)))
                row["r"] = float(np.corrcoef(sim[tau_ok], ref[tau_ok])[0, 1])
                row["range_ref"] = float(np.ptp(ref[tau_ok]))
            rows.append(row)
            ax.set_ylabel("N m / kg")
            ax.set_title(myo_name, fontsize=9, loc="left")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=3, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    tag = f"{args.checkpoint.resolve().parent.parent.name}_it{int(args.checkpoint.stem.split('_')[1]):05d}"
    fig.suptitle(f"{tag}: joint torque, {trial_path.name} ({n} steps, ended by {', '.join(ended) or 'unknown'})", fontsize=10)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / f"{tag}_torque.png", dpi=130)
    plt.close(fig)
    angle_axes[0, 0].legend(fontsize=7, ncol=2, loc="upper right")
    for bx in angle_axes[-1]:
        bx.set_xlabel("time (s)")
    angle_fig.suptitle(f"{tag}: joint angle (red dashed: joint limit), {trial_path.name}", fontsize=10)
    angle_fig.tight_layout()
    angle_fig.savefig(args.out / f"{tag}_angle.png", dpi=130)
    plt.close(angle_fig)

    grf_rows = []
    fig, axes = plt.subplots(3, 2, figsize=(13, 7), sharex=True)
    for c, side in enumerate(("right", "left")):
        for r, (label, axis) in enumerate(GRF_AXES):
            sim, ref = grf_sim[:, c] @ axis, grf_fb[:, c] @ axis
            ax = axes[r, c]
            ax.plot(t, grf[:, c] @ axis / (mass * G), color="tab:blue", lw=0.5, alpha=0.3)
            ax.plot(t, sim, color="tab:blue", lw=1.5, label="policy")
            ax.plot(t, ref, color="k", lw=1.2, label="ForceBody (measured)")
            ax.set_title(f"{side} foot, {label}", fontsize=9, loc="left")
            ax.set_ylabel("body weight")
            ax.grid(alpha=0.3)
            grf_rows.append({"foot": side, "component": label, "rmse": float(np.sqrt(np.mean((sim - ref)[grf_ok] ** 2))),
                             "r": float(np.corrcoef(sim[grf_ok], ref[grf_ok])[0, 1]),
                             "peak_sim": float(sim[grf_ok].max()), "peak_ref": float(ref[grf_ok].max())})
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"{tag}: ground reaction force, {trial_path.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / f"{tag}_grf.png", dpi=130)
    plt.close(fig)

    np.savez_compressed(args.out / f"{tag}_dynamics.npz", t=t, frames=frames, qpos=qpos, qfrc=qfrc, grf_sim_n=grf,
                        model_mass=mass, trial=str(trial_path), cutoff=args.cutoff)

    print(f"[eval] {tag} on {trial_path.name}: {n} steps ({n * dt:.2f} s), ended by {', '.join(ended) or 'unknown'}")
    print(f"       masses: model {mass:.1f} kg, subject {m_dyn:.1f} kg (dynamics); low-pass {args.cutoff:g} Hz; "
          f"frames usable: torque {tau_ok.mean():.0%}, sigma {mc_ok.mean():.0%}, GRF {grf_ok.mean():.0%}")
    if not label_usable:
        print("       [warn] torque_label_usable is False for this trial: its torque labels should not be used")
    total_sim, total_ref = grf_sim.sum(1)[:, 2].mean(), grf_fb.sum(1)[:, 2].mean()
    print(f"       mean total vertical GRF: policy {total_sim:.3f} BW, ForceBody {total_ref:.3f} BW (both should be near 1)")
    if has_tau:
        print("       coordinate           angle RMSE  (bias)   at limit | torque RMSE  range(ref)      r   mean sigma  within 2 sigma   (N m / kg)")
        for row in rows:
            note = "   locked in ForceBody" if row["ref_locked"] else ""
            print(f"       {row['coordinate']:18s} {row['angle_rmse_deg']:8.1f} deg ({row['angle_bias_deg']:+5.1f}) {row['at_limit']:9.0%} | "
                  f"{row['rmse']:10.3f} {row['range_ref']:11.3f} {row['r']:7.2f} {row.get('mean_sigma', float('nan')):11.3f} "
                  f"{row.get('within_2sigma', float('nan')):14.0%}{note}")
        print("       (torque statistics use the same low-pass on both sides; 'at limit' = policy within 2 deg of a joint limit)")
    print("       GRF (body weight)            RMSE        r    peak policy   peak ForceBody")
    for row in grf_rows:
        print(f"       {row['foot']:5s} {row['component']:13s} {row['rmse']:10.3f} {row['r']:8.2f} {row['peak_sim']:11.2f} {row['peak_ref']:14.2f}")
    print(f"       figures: {args.out}/{tag}_torque.png, _grf.png, _angle.png")
    env.close()


if __name__ == "__main__":
    main()
