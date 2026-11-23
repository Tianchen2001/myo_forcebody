"""Reward and early termination as defined in the official MuscleMimic code.

Source: amathislab/musclemimic
  reward       musclemimic/core/reward/trajectory_based.py  (MimicReward)
  termination  musclemimic/core/terminal_state_handler/enhanced_fullbody.py
               (MeanRelativeSiteDeviationWithRootTerminalStateHandler)
  values       fullbody/conf_fullbody.yaml

The terms plug into the MyoSuite mjlab Mimic environment: the clip, its random
start frame per environment and the frame counter all come from MyoSuite, so
these terms see exactly the same reference frame as the rest of the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import mujoco
import numpy as np
import torch

from myo_forcebody.tasks import mm_math

ENTITY = "mimic_fullbody_robot"
VARIANT = "fullbody"

# Order of conf_fullbody.yaml; the first site is the reference (pelvis).
SITES_FOR_MIMIC = (
    "pelvis_mimic", "upper_body_mimic", "head_mimic",
    "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic",
    "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
    "left_hip_mimic", "left_knee_mimic", "left_ankle_mimic", "left_toes_mimic",
    "right_hip_mimic", "right_knee_mimic", "right_ankle_mimic", "right_toes_mimic",
)


@dataclass(frozen=True)
class RewardCfg:
    """Weights and exponential scales of MimicReward (conf_fullbody.yaml + class defaults)."""

    qpos_w: float = 0.1
    qvel_w: float = 0.1
    root_pos_w: float = 0.1
    root_vel_w: float = 0.1
    rpos_w: float = 0.6
    rquat_w: float = 0.01
    rvel_w: float = 0.1  # applied to the angular and to the linear part separately
    qpos_exp: float = 10.0
    qvel_exp: float = 2.0
    root_pos_exp: float = 10.0
    root_vel_exp: float = 10.0
    rpos_exp: float = 100.0
    rquat_exp: float = 10.0
    rvel_exp: float = 0.1

    @property
    def maximum(self) -> float:
        return (self.qpos_w + self.qvel_w + self.root_pos_w + self.root_vel_w
                + self.rpos_w + self.rquat_w + 2 * self.rvel_w)


@dataclass(frozen=True)
class TerminationCfg:
    """Thresholds of the early termination.

    conf_fullbody.yaml trains with 1.0 m / 1.0 m / 1.0 rad and validates with
    0.5 m on the sites; the paper states 0.5 m for sites and root. Defaults here
    follow the paper. Override with FB_MM_SITE_THR, FB_MM_ROOT_THR, FB_MM_ORI_THR.
    """

    site: float = 0.5  # mean deviation of pelvis-relative site positions (m)
    root: float = 0.5  # deviation of the pelvis site from the reference (m)
    orientation: float = 1.0  # root orientation, geodesic angle (rad)

    @staticmethod
    def from_env() -> "TerminationCfg":
        d = TerminationCfg()
        return TerminationCfg(
            site=float(os.environ.get("FB_MM_SITE_THR", d.site)),
            root=float(os.environ.get("FB_MM_ROOT_THR", d.root)),
            orientation=float(os.environ.get("FB_MM_ORI_THR", d.orientation)),
        )


# ---------------------------------------------------------------------------
# Reference: everything the terms need, for every clip frame
# ---------------------------------------------------------------------------


def _scene_ids(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, int]:
    """Site ids (official order), their body ids and the root body id in *model*."""
    by_name = {model.site(i).name.split("/")[-1]: i for i in range(model.nsite)}
    missing = [n for n in SITES_FOR_MIMIC if n not in by_name]
    if missing:
        raise RuntimeError(f"mimic sites missing from the model: {missing}")
    site_ids = np.array([by_name[n] for n in SITES_FOR_MIMIC])
    body_ids = model.site_bodyid[site_ids]
    root_ids = np.unique(model.body_rootid[body_ids])
    if root_ids.size != 1:
        raise RuntimeError("mimic sites do not share one root body")
    return site_ids, body_ids, int(root_ids[0])


def build_reference(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray) -> dict[str, torch.Tensor]:
    """Forward kinematics of the clip on CPU MuJoCo, then the same relative
    quantities as used for the simulation state. One row per clip frame."""
    site_ids, body_ids, root_id = _scene_ids(model)
    data = mujoco.MjData(model)
    n = qpos.shape[0]
    pos = np.empty((n, len(site_ids), 3))
    mat = np.empty((n, len(site_ids), 3, 3))
    cvel = np.empty((n, len(site_ids), 6))
    com = np.empty((n, 3))
    for i in range(n):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        pos[i] = data.site_xpos[site_ids]
        mat[i] = data.site_xmat[site_ids].reshape(-1, 3, 3)
        cvel[i] = data.cvel[body_ids]
        com[i] = data.subtree_com[root_id]

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float32)

    site_vel = mm_math.site_velocities(t(pos), t(cvel), t(com))
    rpos, rangles, rvel = mm_math.relative_site_quantities(t(pos), t(mat), site_vel)
    q, v = t(qpos), t(qvel)
    return {
        "rpos": rpos, "rangles": rangles, "rvel": rvel,
        "qpos": q, "qvel": v,
        "root_vel": mm_math.root_local_velocity(q, v),
        "pelvis_site": t(pos[:, 0]),
    }


# ---------------------------------------------------------------------------
# Per-environment state shared by the reward and the termination
# ---------------------------------------------------------------------------

_cache: dict[int, dict[str, Any]] = {}


def _setup(env: Any) -> dict[str, Any]:
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import _resolve_mimic_mjlab_ids

    myo = _resolve_mimic_mjlab_ids(env, ENTITY, VARIANT)
    source = myo.get("clip_source")
    clip = myo.get("clip")
    if source is None or clip is None or isinstance(clip, (tuple, list)):
        raise RuntimeError("the MuscleMimic terms need exactly one motion clip (MYOSUITE_MIMIC_CLIP)")
    model = env.sim.mj_model
    site_ids, body_ids, root_id = _scene_ids(model)
    device = env.device
    ref = build_reference(model, np.asarray(clip.qpos, dtype=np.float64), np.asarray(clip.qvel, dtype=np.float64))
    state = {
        "ref": {k: v.to(device) for k, v in ref.items()},
        "site_ids": torch.as_tensor(site_ids, device=device, dtype=torch.long),
        "body_ids": torch.as_tensor(body_ids, device=device, dtype=torch.long),
        "root_id": root_id,
        "step": None,
    }
    _cache[id(env)] = state
    return state


def _errors(env: Any) -> dict[str, torch.Tensor]:
    """All tracking errors of the current step, computed once per step."""
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import _resolve_mimic_mjlab_ids

    state = _cache.get(id(env)) or _setup(env)
    step = env.episode_length_buf
    tag = (int(env.common_step_counter), int(step.sum()))
    if state["step"] == tag:
        return state["errors"]

    source = _resolve_mimic_mjlab_ids(env, ENTITY, VARIANT)["clip_source"]
    frame = source.frame_indices(step)
    ref = {k: v[frame] for k, v in state["ref"].items()}

    d = env.scene[ENTITY].data.data
    pos = d.site_xpos[:, state["site_ids"]]
    mat = d.site_xmat[:, state["site_ids"]].reshape(pos.shape[0], -1, 3, 3)
    cvel = d.cvel[:, state["body_ids"]]
    com = d.subtree_com[:, state["root_id"]]
    rpos, rangles, rvel = mm_math.relative_site_quantities(pos, mat, mm_math.site_velocities(pos, cvel, com))
    qpos, qvel = d.qpos, d.qvel

    not_quat = torch.ones(qpos.shape[1], dtype=torch.bool, device=qpos.device)
    not_quat[3:7] = False
    errors = {
        # MimicReward distances (mean squared errors; quaternion as geodesic angle)
        "qpos": (qpos[:, not_quat] - ref["qpos"][:, not_quat]).square().mean(-1)
        + mm_math.quat_angle(qpos[:, 3:7], ref["qpos"][:, 3:7]),
        "qvel": (qvel - ref["qvel"]).square().mean(-1),
        "rpos": (rpos - ref["rpos"]).square().mean((-1, -2)),
        "rangles": (rangles - ref["rangles"]).square().mean((-1, -2)),
        "rvel_rot": (rvel[..., :3] - ref["rvel"][..., :3]).square().mean((-1, -2)),
        "rvel_lin": (rvel[..., 3:] - ref["rvel"][..., 3:]).square().mean((-1, -2)),
        "root_pos": (qpos[:, :3] - ref["qpos"][:, :3]).square().mean(-1),
        "root_vel": (mm_math.root_local_velocity(qpos, qvel) - ref["root_vel"]).square().mean(-1),
        # Termination distances
        "site_dev": (rpos - ref["rpos"]).norm(dim=-1).mean(-1),
        "root_dev": (pos[:, 0] - ref["pelvis_site"]).norm(dim=-1),
        "root_ori": mm_math.quat_angle(qpos[:, 3:7], ref["qpos"][:, 3:7]),
        "clip_end": source.clip_end(step),
    }
    state["step"], state["errors"] = tag, errors
    return errors


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------


def reward_components(env: Any, cfg: RewardCfg = RewardCfg()) -> dict[str, torch.Tensor]:
    """exp(-scale * error) of every reward term, each in (0, 1], shape (N,)."""
    e = _errors(env)
    return {
        "qpos": torch.exp(-cfg.qpos_exp * e["qpos"]),
        "qvel": torch.exp(-cfg.qvel_exp * e["qvel"]),
        "root_pos": torch.exp(-cfg.root_pos_exp * e["root_pos"]),
        "root_vel": torch.exp(-cfg.root_vel_exp * e["root_vel"]),
        "rpos": torch.exp(-cfg.rpos_exp * e["rpos"]),
        "rquat": torch.exp(-cfg.rquat_exp * e["rangles"]),
        "rvel_rot": torch.exp(-cfg.rvel_exp * e["rvel_rot"]),
        "rvel_lin": torch.exp(-cfg.rvel_exp * e["rvel_lin"]),
    }


def mimic_reward(cfg: RewardCfg = RewardCfg()) -> Callable[[Any], torch.Tensor]:
    """MimicReward without the penalty terms (maximum 1.21 per step with the defaults).

    The action out-of-bounds penalty of the official reward is not included here:
    this environment still maps actions through a sigmoid, which has no bounds.
    """

    def _fn(env: Any) -> torch.Tensor:
        r = reward_components(env, cfg)
        total = (
            cfg.qpos_w * r["qpos"] + cfg.qvel_w * r["qvel"]
            + cfg.root_pos_w * r["root_pos"] + cfg.root_vel_w * r["root_vel"]
            + cfg.rpos_w * r["rpos"] + cfg.rquat_w * r["rquat"]
            + cfg.rvel_w * r["rvel_rot"] + cfg.rvel_w * r["rvel_lin"]
        )
        return torch.nan_to_num(total.clamp_min(0.0), nan=0.0)

    return _fn


def mimic_termination(cfg: TerminationCfg | None = None) -> Callable[[Any], torch.Tensor]:
    """True where the relative site deviation, the root position or the root
    orientation exceeds its threshold. Never fires past the end of the clip."""
    cfg = cfg or TerminationCfg.from_env()

    def _fn(env: Any) -> torch.Tensor:
        e = _errors(env)
        deviated = (e["site_dev"] > cfg.site) | (e["root_dev"] > cfg.root) | (e["root_ori"] > cfg.orientation)
        return deviated & ~e["clip_end"]

    return _fn


def metric(name: str) -> Callable[[Any], torch.Tensor]:
    """One raw error for logging, shape (N,): site_dev, root_dev (m), root_ori (rad)."""

    def _fn(env: Any) -> torch.Tensor:
        return _errors(env)[name]

    return _fn
