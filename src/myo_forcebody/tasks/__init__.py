"""mjlab tasks of this project (loaded through the ``mjlab.tasks`` entry point).

Tasks are registered only when a motion clip is given through the environment
variable ``MYOSUITE_MIMIC_CLIP``, like the MyoSuite task they are built on.

  fbMuscleMimic-S1-v0   MyoSuite ``myoMimicFullbody-v0`` with the reward and the
                        early termination replaced by the official MuscleMimic
                        definitions (pelvis-relative site terms). Observations,
                        action mapping and PPO settings are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_TASK = "myoMimicFullbody-v0"
S1_TASK = "fbMuscleMimic-S1-v0"


def _stage1(env_cfg):
    """Swap reward and deviation termination for the official definitions."""
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg

    from myo_forcebody.tasks import mm_terms

    ctrl_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    # mjlab multiplies every reward by the control step; undo it so that the
    # logged reward is the per-step MimicReward (maximum 1.21).
    env_cfg.rewards = {"mimic": RewardTermCfg(func=mm_terms.mimic_reward(), weight=1.0 / ctrl_dt)}
    if "mimic_deviation" not in env_cfg.terminations:
        raise RuntimeError(f"{BASE_TASK} has no 'mimic_deviation' termination to replace")
    env_cfg.terminations["mimic_deviation"] = TerminationTermCfg(func=mm_terms.mimic_termination())
    return env_cfg


def _register() -> None:
    raw = os.environ.get("MYOSUITE_MIMIC_CLIP") or os.environ.get("MIMIC_CLIP")
    if not raw or not Path(raw).expanduser().is_file():
        return
    import myosuite.envs.myo.backends.mjlab  # noqa: F401  registers BASE_TASK for the clip
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, register_mjlab_task

    if BASE_TASK not in list_tasks() or S1_TASK in list_tasks():
        return
    register_mjlab_task(
        task_id=S1_TASK,
        env_cfg=_stage1(load_env_cfg(BASE_TASK)),
        play_env_cfg=_stage1(load_env_cfg(BASE_TASK, play=True)),
        rl_cfg=load_rl_cfg(BASE_TASK),
    )


_register()
