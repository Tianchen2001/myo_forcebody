"""Relative site quantities as defined in the official MuscleMimic code.

Port of ``calculate_relative_site_quantities`` and the helpers it calls
(amathislab/musclemimic, ``loco_mujoco/core/utils/math.py``) to batched torch.
The same functions are used for the simulation state and for the reference clip,
so both sides of every error term go through identical code.

Shapes: B = batch (environments or clip frames), S = tracked sites, with the
main site (pelvis) at index 0.
"""

from __future__ import annotations

import torch


def site_velocities(
    site_pos: torch.Tensor,  # (B, S, 3) world
    body_cvel: torch.Tensor,  # (B, S, 6) cvel of each site's body: [angular, linear]
    root_com: torch.Tensor,  # (B, 3) subtree_com of the root body
) -> torch.Tensor:
    """World-frame 6D velocity [angular, linear] of each site (mj_objectVelocity)."""
    ang = body_cvel[..., :3]
    lin = body_cvel[..., 3:]
    offset = site_pos - root_com[:, None, :]
    lin_at_site = lin - torch.cross(offset, ang, dim=-1)
    return torch.cat([ang, lin_at_site], dim=-1)


def matrix_to_rotvec(mat: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (..., 3, 3) to rotation vectors (..., 3), via quaternions."""
    m = mat.reshape(-1, 3, 3)
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    # Four candidate quaternions (w, x, y, z), each scaled by 4 * one component.
    cand = torch.stack(
        [
            torch.stack([1 + m00 + m11 + m22, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], -1),
            torch.stack([m[:, 2, 1] - m[:, 1, 2], 1 + m00 - m11 - m22, m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0]], -1),
            torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 0, 1] + m[:, 1, 0], 1 - m00 + m11 - m22, m[:, 1, 2] + m[:, 2, 1]], -1),
            torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1], 1 - m00 - m11 + m22], -1),
        ],
        dim=1,
    )  # (n, 4, 4)
    # Use the candidate with the largest diagonal term (best conditioned).
    diag = torch.stack([cand[:, 0, 0], cand[:, 1, 1], cand[:, 2, 2], cand[:, 3, 3]], -1)
    best = diag.argmax(dim=-1)
    quat = cand[torch.arange(m.shape[0], device=m.device), best]
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    quat = torch.where(quat[:, :1] < 0, -quat, quat)  # w >= 0: angle in [0, pi]
    vec = quat[:, 1:]
    sin_half = vec.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, quat[:, :1])
    # angle / sin(angle / 2), with the limit 2 at angle -> 0.
    scale = torch.where(sin_half > 1e-6, angle / sin_half.clamp_min(1e-12), torch.full_like(angle, 2.0))
    return (vec * scale).reshape(*mat.shape[:-2], 3)


def relative_site_quantities(
    site_pos: torch.Tensor,  # (B, S, 3) world
    site_mat: torch.Tensor,  # (B, S, 3, 3) world
    site_vel: torch.Tensor,  # (B, S, 6) world [angular, linear]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Positions, orientations and velocities of sites 1.. relative to site 0.

    Returns:
        rpos (B, S-1, 3): site position minus main site position (world axes).
        rangles (B, S-1, 3): rotation vector of R_main^T R_site.
        rvel (B, S-1, 6): [angular, linear] relative velocity, as upstream:
            angular = (R_main^T R_site)^T w_site - w_main
            linear  = R_main (v_main - v_site)
    """
    main_pos, other_pos = site_pos[:, :1], site_pos[:, 1:]
    main_mat, other_mat = site_mat[:, :1], site_mat[:, 1:]
    main_vel, other_vel = site_vel[:, :1], site_vel[:, 1:]

    rpos = other_pos - main_pos
    rel_rot = main_mat.transpose(-1, -2) @ other_mat  # (B, S-1, 3, 3)
    rangles = matrix_to_rotvec(rel_rot)

    lin = (main_mat @ (main_vel[..., 3:] - other_vel[..., 3:])[..., None])[..., 0]
    ang = (rel_rot.transpose(-1, -2) @ other_vel[..., :3, None])[..., 0] - main_vel[..., :3]
    rvel = torch.cat([ang, lin], dim=-1)
    return rpos, rangles, rvel


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Unit quaternions (..., 4) as (w, x, y, z) to rotation matrices (..., 3, 3)."""
    q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def quat_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (rad) between unit quaternions (..., 4)."""
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dot = (q1 * q2).sum(-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def root_local_velocity(qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
    """Root velocity as upstream: linear rotated into the root frame, angular as stored.

    qpos (B, nq) with the free joint first, qvel (B, nv). Returns (B, 6).
    """
    rot = quat_to_matrix(qpos[:, 3:7])
    lin_local = (rot.transpose(-1, -2) @ qvel[:, :3, None])[..., 0]
    return torch.cat([lin_local, qvel[:, 3:6]], dim=-1)
