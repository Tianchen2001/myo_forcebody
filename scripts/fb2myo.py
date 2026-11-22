"""ForceBody trial (Rajagopal q_gt, 37 DoF) -> MyoFullBody motion clip (+ optional video).

  MUJOCO_GL=egl uv run python scripts/fb2myo.py TRIAL.npz --out clips/NAME.npz --video videos/NAME.mp4

Kinematics only: every frame is forward kinematics on the written qpos, nothing is simulated.
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from myosuite.biomechanics.opensim_motion import OpenSimMotion, opensim_motion_to_qpos
from myosuite.core.trajectory_io import motion_clip_from_states
from myosuite.integrations.musclemimic.fullbody_model import (
    compile_mimic_fullbody_mjmodel, default_mimic_fullbody_config)
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

# ---- block 0: names -----------------------------------------------------------------------------
# ForceBody stores Rajagopal coordinates under nimble names (<joint>_<index>); MyoSuite matches
# OpenSim coordinate names.
_SIDE = {"hip_{s}_0": "hip_flexion_{s}", "hip_{s}_1": "hip_adduction_{s}", "hip_{s}_2": "hip_rotation_{s}",
         "walker_knee_{s}_0": "knee_angle_{s}", "ankle_{s}_0": "ankle_angle_{s}",
         "subtalar_{s}_0": "subtalar_angle_{s}", "mtp_{s}_0": "mtp_angle_{s}",
         "acromial_{s}_0": "arm_flex_{s}", "acromial_{s}_1": "arm_add_{s}", "acromial_{s}_2": "arm_rot_{s}",
         "elbow_{s}_0": "elbow_flex_{s}", "radioulnar_{s}_0": "pro_sup_{s}",
         "radius_hand_{s}_0": "wrist_flex_{s}", "radius_hand_{s}_1": "wrist_dev_{s}"}
OPENSIM = {"ground_pelvis_0": "pelvis_tilt", "ground_pelvis_1": "pelvis_list", "ground_pelvis_2": "pelvis_rotation",
           "ground_pelvis_3": "pelvis_tx", "ground_pelvis_4": "pelvis_ty", "ground_pelvis_5": "pelvis_tz",
           "back_0": "lumbar_extension", "back_1": "lumbar_bending", "back_2": "lumbar_rotation",
           **{k.format(s=s): v.format(s=s) for s in "rl" for k, v in _SIDE.items()}}
# ForceBody joint centre -> model body with its origin at that joint (for scale and the check)
JOINT_BODY = {f"{j}_{s}": f"{b}_{s}" for s in "rl" for j, b in
              (("hip", "femur"), ("walker_knee", "tibia"), ("ankle", "talus"), ("mtp", "toes"),
               ("acromial", "humerus"), ("elbow", "ulna"), ("radius_hand", "lunate"))}
SEGMENTS = [(f"{a}_{s}", f"{b}_{s}") for s in "rl" for a, b in
            (("hip", "walker_knee"), ("walker_knee", "ankle"), ("ankle", "mtp"),
             ("acromial", "elbow"), ("elbow", "radius_hand"))] + [("hip_l", "hip_r"), ("acromial_l", "acromial_r")]
FLIP_XZ = np.array([-1.0, 1.0, -1.0])        # skel_world <-> addb_lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial")
    ap.add_argument("--out", help="clip npz in MyoSuite motion-clip format")
    ap.add_argument("--video", help="mp4 (set MUJOCO_GL=egl on a headless machine)")
    ap.add_argument("--belt", default="stored", choices=["stored", "framewise", "none"],
                    help="treadmill trials: remove belt motion with the velocity stored in the trial (default), "
                         "with a per-frame estimate from the stance foot, or not at all")
    args = ap.parse_args()

    # ---- block 1: load ---------------------------------------------------------------------------
    D = np.load(args.trial, allow_pickle=True)
    Q = np.asarray(D["q_gt"], float)                               # [T, 37], rad / m, frame addb_lab
    if not np.isfinite(Q).all():
        raise SystemExit(f"q_gt has {(~np.isfinite(Q).all(1)).sum()} non-finite frames; cut the trial first")
    T, fps = len(Q), float(D["fps_true_grid"])
    dt = 1.0 / fps
    names = [OPENSIM[str(n)] for n in D["dof_names"]]
    col = {n: i for i, n in enumerate(names)}
    m, _, _ = compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())
    d = mujoco.MjData(m)
    couplings = JointCouplings.from_model(m)
    adr = lambda joint: int(m.jnt_qposadr[m.joint(joint).id])
    xmat = lambda body: d.xmat[m.body(body).id].reshape(3, 3).copy()

    def fk(q):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)

    fk(m.qpos0)
    P0, T0 = xmat("pelvis"), xmat("torso")
    C = np.round(P0)                           # OpenSim axes (x forward, y up, z right) -> model world
    print(f"[data] {Path(args.trial).name}: {T} frames at {fps:.2f} Hz; model nq={m.nq}, {m.nu} muscles")

    # ---- block 2: MyoSuite converter (root, same-named joints, coupled joints) ---------------------
    motion = OpenSimMotion(time=np.arange(T) * dt, labels=tuple(names), values=Q, in_degrees=False)
    mapped = opensim_motion_to_qpos(m, motion, signs={})           # signs={}: Rajagopal knee is flexion-positive
    qpos = mapped.qpos.copy()
    print(f"[myosuite] {len(mapped.coordinates)} coordinates mapped by name; not mapped: {list(mapped.unmapped)}")

    # ---- block 3: forearm and wrist (different zero pose, names and one sign) -----------------------
    # Rajagopal: pro_sup = 0 is palm forward (supinated), + wrist_dev is ulnar.
    # MyoArm:    pro_sup = 0 is palm facing the thigh,    + deviation is radial.
    for s in "rl":
        qpos[:, adr(f"pro_sup_{s}")] = Q[:, col[f"pro_sup_{s}"]] - np.pi / 2
        qpos[:, adr(f"flexion_{s}")] = Q[:, col[f"wrist_flex_{s}"]]
        qpos[:, adr(f"deviation_{s}")] = -Q[:, col[f"wrist_dev_{s}"]]

    # ---- block 4: lumbar (one ball joint -> five coupled segments) --------------------------------
    target = Rot.from_euler("ZXY", Q[:, [col[f"lumbar_{n}"] for n in ("extension", "bending", "rotation")]]).as_matrix()
    lum = [adr(n) for n in ("flex_extension", "lat_bending", "axial_rotation")]
    q = m.qpos0.copy()

    def torso_in_pelvis(x):                    # rotation of the torso relative to the pelvis, from neutral
        q[lum] = x
        project_joint_couplings(couplings, q[None])
        fk(q)
        return xmat("pelvis").T @ xmat("torso") @ (P0.T @ T0).T

    x, worst = np.zeros(3), 0.0
    for t in range(T):
        for _ in range(6):                     # Newton on the rotation error, warm-started from the last frame
            R = torso_in_pelvis(x)
            err = Rot.from_matrix(R.T @ target[t]).as_rotvec()
            if np.linalg.norm(err) < 1e-6:
                break
            J = np.stack([Rot.from_matrix(R.T @ torso_in_pelvis(x + 1e-4 * e)).as_rotvec() / 1e-4
                          for e in np.eye(3)], axis=1)
            x = x + np.linalg.solve(J, err)
        worst = max(worst, np.linalg.norm(err))
        qpos[t, lum] = x
    print(f"[lumbar] worst IK residual {np.degrees(worst):.4f} deg")

    # ---- block 5: shoulder (flex/add/rot -> plane of elevation / elevation / axial rotation) -------
    fk(m.qpos0)
    for s, sx in (("r", 1.0), ("l", -1.0)):
        a3 = Q[:, [col[f"arm_{n}_{s}"] for n in ("flex", "add", "rot")]]
        R = (Rot.from_rotvec(a3[:, :1] * [0, 0, 1.0]) * Rot.from_rotvec(a3[:, 1:2] * [sx, 0, 0])
             * Rot.from_rotvec(a3[:, 2:] * [0, sx, 0])).as_matrix()          # humerus relative to torso
        yA = T0.T @ d.xaxis[m.joint(f"elv_angle_{s}").id]                    # model axes, in torso axes
        xA = T0.T @ d.xaxis[m.joint(f"shoulder_elv_{s}").id]
        xA = xA - yA * (xA @ yA)
        xA /= np.linalg.norm(xA)
        w = np.cross(xA, yA)
        u = R @ yA                                                           # where the humerus long axis points
        lo, hi = m.jnt_range[m.joint(f"elv_angle_{s}").id]                   # joint range of the plane of elevation,
        lo2, hi2 = m.jnt_range[m.joint(f"shoulder1_r2_{s}").id]              # narrowed by its coupled twin (= -plane)
        lo, hi = max(lo, -hi2), min(hi, -lo2)
        mid = 0.5 * (lo + hi)
        plane = np.arctan2(u @ xA, u @ w)
        plane = (plane - mid + np.pi) % (2 * np.pi) - np.pi + mid            # the turn that contains the joint range
        outside = (plane < lo) | (plane > hi)
        # Outside the range the model cannot hold this plane. Near either end of the excluded arc use the nearest
        # limit; across the middle 70% of the arc slide from one limit to the other through the allowed range, so the
        # joint value stays continuous when the arm swings through the excluded arc (it is near vertical there).
        arc = lo + 2 * np.pi - hi
        along = ((plane - hi) % (2 * np.pi)) / arc                           # 0 at the upper limit, 1 at the lower limit
        plane = np.where(outside, hi + np.clip((along - 0.15) / 0.7, 0, 1) * (lo - hi), plane)
        hanging = np.sin(np.arccos(np.clip(u @ yA, -1, 1))) < np.sin(np.deg2rad(2.0))   # plane undefined: arm hangs straight
        idx = np.arange(T)
        plane = np.interp(idx, idx[~hanging], plane[~hanging]) if (~hanging).any() else np.full(T, np.clip(0.0, lo, hi))
        tilt = np.cos(plane)[:, None] * w + np.sin(plane)[:, None] * xA      # direction the arm tilts in for this plane
        elev = np.arctan2(np.clip(np.sum(u * tilt, 1), 0, None), u @ yA)     # best elevation within that plane
        S = (Rot.from_rotvec(plane[:, None] * yA) * Rot.from_rotvec(elev[:, None] * xA)
             * Rot.from_rotvec(-plane[:, None] * yA)).as_matrix()
        v = (np.swapaxes(S, 1, 2) @ R) @ xA
        rot = np.arctan2(v @ np.cross(yA, xA), v @ xA)
        got = S @ Rot.from_rotvec(rot[:, None] * yA).as_matrix()
        change = np.degrees(np.linalg.norm(Rot.from_matrix(np.swapaxes(got, 1, 2) @ R).as_rotvec(), axis=1))
        print(f"[shoulder] {s}: plane of elevation outside the joint range in {outside.sum()} of {T} frames; "
              f"brought inside, humerus orientation changed by mean {change.mean():.2f} deg, max {change.max():.2f} deg")
        qpos[:, adr(f"elv_angle_{s}")] = plane
        qpos[:, adr(f"shoulder_elv_{s}")] = elev
        qpos[:, adr(f"shoulder_rot_{s}")] = rot

    project_joint_couplings(couplings, qpos)   # scapula, clavicle, lumbar segments follow the new values

    # ---- block 6: subject size and floor -----------------------------------------------------------
    jn = [str(n) for n in D["addb_joint_names"]]
    A = np.asarray(D["addb_joints_dyn"], float) * FLIP_XZ          # joint centres, skel_world -> addb_lab
    jc = lambda name: A[:, jn.index(name)]
    seg = lambda a, b: np.median(np.linalg.norm(jc(a) - jc(b), axis=1))
    subject_leg = 0.5 * sum(seg(f"hip_{s}", f"walker_knee_{s}") + seg(f"walker_knee_{s}", f"ankle_{s}") for s in "rl")
    fk(m.qpos0)
    bpos = lambda b: d.xpos[m.body(b).id]
    model_leg = np.linalg.norm(bpos("femur_r") - bpos("tibia_r")) + np.linalg.norm(bpos("tibia_r") - bpos("talus_r"))
    scale = model_leg / subject_leg
    pelvis = Q[:, [col["pelvis_tx"], col["pelvis_ty"], col["pelvis_tz"]]]
    qpos[:, :3] = (pelvis * scale) @ C.T
    qpos[:, :2] -= qpos[0, :2]

    sole = [g for g in range(m.ngeom) if m.geom_contype[g]
            and m.body(m.geom_bodyid[g]).name in ("calcn_r", "toes_r", "calcn_l", "toes_l")]
    feet = [m.body("calcn_r").id, m.body("calcn_l").id]            # ForceBody contact order: right, left
    body_ids = {b: m.body(b).id for b in JOINT_BODY.values()}
    low, foot_xy, body_xyz = np.zeros(T), np.zeros((T, 2, 2)), np.zeros((T, m.nbody, 3))
    for t in range(T):
        fk(qpos[t])
        zs = []
        for g in sole:
            Rg, size = d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
            reach = (size[0] + abs(Rg[2, 2]) * size[1] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
                     else np.sqrt(np.sum((Rg[2] * size) ** 2)))    # capsule, else ellipsoid
            zs.append(d.geom_xpos[g, 2] - reach)
        low[t], foot_xy[t], body_xyz[t] = min(zs), d.xpos[feet, :2], d.xpos
    shift = -np.percentile(low, 5.0)
    qpos[:, 2] += shift
    print(f"[size] leg {subject_leg:.3f} m (subject) / {model_leg:.3f} m (model): translation x{scale:.3f}; "
          f"z shift {shift:+.3f} m, sole height {100*(low.min()+shift):+.1f} .. {100*(low.max()+shift):+.1f} cm")

    # ---- block 7: treadmill belt -------------------------------------------------------------------
    flag = lambda key: str(np.asarray(D[key]).reshape(-1)[0]).lower() if key in D.files else ""
    if flag("setting") == "treadmill" and args.belt != "none":
        foot_v = np.gradient(foot_xy, dt, axis=0)                  # [T, foot, xy] in the model world
        load = np.nan_to_num(np.asarray(D["grf"], float)[:, :2, 1])            # vertical force, right / left
        stance = load > 0.2 * float(D["mass_kg_dynamics"]) * 9.81
        single = stance & ~stance[:, ::-1]                         # one foot flat on the belt
        slip = lambda v: float(np.median(np.linalg.norm(foot_v - v, axis=-1)[stance]))
        # stored: belt surface velocity in skel_world -> addb_lab -> model world
        belt = np.array([float(D["belt_vx_mps"]), 0.0, float(D["belt_vz_mps"])]) * FLIP_XZ
        v_stored = np.repeat(((C @ belt)[:2] * scale)[None], T, 0)
        # framewise: velocity of the single-support foot, gaps interpolated, 0.5 s moving average
        wgt = single[..., None].astype(float)
        has, idx = single.any(1), np.arange(T)
        raw = (foot_v * wgt).sum(1) / np.maximum(wgt.sum(1), 1e-9)
        k = int(round(0.5 * fps)) | 1
        v_frame = np.stack([np.convolve(np.pad(np.interp(idx, idx[has], raw[has, i]), k // 2, mode="edge"),
                                        np.ones(k) / k, mode="valid") for i in range(2)], 1)
        print(f"[belt] stored {np.linalg.norm(belt):.3f} m/s; framewise estimate "
              f"{np.linalg.norm(v_frame, axis=1).min()/scale:.3f} .. {np.linalg.norm(v_frame, axis=1).max()/scale:.3f} m/s; "
              f"belt_mode '{flag('belt_mode')}'")
        print(f"[belt] stance-foot slip (median m/s): none {slip(0.0):.3f}, stored {slip(v_stored[:, None]):.3f}, "
              f"framewise {slip(v_frame[:, None]):.3f}")
        if slip(v_stored[:, None]) > slip(0.0):
            print("[belt] WARNING: stored belt velocity increases the slip; its frame or sign differs in this trial")
        v_use = v_stored if args.belt == "stored" else v_frame
        qpos[:, :2] -= np.cumsum(v_use, axis=0) * dt - v_use[0] * dt
        print(f"[belt] applied '{args.belt}': root travels {np.linalg.norm(qpos[-1, :2] - qpos[0, :2]):.2f} m in {T*dt:.1f} s")

    # ---- block 8: check against the joint centres shipped with ForceBody ---------------------------
    print("[check] bone direction, model vs ForceBody joint centres (mean deg):")
    for a, b in SEGMENTS:
        u = (jc(b) - jc(a)) @ C.T
        v = body_xyz[:, body_ids[JOINT_BODY[b]]] - body_xyz[:, body_ids[JOINT_BODY[a]]]
        cosang = np.sum(u * v, 1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
        print(f"         {a + ' -> ' + b:30s} {np.degrees(np.arccos(np.clip(cosang, -1, 1))).mean():5.1f}")
    for j in range(m.njnt):
        if m.jnt_limited[j] and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            vq, (lo, hi) = qpos[:, m.jnt_qposadr[j]], m.jnt_range[j]
            out = np.mean((vq < lo - 1e-3) | (vq > hi + 1e-3))
            if out > 0.05:
                print(f"[limits] {m.joint(j).name}: {100*out:.0f}% of frames outside [{np.degrees(lo):.0f}, {np.degrees(hi):.0f}] deg")

    # ---- block 9: save ---------------------------------------------------------------------------
    if args.out:
        sites = tuple(default_mimic_fullbody_config().mimic_site_names)
        clip = motion_clip_from_states(m, qpos, dt, sites)        # adds qvel and the mimic-site trajectories
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, qpos=clip.qpos, qvel=clip.qvel, site_xpos=clip.site_xpos,
                 site_names=np.asarray(sites, dtype=str), frequency=np.float64(fps),
                 source_trial=np.asarray(Path(args.trial).name), translation_scale=np.float64(scale),
                 belt=np.asarray(args.belt))
        print(f"[save] {out}: qpos {clip.qpos.shape}, qvel {clip.qvel.shape}, {len(sites)} sites")

    # ---- block 10: video -------------------------------------------------------------------------
    if args.video:
        import imageio.v2 as imageio
        m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
        renderer = mujoco.Renderer(m, height=720, width=1280)
        cam = mujoco.MjvCamera()
        cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, m.body("pelvis").id
        cam.distance, cam.elevation = 3.2, -10.0
        travel = qpos[-1, :2] - qpos[0, :2]
        fk(qpos[0])
        heading = travel if np.linalg.norm(travel) > 0.5 else xmat("pelvis")[:2, 0]   # pelvis x axis = forward
        cam.azimuth = float(np.degrees(np.arctan2(heading[1], heading[0]))) + 110.0
        step = max(1, int(round(fps / 25.0)))
        out = Path(args.video)
        out.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(out, fps=fps / step, macro_block_size=None) as writer:
            for t in range(0, T, step):
                d.qpos[:] = qpos[t]
                mujoco.mj_forward(m, d)
                renderer.update_scene(d, camera=cam)
                writer.append_data(renderer.render())
        renderer.close()
        print(f"[video] {out}: {len(range(0, T, step))} frames at {fps/step:.0f} fps")


if __name__ == "__main__":
    main()
