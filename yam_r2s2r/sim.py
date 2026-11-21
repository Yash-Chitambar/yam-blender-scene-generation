"""MuJoCo simulation of the YAM cell with the calibrated top and wrist cameras.

    MUJOCO_GL=egl .venv/bin/python -m yam_r2s2r.sim --out work/episodes/sim0 [--seconds 7 --fps 12]

Physics: the Menagerie YAM position actuators track a smooth two-arm reach / grasp / retract
trajectory (gravity, joint dynamics and the coupled fingers are simulated; the table is visual only).
Cameras are MuJoCo cameras with the real intrinsics (focal and principal point in pixels) mounted at
the calibrated poses: `right_wrist`, `left_wrist` on link_6 (hand-eye) and `top` (the ZED 2).

Writes, in the layout `make_bundle --from-npz` and the Blender stage read:
    <out>/sim.npz                    qpos[T,16] (simulated), ctrl[T,14], time_s[T]
    <out>/mujoco/<camera>/NNNNNN.png           MuJoCo RGB
    <out>/mujoco/<camera>/NNNNNN_robot.png     robot segmentation (255 = robot)
    <out>/sim.json                   tracking error, table clearance, camera model self-check
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from . import scene as scene_mod  # noqa: E402
from .build_dataset import robot_mask  # noqa: E402
from .common import WRIST_RES, Rig, pose_dict  # noqa: E402

PSIGN = [float(v) for v in os.environ.get('YAM_PSIGN', '-1 -1').split()]   # MuJoCo's principal offset is opposite to (c - centre) on both axes: checked against our pinhole projection
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])     # OpenCV optical axes -> MuJoCo camera axes (looks along -z, +y up)

# joint waypoints (model joints, rad) + gripper (0 closed .. 1 open), per phase; the left arm mirrors joint 1, 5 and 6
HOME = [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]
WAYPOINTS = [   # (time fraction, right arm q6, right gripper, left arm q6, left gripper)
    (0.00, HOME, 1.0, HOME, 1.0),
    (0.22, [0.35, 1.25, 1.00, -0.45, 0.25, 0.0], 1.0, [-0.35, 1.25, 1.00, -0.45, -0.25, 0.0], 1.0),
    (0.45, [0.30, 1.45, 0.95, -0.80, 0.20, 0.4], 1.0, [-0.30, 1.45, 0.95, -0.80, -0.20, -0.4], 0.6),
    (0.60, [0.30, 1.45, 0.95, -0.80, 0.20, 0.4], 0.1, [-0.30, 1.45, 0.95, -0.80, -0.20, -0.4], 0.6),
    (0.80, [-0.15, 1.20, 1.10, -0.50, -0.30, 0.0], 0.1, [0.15, 1.30, 1.05, -0.60, 0.30, 0.0], 0.1),
    (1.00, HOME, 0.8, HOME, 0.8),
]


def build(rig, arms, sizes):
    spec, _, _ = scene_mod.build(rig, arms, table=True)
    for side in arms:
        K = rig.K[side]
        w, h = sizes[f"{side}_wrist"]
        s = w / WRIST_RES[0]
        T = rig.T_link6_cam[side].copy()
        T[:3, :3] = T[:3, :3] @ CV_TO_MJ
        p = pose_dict(T)
        body = next(b for b in spec.bodies if b.name == f"{side}/link_6")
        body.add_camera(name=f"{side}_wrist", pos=p["pos"], quat=p["quat"], resolution=[w, h],
                        sensor_size=[w * 1e-6, h * 1e-6], focal_pixel=[K[0, 0] * s, K[1, 1] * s],
                        principal_pixel=[PSIGN[0] * (K[0, 2] * s - w / 2), PSIGN[1] * (K[1, 2] * s - h / 2)])
    if rig.T_world_top is not None and "top" in sizes:
        K, (w0, _) = rig.top_K("HD1080")
        w, h = sizes["top"]
        s = w / w0
        T = rig.T_world_top.copy()
        T[:3, :3] = T[:3, :3] @ CV_TO_MJ
        p = pose_dict(T)
        spec.worldbody.add_camera(name="top", pos=p["pos"], quat=p["quat"], resolution=[w, h],
                                  sensor_size=[w * 1e-6, h * 1e-6], focal_pixel=[K[0, 0] * s, K[1, 1] * s],
                                  principal_pixel=[PSIGN[0] * (K[0, 2] * s - w / 2), PSIGN[1] * (K[1, 2] * s - h / 2)])
    spec.worldbody.add_light(pos=[0.3, 0.3, 2.0], dir=[0, 0, -1], diffuse=[0.45, 0.45, 0.45])
    spec.worldbody.add_light(pos=[-0.5, 0.3, 1.5], dir=[0.5, 0, -1], diffuse=[0.2, 0.2, 0.2])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.ambient = [0.2, 0.2, 0.2]
    return spec.compile()


def reference(tfrac, arms):
    """Minimum-jerk blend between the waypoints -> ctrl (14) and the commanded arm joints."""
    for (t0, r0, g0, l0, h0), (t1, r1, g1, l1, h1) in zip(WAYPOINTS[:-1], WAYPOINTS[1:]):
        if tfrac <= t1:
            u = np.clip((tfrac - t0) / (t1 - t0), 0, 1)
            u = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            q = {"right": (1 - u) * np.array(r0) + u * np.array(r1), "left": (1 - u) * np.array(l0) + u * np.array(l1)}
            g = {"right": (1 - u) * g0 + u * g1, "left": (1 - u) * h0 + u * h1}
            return q, g
    return {"right": np.array(WAYPOINTS[-1][1]), "left": np.array(WAYPOINTS[-1][3])}, {"right": WAYPOINTS[-1][2], "left": WAYPOINTS[-1][4]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seconds", type=float, default=7.0)
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--wrist-width", type=int, default=640)
    p.add_argument("--top-width", type=int, default=640)
    p.add_argument("--table-z", type=float, default=-0.023)
    a = p.parse_args()
    rig = Rig(table_z=a.table_z)
    arms = ("right", "left")
    sizes = {f"{s}_wrist": (a.wrist_width, round(a.wrist_width * WRIST_RES[1] / WRIST_RES[0])) for s in arms}
    sizes["top"] = (a.top_width, round(a.top_width * 1080 / 1920))
    model = build(rig, arms, sizes)
    data = mujoco.MjData(model)
    lo, hi = model.jnt_range[6]
    nq = scene_mod.ARM_NQ

    def ctrl_at(tfrac):
        q, g = reference(tfrac, arms)
        c = np.zeros(model.nu)
        for i, s in enumerate(arms):
            c[i * 7: i * 7 + 6] = q[s]
            c[i * 7 + 6] = lo + g[s] * (hi - lo)
        return c, q

    # start at the first waypoint, settled
    c0, q0 = ctrl_at(0.0)
    for i, s in enumerate(arms):
        data.qpos[i * nq: i * nq + 6] = q0[s]
        data.qpos[i * nq + 6: i * nq + 8] = c0[i * 7 + 6]
    data.ctrl[:] = c0
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    data.time = 0.0
    robot_geoms = np.array([model.geom_bodyid[g] != 0 for g in range(model.ngeom)])
    renderers = {c: mujoco.Renderer(model, sizes[c][1], sizes[c][0]) for c in sizes}
    for c in sizes:
        (a.out / "mujoco" / c).mkdir(parents=True, exist_ok=True)
    n = int(round(a.seconds * a.fps))
    qpos, ctrl, times, track, clear, iou = [], [], [], [], [], {c: [] for c in sizes}
    verts = []
    for g in range(model.ngeom):
        if robot_geoms[g] and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] == 2:
            m = model.geom_dataid[g]
            verts.append((g, model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]]))
    for f in range(n):
        t_end = f / a.fps
        while data.time < t_end - 1e-9:
            data.ctrl[:], _ = ctrl_at(data.time / a.seconds)
            mujoco.mj_step(model, data)
        c, qref = ctrl_at(t_end / a.seconds)
        mujoco.mj_forward(model, data)
        qpos.append(data.qpos.copy())
        ctrl.append(c)
        times.append(t_end)
        track.append(max(np.abs(data.qpos[i * nq: i * nq + 6] - qref[s]).max() for i, s in enumerate(arms)))
        clear.append(min((v @ data.geom_xmat[g].reshape(3, 3).T[:, 2] + data.geom_xpos[g][2]).min() for g, v in verts) - a.table_z)
        for cam, r in renderers.items():
            r.update_scene(data, camera=cam)
            rgb = r.render()
            r.enable_segmentation_rendering()
            r.update_scene(data, camera=cam)
            seg = r.render()
            r.disable_segmentation_rendering()
            ids = seg[..., 0]
            robot = np.zeros(ids.shape, np.uint8)
            ok = ids >= 0
            robot[ok] = np.where(robot_geoms[ids[ok]] & (model.geom_group[ids[ok]] == 2), 255, 0)
            cv2.imwrite(str(a.out / "mujoco" / cam / f"{f:06d}.png"), rgb[..., ::-1])
            cv2.imwrite(str(a.out / "mujoco" / cam / f"{f:06d}_robot.png"), robot)
            if f % 6 == 0:   # camera model self-check: our own pinhole projection of the robot meshes vs MuJoCo's
                cid = model.camera(cam).id
                T = np.eye(4)
                T[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ CV_TO_MJ
                T[:3, 3] = data.cam_xpos[cid]
                w, h = sizes[cam]
                if cam == "top":
                    K, (w0, _) = rig.top_K("HD1080")
                else:
                    K, w0 = rig.K[cam.split("_")[0]], WRIST_RES[0]
                K = K.copy()
                K[:2] *= w / w0
                import yam_r2s2r.build_dataset as bd
                keep = (bd.MASK_DILATE_PX, bd.OWN_GRIPPER_DILATE_PX)
                bd.MASK_DILATE_PX = bd.OWN_GRIPPER_DILATE_PX = 0
                hull = robot_mask(model, mujoco.MjData(model), data.qpos.copy(), T, K, (w, h))
                bd.MASK_DILATE_PX, bd.OWN_GRIPPER_DILATE_PX = keep
                inside = (robot > 0) & (hull > 0)
                iou[cam].append(float(inside.sum() / max(((robot > 0) | (hull > 0)).sum(), 1)))   # MuJoCo robot pixels inside our hulls
    for r in renderers.values():
        r.close()
    qpos, ctrl, times = np.array(qpos), np.array(ctrl), np.array(times)
    np.savez_compressed(a.out / "sim.npz", qpos=qpos, ctrl=ctrl, time_s=times)
    report = {"frames": n, "fps": a.fps, "seconds": a.seconds, "timestep": model.opt.timestep, "cameras": {c: list(s) for c, s in sizes.items()},
              "tracking_error_rad": {"max": float(np.max(track)), "mean": float(np.mean(track))},
              "min_table_clearance_m": float(np.min(clear)),
              "mujoco_vs_projected_robot_iou": {c: float(np.mean(v)) for c, v in iou.items()},
              "note": "the last entry checks MuJoCo's camera (pose + focal/principal point) against the pinhole K and "
                      "world_from_camera the dataset and Blender bundles use; IoU of MuJoCo's robot mask with our projected convex hulls (hulls are a little larger than the meshes)"}
    (a.out / "sim.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
