"""Joint trajectory of the YAM cell -> r2s2r Blender bundles, one per camera.

    python -m yam_r2s2r.make_bundle --out work/episodes/ep0 --from-run <agent_data_rl real run dir>
    python -m yam_r2s2r.make_bundle --out work/episodes/ep0 --from-npz traj.npz      # q[T, arms, 6|7] or qpos[T, 16], t|time_s|dt
    python -m yam_r2s2r.make_bundle --out work/episodes/demo --demo

Writes
    <out>/model.mjb, model.json           the cell (arms + table), world = right arm base
    <out>/episode.npz                     qpos[T, nq], time_s[T]  (r2s2r.appearance.bundle's trajectory contract)
    <out>/<camera>/camera.json            {width, height, K, world_from_camera [T,4,4]} OpenCV, undistorted pinhole
    <out>/<camera>/bundle/                geometry.npz, motion.npz, manifest.json  (r2s2r.appearance.bundle.export)
for cameras right_wrist, left_wrist (moving with the arm, calibrated hand-eye) and top (the ZED 2).
The bundle export is r2s2r's own code, imported from $R2S2R_APPEARANCE_ROOT/src.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from . import scene as scene_mod
from .common import R2S2R_APPEARANCE, SIDES, WRIST_RES, Kin, Rig


def load_trajectory(a, rig, model, arms):
    """-> qpos [T, nq] (model joints), time_s [T]"""
    if a.demo:
        t = np.arange(0, 6.0, 1 / a.fps)
        home = np.array([0, 1.047, 1.047, 0, 0, 0])
        q = np.zeros((len(t), len(arms), 7))
        for i, side in enumerate(arms):
            s = 1.0 if side == "right" else -1.0
            q[:, i, :6] = home
            q[:, i, 0] = 0.5 * s * np.sin(2 * np.pi * t / 6)
            q[:, i, 1] = 1.047 + 0.25 * np.sin(2 * np.pi * t / 3)
            q[:, i, 3] = -0.6 + 0.3 * np.cos(2 * np.pi * t / 3)
            q[:, i, 4] = 0.3 * s * np.sin(2 * np.pi * t / 2)
            q[:, i, 6] = 0.5 + 0.5 * np.sin(2 * np.pi * t / 2)
        frame = "model"
    elif a.from_run:
        d = np.load(Path(a.from_run) / "ticks.npz")
        q6, t = d["q"], d["t"]                       # [T, 2 (right, left), 6] measured, monotonic seconds
        grip = d["grip"] if "grip" in d.files else np.ones(q6.shape[:2])
        order = ("right", "left")
        q = np.zeros((len(t), len(arms), 7))
        for i, side in enumerate(arms):
            q[:, i, :6], q[:, i, 6] = q6[:, order.index(side)], np.clip(grip[:, order.index(side)], 0, 1)
        keep = np.r_[True, np.diff(t) > 0]
        q, t = q[keep], t[keep] - t[keep][0]
        frame = a.q_frame
    else:
        d = np.load(a.from_npz)
        t = d["time_s"] if "time_s" in d.files else d["t"] if "t" in d.files else np.arange(len(d[d.files[0]])) * float(d["dt"])
        if "qpos" in d.files:
            return np.asarray(d["qpos"], float), np.asarray(t, float)
        q = np.asarray(d["q"], float)
        if q.shape[-1] == 6:
            q = np.concatenate([q, np.ones((*q.shape[:-1], 1))], -1)
        frame = a.q_frame
    if a.stride > 1:
        q, t = q[:: a.stride], t[:: a.stride]
    if a.max_frames:
        q, t = q[: a.max_frames], t[: a.max_frames]
    lo, hi = model.jnt_range[6]
    qpos = np.zeros((len(t), model.nq))
    for i, side in enumerate(arms):
        off = rig.offsets[side] if frame == "measured" else 0
        qpos[:, i * scene_mod.ARM_NQ: i * scene_mod.ARM_NQ + 6] = q[:, i, :6] + off
        qpos[:, i * scene_mod.ARM_NQ + 6: i * scene_mod.ARM_NQ + 8] = (lo + q[:, i, 6:7] * (hi - lo))
    return qpos, np.asarray(t, float)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-run")
    src.add_argument("--from-npz")
    src.add_argument("--demo", action="store_true")
    p.add_argument("--q-frame", choices=["measured", "model"], default="measured",
                   help="measured = encoder joints (the rig's joint offsets are added); model = already Menagerie joints")
    p.add_argument("--arms", nargs="+", default=list(SIDES), choices=SIDES)
    p.add_argument("--cameras", nargs="+", default=["right_wrist", "left_wrist", "top"])
    p.add_argument("--wrist-width", type=int, default=640, help="wrist render width (native 1280x1024)")
    p.add_argument("--top-width", type=int, default=640, help="top render width (native 1920x1080)")
    p.add_argument("--fps", type=float, default=10.0, help="--demo only")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--spec", type=Path)
    p.add_argument("--table-z", type=float, default=-0.023)
    a = p.parse_args()
    rig = Rig(*([a.spec] if a.spec else []), table_z=a.table_z)
    arms = tuple(a.arms)
    _, model, info = scene_mod.build(rig, arms, table=True)
    a.out.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(a.out / "model.mjb"))
    (a.out / "model.json").write_text(json.dumps(info, indent=2) + "\n")
    qpos, t = load_trajectory(a, rig, model, arms)
    np.savez_compressed(a.out / "episode.npz", qpos=qpos, time_s=t)
    sys.path.insert(0, str(R2S2R_APPEARANCE / "src"))
    from r2s2r.appearance import bundle
    kin = Kin()
    for cam in a.cameras:
        side = cam.split("_")[0]
        if cam == "top":
            if rig.T_world_top is None:
                continue
            K, (w0, h0) = rig.top_K("HD1080")
            width = a.top_width
            poses = np.repeat(rig.T_world_top[None], len(t), 0)
        else:
            if side not in arms:
                continue
            K, (w0, h0), width = rig.K[side], WRIST_RES, a.wrist_width
            i = arms.index(side)
            q6 = qpos[:, i * scene_mod.ARM_NQ: i * scene_mod.ARM_NQ + 6]   # model joints: no offsets again
            poses = np.stack([rig.T_world_base[side] @ kin.fk(q) @ rig.T_link6_cam[side] for q in q6])
        s = width / w0
        Ks = K.copy()
        Ks[:2] *= s
        cdir = a.out / cam
        cdir.mkdir(exist_ok=True)
        (cdir / "camera.json").write_text(json.dumps(
            {"width": width, "height": int(round(h0 * s)), "K": Ks.tolist(), "world_from_camera": poses.tolist()}) + "\n")
        if (cdir / "bundle").exists():
            import shutil
            shutil.rmtree(cdir / "bundle")
        m = bundle.export(a.out / "model.mjb", a.out / "episode.npz", cdir / "bundle", camera_json=cdir / "camera.json")
        print(f"{cam}: {m['frames']} frames, {m['width']}x{m['height']}, {len(m['geoms'])} geoms")


if __name__ == "__main__":
    main()
