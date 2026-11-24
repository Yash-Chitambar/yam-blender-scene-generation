"""Scan session -> r2s2r appearance dataset (the `benchmark.json` contract).

A session is what `yam_r2s2r.scan` writes (and what agent_data_rl's calib.py `collect` writes):

    <session>/<side>/NNN.jpg|png      raw 1280x1024 wrist frame, arm stationary
    <session>/<side>/NNN.json         {"q": [7 measured joints], "t": unix seconds, ...}
    <session>/session.json            optional: {"condition": "day_1400", "table_z": ..., "other_arm_q": {...}}

Output (paths relative to benchmark.json, same keys r2s2r's splat_cuda/nht/sfm loaders read):

    <out>/images/<cond>_<side>_NNN.png              undistorted colour (pinhole K kept)
    <out>/images/<cond>_<side>_NNN_background.png   255 = static scene (robot and invalid pixels removed)
    <out>/images/<cond>_<side>_NNN_foreground.png   255 = robot
    <out>/images/<cond>_<side>_NNN_valid.png        255 = pixel has real image data after undistortion
    <out>/benchmark.json                            frames: K, width, height, world_from_camera (OpenCV), split, ...
    <out>/views/<side>_wrist.json                   per-camera manifests for `r2s2r.appearance.sfm`
    <out>/model.mjb, model.json                     the YAM cell model the masks came from

    python -m yam_r2s2r.build_dataset --session S [S2 ...] --out work/dataset
"""
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from . import scene as scene_mod
from .common import SIDES, TABLE_Z, Kin, Rig, camera_in_world

MASK_DILATE_PX = 12
# The camera's own gripper fills the bottom of every wrist frame, and the real fingers (FlexPoint
# soft tips, webcam cable) are larger than the model's linear_4310 fingers: mask them generously.
OWN_GRIPPER_DILATE_PX = 60
ROBOT_GROUPS = (2,)


def _own_gripper_bodies(model, side):
    root = model.body(f"{side}/link_6").id
    out = set()
    for b in range(model.nbody):
        c = b
        while c > 0 and c != root:
            c = model.body_parentid[c]
        if c == root:
            out.add(b)
    return out


def robot_mask(model, data, qpos, T_world_cam, K, size, own_side=None, skip=("table",)):
    """Projected convex hulls of the robot's visual geoms, as seen by one camera."""
    w, h = size
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mask = np.zeros((h, w), np.uint8)
    own = np.zeros((h, w), np.uint8)
    own_bodies = _own_gripper_bodies(model, own_side) if own_side else set()
    R, t = T_world_cam[:3, :3].T, -T_world_cam[:3, :3].T @ T_world_cam[:3, 3]
    for g in range(model.ngeom):
        if model.geom_group[g] not in ROBOT_GROUPS or model.geom(g).name in skip or model.geom_rgba[g, 3] == 0:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        m = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        pw = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        pc = pw @ R.T + t
        pc = pc[pc[:, 2] > 0.015]
        if len(pc) < 3:
            continue
        uv = pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        if (uv[:, 0].max() < 0 or uv[:, 0].min() > w or uv[:, 1].max() < 0 or uv[:, 1].min() > h):
            continue
        hull = cv2.convexHull(np.clip(uv, -4 * w, 5 * w).astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(own if model.geom_bodyid[g] in own_bodies else mask, hull, 255)
    for m_, px in ((mask, MASK_DILATE_PX), (own, OWN_GRIPPER_DILATE_PX)):
        if px and m_.any():
            m_[:] = cv2.dilate(m_, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1,) * 2))
    return np.maximum(mask, own)


def load_session(path):
    path = Path(path)
    meta = json.loads((path / "session.json").read_text()) if (path / "session.json").exists() else {}
    views = []
    for side in SIDES:
        for j in sorted((path / side).glob("*.json")) if (path / side).is_dir() else []:
            img = next((j.with_suffix(e) for e in (".png", ".jpg") if j.with_suffix(e).exists()), None)
            if img is None:
                continue
            rec = json.loads(j.read_text())
            if "q" not in rec:
                continue
            views.append({"side": side, "id": j.stem, "image": img, "q": rec["q"], "t": rec.get("t"),
                          "other_q": rec.get("other_q")})
    for j in sorted((path / "top").glob("*.json")) if (path / "top").is_dir() else []:
        if j.with_suffix(".png").exists():
            rec = json.loads(j.read_text())
            views.append({"side": "top", "id": j.stem, "image": j.with_suffix(".png"), "q": None, "t": rec.get("t"),
                          "other_q": None})
    return meta, views


def build(sessions, out, rig, solves=None, val_every=7, test_every=0):
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=False)
    (out / "views").mkdir()
    for side, path in (solves or {}).items():
        rig.use_solve(side, path)
    kin = Kin()
    arms = tuple(rig.sides)
    _, model, info = scene_mod.build(rig, arms, table=False)
    data = mujoco.MjData(model)
    mujoco.mj_saveModel(model, str(out / "model.mjb"))
    (out / "model.json").write_text(json.dumps(info, indent=2) + "\n")
    frames, conditions = [], []
    for si, session in enumerate(sessions):
        meta, views = load_session(session)
        cond = meta.get("condition") or Path(session).name
        conditions.append({"condition": cond, "session": str(Path(session).resolve()), "meta": meta,
                           "views": len(views)})
        rest = {s: np.asarray(v, float) for s, v in {**(meta.get("other_arm_q") or {}), **(meta.get("rest_q") or {})}.items()}
        for vi, v in enumerate(views):
            side = v["side"]
            raw = cv2.imread(str(v["image"]))
            h, w = raw.shape[:2]
            if side == "top":   # static rectified ZED: both arms at their rest pose, no distortion
                K, _ = rig.top_K("HD1080")
                T = rig.T_world_top
                q_meas = {o: rest[o] for o in arms if o in rest}
                qpos = scene_mod.qpos_from_measured(rig, model, arms, q_meas)
                fg = robot_mask(model, data, qpos, T, K, (w, h))
                valid = np.full((h, w), 255, np.uint8)
                bg = np.where(fg == 0, 255, 0).astype(np.uint8)
                tag = f"{cond}_top_{v['id']}"
                cv2.imwrite(str(out / "images" / f"{tag}.png"), raw)
                for suffix, m in (("background", bg), ("foreground", fg), ("valid", valid)):
                    cv2.imwrite(str(out / "images" / f"{tag}_{suffix}.png"), m)
                frames.append({
                    "image": f"images/{tag}.png", "mask": f"images/{tag}_background.png",
                    "foreground_mask": f"images/{tag}_foreground.png", "valid_mask": f"images/{tag}_valid.png",
                    "episode": cond, "camera": "top", "group": f"e{si}_g{vi:03d}", "split": "train",
                    "timestamp_s": float(v["t"] or vi), "frame_index": vi, "K": K.tolist(), "width": w, "height": h,
                    "world_from_camera": T.tolist(), "qpos": qpos.tolist(),
                    "q_measured": {k: x.tolist() for k, x in q_meas.items()}, "other_arm_masked": len(q_meas) == len(arms),
                    "background_fraction": float((bg > 0).mean()), "saturated_fraction": float((raw.min(-1) > 250).mean()),
                    "sharpness_laplacian_var": None})
                continue
            K, dist = rig.K[side], rig.dist[side]
            gain = ((meta.get("cameras") or {}).get(side) or {}).get("color_gain_bgr")
            if gain:
                raw = np.clip(raw.astype(np.float32) * np.asarray(gain, np.float32), 0, 255).astype(np.uint8)
            image = cv2.undistort(raw, K, dist)
            valid = cv2.undistort(np.full((h, w), 255, np.uint8), K, dist, None, K)
            valid = cv2.erode((valid > 250).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
            T = camera_in_world(rig, kin, side, v["q"])
            q_meas = {side: np.asarray(v["q"], float)}
            for o in arms:
                if o != side:
                    oq = v["other_q"] if v["other_q"] is not None else rest.get(o)
                    if oq is not None:
                        q_meas[o] = np.asarray(oq, float)
            qpos = scene_mod.qpos_from_measured(rig, model, arms, q_meas)
            fg = robot_mask(model, data, qpos, T, K, (w, h), own_side=side)
            bg = np.where((fg == 0) & (valid > 0), 255, 0).astype(np.uint8)
            tag = f"{cond}_{side}_{v['id']}"
            cv2.imwrite(str(out / "images" / f"{tag}.png"), image)
            cv2.imwrite(str(out / "images" / f"{tag}_background.png"), bg)
            cv2.imwrite(str(out / "images" / f"{tag}_foreground.png"), fg)
            cv2.imwrite(str(out / "images" / f"{tag}_valid.png"), valid)
            n = len([f for f in frames if f["camera"] == f"{side}_wrist" and f["episode"] == cond])
            split = ("test" if test_every and n % test_every == test_every - 1 else
                     "validation" if val_every and n % val_every == val_every - 1 else "train")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            frames.append({
                "image": f"images/{tag}.png", "mask": f"images/{tag}_background.png",
                "foreground_mask": f"images/{tag}_foreground.png", "valid_mask": f"images/{tag}_valid.png",
                "episode": cond, "camera": f"{side}_wrist", "group": f"e{si}_g{vi:03d}", "split": split,
                "timestamp_s": float(v["t"]) if v["t"] is not None else float(vi), "frame_index": vi,
                "K": K.tolist(), "width": w, "height": h, "world_from_camera": T.tolist(),
                "qpos": qpos.tolist(), "q_measured": {k: x.tolist() for k, x in q_meas.items()},
                "other_arm_masked": len(q_meas) == len(arms),
                "background_fraction": float((bg > 0).mean()),
                "saturated_fraction": float((image.min(-1) > 250).mean()),
                "sharpness_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F)[bg > 0].var()) if (bg > 0).any() else None,
            })
    bench = {"schema_version": 1, "robot": "i2rt_yam", "world_frame": "right arm base, metres",
             "camera_convention": "opencv world_from_camera; images undistorted (pinhole K)",
             "table_z": rig.table_z, "rig_spec": str(rig.spec_path), "solves": {k: str(v) for k, v in (solves or {}).items()},
             "conditions": conditions, "frames": frames}
    (out / "benchmark.json").write_text(json.dumps(bench, indent=1) + "\n")
    for cam in sorted({f["camera"] for f in frames}):
        sel = []
        for f in frames:
            if f["camera"] == cam:
                g = dict(f)
                for key in ("image", "mask", "foreground_mask", "valid_mask"):
                    g[key] = "../" + g[key]
                sel.append(g)
        (out / "views" / f"{cam}.json").write_text(json.dumps(
            {"K": sel[0]["K"], "width": sel[0]["width"], "height": sel[0]["height"], "frames": sel}, indent=1) + "\n")
    summary = {"frames": len(frames), "by_camera": {c: sum(f["camera"] == c for f in frames) for c in {f["camera"] for f in frames}},
               "by_split": {s: sum(f["split"] == s for f in frames) for s in ("train", "validation", "test")}}
    print(json.dumps(summary))
    return bench


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--spec", type=Path)
    p.add_argument("--table-z", type=float, default=TABLE_Z)
    p.add_argument("--solve-right", type=Path, help="calib.py solve_right.json to use instead of the spec")
    p.add_argument("--solve-left", type=Path)
    p.add_argument("--val-every", type=int, default=7)
    a = p.parse_args()
    rig = Rig(*([a.spec] if a.spec else []), table_z=a.table_z)
    solves = {s: v for s, v in (("right", a.solve_right), ("left", a.solve_left)) if v}
    build(a.session, a.out, rig, solves, a.val_every)


if __name__ == "__main__":
    main()
