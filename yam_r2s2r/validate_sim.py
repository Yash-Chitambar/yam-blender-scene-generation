"""Compare the MuJoCo simulation with the Blender renders of the same episode.

    .venv/bin/python -m yam_r2s2r.validate_sim --episode work/episodes/sim0 --robot-only work/sim_val/robot_only \
        --renders work/sim_val/composite --out work/sim_val

Geometry: per camera and frame, the robot silhouette MuJoCo rendered (segmentation) against the one
Blender rendered from the bundle (alpha of a robot-only pass). Same model, same joint angles, two
independent camera implementations (MuJoCo focal/principal-pixel camera on the link vs the bundle's
world_from_camera + K), so a high IoU says the cameras and kinematics agree.
Output: validation.json, side_by_side.mp4 (MuJoCo | one column per lighting recipe; rows = cameras),
stills.jpg.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CAMS = ("top", "right_wrist", "left_wrist")


def label(im, text):
    cv2.putText(im, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(im, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", required=True, type=Path)
    p.add_argument("--robot-only", required=True, type=Path)
    p.add_argument("--renders", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tile-width", type=int, default=400)
    a = p.parse_args()
    sim = json.loads((a.episode / "sim.json").read_text())
    report = {"episode": str(a.episode), "sim": {k: sim[k] for k in ("frames", "fps", "tracking_error_rad", "min_table_clearance_m")},
              "silhouette": {}}
    for cam in CAMS:
        rdir = next((a.robot_only / cam).glob("*/render.json")).parent
        frames = json.loads((rdir / "render.json").read_text())["frames"]
        ious, shifts, edge = [], [], []
        for f in frames:
            mj = cv2.imread(str(a.episode / "mujoco" / cam / f"{f:06d}_robot.png"), 0) > 127
            bl = cv2.imread(str(rdir / f"{f:06d}.png"), cv2.IMREAD_UNCHANGED)[..., 3] > 127
            if not (mj.any() and bl.any()):
                continue
            ious.append((mj & bl).sum() / (mj | bl).sum())
            cm, cb = np.argwhere(mj).mean(0), np.argwhere(bl).mean(0)
            shifts.append(float(np.linalg.norm(cm - cb)))
            # symmetric mean distance between the two outlines, in pixels
            em = cv2.Canny(mj.astype(np.uint8) * 255, 50, 150) > 0
            eb = cv2.Canny(bl.astype(np.uint8) * 255, 50, 150) > 0
            dm = cv2.distanceTransform((~em).astype(np.uint8), cv2.DIST_L2, 3)
            db = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3)
            edge.append(float((db[em].mean() + dm[eb].mean()) / 2))
        report["silhouette"][cam] = {"frames": len(ious), "iou_mean": float(np.mean(ious)), "iou_min": float(np.min(ious)),
                                     "centroid_shift_px_mean": float(np.mean(shifts)), "outline_distance_px_mean": float(np.mean(edge)),
                                     "outline_distance_px_max": float(np.max(edge))}
    recipes = sorted(d.name for d in (a.renders / a.episode.name).iterdir() if d.is_dir())
    order = [r for key in ("day", "afternoon", "night") for r in recipes if key in r] or recipes
    n = sim["frames"]
    tw = a.tile_width
    a.out.mkdir(parents=True, exist_ok=True)
    writer, stills = None, []
    for f in range(n):
        rows = []
        for cam in CAMS:
            mj = cv2.imread(str(a.episode / "mujoco" / cam / f"{f:06d}.png"))
            th = round(tw * mj.shape[0] / mj.shape[1])
            tiles = [label(cv2.resize(mj, (tw, th)), f"MuJoCo {cam}")]
            for r in order:
                im = cv2.imread(str(a.renders / a.episode.name / r / cam / f"{f:06d}.png"))
                tiles.append(label(cv2.resize(im, (tw, th)), "Blender " + r.replace("light_", "")))
            rows.append(np.hstack(tiles))
        frame = np.vstack(rows)
        if writer is None:
            writer = cv2.VideoWriter(str(a.out / "side_by_side.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), sim["fps"],
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if f in (n // 6, n // 2, int(n * 0.62)):
            stills.append(frame)
    writer.release()
    cv2.imwrite(str(a.out / "stills.jpg"), np.vstack(stills))
    report["recipes"] = order
    report["video"] = str(a.out / "side_by_side.mp4")
    (a.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["silhouette"], indent=1))


if __name__ == "__main__":
    main()
