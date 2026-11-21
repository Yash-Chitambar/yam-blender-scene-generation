"""Check the camera pose chain of a session against a checkerboard lying on the table.

Works on sessions whose NNN.json carry `wrist_corners` (calib.py `collect` runs). The corners are
triangulated across all views with the FK camera poses; the reprojection error and the spread of the
triangulated board say how good the poses are for reconstruction, and the board height gives table_z.

    python -m yam_r2s2r.check_poses --session <run> [--solve-right ... --solve-left ...]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .common import SIDES, Kin, Rig, camera_in_world

SQUARE = 0.015


def triangulate(Ps, uvs):
    A = []
    for P, (u, v) in zip(Ps, uvs):
        A += [u * P[2] - P[0], v * P[2] - P[1]]
    X = np.linalg.svd(np.asarray(A))[2][-1]
    return X[:3] / X[3]


def check(session, rig, sides=SIDES):
    kin = Kin()
    obs = []   # (side, P 3x4 normalised, corners normalised Nx2, K)
    for side in sides:
        for j in sorted((Path(session) / side).glob("*.json")):
            rec = json.loads(j.read_text())
            if not rec.get("wrist_corners"):
                continue
            c = np.asarray(rec["wrist_corners"], np.float64).reshape(-1, 1, 2)
            n = cv2.undistortPoints(c, rig.K[side], rig.dist[side]).reshape(-1, 2)
            T = camera_in_world(rig, kin, side, rec["q"])
            obs.append((side, np.linalg.inv(T)[:3], n, rig.K[side]))
    if len(obs) < 2:
        raise SystemExit("needs at least two views with wrist_corners")
    npts = len(obs[0][2])
    X = np.array([triangulate([o[1] for o in obs], [o[2][i] for o in obs]) for i in range(npts)])
    out = {"views": len(obs), "corners": npts}
    for side in sides:
        errs = []
        for s, P, n, K in obs:
            if s != side:
                continue
            pc = X @ P[:, :3].T + P[:, 3]
            errs.append(np.linalg.norm((pc[:, :2] / pc[:, 2:] - n) * [K[0, 0], K[1, 1]], axis=1))
        if errs:
            e = np.concatenate(errs)
            out[f"reproj_px_{side}"] = {"mean": float(e.mean()), "p95": float(np.quantile(e, 0.95)), "views": len(errs)}
    c = X - X.mean(0)
    normal = np.linalg.svd(c)[2][-1]
    out["board_center_world_m"] = X.mean(0).round(4).tolist()
    out["board_plane_rms_mm"] = float(np.sqrt(((c @ normal) ** 2).mean()) * 1e3)
    out["board_tilt_from_table_deg"] = float(np.degrees(np.arccos(abs(normal[2]))))
    # 8 x 7 inner corners, row-major, 15 mm squares
    d = np.linalg.norm(np.diff(X.reshape(7, 8, 3), axis=1), axis=-1)
    out["square_size_mm"] = {"mean": float(d.mean() * 1e3), "true": SQUARE * 1e3}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True, type=Path)
    p.add_argument("--spec", type=Path)
    p.add_argument("--solve-right", type=Path)
    p.add_argument("--solve-left", type=Path)
    a = p.parse_args()
    rig = Rig(*([a.spec] if a.spec else []))
    for s, v in (("right", a.solve_right), ("left", a.solve_left)):
        if v:
            rig.use_solve(s, v)
    print(json.dumps(check(a.session, rig), indent=1))


if __name__ == "__main__":
    main()
