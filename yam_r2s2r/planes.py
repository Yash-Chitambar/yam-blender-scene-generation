"""Planar background surfaces (walls) from the scan: fit a vertical plane to the SfM points, project the
training images onto it, and add it to env.json as a textured, relightable surface.

    .venv/bin/python -m yam_r2s2r.planes wall --dataset work/dataset --env work/env --name back_wall \
        --region 0.45 -1.5 -0.05 1.2 2.2 1.3          # xmin ymin zmin xmax ymax zmax: where the wall's points are

env.json gets  surfaces: [{name, texture, origin_m, u_axis, v_axis, size_m [u, v], covered_fraction}]
(a rectangle: origin + a*u_axis + b*v_axis, texture u along u_axis, v along v_axis). blender/render_dr.py
builds each one as a textured quad that Cycles lights like the table.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def fit_vertical_plane(pts, iters=2000, tol=0.015, seed=0):
    """RANSAC plane with a horizontal normal. Returns (normal, d) with normal . p = d, and the inlier mask."""
    rng = np.random.default_rng(seed)
    best = (0, None, None)
    xy = pts[:, :2]
    for _ in range(iters):
        a, b = xy[rng.choice(len(xy), 2, replace=False)]
        t = b - a
        if np.linalg.norm(t) < 0.1:
            continue
        n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
        inl = np.abs(xy @ n - a @ n) < tol
        if inl.sum() > best[0]:
            best = (int(inl.sum()), n, inl)
    _, n, inl = best
    c = xy[inl].mean(0)
    w, v = np.linalg.eigh(np.cov((xy[inl] - c).T))
    n = v[:, 0]
    inl = np.abs((xy - c) @ n) < tol
    return np.array([n[0], n[1], 0.0]), float(c @ n), inl


def mosaic(dataset, frames, origin, eu, ev, size, ppm, percentile=40, min_views=2, min_incidence=20):
    """Per-pixel percentile of the training images projected on the rectangle origin + a eu + b ev."""
    W, H = int(round(size[0] * ppm)), int(round(size[1] * ppm))
    normal = np.cross(eu, ev)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    # texture pixel -> world: image row 0 is the top (b = size_v)
    P = origin[None, None] + (uu[..., None] / ppm) * eu + ((H - 1 - vv[..., None]) / ppm) * ev
    stack = np.zeros((len(frames), H, W, 3), np.uint8)
    seen = np.zeros((len(frames), H, W), bool)
    for i, f in enumerate(frames):
        img = cv2.imread(str(dataset / f["image"]))
        mask = cv2.imread(str(dataset / f["mask"]), 0)
        K = np.asarray(f["K"])
        Twc = np.asarray(f["world_from_camera"])
        pc = (P - Twc[:3, 3]) @ Twc[:3, :3]
        z = pc[..., 2]
        with np.errstate(all="ignore"):
            mx = (K[0, 0] * pc[..., 0] / z + K[0, 2]).astype(np.float32)
            my = (K[1, 1] * pc[..., 1] / z + K[1, 2]).astype(np.float32)
        ok = (z > 0.08) & (mx >= 0) & (mx < f["width"] - 1) & (my >= 0) & (my < f["height"] - 1)
        mx[~ok], my[~ok] = -1, -1
        ray = P - Twc[:3, 3]
        inc = np.degrees(np.arcsin(np.abs(ray @ normal) / np.linalg.norm(ray, axis=-1)))
        m = cv2.remap(mask, mx, my, cv2.INTER_NEAREST, borderValue=0)
        stack[i] = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0)
        seen[i] = ok & (m > 0) & (inc > min_incidence)
    count = seen.sum(0)
    values = np.where(seen[..., None], stack.astype(np.float32), np.nan)
    with np.errstate(all="ignore"):
        tex = np.nan_to_num(np.nanpercentile(values, percentile, axis=0)).astype(np.uint8)
    covered = count >= min_views
    if covered.any():
        tex[~covered] = np.median(tex[covered], axis=0).astype(np.uint8)
        tex = cv2.inpaint(tex, cv2.erode((~covered).astype(np.uint8) * 255, np.ones((3, 3), np.uint8)), 5, cv2.INPAINT_TELEA)
    return tex, covered


def cmd_wall(a):
    env_path = a.env / "env.json"
    env = json.loads(env_path.read_text())
    bench = json.loads((a.dataset / "benchmark.json").read_text())
    pts = np.load(env["points"])["points"]
    lo, hi = np.array(a.region[:3]), np.array(a.region[3:])
    sel = pts[((pts >= lo) & (pts <= hi)).all(1)]
    if len(sel) < 50:
        raise SystemExit(f"only {len(sel)} SfM points in the region")
    n, d, inl = fit_vertical_plane(sel)
    cams = np.array([np.asarray(f["world_from_camera"])[:3, 3] for f in bench["frames"]])
    if (cams @ n - d).mean() > 0:          # normal points away from the cameras, into the wall
        n, d = -n, -d
    eu = np.cross([0, 0, 1.0], -n)         # along the wall, so that (eu, ev=up, toward the cameras) is right-handed
    eu /= np.linalg.norm(eu)
    ev = np.array([0, 0, 1.0])
    p = sel[inl]
    s = p @ eu
    s0, s1 = (a.span if a.span else (float(np.quantile(s, 0.01)) - 0.15, float(np.quantile(s, 0.99)) + 0.15))
    z0 = a.bottom if a.bottom is not None else env["table_z"] - 0.02
    z1 = a.top if a.top is not None else float(np.quantile(p[:, 2], 0.995)) + 0.1
    origin = n * d + eu * s0 + ev * z0
    frames = [f for f in bench["frames"] if f["split"] == "train"]
    tex, covered = mosaic(a.dataset, frames, origin, eu, ev, (s1 - s0, z1 - z0), a.px_per_m)
    name = a.name
    cv2.imwrite(str(a.env / f"{name}.png"), tex)
    surf = {"name": name, "texture": f"{name}.png", "origin_m": origin.round(5).tolist(), "u_axis": eu.round(6).tolist(),
            "v_axis": ev.tolist(), "size_m": [round(s1 - s0, 4), round(z1 - z0, 4)], "normal_toward_cameras": (-n).round(6).tolist(),
            "covered_fraction": float(covered.mean()), "plane_inliers": int(inl.sum()), "plane_points_in_region": int(len(sel)),
            "roughness": 0.7}
    env["surfaces"] = [x for x in env.get("surfaces", []) if x["name"] != name] + [surf]
    env_path.write_text(json.dumps(env, indent=2) + "\n")
    print(json.dumps(surf))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("wall")
    c.add_argument("--dataset", required=True, type=Path)
    c.add_argument("--env", required=True, type=Path, help="folder with env.json (from reconstruct texture --sfm)")
    c.add_argument("--name", default="back_wall")
    c.add_argument("--region", type=float, nargs=6, required=True)
    c.add_argument("--span", type=float, nargs=2, help="extent along the wall (m, along u_axis); default: from the points")
    c.add_argument("--bottom", type=float)
    c.add_argument("--top", type=float)
    c.add_argument("--px-per-m", type=float, default=400)
    a = p.parse_args()
    cmd_wall(a)


if __name__ == "__main__":
    main()
