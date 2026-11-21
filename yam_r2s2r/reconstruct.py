"""Geometry of the scanned environment from the dataset: metric SfM points and the table texture.

    .venv/bin/python -m yam_r2s2r.reconstruct sfm     --dataset work/dataset --out work/env/sfm
    .venv/bin/python -m yam_r2s2r.reconstruct texture --dataset work/dataset --out work/env [--sfm work/env/sfm]

`sfm` runs r2s2r's own `appearance.sfm` (COLMAP features, triangulation with the FK camera poses held
fixed, so the points stay in the robot's metric world frame) and then writes the `points.npz`
(`points`, `colors` in [0,1]) that r2s2r's `splat_cuda` / `nht` training takes as `--points`.
r2s2r has no script for that last conversion.

`texture` projects every training image onto the table plane and takes the per-pixel median: the
robot, its shadow and moving highlights drop out. The result is a top-down table image plus its
placement in the world, which the Blender stage lights with Cycles (so the table relights with the
randomized sun / lamps, unlike a baked radiance field).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .common import R2S2R_APPEARANCE


def cmd_sfm(a):
    sys.path.insert(0, str(R2S2R_APPEARANCE / "src"))
    import trimesh
    from r2s2r.appearance import sfm   # needs pycolmap
    views = sorted((a.dataset / "views").glob("*.json"))
    sfm.reconstruct(views, a.out)
    cloud = trimesh.load(a.out / "points.ply")
    xyz = np.asarray(cloud.vertices, np.float32)
    rgb = np.asarray(cloud.colors, np.float32)[:, :3] / 255
    lo, hi = np.array(a.bounds[:3]), np.array(a.bounds[3:])
    keep = np.isfinite(xyz).all(1) & (xyz >= lo).all(1) & (xyz <= hi).all(1)
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) > 20:   # statistical outlier removal
        from scipy.spatial import cKDTree
        d = cKDTree(xyz).query(xyz, k=min(9, len(xyz)))[0][:, 1:].mean(1)
        ok = d < d.mean() + 2.5 * d.std()
        xyz, rgb = xyz[ok], rgb[ok]
    np.savez_compressed(a.out / "points.npz", points=xyz, colors=rgb)
    report = json.loads((a.out / "report.json").read_text())
    report["points_kept"] = int(len(xyz))
    bench = json.loads((a.dataset / "benchmark.json").read_text())
    near = xyz[np.abs(xyz[:, 2] - bench["table_z"]) < 0.04]
    if len(near) > 30:   # table height from the points near the expected plane (robust: median)
        report["table_z_from_points"] = float(np.median(near[:, 2]))
        report["table_points"] = int(len(near))
    (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def cmd_texture(a):
    bench = json.loads((a.dataset / "benchmark.json").read_text())
    z = a.table_z if a.table_z is not None else bench["table_z"]
    frames = [f for f in bench["frames"] if f["split"] == "train" and (not a.condition or f["episode"] == a.condition)]
    if not frames:
        raise SystemExit("no training frames (check --condition)")
    x0, x1, y0, y1 = a.extent
    ppm = a.px_per_m
    W, H = int(round((x1 - x0) * ppm)), int(round((y1 - y0) * ppm))
    # texture pixel (u, v) -> world: x = x0 + u / ppm, y = y1 - v / ppm (image up = world +y)
    M = np.array([[1 / ppm, 0, x0], [0, -1 / ppm, y1], [0, 0, 1.0]])
    stack = np.zeros((len(frames), H, W, 3), np.uint8)
    seen = np.zeros((len(frames), H, W), bool)
    for i, f in enumerate(frames):
        img = cv2.imread(str(a.dataset / f["image"]))
        mask = cv2.imread(str(a.dataset / f["mask"]), 0)
        K = np.asarray(f["K"])
        T = np.linalg.inv(np.asarray(f["world_from_camera"]))
        # plane point (x, y, z) -> pixel: K [r1 r2 (r3 z + t)]
        Hw = K @ np.column_stack([T[:3, 0], T[:3, 1], T[:3, 2] * z + T[:3, 3]])
        Hm = Hw @ M
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        p = np.stack([uu, vv, np.ones_like(uu)], -1).reshape(-1, 3) @ Hm.T
        front = (p[:, 2] > 0.05).reshape(H, W)   # plane point in front of the camera
        warped = cv2.warpPerspective(img, Hm, (W, H), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)
        wmask = cv2.warpPerspective(mask, Hm, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        # grazing views smear: keep rays steeper than --min-elevation
        cam = np.asarray(f["world_from_camera"])[:3, 3]
        wx, wy = x0 + uu / ppm, y1 - vv / ppm
        d = np.stack([wx - cam[0], wy - cam[1], np.full_like(wx, z - cam[2], dtype=float)], -1)
        elev = np.degrees(np.arcsin(np.abs(d[..., 2]) / np.linalg.norm(d, axis=-1)))
        seen[i] = front & (wmask > 0) & (elev > a.min_elevation)
        stack[i] = warped
    count = seen.sum(0)
    # a low percentile instead of the median: on a glossy table the bright samples are reflections
    values = np.where(seen[..., None], stack.astype(np.float32), np.nan)
    with np.errstate(all="ignore"):
        tex = np.nan_to_num(np.nanpercentile(values, a.percentile, axis=0)).astype(np.uint8)
    del values
    covered = count >= a.min_views
    if not covered.any():
        raise SystemExit("no table pixel is seen by enough views")
    ys, xs = np.where(covered)
    fill = np.median(tex[covered], axis=0).astype(np.uint8)
    hole = (~covered).astype(np.uint8) * 255
    tex[~covered] = fill
    tex = cv2.inpaint(tex, cv2.erode(hole, np.ones((3, 3), np.uint8)), 5, cv2.INPAINT_TELEA) if a.inpaint else tex
    a.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out / "table_texture.png"), tex)
    cv2.imwrite(str(a.out / "table_coverage.png"), np.clip(count * 255 // max(count.max(), 1), 0, 255).astype(np.uint8))
    env = {"schema": "yam_r2s2r_env_v1", "world_frame": bench["world_frame"], "table_z": z,
           "table": {"texture": "table_texture.png", "extent_xy_m": [x0, x1, y0, y1], "px_per_m": ppm,
                     "uv": "u along +x, v along +y", "covered_fraction": float(covered.mean()),
                     "covered_bbox_xy_m": [float(x0 + xs.min() / ppm), float(x0 + xs.max() / ppm),
                                           float(y1 - ys.max() / ppm), float(y1 - ys.min() / ppm)],
                     "median_color_srgb": (fill[::-1] / 255).round(4).tolist()},
           "source": {"dataset": str(a.dataset.resolve()), "condition": a.condition, "views": len(frames)}}
    if a.sfm and (a.sfm / "points.npz").exists():
        env["points"] = str((a.sfm / "points.npz").resolve())
    (a.out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    print(json.dumps(env["table"]))


def umeyama(src, dst):
    """Similarity (s, R, t) with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(Y.T @ X / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / X.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def cmd_refine(a):
    """Bundle-adjust the camera poses (intrinsics fixed), then put the result back in the robot's metric
    frame by a similarity fit of the refined camera centres to the FK ones. Rewrites benchmark.json and
    views/*.json in place; the FK poses stay in benchmark_fk.json."""
    sys.path.insert(0, str(R2S2R_APPEARANCE / "src"))
    import pycolmap
    from r2s2r.appearance import sfm
    bpath = a.dataset / "benchmark.json"
    fk_path = a.dataset / "benchmark_fk.json"
    if not fk_path.exists():
        fk_path.write_text(bpath.read_text())
    bench = json.loads(fk_path.read_text())
    work = a.dataset / "refine"
    if work.exists():
        import shutil
        shutil.rmtree(work)
    work.mkdir()
    manifests = []
    for cam in sorted({f["camera"] for f in bench["frames"]}):   # every frame takes part, also validation
        sel = []
        for f in bench["frames"]:
            if f["camera"] == cam:
                g = dict(f, split="train")
                for key in ("image", "mask", "foreground_mask", "valid_mask"):
                    g[key] = "../" + g[key]
                sel.append(g)
        m = work / f"{cam}.json"
        m.write_text(json.dumps({"K": sel[0]["K"], "width": sel[0]["width"], "height": sel[0]["height"], "frames": sel}))
        manifests.append(m)
    sfm.reconstruct(manifests, work / "sfm")
    rec = pycolmap.Reconstruction(work / "sfm" / "sparse")
    before = rec.compute_mean_reprojection_error()
    opt = pycolmap.BundleAdjustmentOptions()
    opt.refine_focal_length = opt.refine_principal_point = opt.refine_extra_params = False
    opt.refine_rig_from_world, opt.refine_sensor_from_rig = True, False
    for _ in range(a.rounds):
        pycolmap.bundle_adjustment(rec, opt)
    after = rec.compute_mean_reprojection_error()
    sources = json.loads((work / "sfm" / "sources.json").read_text())
    refined, fk = {}, {}
    for image in rec.images.values():
        if image.num_points3D < a.min_points:
            continue          # too few observations: keep the FK pose
        key = sources[image.name]["frame"]["image"].replace("../", "")
        refined[key] = np.linalg.inv(np.vstack([image.cam_from_world().matrix(), [0, 0, 0, 1]]))
        fk[key] = np.asarray(sources[image.name]["frame"]["world_from_camera"])
    keys = sorted(refined)
    s, R, t = umeyama(np.array([refined[k][:3, 3] for k in keys]), np.array([fk[k][:3, 3] for k in keys]))
    shift, turn = [], []
    for f in bench["frames"]:
        k = f["image"]
        if k not in refined:
            f["pose_source"] = "fk"
            continue
        T = np.eye(4)
        T[:3, :3] = R @ refined[k][:3, :3]
        T[:3, 3] = s * R @ refined[k][:3, 3] + t
        shift.append(np.linalg.norm(T[:3, 3] - fk[k][:3, 3]))
        turn.append(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ fk[k][:3, :3]) - 1) / 2, -1, 1))))
        f["world_from_camera"], f["pose_source"] = T.tolist(), "bundle_adjusted"
    report = {"frames": len(bench["frames"]), "refined": len(shift), "kept_fk": len(bench["frames"]) - len(shift),
              "mean_reprojection_px_before": float(before), "mean_reprojection_px_after": float(after),
              "similarity_scale": float(s), "camera_shift_mm": {"median": float(np.median(shift) * 1e3), "p95": float(np.quantile(shift, 0.95) * 1e3)},
              "camera_turn_deg": {"median": float(np.median(turn)), "p95": float(np.quantile(turn, 0.95))}}
    bench["pose_refinement"] = report
    bpath.write_text(json.dumps(bench, indent=1) + "\n")
    for cam in sorted({f["camera"] for f in bench["frames"]}):
        sel = []
        for f in bench["frames"]:
            if f["camera"] == cam:
                g = dict(f)
                for key in ("image", "mask", "foreground_mask", "valid_mask"):
                    g[key] = "../" + g[key]
                sel.append(g)
        (a.dataset / "views" / f"{cam}.json").write_text(json.dumps(
            {"K": sel[0]["K"], "width": sel[0]["width"], "height": sel[0]["height"], "frames": sel}, indent=1) + "\n")
    print(json.dumps(report))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("sfm")
    c.add_argument("--dataset", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--bounds", type=float, nargs=6, default=[-3, -4, -1.2, 5, 5, 3.5],
                   help="keep points inside xmin ymin zmin xmax ymax zmax (m, world)")
    c = sub.add_parser("texture")
    c.add_argument("--dataset", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--sfm", type=Path)
    c.add_argument("--condition")
    c.add_argument("--table-z", type=float)
    c.add_argument("--extent", type=float, nargs=4, default=[-0.25, 0.85, -0.45, 1.05], help="x0 x1 y0 y1 (m)")
    c.add_argument("--px-per-m", type=float, default=500)
    c.add_argument("--min-views", type=int, default=3)
    c.add_argument("--min-elevation", type=float, default=25, help="deg: drop grazing rays")
    c.add_argument("--percentile", type=float, default=35, help="per-pixel percentile over views (50 = median)")
    c.add_argument("--no-inpaint", dest="inpaint", action="store_false")
    c = sub.add_parser("refine")
    c.add_argument("--dataset", required=True, type=Path)
    c.add_argument("--rounds", type=int, default=2)
    c.add_argument("--min-points", type=int, default=30)
    a = p.parse_args()
    {"sfm": cmd_sfm, "texture": cmd_texture, "refine": cmd_refine}[a.cmd](a)


if __name__ == "__main__":
    main()
