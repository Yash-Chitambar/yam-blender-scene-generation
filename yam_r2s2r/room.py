"""The scanned room as an r2s2r radiance field: train it, render it at bundle cameras, composite.

Run with the CUDA env (.venv-appearance: r2s2r's pinned torch + gsplat):

    A=.venv-appearance/bin/python
    $A -m yam_r2s2r.room train     --dataset work/dataset --points work/env/sfm/points.npz --out work/room/splat [--steps 15000 --width 640]
    $A -m yam_r2s2r.room render    --weights work/room/splat/step15000.pt --episode work/episodes/demo --out work/room/renders/demo
    $A -m yam_r2s2r.room composite --renders work/renders_fg --room work/room/renders --out work/renders_room

`train` is r2s2r's own `appearance.splat_cuda.train` on our benchmark.json (robot masked out, FK poses fixed).
`render` rasterizes the field at every frame of each camera of an episode (same K and poses as the Blender bundle).
`composite` puts Blender's foreground (robot + relit table, rendered by collect --transparent) over the room.
The field is baked for the light it was scanned in, so the composite scales it by one RGB gain per
(recipe, camera): the gain that makes the field agree with Blender's relit scanned wall/room in the
reference frame render_dr saves. That carries the recipe's exposure, white balance and overall light
level into the background; it cannot move shadows or sun patches inside the room field.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from .common import R2S2R_APPEARANCE

sys.path.insert(0, str(R2S2R_APPEARANCE / "src"))


def _cuda_env():
    import site
    for sp in site.getsitepackages():
        cuda = Path(sp) / "nvidia/cu13"
        if cuda.exists():
            os.environ.setdefault("CUDA_HOME", str(cuda))
            os.environ["PATH"] = f"{cuda / 'bin'}:{Path(sys.executable).parent}:{os.environ['PATH']}"


def to_lin(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def to_srgb(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


def cmd_train(a):
    _cuda_env()
    from r2s2r.appearance import splat_cuda
    splat_cuda.train(a.dataset / "benchmark.json", a.points, a.out, steps=a.steps, width=a.width,
                     checkpoint_every=a.checkpoint_every, loss_mode=a.loss_mode, max_points=a.max_points,
                     rasterize_mode=a.rasterize_mode)
    # held-out score on the last validation renders
    bench = json.loads((a.dataset / "benchmark.json").read_text())
    vdir = a.out / f"validation_{a.steps}"
    psnr = []
    for f in bench["frames"]:
        if f["split"] != "validation":
            continue
        pred = cv2.imread(str(vdir / (Path(f["image"]).stem + ".png")))
        if pred is None:
            continue
        h, w = pred.shape[:2]
        tgt = cv2.resize(cv2.imread(str(a.dataset / f["image"])), (w, h), interpolation=cv2.INTER_AREA)
        m = cv2.resize(cv2.imread(str(a.dataset / f["mask"]), 0), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        mse = ((pred.astype(np.float32) - tgt.astype(np.float32))[m] ** 2).mean() / 255 ** 2
        psnr.append(float(-10 * np.log10(max(mse, 1e-10))))
    report = {"validation_views": len(psnr), "psnr_db_mean": float(np.mean(psnr)), "psnr_db_min": float(np.min(psnr)),
              "note": "masked sRGB PSNR on held-out wrist views, background pixels only"}
    (a.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def cmd_render(a):
    _cuda_env()
    import torch
    from r2s2r.appearance import splat_cuda
    params = {k: v.cuda() for k, v in torch.load(a.weights, map_location="cpu", weights_only=True).items()}
    for cam_dir in sorted(p.parent for p in a.episode.glob("*/camera.json")):
        cam = json.loads((cam_dir / "camera.json").read_text())
        w, h = cam["width"], cam["height"]
        k = np.asarray(cam["K"], np.float32)
        poses = np.asarray(cam["world_from_camera"])
        out = a.out / cam_dir.name
        out.mkdir(parents=True, exist_ok=True)
        dummy = np.zeros((h, w, 3), np.uint8)
        with torch.no_grad():
            for f in range(0, len(poses), a.stride):
                view = (None, np.linalg.inv(poses[f]).astype(np.float32), k, dummy, None)
                rendered, alpha, _ = splat_cuda.rasterize(params, view, 3, "3d", a.rasterize_mode)
                rgb = rendered[0, ..., :3].clamp(0, 1).cpu().numpy()
                op = alpha[0, ..., 0].cpu().numpy()
                cv2.imwrite(str(out / f"{f:06d}.png"), np.uint8(rgb[..., ::-1] * 255))
                cv2.imwrite(str(out / f"{f:06d}_opacity.png"), np.uint8(op.clip(0, 1) * 255))
        print(f"{cam_dir.name}: {len(range(0, len(poses), a.stride))} frames {w}x{h}", flush=True)


def cmd_composite(a):
    rows = [json.loads(l) for l in (a.renders / "index.jsonl").read_text().splitlines()]
    a.out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        src = a.renders / row["dir"]
        room = a.room / row["episode"] / row["camera"]
        meta = json.loads((src / "render.json").read_text())
        if not meta.get("transparent"):
            raise SystemExit(f"{src}: render with collect --transparent first")
        dst = a.out / row["dir"]
        dst.mkdir(parents=True, exist_ok=True)
        gain = np.ones(3)
        ref = src / f"reference_{meta['exposure_frame']:06d}.png"
        bg_ref = room / f"{meta['exposure_frame']:06d}.png"
        if ref.exists() and bg_ref.exists() and not a.no_relight:
            r = to_lin(cv2.imread(str(ref))[..., ::-1] / 255.0)
            b = to_lin(cv2.imread(str(bg_ref))[..., ::-1] / 255.0)
            fg = cv2.imread(str(src / f"{meta['exposure_frame']:06d}.png"), cv2.IMREAD_UNCHANGED)
            op = cv2.imread(str(room / f"{meta['exposure_frame']:06d}_opacity.png"), 0)
            sel = (fg[..., 3] < 8) & (op > 128) & (b.max(-1) < 0.9) & (r.max(-1) < 0.9)
            if sel.sum() > 500:
                gain = np.clip(np.median(r[sel], 0) / np.maximum(np.median(b[sel], 0), 1e-3), 0.05, 20)
        for f in meta["frames"]:
            fg = cv2.imread(str(src / f"{f:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
            bg = cv2.imread(str(room / f"{f:06d}.png"))
            if bg is None:
                raise SystemExit(f"no room render for {row['episode']} {row['camera']} frame {f}: run room render (stride 1)")
            alpha = fg[..., 3:4]
            lin = to_lin(fg[..., 2::-1]) * alpha + to_lin(bg[..., ::-1] / 255.0) * gain * (1 - alpha)
            cv2.imwrite(str(dst / f"{f:06d}.png"), np.uint8(to_srgb(lin)[..., ::-1] * 255 + 0.5))
        meta["room_field"] = {"renders": str(room.resolve()), "linear_rgb_gain": gain.tolist(), "relit": not a.no_relight}
        (dst / "render.json").write_text(json.dumps(meta, indent=1) + "\n")
        with (a.out / "index.jsonl").open("a") as fh:
            fh.write(json.dumps({**row, "room_gain": gain.round(3).tolist()}) + "\n")
        print(f"{row['dir']}: gain {gain.round(2).tolist()}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("train")
    c.add_argument("--dataset", required=True, type=Path)
    c.add_argument("--points", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--steps", type=int, default=15000)
    c.add_argument("--width", type=int, default=640)
    c.add_argument("--checkpoint-every", type=int, default=5000)
    c.add_argument("--loss-mode", default="l1-ssim", choices=["mse", "l1-ssim", "hybrid"])
    c.add_argument("--rasterize-mode", default="antialiased", choices=["classic", "antialiased"])
    c.add_argument("--max-points", type=int, default=600000)
    c = sub.add_parser("render")
    c.add_argument("--weights", required=True, type=Path)
    c.add_argument("--episode", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--stride", type=int, default=1)
    c.add_argument("--rasterize-mode", default="antialiased", choices=["classic", "antialiased"])
    c = sub.add_parser("composite")
    c.add_argument("--renders", required=True, type=Path, help="collect --transparent output")
    c.add_argument("--room", required=True, type=Path, help="folder holding <episode>/<camera>/ room renders")
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--no-relight", action="store_true", help="keep the room field as scanned (no gain)")
    a = p.parse_args()
    {"train": cmd_train, "render": cmd_render, "composite": cmd_composite}[a.cmd](a)


if __name__ == "__main__":
    main()
