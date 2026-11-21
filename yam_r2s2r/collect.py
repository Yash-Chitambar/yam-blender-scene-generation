"""Blender data collection: every episode x camera x lighting recipe.

    python -m yam_r2s2r.collect --episodes work/episodes/* --lighting work/lighting --env work/env/env.json \
        --out work/renders [--cameras right_wrist left_wrist top] [--samples 48] [--device OPTIX]
        [--per-episode 4] [--frames-stride 1] [--transparent] [--video]

Output: <out>/<episode>/<recipe>/<camera>/000000.png ... + render.json, and <out>/index.jsonl with one
line per (episode, recipe, camera): paths, recipe label, exposure. `--per-episode N` gives each episode
N recipes chosen in rotation (default: all). Finished (episode, recipe, camera) folders are skipped, so
the command can be re-run after an interruption.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
BLENDER = os.environ.get("R2S2R_BLENDER", "/home/yashc/r2s2r/.bootstrap/blender/blender-4.5.3-linux-x64/blender")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", nargs="+", required=True, type=Path)
    p.add_argument("--lighting", required=True, type=Path, help="folder of light_*.json")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--env", type=Path)
    p.add_argument("--cameras", nargs="+", default=["right_wrist", "left_wrist", "top"])
    p.add_argument("--per-episode", type=int)
    p.add_argument("--frames-stride", type=int, default=1)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--device", default="OPTIX")
    p.add_argument("--transparent", action="store_true")
    p.add_argument("--video", action="store_true", help="also write <camera>.mp4 (needs ffmpeg)")
    p.add_argument("--blender", default=BLENDER)
    a = p.parse_args()
    recipes = sorted(a.lighting.glob("light_*.json"))
    if not recipes:
        raise SystemExit(f"no light_*.json in {a.lighting}")
    a.out.mkdir(parents=True, exist_ok=True)
    k = 0
    for ep in a.episodes:
        n = a.per_episode or len(recipes)
        chosen = [recipes[(k + i) % len(recipes)] for i in range(n)]
        k += n
        for cam in a.cameras:
            bundle = ep / cam / "bundle"
            if not bundle.exists():
                continue
            frames = json.loads((bundle / "manifest.json").read_text())["frames"]
            todo = [r for r in chosen if not (a.out / ep.name / r.stem / cam / "render.json").exists()]
            for r in todo:   # render_dr writes OUT/<recipe name>/, so give it one recipe and its own folder
                target = a.out / ep.name / r.stem
                cmd = [a.blender, "-b", "--python", str(PKG / "blender/render_dr.py"), "--", "--bundle", str(bundle),
                       "--lighting", str(r), "--out", str(target / f".{cam}"), "--samples", str(a.samples),
                       "--device", a.device, "--frames", *map(str, range(0, frames, a.frames_stride))]
                if a.env:
                    cmd += ["--env", str(a.env)]
                if a.transparent:
                    cmd.append("--transparent")
                res = subprocess.run(cmd, capture_output=True, text=True)
                made = target / f".{cam}" / r.stem
                if res.returncode or not (made / "render.json").exists():
                    print(res.stdout[-2000:], res.stderr[-2000:])
                    raise SystemExit(f"blender failed: {ep.name} {r.stem} {cam}")
                made.rename(target / cam)
                (target / f".{cam}").rmdir()
                meta = json.loads((target / cam / "render.json").read_text())
                with (a.out / "index.jsonl").open("a") as f:
                    f.write(json.dumps({"episode": ep.name, "recipe": r.stem, "camera": cam,
                                        "dir": str((target / cam).relative_to(a.out)), "frames": len(meta["frames"]),
                                        "label": meta["lighting"]["time"]["label"], "hour": meta["lighting"]["time"]["hour"],
                                        "exposure_ev": meta["exposure_ev"]}) + "\n")
                if a.video:
                    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-pattern_type", "glob", "-i",
                                    str(target / cam / "*.png"), "-pix_fmt", "yuv420p", str(target / f"{cam}.mp4")], check=False)
                print(f"{ep.name} {r.stem} {cam}: {len(meta['frames'])} frames ({meta['lighting']['time']['label']})", flush=True)


if __name__ == "__main__":
    main()
