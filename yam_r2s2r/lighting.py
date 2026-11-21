"""Domain-randomized lighting recipes: time of day (sun + sky through windows), room lamps, HDRIs, camera.

    python -m yam_r2s2r.lighting --out work/lighting --n 32 [--seed 0] [--hours 8 12 16 20 23]
                                 [--hdri-dir assets/hdri] [--latitude 37.87] [--day-of-year 324]

One JSON per variant (`<out>/light_0000.json`, schema `yam_r2s2r_lighting_v1`), read by
blender/render_dr.py. Everything is sampled here, in plain Python, so a render is reproducible from
its recipe and the recipe can be logged next to the images as a label.

Model: the cell sits in a box room. Daylight is a physical sky (Nishita) plus a sun lamp whose
elevation and azimuth follow the solar position for the sampled hour, date and latitude; it enters
only through the room's windows, so the patch of sun on the table moves with the hour. Ceiling lamps
(area lights, 2700-6500 K) are on or off. Some variants replace the sky by an HDRI. The camera has
auto exposure (target mean brightness), an exposure bias and a white balance.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "yam_r2s2r_lighting_v1"


def solar_position(hour, day_of_year, latitude_deg):
    """(elevation deg, azimuth deg clockwise from north) at local solar time `hour`."""
    decl = math.radians(-23.44) * math.cos(2 * math.pi * (day_of_year + 10) / 365.0)
    lat, ha = math.radians(latitude_deg), math.radians(15.0 * (hour - 12.0))
    sin_el = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    el = math.asin(max(-1.0, min(1.0, sin_el)))
    az = math.atan2(math.sin(ha), math.cos(ha) * math.sin(lat) - math.tan(decl) * math.cos(lat)) + math.pi
    return math.degrees(el), math.degrees(az) % 360.0


def kelvin_rgb(k):
    """Linear RGB tint of a black body (Tanner Helland fit), normalised to max 1."""
    t = k / 100.0
    r = 255 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    c = np.clip([r, g, b], 0, 255) / 255.0
    c = c ** 2.2
    return (c / c.max()).round(4).tolist()


def sample(rng, index, hour=None, hdris=(), latitude=37.87, day_of_year=None, hdri_prob=0.2):
    hour = float(rng.uniform(0, 24)) if hour is None else float(hour) + float(rng.uniform(-0.5, 0.5))
    hour %= 24
    doy = int(rng.integers(1, 366)) if day_of_year is None else int(day_of_year)
    el, az = solar_position(hour, doy, latitude)
    day = el > -4
    # compass heading of world +x (the direction the right arm faces); unknown, so randomized
    north = float(rng.uniform(0, 360))
    # sun azimuth in the world frame, counter-clockwise from +x
    sun_world_az = (north - az) % 360
    overcast = float(rng.beta(1.2, 2.0))            # 0 clear .. 1 fully overcast
    room = [float(rng.uniform(4, 8)), float(rng.uniform(4, 9)), float(rng.uniform(2.5, 3.4))]
    floor_z = -0.76 + float(rng.uniform(-0.05, 0.05))      # table top is ~0.74 m above the floor
    center = [float(rng.uniform(-0.8, 1.2)), float(rng.uniform(-0.8, 1.4))]
    windows = []
    for wall in rng.permutation(["+x", "-x", "+y", "-y"])[: int(rng.integers(1, 3))]:
        length = room[0] if wall in ("+y", "-y") else room[1]
        n = int(rng.integers(1, 4))
        width = float(rng.uniform(0.9, min(2.2, length / n - 0.3)))
        for k in range(n):
            windows.append({"wall": str(wall), "center_m": float((k + 0.5) / n * length - length / 2),
                            "width_m": width, "sill_m": float(rng.uniform(0.7, 1.1)),
                            "height_m": float(rng.uniform(1.0, 1.6))})
    lamps_on = bool(rng.random() < (0.55 if el > 8 else 0.95))
    lamps = []
    if lamps_on:
        kelvin = float(rng.choice([2700, 3000, 3500, 4000, 5000, 6500])) + float(rng.uniform(-150, 150))
        nx, ny = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        watts = float(rng.uniform(120, 600)) / (nx * ny)       # Blender area-light watts per panel
        for i in range(nx):
            for j in range(ny):
                lamps.append({"position": [center[0] + (i + 0.5) / nx * room[0] - room[0] / 2,
                                           center[1] + (j + 0.5) / ny * room[1] - room[1] / 2, floor_z + room[2] - 0.05],
                              "size_m": [float(rng.uniform(0.3, 1.2)), float(rng.uniform(0.3, 0.6))],
                              "energy_w": watts * float(rng.uniform(0.8, 1.2)), "kelvin": kelvin, "color": kelvin_rgb(kelvin)})
    if rng.random() < 0.25:   # a task lamp near the table
        k = float(rng.uniform(2700, 5000))
        lamps.append({"position": [float(rng.uniform(-0.2, 0.8)), float(rng.uniform(-0.4, 1.0)), float(rng.uniform(0.5, 1.1))],
                      "size_m": [0.08, 0.08], "energy_w": float(rng.uniform(8, 40)), "kelvin": k, "color": kelvin_rgb(k), "aim": [0.3, 0.3, 0.0]})
    use_hdri = bool(len(hdris) and rng.random() < hdri_prob)
    sky = {"mode": "hdri" if use_hdri else "nishita", "strength": float(rng.uniform(0.6, 1.4)),
           "air_density": float(rng.uniform(0.7, 1.6)), "dust_density": float(1 + 6 * overcast),
           "ozone_density": float(rng.uniform(0.6, 1.4))}
    if use_hdri:
        sky.update({"hdri": str(hdris[int(rng.integers(len(hdris)))]), "rotation_deg": float(rng.uniform(0, 360)),
                    "strength": float(rng.uniform(0.3, 2.0))})
    sun_k = 2200 + 3600 * max(0.0, min(1.0, el / 25.0))       # warm near the horizon
    recipe = {
        "schema": SCHEMA, "name": f"light_{index:04d}", "seed": int(rng.integers(0, 2 ** 31 - 1)),
        "time": {"hour": round(hour, 3), "day_of_year": doy, "latitude_deg": latitude, "label": label(hour, el)},
        "sun": {"enabled": bool(day and el > 0.5), "elevation_deg": round(el, 3), "azimuth_compass_deg": round(az, 3),
                "azimuth_world_deg": round(sun_world_az, 3), "north_from_world_x_deg": round(north, 2),
                "strength": float((1 - 0.92 * overcast) * rng.uniform(0.7, 1.2)),
                "angle_deg": float(0.53 + 25 * overcast), "color": kelvin_rgb(sun_k), "overcast": round(overcast, 3)},
        "sky": sky,
        "room": {"enabled": True, "size_m": room, "center_xy_m": center, "floor_z_m": floor_z,
                 "wall_color": (np.clip(rng.normal(0.72, 0.12, 3) * rng.uniform(0.9, 1.0, 3), 0.2, 0.95)).round(3).tolist(),
                 "floor_color": np.clip(rng.uniform(0.06, 0.45) + rng.normal(0, 0.02, 3), 0.02, 0.6).round(3).tolist(), "windows": windows,
                 "blinds": float(rng.uniform(0.0, 0.8)) if rng.random() < 0.3 else 0.0},
        "lamps": lamps,
        "camera": {"auto_exposure_target": float(rng.uniform(0.30, 0.46)), "exposure_bias_ev": float(rng.normal(0, 0.35)),
                   "white_balance_k": float(rng.uniform(3800, 7200)), "auto_white_balance": bool(rng.random() < 0.6)},
        "table": {"roughness": float(rng.uniform(0.12, 0.55)), "specular": float(rng.uniform(0.3, 0.7)),
                  "color_gain": float(rng.uniform(0.75, 1.25))},
        # the YAM: black anodised links and housings, light printed covers
        "robot": {"roughness": float(rng.uniform(0.3, 0.7)), "black": float(rng.uniform(0.015, 0.07)),
                  "white": float(rng.uniform(0.6, 0.85))},
    }
    return recipe


def label(hour, el):
    if el < -6:
        return "night"
    if el < 6:
        return "dawn" if hour < 12 else "dusk"
    return "morning" if hour < 11 else "midday" if hour < 14 else "afternoon"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hours", type=float, nargs="+", help="cycle through these hours (+-0.5 h); default: uniform 0-24")
    p.add_argument("--hdri-dir", type=Path, nargs="*", default=[])
    p.add_argument("--hdri-prob", type=float, default=0.2)
    p.add_argument("--latitude", type=float, default=37.87)
    p.add_argument("--day-of-year", type=int)
    a = p.parse_args()
    hdris = sorted(str(f.resolve()) for d in a.hdri_dir for f in list(d.glob("*.hdr")) + list(d.glob("*.exr")))
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    index = []
    for i in range(a.n):
        r = sample(rng, i, a.hours[i % len(a.hours)] if a.hours else None, hdris, a.latitude, a.day_of_year, a.hdri_prob)
        (a.out / f"{r['name']}.json").write_text(json.dumps(r, indent=1) + "\n")
        index.append({"name": r["name"], "label": r["time"]["label"], "hour": r["time"]["hour"],
                      "sun_elevation_deg": r["sun"]["elevation_deg"], "lamps": len(r["lamps"]), "sky": r["sky"]["mode"]})
    (a.out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    for row in index:
        print(row)


if __name__ == "__main__":
    main()
