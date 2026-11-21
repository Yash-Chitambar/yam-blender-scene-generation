"""Blender (Cycles) render of an r2s2r bundle under domain-randomized lighting recipes.

    blender -b --python blender/render_dr.py -- --bundle EP/right_wrist/bundle --env work/env/env.json \
        --lighting work/lighting/light_0000.json [more.json ...] --out OUT [--samples 48 --device OPTIX]
        [--frames 0 10 20] [--transparent] [--save-blend]

The scene, camera model and robot meshes come from r2s2r's own renderer
(scripts/appearance/render_blender.py: `setup`), imported from --r2s2r-scripts. This script adds what
that renderer lacks: a sun + physical sky that follow the time of day, a box room with windows the
daylight enters through, ceiling lamps, HDRI skies, a relightable textured table from the scan, and a
camera with auto exposure and white balance. One output folder per recipe:

    OUT/<recipe name>/000000.png ...   OUT/<recipe name>/render.json (recipe + exposure actually used)
"""
import argparse
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

DEFAULT_R2S2R_SCRIPTS = "/home/yashc/r2s2r_handoff/1_room_appearance/composite_handoff_v2/scripts/appearance"


def load_upstream(folder):
    spec = importlib.util.spec_from_file_location("r2s2r_render_blender", str(Path(folder) / "render_blender.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def principled(name, color, roughness=0.6, specular=0.5, emission=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*color[:3], 1)
    b.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = specular
    return mat


def add_mesh(scene, name, verts, faces, mat, uvs=None, camera_visible=True):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    if uvs is not None:
        layer = mesh.uv_layers.new(name="uv")
        for poly in mesh.polygons:
            for li, vi in zip(poly.loop_indices, poly.vertices):
                layer.data[li].uv = uvs[vi]
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.visible_camera = camera_visible
    return obj


def wall_quads(length, height, holes):
    """Rectangles (u0, u1, v0, v1) covering [-L/2, L/2] x [0, H] minus the window holes."""
    out, u = [], -length / 2
    for h in sorted(holes, key=lambda h: h[0]):
        a, b, lo, hi = h
        a, b = max(a, -length / 2 + 0.05), min(b, length / 2 - 0.05)
        if b <= a or a < u:
            continue
        out += [(u, a, 0, height), (a, b, 0, lo), (a, b, hi, height)]
        u = b
    out.append((u, length / 2, 0, height))
    return [q for q in out if q[1] - q[0] > 1e-4 and q[3] - q[2] > 1e-4]


def build_room(scene, room, transparent):
    sx, sy, sz = room["size_m"]
    cx, cy = room["center_xy_m"]
    z0 = room["floor_z_m"]
    wall = principled("room_wall", room["wall_color"], 0.85, 0.2)
    floor = principled("room_floor", room["floor_color"], 0.6, 0.3)
    vis = not transparent
    add_mesh(scene, "room_floor", [(cx - sx / 2, cy - sy / 2, z0), (cx + sx / 2, cy - sy / 2, z0),
                                   (cx + sx / 2, cy + sy / 2, z0), (cx - sx / 2, cy + sy / 2, z0)], [(0, 1, 2, 3)], floor,
             camera_visible=vis)
    add_mesh(scene, "room_ceiling", [(cx - sx / 2, cy - sy / 2, z0 + sz), (cx + sx / 2, cy - sy / 2, z0 + sz),
                                     (cx + sx / 2, cy + sy / 2, z0 + sz), (cx - sx / 2, cy + sy / 2, z0 + sz)], [(3, 2, 1, 0)],
             wall, camera_visible=vis)
    blinds = room.get("blinds", 0.0)
    # wall name -> (origin on the wall's centre line at the floor, along-wall unit vector, wall length)
    walls = {"+x": ((cx + sx / 2, cy, z0), (0, 1, 0), sy), "-x": ((cx - sx / 2, cy, z0), (0, 1, 0), sy),
             "+y": ((cx, cy + sy / 2, z0), (1, 0, 0), sx), "-y": ((cx, cy - sy / 2, z0), (1, 0, 0), sx)}
    for name, (origin, along, length) in walls.items():
        holes = []
        for w in room["windows"]:
            if w["wall"] != name:
                continue
            top = w["sill_m"] + w["height_m"]
            lo = w["sill_m"] + blinds * 0 + 0.0
            hi = min(sz - 0.1, top) - blinds * w["height_m"] * 0.0
            lo2 = lo
            hi2 = lo + (hi - lo) * (1 - blinds)        # blinds come down from the top
            holes.append((w["center_m"] - w["width_m"] / 2, w["center_m"] + w["width_m"] / 2, lo2, hi2))
        verts, faces = [], []
        o, a = Vector(origin), Vector(along)
        for (u0, u1, v0, v1) in wall_quads(length, sz, holes):
            n = len(verts)
            verts += [o + a * u0 + Vector((0, 0, v0)), o + a * u1 + Vector((0, 0, v0)),
                      o + a * u1 + Vector((0, 0, v1)), o + a * u0 + Vector((0, 0, v1))]
            faces.append((n, n + 1, n + 2, n + 3))
        add_mesh(scene, f"room_wall_{name}", verts, faces, wall, camera_visible=vis)
        for (u0, u1, v0, v1) in holes:   # light portals: Cycles samples the sky through the windows
            lamp = bpy.data.lights.new("portal", "AREA")
            lamp.shape = "RECTANGLE"
            lamp.size, lamp.size_y = max(u1 - u0, 0.05), max(v1 - v0, 0.05)
            lamp.cycles.is_portal = True
            obj = bpy.data.objects.new("portal", lamp)
            scene.collection.objects.link(obj)
            centre = o + a * ((u0 + u1) / 2) + Vector((0, 0, (v0 + v1) / 2))
            inward = Vector((cx, cy, centre.z)) - Vector((centre.x, centre.y, centre.z))
            inward = Vector((inward.x if name in ("+x", "-x") else 0, inward.y if name in ("+y", "-y") else 0, 0)).normalized()
            x_axis, y_axis = a, Vector((0, 0, 1))
            z_axis = -inward                         # area lights emit along local -Z
            if x_axis.cross(y_axis).dot(z_axis) < 0:
                x_axis = -x_axis
            obj.matrix_world = Matrix(((x_axis.x, y_axis.x, z_axis.x, centre.x), (x_axis.y, y_axis.y, z_axis.y, centre.y),
                                       (x_axis.z, y_axis.z, z_axis.z, centre.z), (0, 0, 0, 1)))


def build_world(scene, recipe):
    world = scene.world
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    for link in list(bg.inputs["Color"].links):
        nt.links.remove(link)
    sky, sun = recipe["sky"], recipe["sun"]
    el = math.radians(sun["elevation_deg"])
    if sky["mode"] == "hdri":
        tex = nt.nodes.new("ShaderNodeTexEnvironment")
        tex.image = bpy.data.images.load(sky["hdri"])
        mapping, coord = nt.nodes.new("ShaderNodeMapping"), nt.nodes.new("ShaderNodeTexCoord")
        mapping.inputs["Rotation"].default_value = (0, 0, math.radians(sky.get("rotation_deg", 0)))
        nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = sky["strength"]
        return "hdri"
    tex = nt.nodes.new("ShaderNodeTexSky")
    kinds = [e.identifier for e in tex.bl_rna.properties["sky_type"].enum_items]
    tex.sky_type = "NISHITA" if "NISHITA" in kinds else next(k for k in kinds if "SCATTER" in k)
    tex.sun_disc = False                              # the sun is a lamp: far less noise through windows
    tex.sun_elevation = max(el, math.radians(-10))
    # sky texture: sun_rotation is clockwise from +Y seen from above; recipe azimuth is CCW from +X
    tex.sun_rotation = math.radians(90 - sun["azimuth_world_deg"])
    tex.air_density, tex.ozone_density = sky["air_density"], sky["ozone_density"]
    if hasattr(tex, "dust_density"):
        tex.dust_density = sky["dust_density"]
    elif hasattr(tex, "aerosol_density"):
        tex.aerosol_density = sky["dust_density"]
    nt.links.new(tex.outputs["Color"], bg.inputs["Color"])
    night = sun["elevation_deg"] < -6
    # a little light at night: street / monitor glow, so windows are not pure black
    bg.inputs["Strength"].default_value = sky["strength"] * (1.0 if not night else 0.0)
    if night:
        for link in list(bg.inputs["Color"].links):
            nt.links.remove(link)
        bg.inputs["Color"].default_value = (0.01, 0.012, 0.02, 1)
        bg.inputs["Strength"].default_value = 0.05 * sky["strength"]
    return tex.sky_type


def build_sun(scene, sun):
    if not sun["enabled"]:
        return None
    el, az = math.radians(sun["elevation_deg"]), math.radians(sun["azimuth_world_deg"])
    lamp = bpy.data.lights.new("sun", "SUN")
    # direct normal irradiance, W/m2: clear-sky extinction with air mass
    air_mass = 1 / max(math.sin(el) + 0.50572 * (math.degrees(el) + 6.07995) ** -1.6364, 1e-3)
    lamp.energy = 1050 * (0.7 ** (air_mass ** 0.678)) * sun["strength"]
    lamp.angle = math.radians(sun["angle_deg"])
    lamp.color = sun["color"]
    obj = bpy.data.objects.new("sun", lamp)
    scene.collection.objects.link(obj)
    to_sun = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    obj.rotation_euler = to_sun.to_track_quat("Z", "Y").to_euler()   # the lamp shines along its -Z
    return lamp.energy


def build_lamps(scene, lamps):
    for i, l in enumerate(lamps):
        lamp = bpy.data.lights.new(f"lamp_{i}", "AREA")
        lamp.shape = "RECTANGLE"
        lamp.size, lamp.size_y = l["size_m"]
        lamp.energy, lamp.color = l["energy_w"], l["color"]
        obj = bpy.data.objects.new(f"lamp_{i}", lamp)
        scene.collection.objects.link(obj)
        obj.location = l["position"]
        if l.get("aim"):
            d = Vector(l["position"]) - Vector(l["aim"])
            obj.rotation_euler = d.to_track_quat("Z", "Y").to_euler()


def build_table(scene, env, env_root, recipe):
    """Textured plane at the scanned table height (UVs from the texture's world placement)."""
    t = env["table"]
    x0, x1, y0, y1 = t["extent_xy_m"]
    cx0, cx1, cy0, cy1 = t.get("crop_xy_m") or t.get("covered_bbox_xy_m") or t["extent_xy_m"]
    z = env["table_z"]
    look = recipe["table"]
    mat = principled("scan_table", [0.3, 0.3, 0.3], look["roughness"], look["specular"])
    nt = mat.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str((env_root / t["texture"]).resolve()))
    tex.extension = "EXTEND"
    gain = nt.nodes.new("ShaderNodeMixRGB")
    gain.blend_type = "MULTIPLY"
    gain.inputs["Fac"].default_value = 1.0
    g = look["color_gain"]
    gain.inputs["Color2"].default_value = (g, g, g, 1)
    nt.links.new(tex.outputs["Color"], gain.inputs["Color1"])
    nt.links.new(gain.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    uv = lambda x, y: ((x - x0) / (x1 - x0), (y - y0) / (y1 - y0))
    corners = [(cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1)]
    thick = 0.04
    verts = [(x, y, z) for x, y in corners] + [(x, y, z - thick) for x, y in corners]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    uvs = [uv(x, y) for x, y in corners] * 2
    return add_mesh(scene, "scan_table", verts, faces, mat, uvs)


def build_surfaces(scene, env, env_root):
    """Scanned planar background surfaces (walls): textured quads, lit by Cycles like everything else."""
    for sf in env.get("surfaces", []):
        mat = principled("scan_" + sf["name"], [0.6, 0.6, 0.6], sf.get("roughness", 0.7), 0.3)
        nt = mat.node_tree
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str((env_root / sf["texture"]).resolve()))
        tex.extension = "EXTEND"
        nt.links.new(tex.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
        o, eu, ev = Vector(sf["origin_m"]), Vector(sf["u_axis"]), Vector(sf["v_axis"])
        su, sv = sf["size_m"]
        verts = [o, o + eu * su, o + eu * su + ev * sv, o + ev * sv]
        add_mesh(scene, "scan_" + sf["name"], verts, [(0, 1, 2, 3)], mat, [(0, 0), (1, 0), (1, 1), (0, 1)])


def render_probe(scene, path):
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(path), check_existing=False)
    px = np.empty(img.size[0] * img.size[1] * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    px = px.reshape(-1, 4)
    w = px[:, 3] if px[:, 3].min() < 1 else np.ones(len(px))
    if w.sum() < 1:
        return None
    luma = px[:, :3] @ np.array([0.2126, 0.7152, 0.0722])
    return float((luma * w).sum() / w.sum())


def lin(v):
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def auto_exposure(scene, cam, tmp):
    """Camera-like auto exposure: bring the mean image brightness to the target, then add the bias."""
    samples, pct = scene.cycles.samples, scene.render.resolution_percentage
    scene.cycles.samples, scene.render.resolution_percentage = 12, 25
    ev = 0.0
    scene.view_settings.exposure = ev
    try:
        for i in range(4):
            mean = render_probe(scene, Path(tmp) / f"probe_{i}.png")
            if mean is None:
                break
            step = math.log2(lin(cam["auto_exposure_target"]) / max(lin(mean), 1e-6))
            step = max(-8.0, min(8.0, step))
            ev = max(-20.0, min(20.0, ev + step))
            scene.view_settings.exposure = ev
            if abs(step) < 0.15:
                break
    finally:
        scene.cycles.samples, scene.render.resolution_percentage = samples, pct
    ev += cam.get("exposure_bias_ev", 0.0)
    scene.view_settings.exposure = ev
    return ev


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--lighting", required=True, type=Path, nargs="+")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--env", type=Path, help="env.json from yam_r2s2r.reconstruct texture (scanned table)")
    p.add_argument("--appearance", type=Path, help="extra r2s2r appearance recipe (materials, body_assets, ...)")
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--device", choices=["CPU", "CUDA", "OPTIX", "METAL", "HIP"], default="OPTIX")
    p.add_argument("--frames", type=int, nargs="+")
    p.add_argument("--transparent", action="store_true",
                   help="RGBA foreground only (room hidden from the camera), to composite over an r2s2r room field")
    p.add_argument("--robot-only", action="store_true", help="validation: only the robot meshes (no table, no walls)")
    p.add_argument("--exposure-frame", type=int, help="frame the auto exposure is measured on (default: middle)")
    p.add_argument("--save-blend", action="store_true")
    p.add_argument("--r2s2r-scripts", default=DEFAULT_R2S2R_SCRIPTS)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
    rb = load_upstream(a.r2s2r_scripts)
    env = json.loads(a.env.read_text()) if a.env else None
    motion = np.load(a.bundle / "motion.npz", allow_pickle=False)
    for recipe_path in a.lighting:
        recipe = json.loads(recipe_path.read_text())
        if recipe.get("schema") != "yam_r2s2r_lighting_v1":
            raise ValueError(f"{recipe_path}: not a yam_r2s2r lighting recipe")
        out = a.out / recipe["name"]
        out.mkdir(parents=True, exist_ok=True)
        # r2s2r appearance recipe: no default lights (ours follow), robot look, scanned table replaces the model's
        cfg = json.loads(a.appearance.read_text()) if a.appearance else {}
        cfg.setdefault("lights", [])
        cfg.setdefault("world_strength", 1.0)
        robot = recipe["robot"]
        base = {"roughness": robot["roughness"], "smooth_angle_degrees": 40, "weld_coincident_vertices": True}
        sm = cfg.setdefault("source_materials", {})
        sm.setdefault("*black", {**base, "color": [robot["black"]] * 3})
        sm.setdefault("*white", {**base, "color": [robot["white"]] * 3})
        if a.robot_only:
            env = None
        if env or a.robot_only:
            cfg.setdefault("hide_geoms", []).append("table")
        (out / "appearance.json").write_text(json.dumps(cfg, indent=1) + "\n")
        scene, objects, camobj, meta, replacements = rb.setup(a.bundle, out / "appearance.json", a.samples, a.device)
        scene.cycles.seed = recipe["seed"] % (2 ** 31)
        sky_kind = build_world(scene, recipe)
        sun_energy = build_sun(scene, recipe["sun"]) if recipe["sky"]["mode"] != "hdri" else None
        if recipe["room"]["enabled"]:
            build_room(scene, recipe["room"], a.transparent)
        build_lamps(scene, recipe["lamps"])
        if env:
            build_table(scene, env, a.env.parent, recipe)
            build_surfaces(scene, env, a.env.parent)
        cam = recipe["camera"]
        vs = scene.view_settings
        if hasattr(vs, "use_white_balance"):
            vs.use_white_balance = True
            wb = cam["white_balance_k"]
            if cam.get("auto_white_balance"):   # an auto camera roughly neutralises the dominant light
                lit_by_lamps = recipe["lamps"] and (recipe["sun"]["elevation_deg"] < 3 or not recipe["room"]["windows"])
                wb = 0.5 * wb + 0.5 * (recipe["lamps"][0]["kelvin"] if lit_by_lamps else 6000)
            vs.white_balance_temperature = wb
        if a.transparent:
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = "RGBA"
        frames = a.frames if a.frames is not None else list(range(meta["frames"]))

        def pose(f):
            for obj, matrix in zip(objects, motion["geom_world"][f]):
                obj.matrix_world = Matrix(matrix.tolist())
            camobj.matrix_world = Matrix(motion["camera_world"][f].tolist())
            for obj, body_id, local in replacements:
                obj.matrix_world = Matrix(motion["body_world"][f, body_id].tolist()) @ local

        ef = a.exposure_frame if a.exposure_frame is not None else frames[len(frames) // 2]
        pose(ef)
        backdrop = [o for o in scene.objects if o.name.startswith("room_") or
                    (o.name.startswith("scan_") and o.name != "scan_table")]
        if a.transparent:   # exposure is metered on the whole view, like the real camera
            scene.render.film_transparent = False
            for o in backdrop:
                o.visible_camera = True
        with tempfile.TemporaryDirectory() as tmp:
            ev = auto_exposure(scene, cam, tmp)
        if a.transparent:
            # one opaque frame with the relit scanned wall / room: the reference the room-field composite
            # matches its brightness and colour to (yam_r2s2r.room composite)
            scene.render.image_settings.color_mode = "RGB"
            scene.render.filepath = str((out / f"reference_{ef:06d}.png").resolve())
            bpy.ops.render.render(write_still=True)
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = "RGBA"
            for o in backdrop:
                o.visible_camera = False
        for f in frames:
            pose(f)
            scene.render.filepath = str((out / f"{f:06d}.png").resolve())
            bpy.ops.render.render(write_still=True)
        if a.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=str((out / "scene.blend").resolve()))
        (out / "render.json").write_text(json.dumps({
            "blender": bpy.app.version_string, "device": a.device, "samples": a.samples, "frames": frames,
            "time_s": motion["time_s"][frames].tolist(), "bundle": str(a.bundle.resolve()),
            "env": str(a.env.resolve()) if a.env else None, "transparent": a.transparent,
            "exposure_ev": ev, "exposure_frame": ef, "sky": sky_kind, "sun_irradiance_w_m2": sun_energy,
            "width": meta["width"], "height": meta["height"], "K": meta["K"], "lighting": recipe}, indent=1) + "\n")
        print(f"[render_dr] {recipe['name']}: {len(frames)} frames, exposure {ev:+.2f} EV, {recipe['time']['label']}", flush=True)


if __name__ == "__main__":
    main()
