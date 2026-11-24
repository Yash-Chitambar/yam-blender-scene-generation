"""MuJoCo model of the YAM cell (one or two arms on a table) in the world frame = right arm base.

The model is what r2s2r's `appearance.bundle` exports to Blender, and what the robot masks of the
reconstruction dataset are projected from. Joints are Menagerie (model) joints: measured + offsets.

    python -m yam_r2s2r.scene --out work/scene [--arms right left] [--table-z -0.023]
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .common import MENAGERIE_YAM, TABLE_Z, Rig, pose_dict

ARM_NQ = 8          # 6 arm joints + 2 coupled finger slides
TABLE_SIZE = (1.6, 1.8, 0.04)     # x, y, thickness (m)


def build(rig, arms=("right", "left"), table=True, table_center=None, hide_d405=True):
    """Returns (MjSpec, info). qpos layout: ARM_NQ values per arm, in the order of `arms`."""
    spec = mujoco.MjSpec()
    spec.modelname = "yam_cell_" + "_".join(arms)
    spec.compiler.degree = False
    for side in arms:
        arm = mujoco.MjSpec.from_file(str(MENAGERIE_YAM))
        if hide_d405:   # the model's D405 is a placeholder: the real cell has a UVC webcam there
            for g in arm.geoms:
                if g.parent.name == "wrist_camera":
                    g.rgba = [0, 0, 0, 0]
        for k in list(arm.keys):
            arm.delete(k)
        p = pose_dict(rig.T_world_base[side])
        frame = spec.worldbody.add_frame(pos=p["pos"], quat=p["quat"])
        spec.attach(arm, prefix=f"{side}/", frame=frame)
    if table:
        ys = [rig.T_world_base[s][1, 3] for s in arms]
        c = table_center or (0.30, float(np.mean(ys)))
        spec.worldbody.add_geom(
            name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, TABLE_SIZE[2] / 2],
            pos=[c[0], c[1], rig.table_z - TABLE_SIZE[2] / 2], rgba=[0.55, 0.5, 0.45, 1],
            group=2, contype=0, conaffinity=0)
    model = spec.compile()
    assert model.nq == ARM_NQ * len(arms), f"unexpected nq {model.nq}"
    info = {"arms": list(arms), "nq": int(model.nq), "arm_nq": ARM_NQ,
            "table_z": rig.table_z, "world": "right arm base",
            "finger_range_m": model.jnt_range[6].tolist(),
            "qpos_layout": "per arm: joint1..6 (model = measured + joint offsets), left_finger, right_finger"}
    return spec, model, info


def qpos_from_measured(rig, model, arms, q_measured):
    """{side: q7 measured (6 joints rad + gripper 0 closed..1 open)} -> model qpos."""
    lo, hi = model.jnt_range[6]
    out = np.zeros(ARM_NQ * len(arms))
    for i, side in enumerate(arms):
        q = np.asarray(q_measured.get(side, np.zeros(7)), float)
        out[i * ARM_NQ: i * ARM_NQ + 6] = q[:6] + (rig.offsets[side] if side in q_measured else 0)
        grip = q[6] if len(q) > 6 else 1.0
        out[i * ARM_NQ + 6: i * ARM_NQ + 8] = lo + np.clip(grip, 0, 1) * (hi - lo)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--arms", nargs="+", default=["right", "left"], choices=["right", "left"])
    p.add_argument("--spec", type=Path)
    p.add_argument("--table-z", type=float, default=TABLE_Z)
    p.add_argument("--no-table", action="store_true", help="the table comes from the environment bundle instead")
    a = p.parse_args()
    rig = Rig(*([a.spec] if a.spec else []), table_z=a.table_z)
    spec, model, info = build(rig, tuple(a.arms), table=not a.no_table)
    a.out.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(a.out / "model.mjb"))
    (a.out / "model.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
