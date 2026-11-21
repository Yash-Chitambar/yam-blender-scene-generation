"""Rig description, frames and kinematics shared by every stage. No hardware access here.

Frames (all metres, OpenCV optical camera frames: +z forward, +y down):
  world          = right arm base (agent_data_rl convention); the table top is z = table_z
                   (about -0.023 m on duchamp3: the bases sit on mounting plates)
  T_world_cam    = T_world_base(side) @ FK_link_6(q_measured + joint_offsets) @ T_link6_cam
Quaternions in the rig spec are [w, x, y, z].
"""
import json
import os
from pathlib import Path

import mujoco
import numpy as np

PKG = Path(__file__).resolve().parents[1]
AGENT_REPO = Path(os.environ.get("YAM_AGENT_REPO", "/home/yashc/agent_data_rl"))
R2S2R_APPEARANCE = Path(os.environ.get(
    "R2S2R_APPEARANCE_ROOT", "/home/yashc/r2s2r_handoff/1_room_appearance/composite_handoff_v2"))
MENAGERIE_YAM = AGENT_REPO / "third_party/mujoco_menagerie/i2rt_yam/yam_linear_4310.xml"
DEFAULT_SPEC = AGENT_REPO / "blind_insertion/configs/real_cameras_duchamp3.json"
SIDES = ("right", "left")
WRIST_RES = (1280, 1024)


def T_from(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def T_inv(T):
    return T_from(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def T_pose(p):
    """{pos, quat wxyz} -> 4x4."""
    R = np.zeros(9)
    q = np.asarray(p["quat"], float)
    mujoco.mju_quat2Mat(R, q / np.linalg.norm(q))
    return T_from(R.reshape(3, 3), np.asarray(p["pos"], float))


def pose_dict(T):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(T[:3, :3]).flatten())
    return {"pos": T[:3, 3].tolist(), "quat": q.tolist()}


class Rig:
    """Calibration of one YAM cell, read from the agent_data_rl camera spec."""

    def __init__(self, spec_path=DEFAULT_SPEC, table_z=-0.023):
        self.spec_path = Path(spec_path)
        s = json.loads(self.spec_path.read_text())
        self.spec = s
        self.table_z = float(table_z)
        self.sides = [x for x in SIDES if s["wrist"].get(x, {}).get("pose_in_link6")]
        self.T_world_base = {"right": np.eye(4), "left": T_pose(s["arms"]["left_base_in_right_base"])}
        self.offsets = {x: np.asarray(s["arms"][x]["joint_offsets_rad"], float) for x in self.sides}
        self.T_link6_cam = {x: T_pose(s["wrist"][x]["pose_in_link6"]) for x in self.sides}
        self.K, self.dist = {}, {}
        for x in self.sides:
            i = s["wrist"][x]["intrinsics"]
            self.K[x] = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])
            self.dist[x] = np.asarray(s["wrist"][x]["distortion"], float)
        top = s.get("top", {})
        self.T_world_top = T_pose(top["extrinsics_cam_in_right_base"]) if top.get("extrinsics_cam_in_right_base") else None

    def top_K(self, mode="HD1080"):
        v = self.spec["top"]["intrinsics_rectified"][mode]
        return np.array([[v["fx"], 0, v["cx"]], [0, v["fy"], v["cy"]], [0, 0, 1.0]]), tuple(v["resolution"])

    def use_solve(self, side, solve_json):
        """Replace this side's intrinsics, hand-eye and joint offsets by a calib.py solve_<side>.json
        (use the solve that belongs to the session being processed)."""
        d = json.loads(Path(solve_json).read_text())
        m = d["menagerie"]
        w = m.get("with_joint_offsets")
        self.K[side], self.dist[side] = np.asarray(d["K"], float), np.asarray(d["dist"], float)
        self.T_link6_cam[side] = np.asarray((w or m)["T_link6_cam"], float)
        self.offsets[side] = np.asarray(w["joint_offsets_rad"], float) if w else np.zeros(6)


class Kin:
    """FK of Menagerie link_6 for one arm (the frame the hand-eye calibration uses)."""

    def __init__(self, path=MENAGERIE_YAM):
        self.m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(self.m)
        self.b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        assert self.b >= 0, "the Menagerie YAM model has no body link_6"

    def fk(self, q6):
        self.d.qpos[:] = 0
        self.d.qpos[:6] = q6
        mujoco.mj_kinematics(self.m, self.d)
        return T_from(self.d.xmat[self.b].reshape(3, 3), self.d.xpos[self.b])


def camera_in_world(rig, kin, side, q_measured):
    """World pose (OpenCV) of the wrist camera of `side` at measured (encoder) joints."""
    q = np.asarray(q_measured, float)[:6] + rig.offsets[side]
    return rig.T_world_base[side] @ kin.fk(q) @ rig.T_link6_cam[side]
