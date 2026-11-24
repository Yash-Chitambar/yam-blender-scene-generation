"""Stop-and-shoot wrist-camera scan of the workspace with one YAM arm (or both, one after the other).

Run it with agent_data_rl's venv, from anywhere:

    /home/yashc/agent_data_rl/.venv/bin/python -m yam_r2s2r.scan plan --side right --out work/plan_right.json
    /home/yashc/agent_data_rl/.venv/bin/python -m yam_r2s2r.scan run  --side right --out work/sessions/day_1400 --condition day_1400

`plan` touches no hardware: it only solves IK and runs the rig's table/reach checks.
`run` MOVES THE ARM. Before it: clamp both bases, start scripts/real/estop.py in another terminal,
park the other arm at its folded rest (rigctl.py), clear the table of anything not part of the scene.
All motion goes through agent_data_rl's safety layer (safety.SafeArm + calib.Mover): PD hold from
connect, table clearance, reach <= 0.42 m, <= 0.3 rad/s, tracking-error abort, estop file, and the arm
returns to where it started. A view whose pose or path fails a check is skipped, never forced.

Each view: move, settle, average the joints, drop buffered frames, save
    <out>/<side>/NNN.png + NNN.json {q (7, measured), t, kind, target, camera_pose_planned}
and <out>/session.json (condition, exposure settings, other arm pose assumption, plan).
One scan per lighting condition (time of day): use a new --out and --condition each time.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .common import AGENT_REPO, DEFAULT_SPEC, SIDES, TABLE_Z

sys.path.insert(0, str(AGENT_REPO / "scripts/real"))
sys.path.insert(0, str(AGENT_REPO))
import calib  # noqa: E402  (agent_data_rl/scripts/real)
import rig as rigmod  # noqa: E402
import safety  # noqa: E402
from blind_insertion.assets.real_camera import load_spec  # noqa: E402

SEEDS = ([0, 1.0, 1.0, -0.5, 0, 0], [0, 1.6, 1.3, -0.9, 0, 0], [0, 0.8, 0.6, 0.2, 0, 0],
         [0.6, 1.0, 1.0, -0.5, 0, 0], [-0.6, 1.0, 1.0, -0.5, 0, 0])


def T_pose(p):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(p["quat"], float))
    return calib.T_from(R.reshape(3, 3), np.asarray(p["pos"], float))


def lateral_ok(checker, side, q6, slack=0.0):
    """calib.Mover.lateral_ok without a connected arm (same limits)."""
    c = checker
    c.d.qpos[:] = 0
    c.d.qpos[:6] = q6
    mujoco.mj_kinematics(c.m, c.d)
    l6 = mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_BODY, "link6")
    reach = float(np.linalg.norm(c.d.xpos[l6, :2]))
    return (float((c.d.xpos[2:, 1] * calib.INWARD[side]).max()) <= calib.MAX_INWARD + slack
            and reach <= calib.MAX_REACH + slack)


def path_why(checker, side, q_from, q_to, n=25):
    for a in np.linspace(0, 1, n):
        q = (1 - a) * np.asarray(q_from) + a * np.asarray(q_to)
        why = checker.violation(q)
        if why:
            return f"path at {a:.2f}: {why}"
        if not lateral_ok(checker, side, q, slack=calib.PATH_SLACK):
            return f"path at {a:.2f}: reach or inward limit"
    return None


def candidates(side, rng, n_table, n_room, grid):
    """Camera goals in the arm's base frame: (kind, p_cam, p_target, roll)."""
    inward = calib.INWARD[side]
    out = []
    targets = [np.array([x, inward * y, 0.0]) for x in grid["x"] for y in grid["y"]]
    for _ in range(n_table):
        tgt = targets[rng.integers(len(targets))] + np.r_[rng.uniform(-0.03, 0.03, 2), 0]
        elev = np.radians(rng.uniform(30, 80))
        toward = -tgt[:2] / np.linalg.norm(tgt[:2])
        az = np.radians(rng.uniform(-75, 75))
        h = np.array([np.cos(az) * toward[0] - np.sin(az) * toward[1], np.sin(az) * toward[0] + np.cos(az) * toward[1]])
        d = rng.uniform(0.22, 0.38)
        p = tgt + d * np.array([np.cos(elev) * h[0], np.cos(elev) * h[1], np.sin(elev)])
        out.append(("table", p, tgt, np.radians(rng.uniform(-25, 25))))
    for _ in range(n_room):   # look outward and slightly down: walls, shelves, the far table edge
        az = np.radians(rng.uniform(-110, 110))
        # camera positions spread over the reachable volume: parallax is what the room field needs
        r, pa = rng.uniform(0.14, 0.38), np.radians(rng.uniform(-70, 70))
        p = np.array([r * np.cos(pa), r * np.sin(pa) + inward * 0.03, rng.uniform(0.16, 0.40)])
        tgt = p + np.array([np.cos(az), np.sin(az), -np.tan(np.radians(rng.uniform(-20, 40)))]) * 1.5
        out.append(("room", p, tgt, np.radians(rng.uniform(-10, 10))))
    return out


def make_plan(side, spec, q_rest_meas, n_table, n_room, seed, grid, kin=None, checker=None):
    """IK + safety filter, then a greedy short-path order. Joints returned are encoder (measured) frame."""
    kin = kin or calib.Kin()
    checker = checker or calib.calib_checker(calib.PLAN_CLEARANCE)
    path_check = calib.calib_checker(calib.PATH_CLEARANCE)
    X = T_pose(spec["wrist"][side]["pose_in_link6"])
    off = rigmod.joint_offsets(side)
    R_ref = (kin.fk(np.asarray(q_rest_meas[:6]) + off) @ X)[:3, :3]
    if abs(R_ref[2, 2]) > 0.95:   # camera looks straight up/down at rest: use a level reference instead
        R_ref = np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]])
    rng = np.random.default_rng(seed)
    views, rejected = [], {"ik": 0, "clearance": 0, "reach": 0}
    for kind, p, tgt, roll in candidates(side, rng, n_table * 4, n_room * 4, grid):
        T_cam = calib.look_at(p, tgt, R_ref, roll)
        qm = kin.ik(T_cam @ calib.T_inv(X), seeds=[np.asarray(q_rest_meas[:6]) + off, *SEEDS], iters=200)
        if qm is None:
            rejected["ik"] += 1
            continue
        q = qm - off
        if checker.violation(q):
            rejected["clearance"] += 1
            continue
        if not lateral_ok(checker, side, q):
            rejected["reach"] += 1
            continue
        views.append({"kind": kind, "q": q.tolist(), "target": np.asarray(tgt).round(4).tolist(),
                      "camera_pose_planned": T_cam.round(6).tolist()})
    keep = ([v for v in views if v["kind"] == "table"][:n_table] + [v for v in views if v["kind"] == "room"][:n_room])
    # greedy order by joint distance, starting at rest; a view is dropped when no checked path reaches it
    order, cur, left, unreachable = [], np.asarray(q_rest_meas[:6], float), list(keep), 0
    while left:
        left.sort(key=lambda v: np.abs(np.asarray(v["q"]) - cur).max())
        nxt = next((v for v in left if path_why(path_check, side, cur, v["q"]) is None), None)
        if nxt is None:
            unreachable = len(left)
            break
        left.remove(nxt)
        order.append(nxt)
        cur = np.asarray(nxt["q"], float)
    return {"side": side, "seed": seed, "q_rest": list(map(float, q_rest_meas[:6])), "views": order,
            "counts": {"table": sum(v["kind"] == "table" for v in order), "room": sum(v["kind"] == "room" for v in order)},
            "rejected": rejected, "unreachable_from_path": unreachable,
            "frame": "joints: encoder frame (as the motors read); targets and camera poses: this arm's base frame"}


def use_rig(a):
    """Point agent_data_rl's rig module (joint offsets, wrist camera ports, ZED serial) at this cell."""
    rigmod.SPEC = Path(a.spec)
    for side, dev in getattr(a, "cam_device", []):
        rigmod.WRIST_CAM[side] = dev
    if getattr(a, "zed_serial", None):
        rigmod.ZED_SERIAL = a.zed_serial


def cmd_plan(a):
    use_rig(a)
    spec = load_spec(str(rigmod.SPEC))
    grid = {"x": a.grid_x, "y": a.grid_y}
    plans = {}
    for side in (SIDES if a.side == "both" else (a.side,)):
        plans[side] = make_plan(side, spec, np.zeros(6), a.table_views, a.room_views, a.seed, grid)
        p = plans[side]
        print(f"{side}: {len(p['views'])} views ({p['counts']}), rejected {p['rejected']}, "
              f"unreachable {p['unreachable_from_path']}", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(plans, indent=1) + "\n")


class Cam(calib.WristCam):
    """Wrist camera with exposure and white balance locked for the whole scan (a radiance field needs
    constant photometry). Settings are restored to automatic on close: other rig scripts expect that."""

    def _mean(self, n=5):
        for _ in range(n):
            ok, f = self.cap.read()
        return f

    def lock(self, target=115.0):
        """Manual exposure tuned so the mean brightness of the current view is `target` (0-255), and the
        white balance temperature that makes that view most neutral. The driver does not report what its
        auto mode chose, so both are searched."""
        for _ in range(20):
            self.cap.read()
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)     # V4L2: 1 = manual, 3 = auto
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        exposure = 100.0
        for _ in range(8):
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            mean = float(cv2.cvtColor(self._mean(), cv2.COLOR_BGR2GRAY).mean())
            if abs(mean - target) < 0.08 * target:
                break
            exposure = float(np.clip(exposure * np.clip(target / max(mean, 1.0), 0.4, 2.5), 3, 2500))
        best = None
        for k in range(2800, 6600, 400):
            self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, k)
            f = self._mean(6).reshape(-1, 3).mean(0)
            cast = abs(float(f[2] - f[0])) + abs(float(f[1] - (f[0] + f[2]) / 2))
            if best is None or cast < best[0]:
                best = (cast, k)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, best[1])
        f = self._mean(6)
        self.settings = {"exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE), "gain": self.cap.get(cv2.CAP_PROP_GAIN),
                         "wb_temperature": self.cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
                         "mean_brightness_at_lock": float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()),
                         "mean_bgr_at_lock": f.reshape(-1, 3).mean(0).round(1).tolist(),
                         # manual white balance leaves a green cast on these webcams: a fixed grey-world gain,
                         # measured once on the lock view (white wall + black table), applied by build_dataset
                         "color_gain_bgr": (f.reshape(-1, 3).mean() / f.reshape(-1, 3).mean(0)).round(4).tolist(),
                         "locked": {"auto_exposure": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                                    "auto_wb": self.cap.get(cv2.CAP_PROP_AUTO_WB)}}
        return self.settings

    def close(self):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        super().close()


ZED_GRAB = """
import sys, cv2, pyzed.sl as sl
sys.path.insert(0, sys.argv[2])
import camshare
share = camshare.claim("zed")
cam = sl.Camera()
p = sl.InitParameters()
p.camera_resolution = sl.RESOLUTION.HD1080
p.depth_mode = sl.DEPTH_MODE.NONE
p.set_from_serial_number(int(sys.argv[3]))
assert cam.open(p) == sl.ERROR_CODE.SUCCESS, "ZED open failed"
img = sl.Mat()
for _ in range(20):
    cam.grab()
cam.retrieve_image(img, sl.VIEW.LEFT)
cv2.imwrite(sys.argv[1], img.get_data()[:, :, :3])
cam.close()
share.release()
"""


def grab_top(out):
    """One rectified HD1080 frame of the top ZED 2, in its own process (a failed ZED open can crash)."""
    import subprocess
    (out / "top").mkdir(exist_ok=True)
    path = out / "top" / "000.png"
    r = subprocess.run([sys.executable, "-c", ZED_GRAB, str(path), str(AGENT_REPO / "scripts/real"), str(rigmod.ZED_SERIAL)],
                       capture_output=True, text=True, timeout=120)
    if r.returncode or not path.exists():
        print(f"top camera: no frame ({(r.stderr or r.stdout).strip().splitlines()[-1:]}); continuing without it", flush=True)
        return False
    (out / "top" / "000.json").write_text(json.dumps({"camera": "top", "t": time.time(), "mode": "HD1080 rectified left"}))
    print("top camera: saved", flush=True)
    return True


def cmd_run(a):
    out = Path(a.out)
    use_rig(a)
    spec = load_spec(str(rigmod.SPEC))
    grid = {"x": a.grid_x, "y": a.grid_y}
    sides = SIDES if a.side == "both" else (a.side,)
    out.mkdir(parents=True, exist_ok=True)
    session = {"condition": a.condition or out.name, "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
               "rig_spec": str(rigmod.SPEC), "table_z": a.table_z, "notes": a.notes, "cameras": {}, "plans": {},
               "other_arm_q": {}, "rest_q": {}, "other_arm_q_note": "the arm that is not scanning is assumed parked at the folded rest (all zeros)"}
    if (out / "session.json").exists():   # adding a side to an existing session (e.g. after a camera dropout): keep the rest
        old = json.loads((out / "session.json").read_text())
        for k in ("cameras", "plans", "other_arm_q", "rest_q"):
            session[k] = {s: v for s, v in old.get(k, {}).items() if s not in sides}
        session["time"] = old.get("time", session["time"])
    if not a.yes:
        print("This moves the arm. Bases clamped, estop.py running, other arm parked, table as you want it scanned?")
        if input("type 'go' to start > ").strip().lower() != "go":
            raise SystemExit("nothing moved")
    if a.zed:   # before any arm connects: opening the ZED stalls i2rt's motor thread (see snapshot.Zed)
        grab_top(out)
    kin, checker = calib.Kin(), calib.calib_checker(calib.PLAN_CLEARANCE)
    for side in sides:
        (out / side).mkdir(exist_ok=True)
        other = "left" if side == "right" else "right"
        session["other_arm_q"][other] = [0.0] * 7

        def body(mover, q_rest, side=side):
            session["rest_q"][side] = np.asarray(q_rest, float).tolist()   # where this arm sits while the other scans
            plan = make_plan(side, spec, q_rest, a.table_views, a.room_views, a.seed, grid, kin, checker)
            session["plans"][side] = plan
            print(f"{side}: {len(plan['views'])} views planned ({plan['counts']})", flush=True)
            cam = Cam(side)
            try:
                if a.lock_exposure:
                    session["cameras"][side] = cam.lock()
                n = len(list((out / side).glob("*.json")))
                for i, v in enumerate(plan["views"]):
                    q_goal = np.asarray(v["q"], float)
                    why = mover.path_ok(mover.arm.read()[:6], q_goal, check=mover.path_check)
                    if why:
                        print(f"  view {i}: skipped ({why})", flush=True)
                        continue
                    mover.move_to(q_goal, max_vel=a.max_vel)
                    time.sleep(a.settle)
                    q = mover.joints()
                    frame = cam.grab()
                    sharp = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                    tag = f"{n:03d}"
                    cv2.imwrite(str(out / side / f"{tag}.png"), frame)
                    (out / side / f"{tag}.json").write_text(json.dumps(
                        {"q": q.tolist(), "t": time.time(), "kind": v["kind"], "target": v["target"],
                         "camera_pose_planned": v["camera_pose_planned"], "q_commanded": v["q"],
                         "max_tracking_lag_rad": mover.max_lag.tolist(), "sharpness_laplacian_var": sharp}))
                    n += 1
                    print(f"  view {i + 1}/{len(plan['views'])} {v['kind']}: saved {tag} (sharpness {sharp:.0f})", flush=True)
                    (out / "session.json").write_text(json.dumps(session, indent=1) + "\n")
            finally:
                cam.close()

        calib.supervised(side, kin, checker, body)
    (out / "session.json").write_text(json.dumps(session, indent=1) + "\n")
    print(f"done: {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        c = sub.add_parser(name)
        c.add_argument("--spec", default=str(DEFAULT_SPEC), help="rig camera spec (default $YAM_RIG_SPEC or duchamp3)")
        c.add_argument("--table-z", type=float, default=TABLE_Z, help="table top in the right base frame (default $YAM_TABLE_Z)")
        c.add_argument("--side", choices=["right", "left", "both"], required=True)
        c.add_argument("--out", required=name == "run")
        c.add_argument("--table-views", type=int, default=45)
        c.add_argument("--room-views", type=int, default=15)
        c.add_argument("--seed", type=int, default=0)
        c.add_argument("--grid-x", type=float, nargs="+", default=[0.22, 0.30, 0.38], help="table targets, forward (m)")
        c.add_argument("--grid-y", type=float, nargs="+", default=[-0.10, 0.0, 0.12, 0.24],
                       help="table targets, toward the other arm (m)")
    c.add_argument("--zed-serial", type=int, help="top ZED serial (default: agent_data_rl rig.py ZED_SERIAL)")
    c.add_argument("--condition", help="lighting label, e.g. day_1400, dusk_roomlights, night")
    c.add_argument("--notes", default="")
    c.add_argument("--max-vel", type=float, default=0.3)
    c.add_argument("--settle", type=float, default=0.4, help="extra settle time (s) after calib.Mover's 0.6 s")
    c.add_argument("--no-lock-exposure", dest="lock_exposure", action="store_false")
    c.add_argument("--zed", action="store_true", help="also save one frame of the top ZED 2 (taken before the arms connect)")
    c.add_argument("--yes", action="store_true", help="skip the 'go' prompt")
    c.add_argument("--cam-device", nargs=2, action="append", metavar=("SIDE", "PATH"), default=[],
                   help="wrist camera device for a side, when its cable is on another USB port than agent_data_rl's rig.py says")
    a = p.parse_args()
    {"plan": cmd_plan, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
