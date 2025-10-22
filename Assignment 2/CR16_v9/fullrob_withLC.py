#!/usr/bin/env python3
# room_only_fixed.py — Z-positioning consistency fixes with FULL CONTROL (+ safety curtain, avoidance, RMRC)
from __future__ import annotations

from pathlib import Path
import math
import time
import importlib.util
import inspect
import threading
import tkinter as tk

import numpy as np
import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
from spatialmath import SE3

from room_utils import apply_swift_browser_fix, make_room
from ir_support.robots.LinearUR3 import LinearUR3
from ir_support import RectangularPrism, line_plane_intersection

from spatialmath.base import tr2rpy, tr2delta
from spatialmath import UnitQuaternion

# --- asset helpers -----------------------------------------------------------
try:
    import trimesh
except Exception:
    trimesh = None

from itertools import combinations


def _np(v):
    import numpy as _np
    return _np.array(v, dtype=float)


def mesh_bounds_info(path: Path, scale_vec):
    """
    Returns (z_lift, size_m) where z_lift raises the mesh so its lowest vertex sits at z=0,
    using the SAME scale you pass to sg.Mesh (either scalar or [sx,sy,sz]).
    If trimesh is unavailable, returns (0.0, None).
    """
    if trimesh is None:
        return 0.0, None

    tm = trimesh.load_mesh(str(path), process=False)

    bmin = np.asarray(tm.bounds[0], dtype=float)
    bmax = np.asarray(tm.bounds[1], dtype=float)

    sv = np.asarray(scale_vec, dtype=float)
    if sv.size == 1:
        sv = np.repeat(float(sv), 3)
    elif sv.size != 3:
        raise ValueError("scale_vec must be a scalar or a 3-vector")

    size_m = (bmax - bmin) * sv
    z_lift = -bmin[2] * sv[2]

    return float(z_lift), size_m


# ===== VERIFIED SCALES (Based on YOUR actual STL verification output) =====
SCALES = {
    # Already in meters - no conversion needed
    "Stove.stl": [1.0, 1.0, 1.0],
    "table.stl": [1.0, 1.0, 1.0],
    "pepper_grinder.stl": [1.0, 1.0, 1.0],
    "barrier.stl": [1.0, 1.0, 1.0],

    # In millimeters - convert to meters
    "rightwayup.stl": [1.0, 1.0, 1.0],
    "Potwithoutthelid.stl": [0.001, 0.001, 0.001],
    "jugfixed.stl": [1.0, 1.0, 1.0],
    "Fruit_and_Vegetables_Tray.stl": [0.001, 0.001, 0.001],

    # In centimeters - convert to meters
    "beef.stl": [0.01, 0.01, 0.01],
    "chicken.stl": [0.01, 0.01, 0.01],
}

# =====================================================================
# POSITION CONTROL CENTER
# =====================================================================
# Change X, Y positions here (in meters, relative to room center)
POSITIONS = {
    # Format: "NAME": (x = red, y = green -closer to back wall)
    "STOVE": (0.0, -2.4),           # Back wall
    "TABLE1": (-1.3, -2.5),         # First table
    "TABLE2": (1.3, 0.5),           # Work table (second table = grocery table)
    "POT": (-1.15, -1),             # On table 1
    "JUG": (0.4, -1.3),             # On table 2 (1.50 - 0.3)
    "PEPPER": (0.35, -0.7),         # On table 2 (1.50 - 0.3)
    "BEEF": (0.2, -1),              # On table 2
    "FRUIT_VEG": (0.3, -0.5),       # On table 2
    "CHICKEN": (0.25, -1),          # On table 2 (1.50 + 0.3)
    "UR3": (0, -0.5),               # LinearUR3 rail position
    "CR3": (-0.5, -0.4),            # CR3 robot
    "CR16": (-0.5, -1.5),           # CR16 robot
    "BARRIER": (-0.75, 1.25),
}

# HEIGHT ADJUSTMENTS (in meters)
HEIGHT_OFFSETS = {
    # Robots
    "CR3": -0.02,      # On table 1
    "CR16": -0.02,     # On table 1
    "UR3": 0,          # On floor

    # Tables
    "STOVE": 0.0,    # On floor
    "TABLE1": 0.0,   # On floor
    "TABLE2": 0.0,   # On floor

    # Items on tables
    "POT": -0.02,        # On table 1
    "JUG": -0.02,        # On table 2
    "PEPPER": -0.02,     # On table 2
    "BEEF": -0.02,       # On table 2
    "FRUIT_VEG": -0.02,  # On table 2
    "CHICKEN": -0.02,    # On table 2
    "BARRIER": 0.0,
}

# ===== Motion / sim timing =====
FPS = 60
DT = 1.0 / FPS

# ========================= hardware =========================
import serial  # pip install pyserial


def start_arduino_estop_listener(safety, port="COM3", baud=115200, verbose=True):
    """
    Listen for lines like 'E,1' from the Arduino.
    On 'E,1' -> engage_e_stop() (latched). 'E,0' is ignored (resume via GUI).
    """
    def _worker():
        try:
            ser = serial.Serial(port, baudrate=baud, timeout=0.05)
            time.sleep(2.0)  # Uno resets on serial open; give it time to boot
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            print(f"[E-STOP] Listening on {port} @ {baud}")
        except Exception as e:
            print(f"[E-STOP] Serial open failed ({port}): {e}")
            return

        while True:
            try:
                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue
                if verbose:
                    print(f"[E-STOP] RX: {line}")
                if line.startswith("E,"):
                    parts = line.split(",", 1)
                    if len(parts) == 2 and parts[1].strip() == "1":
                        if not safety.e_stop_engaged:
                            print("🔴 [E-STOP] Arduino button pressed → E-STOP engaged")
                        safety.engage_e_stop()
            except Exception as e:
                print(f"[E-STOP] Listener error: {e}")
                break

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


# ===== Safety Controller =====
class SafetyController:
    def __init__(self):
        self._lock = threading.Lock()
        self._run_evt = threading.Event()
        self._run_evt.set()
        self.e_stop_engaged = False

    def engage_e_stop(self):
        with self._lock:
            self.e_stop_engaged = True
            self._run_evt.clear()

    def disengage_e_stop(self):
        with self._lock:
            self.e_stop_engaged = False

    def resume(self):
        with self._lock:
            if not self.e_stop_engaged:
                self._run_evt.set()

    def is_running(self) -> bool:
        return self._run_evt.is_set() and not self.e_stop_engaged

    def block_until_allowed(self, env, dt: float):
        while not self._run_evt.wait(timeout=dt):
            try:
                env.step(dt)
            except Exception:
                time.sleep(dt)


def launch_safety_gui(safety: SafetyController):
    root = tk.Tk()
    root.title("Safety Panel")
    root.geometry("280x190")
    try:
        root.wm_attributes("-topmost", True)
    except Exception:
        pass

    status = tk.StringVar(value="RUNNING")

    def refresh_label():
        if safety.e_stop_engaged:
            status.set("E-STOP ENGAGED")
        elif safety.is_running():
            status.set("RUNNING")
        else:
            status.set("READY (disengaged, press RESUME)")

    def on_estop():
        if not safety.e_stop_engaged:
            safety.engage_e_stop()
        else:
            safety.disengage_e_stop()
        refresh_label()

    def on_resume():
        safety.resume()
        refresh_label()

    font_btn = ("Segoe UI", 16, "bold")
    btn_estop = tk.Button(root, text="E-STOP", command=on_estop,
                          bg="#b30000", fg="white", font=font_btn, height=2)
    btn_resume = tk.Button(root, text="RESUME", command=on_resume,
                           bg="#006400", fg="white", font=font_btn, height=2)
    lbl = tk.Label(root, textvariable=status, font=("Segoe UI", 12))

    btn_estop.pack(fill="x", padx=10, pady=(12, 6))
    btn_resume.pack(fill="x", padx=10, pady=6)
    lbl.pack(pady=(8, 6))

    refresh_label()
    root.mainloop()


# ---------- robot load helpers ----------

def _load_robot_class(pyfile: Path, prefer_names: tuple[str, ...]) -> type:
    if not pyfile.exists():
        raise FileNotFoundError(f"Robot file not found: {pyfile}")

    spec = importlib.util.spec_from_file_location("user_robot_module", str(pyfile))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    for name in prefer_names:
        if hasattr(mod, name) and inspect.isclass(getattr(mod, name)):
            return getattr(mod, name)

    candidates = []
    RobotBase = getattr(rtb, "Robot", tuple())
    for name, obj in vars(mod).items():
        if inspect.isclass(obj):
            try:
                if issubclass(obj, (rtb.ERobot, rtb.DHRobot, RobotBase)):
                    candidates.append((name, obj))
            except Exception:
                pass

    if candidates:
        print(f"[loader] Using {candidates[0][0]} from {pyfile.name}")
        return candidates[0][1]

    raise ImportError(f"No robot class found in {pyfile.name}.")


# ---------- geometry + collision ----------

def is_intersection_point_inside_triangle(intersect_p, triangle_verts):
    """Check if intersection point is inside a triangle using barycentric coordinates"""
    u = triangle_verts[1, :] - triangle_verts[0, :]
    v = triangle_verts[2, :] - triangle_verts[0, :]

    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)

    w = intersect_p - triangle_verts[0, :]
    wu = np.dot(w, u)
    wv = np.dot(w, v)

    D = uv * uv - uu * vv

    if abs(D) < 1e-9:
        return False

    # Get and test parametric coords (s and t)
    s = (uv * wv - vv * wu) / D
    if s < 0.0 or s > 1.0:
        return False

    t = (uv * wu - uu * wv) / D
    if t < 0.0 or (s + t) > 1.0:
        return False

    return True


def get_link_points(robot, q=None):
    """Return Nx3 array of joint positions (base + each link end) in world frame"""
    if q is None:
        q = robot.q

    T_all = robot.fkine_all(q)

    base_pos = robot.base.t if hasattr(robot.base, 't') else robot.base.A[:3, 3]
    pts = [base_pos.tolist() if hasattr(base_pos, 'tolist') else list(base_pos)]

    if hasattr(T_all, '__iter__'):
        for T in T_all:
            if hasattr(T, 't'):
                pts.append(T.t.tolist())
            elif hasattr(T, 'A'):
                pts.append(T.A[:3, 3].tolist())
            else:
                pts.append(T[:3, 3].tolist())
    else:
        if hasattr(T_all, 't'):
            pts.append(T_all.t.tolist())
        elif hasattr(T_all, 'A'):
            pts.append(T_all.A[:3, 3].tolist())

    return np.asarray(pts, dtype=float)


def check_collision(robot, q, obstacles_list, env=None, visualize=False, robot_name="Robot"):
    """
    Check if robot at configuration q collides with any obstacles.
    Returns True if collision detected, False otherwise.
    """
    pts = get_link_points(robot, q)

    for i in range(len(pts) - 1):
        link_start = pts[i]
        link_end = pts[i + 1]

        for obs_idx, (faces, vertices, face_normals) in enumerate(obstacles_list):
            for j, face in enumerate(faces):
                vert_on_plane = vertices[face][0]
                intersect_p, check = line_plane_intersection(
                    face_normals[j],
                    vert_on_plane,
                    link_start,
                    link_end,
                )

                if check == 1:
                    triangle_list = np.array(list(combinations(face, 3)), dtype=int)
                    for triangle in triangle_list:
                        if is_intersection_point_inside_triangle(intersect_p, vertices[triangle]):
                            print(f"🔴 [{robot_name}] COLLISION DETECTED!")
                            print(
                                f"   Link {i} collided with obstacle {obs_idx+1} at position: "
                                f"({intersect_p[0]:.3f}, {intersect_p[1]:.3f}, {intersect_p[2]:.3f})"
                            )

                            if visualize and env is not None:
                                collision_sphere = sg.Sphere(radius=0.05, color=[1.0, 0.0, 0.0, 1.0])
                                collision_sphere.T = SE3(intersect_p[0], intersect_p[1], intersect_p[2]).A
                                env.add(collision_sphere)
                            return True
    return False


# ---------- teach panel (unchanged) ----------

def teach_multi_swift(robots: dict, env, safety, default="CR16", dt=0.02):
    """Swift teach panel for multiple robots (e.g., CR16 and CR3)."""
    active = {"name": default if default in robots else next(iter(robots.keys()))}

    def is_prismatic(link):
        return bool(getattr(link, "isprismatic", getattr(link, "sigma", 0) == 1))

    def qlim_pair(link):
        ql = getattr(link, "qlim", None)
        if ql is None:
            return None, None
        arr = np.asarray(ql).ravel()
        return (float(arr[0]), float(arr[1])) if arr.size >= 2 else (None, None)

    def current_T(rbt):
        T = rbt.fkine(rbt.q)
        return T.A if hasattr(T, "A") else np.asarray(T)

    def pose_strings(rbt):
        T = current_T(rbt)
        xyz = T[:3, 3]
        rpy = tr2rpy(T, unit="deg")  # ZYX
        qobj = UnitQuaternion(T[:3, :3])
        w = float(getattr(qobj, "s", 1.0))
        v = getattr(qobj, "v", np.array([0.0, 0.0, 0.0], dtype=float))
        x, y, z = [float(c) for c in np.ravel(v)]
        return (
            f"X: {xyz[0]:.3f} m",
            f"Y: {xyz[1]:.3f} m",
            f"Z: {xyz[2]:.3f} m",
            f"Roll (φ): {rpy[0]:.2f}°",
            f"Pitch (θ): {rpy[1]:.2f}°",
            f"Yaw (ψ): {rpy[2]:.2f}°",
            f"Quat: w={w:.3f}, x={x:.3f}, y={y:.3f}, z={z:.3f}",
        )

    header = swift.Label(f"Teach Panel – ACTIVE: {active['name']}")
    env.add(header)

    def set_active(name):
        active["name"] = name
        header.desc = f"Teach Panel – ACTIVE: {name}"
        for nm, lbl in section_labels.items():
            lbl.desc = f"{nm} sliders [{'ACTIVE' if nm == active['name'] else 'inactive'}]"

    for name in robots.keys():
        env.add(swift.Button(desc=f"Control {name}", cb=lambda *_, n=name: set_active(n)))

    lbl_x = swift.Label(""); env.add(lbl_x)
    lbl_y = swift.Label(""); env.add(lbl_y)
    lbl_z = swift.Label(""); env.add(lbl_z)
    lbl_r = swift.Label(""); env.add(lbl_r)
    lbl_p = swift.Label(""); env.add(lbl_p)
    lbl_yw = swift.Label(""); env.add(lbl_yw)
    lbl_q = swift.Label(""); env.add(lbl_q)

    def refresh_labels():
        rbt = robots[active["name"]]
        x_s, y_s, z_s, r_s, p_s, yaw_s, q_s = pose_strings(rbt)
        lbl_x.desc, lbl_y.desc, lbl_z.desc = x_s, y_s, z_s
        lbl_r.desc, lbl_p.desc, lbl_yw.desc = r_s, p_s, yaw_s
        lbl_q.desc = q_s

    lbl_estop = swift.Label("E-STOP: DISENGAGED"); env.add(lbl_estop)

    def on_estop(*_):
        safety.engage_e_stop()
        lbl_estop.desc = "E-STOP: ENGAGED"

    def on_resume(*_):
        safety.disengage_e_stop()
        safety.resume()
        lbl_estop.desc = "E-STOP: DISENGAGED"

    env.add(swift.Button(desc="E-STOP", cb=on_estop))
    env.add(swift.Button(desc="Resume", cb=on_resume))

    section_labels = {}
    for name, rbt in robots.items():
        section_label = swift.Label(f"{name} sliders [{'ACTIVE' if name == active['name'] else 'inactive'}]")
        section_labels[name] = section_label
        env.add(section_label)

        offsets = [float(getattr(L, "offset", 0.0)) for L in rbt.links]

        for j in range(rbt.n):
            link = rbt.links[j]
            pris = is_prismatic(link)
            ql0, ql1 = qlim_pair(link)
            off = offsets[j]

            if pris:
                ui_min = ql0 if ql0 is not None else -1.0
                ui_max = ql1 if ql1 is not None else 1.0
                ui_min, ui_max = sorted((ui_min, ui_max))
                value = float(rbt.q[j])

                def on_change(v, name=name, rbt=rbt, j=j, ui_min=ui_min, ui_max=ui_max):
                    if (active["name"] != name) or (not safety.is_running()):
                        return
                    q = np.asarray(rbt.q, float)
                    q[j] = float(v)
                    if ui_min is not None:
                        q[j] = float(np.clip(q[j], ui_min, ui_max))
                    rbt.q = q.tolist()
                    try:
                        env.step(0)
                    except Exception:
                        pass
                    refresh_labels()

                env.add(swift.Slider(cb=on_change, min=ui_min, max=ui_max,
                                     step=0.001, value=value, desc=f"{name} J{j+1}", unit=" m"))
            else:
                if ql0 is not None and ql1 is not None:
                    ui_min = float(np.rad2deg(ql0 + off))
                    ui_max = float(np.rad2deg(ql1 + off))
                else:
                    ui_min, ui_max = -180.0, 180.0
                ui_min, ui_max = sorted((ui_min, ui_max))
                value = float(np.rad2deg(rbt.q[j] + off))

                def on_change(v, name=name, rbt=rbt, j=j, off=off, ql0=ql0, ql1=ql1, ui_min=ui_min, ui_max=ui_max):
                    if (active["name"] != name) or (not safety.is_running()):
                        return
                    q = np.asarray(rbt.q, float)
                    q_int = np.deg2rad(float(v)) - off
                    if ql0 is not None and ql1 is not None:
                        q_int = float(np.clip(q_int, ql0, ql1))
                    q[j] = q_int
                    rbt.q = q.tolist()
                    try:
                        env.step(0)
                    except Exception:
                        pass
                    refresh_labels()

                env.add(swift.Slider(cb=on_change, min=ui_min, max=ui_max,
                                     step=1.0, value=float(np.clip(value, ui_min, ui_max)),
                                     desc=f"{name} J{j+1}", unit="°"))

    def _label_refresher():
        while True:
            try:
                refresh_labels()
            except Exception:
                pass
            time.sleep(dt)

    threading.Thread(target=_label_refresher, daemon=True).start()
    refresh_labels()


# ---------- safety curtain + helpers ----------
LIGHT_CURTAIN_Y = 0.8   # y-plane that acts like a light curtain (increase to move further from back wall)


def draw_light_curtain(env, x_min=-2.0, x_max=2.0, y=LIGHT_CURTAIN_Y, z=1.0, n=20):
    """Visualize the light curtain as a row of translucent spheres."""
    for i in range(n):
        x = x_min + (x_max - x_min) * (i / (n - 1))
        s = sg.Sphere(radius=0.03, color=[1.0, 1.0, 0.0, 0.25])
        s.T = SE3(x, y, z).A
        env.add(s)


def start_light_curtain_monitor(robots: dict, safety: SafetyController, dt=0.02):
    """Background thread: if any robot's TCP crosses the Y curtain → E‑STOP."""
    def _worker():
        tripped = False
        while True:
            try:
                for name, rbt in robots.items():
                    T = rbt.fkine(rbt.q)
                    y = float((T.A if hasattr(T, 'A') else T)[1, 3])
                    if y > LIGHT_CURTAIN_Y:
                        if not tripped:
                            print(f"🔶 [LightCurtain] Triggered by {name} (y={y:.3f}) → E‑STOP")
                        tripped = True
                        safety.engage_e_stop()
                        break
                time.sleep(dt)
            except Exception:
                time.sleep(dt)
    threading.Thread(target=_worker, daemon=True).start()


# ---------- simple planner + RMRC ----------
from roboticstoolbox import trapezoidal


def will_path_collide(robot, q_start, q_goal, obstacles_list, samples=30, name="robot"):
    for k in range(samples):
        s = k / (samples - 1)
        q = (1 - s) * np.asarray(q_start) + s * np.asarray(q_goal)
        if check_collision(robot, q, obstacles_list, robot_name=name):
            return True
    return False


def trap_qmatrix(q_start, q_goal, n_steps, n_joints):
    s = trapezoidal(0, 1, n_steps).q
    Q = np.empty((n_steps, n_joints))
    for i in range(n_steps):
        Q[i, :] = (1 - s[i]) * q_start + s[i] * q_goal
    return Q


def plan_with_go_high(robot, q_start, T_goal, obstacles_list, steps=60, name="robot"):
    """Plan q-trajectory; if a straight joint path collides, insert 'go-high' via-points."""
    sol = robot.ikine_LM(T_goal, q0=q_start)
    q_goal = sol.q

    if not will_path_collide(robot, q_start, q_goal, obstacles_list, samples=25, name=name):
        return trap_qmatrix(np.asarray(q_start), np.asarray(q_goal), steps, robot.n)

    print(f"🟡 [{name}] Predicted collision on direct path → inserting via-points")

    T_now = robot.fkine(q_start)
    z_now = float((T_now.A if hasattr(T_now, 'A') else T_now)[2, 3])
    z_goal = float((T_goal.A if hasattr(T_goal, 'A') else T_goal)[2, 3])

    z_clear = max(z_now, z_goal) + 0.35  # raise above both

    # Stay at current XY, raise Z
    T_up1 = SE3(T_now.t[0], T_now.t[1], z_clear) @ SE3(T_now.R)
    q_up1 = robot.ikine_LM(T_up1, q0=q_start).q

    # Move above goal at Z_clear
    T_up2 = SE3(T_goal.t[0], T_goal.t[1], z_clear) @ SE3(T_goal.R)
    q_up2 = robot.ikine_LM(T_up2, q0=q_up1).q

    Q1 = trap_qmatrix(np.asarray(q_start), np.asarray(q_up1), steps // 3, robot.n)
    Q2 = trap_qmatrix(np.asarray(q_up1), np.asarray(q_up2), steps // 3, robot.n)
    Q3 = trap_qmatrix(np.asarray(q_up2), np.asarray(q_goal), steps - (2 * (steps // 3)), robot.n)
    return np.vstack((Q1, Q2, Q3))


def rmrc_to_pose(robot, T_target, env, safety, steps=80, gain=0.6, name="robot"):
    """Resolved-rate servo from current pose to T_target."""
    q = np.asarray(robot.q, float)
    for _ in range(steps):
        safety.block_until_allowed(env, DT)
        T_now = robot.fkine(q)
        Delta = tr2delta(T_now.A, T_target.A)  # 6×1 twist error
        v = gain * Delta
        J = robot.jacobe(q)
        dq = np.linalg.pinv(J) @ v
        q = q + dq * DT
        robot.q = q
        env.step(DT)
        time.sleep(DT)


# ---------- main ----------

def main():
    # Launch Swift
    apply_swift_browser_fix()
    env = swift.Swift()
    env.launch(realtime=True, browser=None, host="127.0.0.1", port=52100, ws_port=53100)

    safety = SafetyController()
    threading.Thread(target=launch_safety_gui, args=(safety,), daemon=True).start()
    start_arduino_estop_listener(safety, port="COM9", baud=115200)  # change COM port if needed

    # Room Constants
    ROOM_W = 6.0
    ROOM_D = 6.0
    FLOOR_TOP = 0.005  # All Z measurements relative to this

    make_room(
        env,
        room_w=ROOM_W,
        room_d=ROOM_D,
        floor_t=0.10,
        open_side="+Y",
        floor_top=FLOOR_TOP,
    )

    # Helper: Add mesh with consistent Z positioning
    def add_mesh(obj_name: str, filename: str, rotation_z: float = 0.0,
                 z_base: float = FLOOR_TOP, color=None, extra_rotation=None):
        x, y = POSITIONS.get(obj_name, (0.0, 0.0))
        height_offset = HEIGHT_OFFSETS.get(obj_name, 0.0)

        path = Path(__file__).parent / "assets" / filename
        scale = SCALES.get(filename, [1.0, 1.0, 1.0])

        z_lift, size_m = mesh_bounds_info(path, scale)

        if size_m is not None:
            print(f"{obj_name:15s} {filename:30s} size (m): X={size_m[0]:.3f}  Y={size_m[1]:.3f}  Z={size_m[2]:.3f}")
            top_z = z_base + z_lift + float(size_m[2]) + height_offset
        else:
            print(f"{obj_name:15s} {filename:30s} [trimesh unavailable]")
            top_z = z_base + z_lift + height_offset

        mesh = sg.Mesh(
            filename=str(path),
            scale=scale,
            color=color or [0.7, 0.7, 0.7, 1.0],
        )

        T = SE3(x, y, z_base + z_lift + height_offset) @ SE3.Rz(rotation_z)
        if extra_rotation is not None:
            T = T @ extra_rotation

        mesh.T = T
        env.add(mesh)

        return mesh, top_z

    # Add all objects with named positioning
    SMALL_GAP = 0.003  # Small gap to prevent Z-fighting
    TABLE_HEIGHT = 0.45  # Hardcoded table surface height (same as CR3)

    obstacles_list = []

    # Floor objects
    stove, _ = add_mesh(
        "STOVE", "Stove.stl",
        rotation_z=math.pi,
        color=[0.70, 0.70, 0.70, 1.0],
    )

    table, table_top_z = add_mesh(
        "TABLE1", "table.stl",
        rotation_z=math.pi / 2,
        color=[0.50, 0.50, 0.50, 1.0],
    )
    print(f"✓ Table 1 top surface at Z = {table_top_z:.4f} m")

    # Add table 1 collision geometry
    table1_x, table1_y = POSITIONS["TABLE1"]
    table1_lwh = [1.5, 1.0, 0.8]  # Approximate table dimensions
    table1_center = [table1_x, table1_y, table_top_z / 2]
    vertices_t1, faces_t1, normals_t1 = RectangularPrism(
        table1_lwh[0], table1_lwh[1], table1_lwh[2], center=table1_center
    ).get_data()
    obstacles_list.append((faces_t1, vertices_t1, normals_t1))

    work_table, table2_top_z = add_mesh(
        "TABLE2", "rightwayup.stl",
        rotation_z=0.0,
        extra_rotation=SE3.RPY([-90, 0, 0], order='xyz', unit='deg'),
        color=[0.50, 0.50, 0.50, 1.0],
    )
    print(f"✓ Table 2 top surface at Z = {table2_top_z:.4f} m")

    # Add table 2 collision geometry
    table2_x, table2_y = POSITIONS["TABLE2"]
    table2_lwh = [1.5, 1.0, 0.8]
    table2_center = [table2_x, table2_y, table2_top_z / 2]
    vertices_t2, faces_t2, normals_t2 = RectangularPrism(
        table2_lwh[0], table2_lwh[1], table2_lwh[2], center=table2_center
    ).get_data()
    obstacles_list.append((faces_t2, vertices_t2, normals_t2))

    # Objects on table 1
    pot, _ = add_mesh(
        "POT", "Potwithoutthelid.stl",
        z_base=TABLE_HEIGHT + SMALL_GAP,
        color=[1.0, 0.0, 0.0, 1.0],
    )

    # Objects on table 2
    jug, _ = add_mesh(
        "JUG", "jugfixed.stl",
        z_base=TABLE_HEIGHT + SMALL_GAP,
        color=[0.8, 0.9, 1.0, 1.0],
    )

    pepper_grinder, _ = add_mesh(
        "PEPPER", "pepper_grinder.stl",
        z_base=TABLE_HEIGHT + SMALL_GAP,
        color=[0.2, 0.2, 0.2, 1.0],
    )

    beef, _ = add_mesh(
        "BEEF", "beef.stl",
        z_base=TABLE_HEIGHT + SMALL_GAP,
        color=[0.8, 0.3, 0.3, 1.0],
    )

    fruit_veg_tray, _ = add_mesh(
        "FRUIT_VEG", "Fruit_and_Vegetables_Tray.stl",
        z_base=TABLE_HEIGHT + SMALL_GAP,
        color=[0.4, 0.7, 0.3, 1.0],
    )

    chicken, _ = add_mesh(
        "CHICKEN", "chicken.stl",
        z_base=TABLE_HEIGHT + SMALL_GAP,
        color=[1.0, 0.9, 0.7, 1.0],
    )

    # Barrier visual (already part of safety equipment models)
    barrier_mesh, barrier_top_z = add_mesh(
        "BARRIER", "barrier.stl",
        rotation_z=0.0,
        z_base=FLOOR_TOP + SMALL_GAP,
        color=[0.90, 0.20, 0.20, 1.0],
    )

    # Linear UR3 on rail
    ur3 = LinearUR3()
    ur3_x, ur3_y = POSITIONS["UR3"]
    RAIL_Z = FLOOR_TOP + SMALL_GAP + HEIGHT_OFFSETS["UR3"]
    YAW = math.pi / 2

    ur3.base = SE3(ur3_x, ur3_y, RAIL_Z) @ SE3.Rz(YAW) @ ur3.base
    ur3.add_to_env(env)
    print(f"✓ UR3 base at Z = {RAIL_Z:.4f} m")

    # CR3 Robot
    CR3_FILE = Path(__file__).parent / "Cr3UR3editon.py"
    CR3Class = _load_robot_class(CR3_FILE, ("CR3", "Cr3UR3editon", "DobotCR3", "RobotCR3"))
    cr3 = CR3Class()

    cr3_x, cr3_y = POSITIONS["CR3"]
    CR3_Z = TABLE_HEIGHT + SMALL_GAP + HEIGHT_OFFSETS["CR3"]
    CR3_YAW = -math.pi / 2

    print(f"✓ CR3 base at Z = {CR3_Z:.4f} m (offset={HEIGHT_OFFSETS['CR3']:.4f})")

    base0_cr3 = getattr(cr3, "base", SE3())
    cr3.base = SE3(cr3_x, cr3_y, CR3_Z) @ SE3.Rz(CR3_YAW) @ base0_cr3

    try:
        q_spawn = cr3.q.copy()
    except Exception:
        q_spawn = np.zeros(getattr(cr3, "n", 6))
    cr3.q = q_spawn
    if hasattr(cr3, "qtest"):
        cr3.qtest = q_spawn

    if hasattr(cr3, "add_to_env"):
        cr3.add_to_env(env)
    else:
        env.add(cr3)
    env.step(0.02)

    # CR16 Robot
    CR16_FILE = Path(__file__).parent / "CR16Creator.py"
    try:
        CR16Class = _load_robot_class(CR16_FILE, ("CR16", "DobotCR16", "RobotCR16", "Cr16", "Cr16UR3Edition"))
        try:
            cr16 = CR16Class()
        except TypeError:
            cr16 = CR16Class(use_mesh=False)
    except Exception as e:
        print("[CR16] Load skipped:", e)
        cr16 = None

    if cr16 is not None:
        cr16_x, cr16_y = POSITIONS["CR16"]
        CR16_Z = TABLE_HEIGHT + SMALL_GAP + HEIGHT_OFFSETS["CR16"]
        CR16_YAW = +math.pi / 2

        print(f"✓ CR16 base at Z = {CR16_Z:.4f} m (offset={HEIGHT_OFFSETS['CR16']:.4f})")

        base0_cr16 = getattr(cr16, "base", SE3())
        cr16.base = SE3(cr16_x, cr16_y, CR16_Z) @ SE3.Rz(CR16_YAW) @ base0_cr16

        try:
            if hasattr(cr16, "q_home"):
                q_spawn16 = cr16.q_home
            elif hasattr(cr16, "qtest"):
                q_spawn16 = cr16.qtest
            else:
                q_spawn16 = cr16.q.copy()
        except Exception:
            q_spawn16 = np.zeros(getattr(cr16, "n", 6))
        cr16.q = q_spawn16

        try:
            if hasattr(cr16, "add_to_env"):
                cr16.add_to_env(env)
            else:
                from ir_support import CylindricalDHRobotPlot
                CylindricalDHRobotPlot(cr16).add_to_env(env)
            env.step(0.02)
        except Exception as e:
            print("[CR16] Visual add failed:", e)

    # ---- Teach panel for CR16 + CR3 ----
    robots = {}
    if cr16 is not None:
        robots["CR16"] = cr16
    if cr3 is not None:
        robots["CR3"] = cr3
    robots["UR3"] = ur3

    if robots:
        teach_multi_swift(robots, env, safety, default="CR16", dt=DT)

    # Safety curtain visual + background monitor
    draw_light_curtain(env, x_min=-2.0, x_max=2.0, y=LIGHT_CURTAIN_Y, z=1.0, n=25)
    start_light_curtain_monitor(robots, safety, dt=0.03)

    # -------------------------
    # Scene Summary
    # -------------------------
    env.step(0.02)
    print("
" + "="*70)
    print("[Scene] All objects positioned with full control")
    print(f"  FLOOR_TOP = {FLOOR_TOP:.4f} m")
    print(f"  TABLE_HEIGHT (hardcoded) = {TABLE_HEIGHT:.4f} m")
    print(f"  Table 1 top = {table_top_z:.4f} m")
    print(f"  Table 2 top = {table2_top_z:.4f} m")
    print(f"
[Collision Detection] Monitoring {len(obstacles_list)} obstacle(s)")
    print("
💡 To reposition objects:")
    print("   • Edit POSITIONS dictionary for X,Y coordinates")
    print("   • Edit HEIGHT_OFFSETS dictionary for Z adjustments")
    print("   • Edit TABLE_HEIGHT constant to change table surface height")
    print("="*70 + "
")

    # -------------------------
    # UR3 Movement to Beef (with avoidance + RMRC final approach)
    # -------------------------
    interpolation = 2           # 1 = Quintic Polynomial, 2 = Trapezoidal Velocity
    steps = 60                  # steps per segment

    beef_x, beef_y = POSITIONS["BEEF"]
    beef_z = TABLE_HEIGHT + 0.15  # approach above table

    T_beef = SE3(beef_x, beef_y, beef_z) @ SE3.Rx(math.pi)

    q_current = ur3.q.copy()

    print("
[UR3] Planning path to beef with collision check...")
    q_path = plan_with_go_high(ur3, q_current, T_beef, obstacles_list, steps=steps, name="UR3")

    print(f"
[UR3] Executing path to beef at ({beef_x:.2f}, {beef_y:.2f}, {beef_z:.2f})...")
    for q in q_path:
        safety.block_until_allowed(env, DT)
        ur3.q = q
        env.step(DT)
        time.sleep(DT)

    print("[UR3] At approach pose above beef. Switching to RMRC descend...")

    # RMRC: descend closer to the object (maintain orientation)
    T_touch = SE3(beef_x, beef_y, TABLE_HEIGHT + 0.05) @ SE3.Rx(math.pi)
    rmrc_to_pose(ur3, T_touch, env, safety, steps=80, gain=0.7, name="UR3")

    print("[UR3] Beef is now attached to end-effector")

    # Attach beef to UR3 end-effector with offset
    BEEF_OFFSET_X = -0.3
    BEEF_OFFSET_Y = -0.09
    BEEF_OFFSET_Z = 0
    T_offset = SE3(BEEF_OFFSET_X, BEEF_OFFSET_Y, BEEF_OFFSET_Z)

    T_ee = ur3.fkine(ur3.q)
    beef.T = T_ee @ T_offset
    env.step(DT)
    time.sleep(0.5)

    # Move UR3 to a new position with beef attached
    print("
[UR3] Moving to new position with beef attached...")
    target_x = -beef_x - 0.2
    target_y = beef_y
    target_z = beef_z + 0.05

    T_target = SE3(target_x, target_y, target_z) @ SE3.Rx(math.pi)

    q_current = ur3.q.copy()
    sol_target = ur3.ikine_LM(T_target, q0=q_current)
    q_target = sol_target.q
    q_matrix_move = trap_qmatrix(np.asarray(q_current), np.asarray(q_target), steps, ur3.n)

    for q in q_matrix_move:
        safety.block_until_allowed(env, DT)
        ur3.q = q
        T_ee = ur3.fkine(ur3.q)
        beef.T = T_ee @ T_offset
        env.step(DT)
        time.sleep(DT)

    print("[UR3] Movement complete - beef followed end-effector with offset!")

    print("
[UR3] Releasing beef...")
    T_beef_final = beef.T @ SE3.Rx(math.pi).A

    # -------------------------
    # The remaining sequence (chicken/pepper/jug, CR3 & CR16 legs)
    # is left functionally identical to your original script to
    # preserve behavior, while benefitting from the global safety
    # curtain and cleaned CR16 duplication. If you'd like, we can
    # propagate the go‑high avoidance + RMRC pattern to every leg.
    # -------------------------

    # UR3 → Chicken
    print("
[UR3] Moving to chicken...")
    chicken_x, chicken_y = POSITIONS["CHICKEN"]
    chicken_z = TABLE_HEIGHT + 0.15
    T_chicken = SE3(chicken_x, chicken_y, chicken_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    sol = ur3.ikine_LM(T_chicken, q0=q_current)
    q_chicken = sol.q
    q_matrix = trap_qmatrix(np.asarray(q_current), np.asarray(q_chicken), steps, ur3.n)
    for q in q_matrix:
        safety.block_until_allowed(env, DT)
        ur3.q = q
        env.step(DT)
        time.sleep(DT)
    print("[UR3] Reached chicken position!")
    print("[UR3] Chicken is now attached to end-effector")

    CHICKEN_OFFSET_X = -0.3
    CHICKEN_OFFSET_Y = -0.09
    CHICKEN_OFFSET_Z = 0
    T_offset_chicken = SE3(CHICKEN_OFFSET_X, CHICKEN_OFFSET_Y, CHICKEN_OFFSET_Z)
    T_ee = ur3.fkine(ur3.q)
    chicken.T = T_ee @ T_offset_chicken
    env.step(DT)
    time.sleep(0.5)

    print("
[UR3] Moving to place chicken...")
    target_chicken_x = -chicken_x - 0.2
    target_chicken_y = chicken_y
    target_chicken_z = chicken_z + 0.05
    T_target_chicken = SE3(target_chicken_x, target_chicken_y, target_chicken_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    sol_target_chicken = ur3.ikine_LM(T_target_chicken, q0=q_current)
    q_target_chicken = sol_target_chicken.q
    q_matrix_move_chicken = trap_qmatrix(np.asarray(q_current), np.asarray(q_target_chicken), steps, ur3.n)
    for q in q_matrix_move_chicken:
        safety.block_until_allowed(env, DT)
        ur3.q = q
        T_ee = ur3.fkine(ur3.q)
        chicken.T = T_ee @ T_offset_chicken
        beef.T = T_beef_final
        env.step(DT)
        time.sleep(DT)
    print("[UR3] Chicken placement complete!")
    print("
[UR3] Releasing chicken...")
    T_chicken_final = chicken.T @ SE3.Rx(math.pi).A

    # UR3 → Pepper
    print("
[UR3] Moving to pepper...")
    pepper_x, pepper_y = POSITIONS["PEPPER"]
    pepper_z = TABLE_HEIGHT + 0.15
    T_pepper = SE3(pepper_x, pepper_y, pepper_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    sol = ur3.ikine_LM(T_pepper, q0=q_current)
    q_pepper = sol.q
    q_matrix = trap_qmatrix(np.asarray(q_current), np.asarray(q_pepper), steps, ur3.n)
    for q in q_matrix:
        safety.block_until_allowed(env, DT)
        ur3.q = q
        env.step(DT)
        time.sleep(DT)
    print("[UR3] Reached pepper position!")
    print("[UR3] Pepper is now attached to end-effector")

    PEPPER_OFFSET_X = -0.3
    PEPPER_OFFSET_Y = -0.09
    PEPPER_OFFSET_Z = 0
    T_offset_pepper = SE3(PEPPER_OFFSET_X, PEPPER_OFFSET_Y, PEPPER_OFFSET_Z)
    T_ee = ur3.fkine(ur3.q)
    pepper_grinder.T = T_ee @ T_offset_pepper
    env.step(DT)
    time.sleep(0.5)

    print("
[UR3] Moving to place pepper...")
    target_pepper_x = -pepper_x - 0.2
    target_pepper_y = pepper_y
    target_pepper_z = pepper_z + 0.05
    T_target_pepper = SE3(target_pepper_x, target_pepper_y, target_pepper_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    sol_target_pepper = ur3.ikine_LM(T_target_pepper, q0=q_current)
    q_target_pepper = sol_target_pepper.q
    q_matrix_move_pepper = trap_qmatrix(np.asarray(q_current), np.asarray(q_target_pepper), steps, ur3.n)
    for q in q_matrix_move_pepper:
        safety.block_until_allowed(env, DT)
        ur3.q = q
        T_ee = ur3.fkine(ur3.q)
        pepper_grinder.T = T_ee @ T_offset_pepper
        beef.T = T_beef_final
        chicken.T = T_chicken_final
        env.step(DT)
        time.sleep(DT)
    print("[UR3] Pepper placement complete!")
    print("
[UR3] Releasing pepper...")
    T_pepper_final = pepper_grinder.T @ SE3.Rx(math.pi).A

    # UR3 → Jug
    print("
[UR3] Moving to jug...")
    jug_x, jug_y = POSITIONS["JUG"]
    jug_z = TABLE_HEIGHT + 0.15
    T_jug = SE3(jug_x, jug_y, jug_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    sol = ur3.ikine_LM(T_jug, q0=q_current)
    q_jug = sol.q
    q_matrix = trap_qmatrix(np.asarray(q_current), np.asarray(q_jug), steps, ur3.n)
    for q in q_matrix:
        safety.block_until_allowed(env, DT)
        ur3.q = q
        env.step(DT)
        time.sleep(DT)
    print("[UR3] Reached jug position!")
    print("[UR3] Jug is now attached to end-effector")

    JUG_OFFSET_X = -0.3
    JUG_OFFSET_Y = -0.09
    JUG_OFFSET_Z = 0
    T_offset_jug = SE3(JUG_OFFSET_X, JUG_OFFSET_Y, JUG_OFFSET_Z)
    T_ee = ur3.fkine(ur3.q)
    jug.T = T_ee @ T_offset_jug
    env.step(DT)
    time.sleep(0.5)

    print("
[UR3] Moving to place jug...")
    target_jug_x = -jug_x - 0.2
    target_jug_y = jug_y
    target_jug_z = jug_z + 0.05
    T_target_jug = SE3(target_jug_x, target_jug_y, target_jug_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    sol_target_jug = ur3.ikine_LM(T_target_jug, q0=q_current)
    q_target_jug = sol_target_jug.q
    q_matrix_move_jug = trap_qmatrix(np.asarray(q_current), np.asarray(q_target_jug), steps, ur3.n)
    for q in q_matrix_move_jug:
        safety.block_until_allowed(env, DT)
        ur3.q = q
        T_ee = ur3.fkine(ur3.q)
        jug.T = T_ee @ T_offset_jug
        beef.T = T_beef_final
        chicken.T = T_chicken_final
        pepper_grinder.T = T_pepper_final
        env.step(DT)
        time.sleep(DT)
    print("[UR3] Jug placement complete!")

    print("
[UR3] Releasing jug and returning to neutral position...")
    T_jug_final = jug.T @ SE3.Rx(math.pi).A

    retreat_x, retreat_y, retreat_z = 0.0, -0.5, jug_z + 0.3
    T_retreat_final = SE3(retreat_x, retreat_y, retreat_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    sol_retreat_final = ur3.ikine_LM(T_retreat_final, q0=q_current)
    q_retreat_final = sol_retreat_final.q
    q_matrix_retreat_final = trap_qmatrix(np.asarray(q_current), np.asarray(q_retreat_final), steps, ur3.n)
    for q in q_matrix_retreat_final:
        safety.block_until_allowed(env, DT)
        ur3.q = q
        beef.T = T_beef_final
        chicken.T = T_chicken_final
        pepper_grinder.T = T_pepper_final
        jug.T = T_jug_final
        env.step(DT)
        time.sleep(DT)

    print("[UR3] All tasks complete - beef, chicken, pepper, and jug placed!")

    # -------------------------
    # CR3 pick & place sequence (unchanged motion style)
    # -------------------------
    print("
[CR3] Starting pick-and-place sequence...")
    print("[CR3] Moving to beef...")

    beef_pickup_x = -beef_x - 0.2
    beef_pickup_y = beef_y
    beef_pickup_z = beef_z + 0.05 + 0.15
    T_beef_pickup = SE3(beef_pickup_x, beef_pickup_y, beef_pickup_z) @ SE3.Rx(math.pi)

    q_current_cr3 = cr3.q.copy()
    sol_cr3 = cr3.ikine_LM(T_beef_pickup, q0=q_current_cr3)
    q_beef_pickup = sol_cr3.q

    q_matrix_cr3 = trap_qmatrix(np.asarray(q_current_cr3), np.asarray(q_beef_pickup), steps, cr3.n)
    for q in q_matrix_cr3:
        safety.block_until_allowed(env, DT)
        cr3.q = q
        env.step(DT)
        time.sleep(DT)

    print("[CR3] Reached beef position!")

    CR3_OFFSET_X = 0.0
    CR3_OFFSET_Y = 0.0
    CR3_OFFSET_Z = -0.05
    T_offset_cr3 = SE3(CR3_OFFSET_X, CR3_OFFSET_Y, CR3_OFFSET_Z)

    pot_x, pot_y = POSITIONS["POT"]
    pot_z = TABLE_HEIGHT + 0.2

    print("[CR3] Moving beef to pot...")
    T_pot = SE3(pot_x, pot_y, pot_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    sol_pot = cr3.ikine_LM(T_pot, q0=q_current_cr3)
    q_pot = sol_pot.q
    q_matrix_to_pot = trap_qmatrix(np.asarray(q_current_cr3), np.asarray(q_pot), steps, cr3.n)
    for q in q_matrix_to_pot:
        safety.block_until_allowed(env, DT)
        cr3.q = q
        T_ee_cr3 = cr3.fkine(cr3.q)
        beef.T = (T_ee_cr3 @ T_offset_cr3 @ SE3.Rx(math.pi)).A
        env.step(DT)
        time.sleep(DT)

    print("[CR3] Beef placed in pot!")
    beef.T = SE3(pot_x, pot_y, pot_z - 0.1).A

    print("[CR3] Moving to chicken...")
    chicken_pickup_x = -chicken_x - 0.2
    chicken_pickup_y = chicken_y
    chicken_pickup_z = chicken_z + 0.05 + 0.15
    T_chicken_pickup = SE3(chicken_pickup_x, chicken_pickup_y, chicken_pickup_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    sol_chicken = cr3.ikine_LM(T_chicken_pickup, q0=q_current_cr3)
    q_chicken_pickup = sol_chicken.q
    q_matrix_cr3 = trap_qmatrix(np.asarray(q_current_cr3), np.asarray(q_chicken_pickup), steps, cr3.n)
    for q in q_matrix_cr3:
        safety.block_until_allowed(env, DT)
        cr3.q = q
        env.step(DT)
        time.sleep(DT)

    print("[CR3] Moving chicken to pot...")
    q_current_cr3 = cr3.q.copy()
    sol_pot = cr3.ikine_LM(T_pot, q0=q_current_cr3)
    q_pot = sol_pot.q
    q_matrix_to_pot = trap_qmatrix(np.asarray(q_current_cr3), np.asarray(q_pot), steps, cr3.n)
    for q in q_matrix_to_pot:
        safety.block_until_allowed(env, DT)
        cr3.q = q
        T_ee_cr3 = cr3.fkine(cr3.q)
        chicken.T = (T_ee_cr3 @ T_offset_cr3 @ SE3.Rx(math.pi)).A
        env.step(DT)
        time.sleep(DT)
    print("[CR3] Chicken placed in pot!")
    chicken.T = SE3(pot_x, pot_y, pot_z - 0.1).A

    print("[CR3] Moving to pepper...")
    pepper_pickup_x = -pepper_x - 0.2
    pepper_pickup_y = pepper_y
    pepper_pickup_z = pepper_z + 0.05 + 0.15
    T_pepper_pickup = SE3(pepper_pickup_x, pepper_pickup_y, pepper_pickup_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    sol_pepper = cr3.ikine_LM(T_pepper_pickup, q0=q_current_cr3)
    q_pepper_pickup = sol_pepper.q
    q_matrix_cr3 = trap_qmatrix(np.asarray(q_current_cr3), np.asarray(q_pepper_pickup), steps, cr3.n)
    for q in q_matrix_cr3:
        safety.block_until_allowed(env, DT)
        cr3.q = q
        env.step(DT)
        time.sleep(DT)

    print("[CR3] Moving pepper to pot...")
    q_current_cr3 = cr3.q.copy()
    sol_pot = cr3.ikine_LM(T_pot, q0=q_current_cr3)
    q_pot = sol_pot.q
    q_matrix_to_pot = trap_qmatrix(np.asarray(q_current_cr3), np.asarray(q_pot), steps, cr3.n)
    for q in q_matrix_to_pot:
        safety.block_until_allowed(env, DT)
        cr3.q = q
        T_ee_cr3 = cr3.fkine(cr3.q)
        pepper_grinder.T = (T_ee_cr3 @ T_offset_cr3 @ SE3.Rx(math.pi)).A
        env.step(DT)
        time.sleep(DT)
    print("[CR3] Pepper placed in pot!")
    pepper_grinder.T = SE3(pot_x, pot_y, pot_z - 0.1).A

    print("[CR3] Moving to jug...")
    jug_pickup_x = -jug_x - 0.2
    jug_pickup_y = jug_y
    jug_pickup_z = jug_z + 0.05 + 0.15
    T_jug_pickup = SE3(jug_pickup_x, jug_pickup_y, jug_pickup_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    sol_jug = cr3.ikine_LM(T_jug_pickup, q0=q_current_cr3)
    q_jug_pickup = sol_jug.q
    q_matrix_cr3 = trap_qmatrix(np.asarray(q_current_cr3), np.asarray(q_jug_pickup), steps, cr3.n)
    for q in q_matrix_cr3:
        safety.block_until_allowed(env, DT)
        cr3.q = q
        env.step(DT)
        time.sleep(DT)

    print("[CR3] Moving jug to pot...")
    q_current_cr3 = cr3.q.copy()
    sol_pot = cr3.ikine_LM(T_pot, q0=q_current_cr3)
    q_pot = sol_pot.q
    q_matrix_to_pot = trap_qmatrix(np.asarray(q_current_cr3), np.asarray(q_pot), steps, cr3.n)
    for q in q_matrix_to_pot:
        safety.block_until_allowed(env, DT)
        cr3.q = q
        T_ee_cr3 = cr3.fkine(cr3.q)
        jug.T = (T_ee_cr3 @ T_offset_cr3 @ SE3.Rx(math.pi)).A
        env.step(DT)
        time.sleep(DT)

    print("[CR3] Jug placed in pot!")
    jug.T = SE3(pot_x, pot_y, pot_z - 0.1).A
    print("[CR3] All ingredients placed in pot - cooking complete!")

    # CR16 pot transfer to stove (unchanged motion style)
    print("
[CR16] Starting pot transfer to stove...")
    print("[CR16] Moving to pot...")

    pot_x, pot_y = POSITIONS["POT"]
    pot_z = TABLE_HEIGHT + 0.2
    T_pot_pickup = SE3(pot_x, pot_y, pot_z) @ SE3.Rx(math.pi)

    q_current_cr16 = cr16.q.copy()
    sol_cr16 = cr16.ikine_LM(T_pot_pickup, q0=q_current_cr16)
    q_pot_pickup = sol_cr16.q
    q_matrix_cr16 = trap_qmatrix(np.asarray(q_current_cr16), np.asarray(q_pot_pickup), steps, cr16.n)
    for q in q_matrix_cr16:
        safety.block_until_allowed(env, DT)
        cr16.q = q
        env.step(DT)
        time.sleep(DT)

    print("[CR16] Reached pot position!")

    CR16_OFFSET_X = 0.0
    CR16_OFFSET_Y = 0.0
    CR16_OFFSET_Z = -0.05
    T_offset_cr16 = SE3(CR16_OFFSET_X, CR16_OFFSET_Y, CR16_OFFSET_Z)

    stove_x, stove_y = POSITIONS["STOVE"]
    stove_z = FLOOR_TOP + 0.9

    print("[CR16] Moving pot to stove...")
    T_stove = SE3(stove_x, stove_y, stove_z) @ SE3.Rx(math.pi)

    q_current_cr16 = cr16.q.copy()
    sol_stove = cr16.ikine_LM(T_stove, q0=q_current_cr16)
    q_stove = sol_stove.q
    q_matrix_to_stove = trap_qmatrix(np.asarray(q_current_cr16), np.asarray(q_stove), steps, cr16.n)
    for q in q_matrix_to_stove:
        safety.block_until_allowed(env, DT)
        cr16.q = q
        T_ee_cr16 = cr16.fkine(cr16.q)
        pot.T = (T_ee_cr16 @ T_offset_cr16).A
        beef.T = (T_ee_cr16 @ T_offset_cr16 @ SE3(0, 0, -0.1)).A
        chicken.T = (T_ee_cr16 @ T_offset_cr16 @ SE3(0, 0, -0.1)).A
        pepper_grinder.T = (T_ee_cr16 @ T_offset_cr16 @ SE3(0, 0, -0.1)).A
        jug.T = (T_ee_cr16 @ T_offset_cr16 @ SE3(0, 0, -0.1)).A
        env.step(DT)
        time.sleep(DT)

    print("[CR16] Pot placed on stove!")

    pot.T = SE3(stove_x, stove_y, stove_z - 0.05).A
    beef.T = SE3(stove_x, stove_y, stove_z - 0.15).A
    chicken.T = SE3(stove_x, stove_y, stove_z - 0.15).A
    pepper_grinder.T = SE3(stove_x, stove_y, stove_z - 0.15).A
    jug.T = SE3(stove_x, stove_y, stove_z - 0.15).A

    print("[CR16] Pot transfer complete - ready to cook!")

    env.set_camera_pose([1.8, 3.4, 1.6], [0.0, -0.5, 0.8])
    env.hold()


if __name__ == "__main__":
    main()
