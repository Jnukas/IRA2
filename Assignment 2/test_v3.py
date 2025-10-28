#!/usr/bin/env python3
# room_only_fixed.py — Z-positioning consistency fixes with FULL CONTROL
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

from spatialmath.base import tr2rpy
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
    "barrier.stl" : [1.0, 1.0, 1.0],
    
    # In millimeters - convert to meters
    "table2.stl": [1.0, 1.0, 1.0],
    "Potwithoutthelid.stl": [1.0, 1.0, 1.0],
    "jugfixed.stl": [1.0, 1.0, 1.0],
    "Fruit_and_Vegetables_Tray.stl": [1, 1, 1],
    
    # In centimeters - convert to meters
    "beef.stl": [1.0, 1.0, 1.0],
    "chicken.stl": [1.0, 1.0, 1.0],
}

# =====================================================================
# POSITION CONTROL CENTER
# =====================================================================
POSITIONS = {
    "STOVE": (0.0, -2.4),
    

    "TABLE2": (0.80, -0.85),
    
    "JUG": (0.35, -1.3),
    "PEPPER": (0.35, -0.7),
    "BEEF": (0.35, -1),
    "FRUIT_VEG": (0.4, -0.5),
    "CHICKEN": (0.35, -1),

    "TABLE1": (-0.80, -0.85),
    "CR3": (-0.5, -0.4),
    "CR16": (-0.5, -1.5),
    "POT": (-1.15, -1),

    "UR3": (0, -0.5),

    "BARRIER" : (-0.75, 1.25),
}

HEIGHT_OFFSETS = {
    "CR3": -0.02,
    "CR16": -0.02,
    "UR3": 0,
    "STOVE": 0.0,
    "TABLE1": 0.0,
    "TABLE2": 0.0,
    "POT": -0.02,
    "JUG": -0.02,
    "PEPPER": -0.02,
    "BEEF": -0.02,
    "FRUIT_VEG": -0.02,
    "CHICKEN": -0.02,
    "BARRIER" : 0.0,
}

# ===== Motion toggles =====
RUN_WIGGLE_CR3  = True
RUN_WIGGLE_CR16 = True
RUN_RAIL_SLIDE  = True
FPS = 60
DT = 1.0 / FPS

# ========================= hardware
import serial

def start_arduino_estop_listener(safety, port="COM3", baud=115200, verbose=True):
    """
    Listen for lines like 'E,1' from the Arduino.
    On 'E,1' -> engage_e_stop() (latched). 'E,0' is ignored (resume via GUI).
    """
    def _worker():
        try:
            ser = serial.Serial(port, baudrate=baud, timeout=0.05)
            time.sleep(2.0)
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
                    link_end
                )
                
                if check == 1:
                    triangle_list = np.array(list(combinations(face, 3)), dtype=int)
                    for triangle in triangle_list:
                        if is_intersection_point_inside_triangle(intersect_p, vertices[triangle]):
                            print(f"🔴 [{robot_name}] COLLISION DETECTED!")
                            print(f"   Link {i} collided with obstacle {obs_idx+1} at position: ({intersect_p[0]:.3f}, {intersect_p[1]:.3f}, {intersect_p[2]:.3f})")
                            
                            if visualize and env is not None:
                                collision_sphere = sg.Sphere(radius=0.05, color=[1.0, 0.0, 0.0, 1.0])
                                collision_sphere.T = SE3(intersect_p[0], intersect_p[1], intersect_p[2]).A
                                env.add(collision_sphere)
                            return True
    return False


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
        rpy = tr2rpy(T, unit="deg")
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
                ui_max = ql1 if ql1 is not None else  1.0
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


# ===== HELPER FUNCTIONS TO CONSOLIDATE REPETITIVE CODE =====

def compute_ik_trajectory(robot, T_target, q_current, interpolation, steps, robot_name="Robot"):
    """
    Compute IK and generate trajectory. Prints warning if out of limits but proceeds anyway.
    Returns q_matrix for animation.
    """
    from roboticstoolbox import trapezoidal
    
    sol = robot.ikine_LM(T_target, q0=q_current)
    q_target = sol.q
    
    # Check limits and print warning, but proceed anyway (the "hard-coded" workaround)
    q_target_in_limits = not robot.islimit(q_target)
    print(f"[{robot_name}] q_target within joint limits: {q_target_in_limits}")
    if not q_target_in_limits:
        print(f"[{robot_name}] Warning: q_target may be outside limits: {q_target}")
    
    if interpolation == 1:
        q_matrix = rtb.jtraj(q_current, q_target, steps).q
    elif interpolation == 2:
        s = trapezoidal(0, 1, steps).q
        q_matrix = np.empty((steps, robot.n))
        for i in range(steps):
            q_matrix[i, :] = (1 - s[i]) * q_current + s[i] * q_target
    else:
        raise ValueError("interpolation = 1 for Quintic Polynomial, or 2 for Trapezoidal Velocity")
    
    return q_matrix


def animate_robot_movement(robot, q_matrix, safety, env, obstacles_list=None, robot_name="Robot"):
    """
    Animate robot through trajectory with optional collision checking.
    """
    for i, q in enumerate(q_matrix):
        safety.block_until_allowed(env, DT)
        
        if obstacles_list is not None:
            if check_collision(robot, q, obstacles_list, robot_name=robot_name):
                pass  # Collision message already printed
        
        robot.q = q
        env.step(DT)
        time.sleep(DT)


def animate_robot_with_object(robot, q_matrix, obj, T_offset, safety, env, robot_name="Robot"):
    """
    Animate robot through trajectory with object following end-effector.
    """
    for i, q in enumerate(q_matrix):
        safety.block_until_allowed(env, DT)
        robot.q = q
        
        T_ee = robot.fkine(robot.q)
        obj.T = (T_ee @ T_offset).A
        
        env.step(DT)
        time.sleep(DT)


def animate_robot_with_multiple_objects(robot, q_matrix, objects_with_offsets, safety, env, robot_name="Robot"):
    """
    Animate robot through trajectory with multiple objects following.
    objects_with_offsets: list of (object, T_offset) tuples
    """
    for i, q in enumerate(q_matrix):
        safety.block_until_allowed(env, DT)
        robot.q = q
        
        T_ee = robot.fkine(robot.q)
        for obj, T_offset in objects_with_offsets:
            obj.T = (T_ee @ T_offset).A
        
        env.step(DT)
        time.sleep(DT)


def main():
    # -------------------------
    # Launch Swift
    # -------------------------
    apply_swift_browser_fix()
    env = swift.Swift()
    env.launch(realtime=True, browser=None, host="127.0.0.1", port=52100, ws_port=53100)

    safety = SafetyController()
    threading.Thread(target=launch_safety_gui, args=(safety,), daemon=True).start()
    start_arduino_estop_listener(safety, port="COM9", baud=115200)

    # -------------------------
    # Room Constants
    # -------------------------
    ROOM_W = 6.0
    ROOM_D = 6.0
    FLOOR_TOP = 0.005

    make_room(
        env,
        room_w=ROOM_W,
        room_d=ROOM_D,
        floor_t=0.10,
        open_side="+Y",
        floor_top=FLOOR_TOP,
    )

    # ===================================================================
    # CALCULATE TABLE HEIGHTS FROM MESH GEOMETRY
    # ===================================================================
    print("\n" + "="*70)
    print("[Table Heights] Calculating from mesh geometry...")
    print("="*70)
    
    # Calculate table heights - these are now numerical values you can use directly
    TABLE1_HEIGHT = 0.32 
    TABLE2_HEIGHT = 0.32
    
    print("\n✅ TABLE1_HEIGHT = {:.4f} m".format(TABLE1_HEIGHT))
    print("✅ TABLE2_HEIGHT = {:.4f} m".format(TABLE2_HEIGHT))
    print("="*70 + "\n")

    # -------------------------
    # Helper: Add mesh with consistent Z positioning
    # -------------------------
    def add_mesh(obj_name: str, filename: str, rotation_z: float = 0.0,
                 z_base: float = FLOOR_TOP, color=None, extra_rotation=None):
        """Add a mesh with automatic Z positioning and named height offset."""
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

    # -------------------------
    # Add all objects
    # -------------------------
    
    SMALL_GAP = 0.003
    obstacles_list = []
    
    stove, _ = add_mesh(
        "STOVE", "Stove.stl",
        rotation_z=math.pi,
        color=[0.70, 0.70, 0.70, 1.0]
    )

    table, table_top_z = add_mesh(
        "TABLE1", "table.stl",
        rotation_z=math.pi / 2,
        color=[0.50, 0.50, 0.50, 1.0]
    )
    print(f"✓ Table 1 placed at Z = {table_top_z:.4f} m")
    
    # Add table 1 collision geometry
    table1_x, table1_y = POSITIONS["TABLE1"]
    table1_lwh = [1.5, 1.0, 0.8]
    table1_center = [table1_x, table1_y, table_top_z / 2]
    vertices_t1, faces_t1, normals_t1 = RectangularPrism(
        table1_lwh[0], table1_lwh[1], table1_lwh[2], center=table1_center
    ).get_data()
    obstacles_list.append((faces_t1, vertices_t1, normals_t1))

    work_table, table2_top_z = add_mesh(
        "TABLE2", "table2.stl",
        rotation_z=0.0,
        extra_rotation=SE3.RPY([-90, 0, 0], order='xyz', unit='deg'),
        color=[0.50, 0.50, 0.50, 1.0]
    )
    print(f"✓ Table 2 placed at Z = {table2_top_z:.4f} m")
    
    # Add table 2 collision geometry
    table2_x, table2_y = POSITIONS["TABLE2"]
    table2_lwh = [1.5, 1.0, 0.8]
    table2_center = [table2_x, table2_y, table2_top_z / 2]
    vertices_t2, faces_t2, normals_t2 = RectangularPrism(
        table2_lwh[0], table2_lwh[1], table2_lwh[2], center=table2_center
    ).get_data()
    obstacles_list.append((faces_t2, vertices_t2, normals_t2))

    # ===================================================================
    # OBJECTS ON TABLES - Using calculated heights directly!
    # ===================================================================
    # Objects on TABLE1 - use TABLE1_HEIGHT
    pot, _ = add_mesh("POT", "Potwithoutthelid.stl", z_base=TABLE1_HEIGHT + SMALL_GAP, color=[1.0, 0.0, 0.0, 1.0])
    
    # Objects on TABLE2 - use TABLE2_HEIGHT
    jug, _ = add_mesh("JUG", "jugfixed.stl", z_base=TABLE2_HEIGHT + SMALL_GAP, color=[0.8, 0.9, 1.0, 1.0])
    pepper_grinder, _ = add_mesh("PEPPER", "pepper_grinder.stl", z_base=TABLE2_HEIGHT + SMALL_GAP, color=[0.2, 0.2, 0.2, 1.0])
    beef, _ = add_mesh("BEEF", "beef.stl", z_base=TABLE2_HEIGHT + SMALL_GAP, color=[0.8, 0.3, 0.3, 1.0])
    fruit_veg_tray, _ = add_mesh("FRUIT_VEG", "Fruit_and_Vegetables_Tray.stl", z_base=TABLE2_HEIGHT + SMALL_GAP, color=[0.4, 0.7, 0.3, 1.0])
    chicken, _ = add_mesh("CHICKEN", "chicken.stl", z_base=TABLE2_HEIGHT + SMALL_GAP, color=[1.0, 0.9, 0.7, 1.0])
    
    barrier_mesh, barrier_top_z = add_mesh("BARRIER", "barrier.stl", rotation_z=0.0, z_base=FLOOR_TOP + SMALL_GAP, color=[0.90, 0.20, 0.20, 1.0])

    # -------------------------
    # Linear UR3 on rail
    # -------------------------
    ur3 = LinearUR3()
    ur3_x, ur3_y = POSITIONS["UR3"]
    RAIL_Z = FLOOR_TOP + SMALL_GAP + HEIGHT_OFFSETS["UR3"]
    YAW = math.pi / 2

    ur3.base = SE3(ur3_x, ur3_y, RAIL_Z) @ SE3.Rz(YAW) @ ur3.base
    ur3.add_to_env(env)
    print(f"✓ UR3 base at Z = {RAIL_Z:.4f} m")

    # -------------------------
    # CR3 Robot (on TABLE1)
    # -------------------------
    CR3_FILE = Path(__file__).parent / "Cr3UR3editon.py"
    CR3Class = _load_robot_class(CR3_FILE, ("CR3", "Cr3UR3editon", "DobotCR3", "RobotCR3"))
    cr3 = CR3Class()

    cr3_x, cr3_y = POSITIONS["CR3"]
    CR3_Z = TABLE1_HEIGHT + SMALL_GAP + HEIGHT_OFFSETS["CR3"]  # ← Uses TABLE1_HEIGHT
    CR3_YAW = -math.pi / 2
    
    print(f"✓ CR3 base at Z = {CR3_Z:.4f} m (TABLE1_HEIGHT + offset)")

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

    # -------------------------
    # CR16 Robot (on TABLE1)
    # -------------------------
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
        CR16_Z = TABLE1_HEIGHT + SMALL_GAP + HEIGHT_OFFSETS["CR16"]  # ← Uses TABLE1_HEIGHT
        CR16_YAW = +math.pi / 2
        
        print(f"✓ CR16 base at Z = {CR16_Z:.4f} m (TABLE1_HEIGHT + offset)")

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

    # -------------------------
    # Teach panel for CR16 + CR3
    # -------------------------
    robots = {}
    if cr16 is not None:
        robots["CR16"] = cr16
    if cr3 is not None:
        robots["CR3"] = cr3

    if robots:
        teach_multi_swift(robots, env, safety, default="CR16", dt=DT)

    # -------------------------
    # Scene Summary
    # -------------------------
    env.step(0.02)
    print("\n" + "="*70)
    print("[Scene] All objects positioned with calculated table heights")
    print(f"  FLOOR_TOP = {FLOOR_TOP:.4f} m")
    print(f"  TABLE1_HEIGHT (calculated) = {TABLE1_HEIGHT:.4f} m")
    print(f"  TABLE2_HEIGHT (calculated) = {TABLE2_HEIGHT:.4f} m")
    print(f"\n[Collision Detection] Monitoring {len(obstacles_list)} obstacle(s)")
    print("\n💡 To reposition objects:")
    print("   • Edit POSITIONS dictionary for X,Y coordinates")
    print("   • Edit HEIGHT_OFFSETS dictionary for Z adjustments")
    print("   • Swap table STLs - heights auto-calculate!")
    print(f"\n📦 Objects on TABLE1 (use TABLE1_HEIGHT = {TABLE1_HEIGHT:.4f}):")
    print("   - POT, CR3, CR16")
    print(f"\n📦 Objects on TABLE2 (use TABLE2_HEIGHT = {TABLE2_HEIGHT:.4f}):")
    print("   - JUG, PEPPER, BEEF, CHICKEN, FRUIT_VEG")
    print("="*70 + "\n")

    # -------------------------
    # Movement Configuration
    # -------------------------
    interpolation = 2
    steps = 50
    
    # ===================================================================
    # ROBOT MOVEMENTS - Using calculated table heights
    # ===================================================================
    
    # -------------------------
    # UR3 Movement to Beef (on TABLE2)
    # -------------------------
    beef_x, beef_y = POSITIONS["BEEF"]
    beef_z = TABLE2_HEIGHT + 0.15  # ← Uses TABLE2_HEIGHT
    
    T_beef = SE3(beef_x, beef_y, beef_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    print("\n[UR3] Calculating inverse kinematics for beef position...")
    q_matrix = compute_ik_trajectory(ur3, T_beef, q_current, interpolation, steps, "UR3")
    
    print(f"\n[UR3] Moving to beef at ({beef_x:.2f}, {beef_y:.2f}, {beef_z:.2f})...")
    print("Press Ctrl+C to stop\n")
    
    animate_robot_movement(ur3, q_matrix, safety, env, obstacles_list, "UR3")
    
    print("[UR3] Reached beef position!")
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
    
    # Move UR3 with beef attached
    print("\n[UR3] Moving to new position with beef attached...")
    
    target_x = -beef_x - 0.2
    target_y = beef_y
    target_z = beef_z + 0.05
    
    T_target = SE3(target_x, target_y, target_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix_move = compute_ik_trajectory(ur3, T_target, q_current, interpolation, steps, "UR3")
    animate_robot_with_object(ur3, q_matrix_move, beef, T_offset, safety, env, "UR3")
    
    print("[UR3] Movement complete - beef followed end-effector with offset!")
    
    # Release beef at target position
    print("\n[UR3] Releasing beef...")
    T_beef_final = beef.T @ SE3.Rx(math.pi).A
    
    # -------------------------
    # UR3 Movement to Chicken (on TABLE2)
    # -------------------------
    print("\n[UR3] Moving to chicken...")
    
    chicken_x, chicken_y = POSITIONS["CHICKEN"]
    chicken_z = TABLE2_HEIGHT + 0.15  # ← Uses TABLE2_HEIGHT
    
    T_chicken = SE3(chicken_x, chicken_y, chicken_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix = compute_ik_trajectory(ur3, T_chicken, q_current, interpolation, steps, "UR3")
    animate_robot_movement(ur3, q_matrix, safety, env, obstacles_list, "UR3")
    
    print("[UR3] Reached chicken position!")
    
    # Attach chicken with same offset pattern
    CHICKEN_OFFSET_X = -0.3
    CHICKEN_OFFSET_Y = -0.09
    CHICKEN_OFFSET_Z = 0
    T_offset_chicken = SE3(CHICKEN_OFFSET_X, CHICKEN_OFFSET_Y, CHICKEN_OFFSET_Z)
    
    T_ee = ur3.fkine(ur3.q)
    chicken.T = T_ee @ T_offset_chicken
    env.step(DT)
    
    # Move chicken to new position
    print("\n[UR3] Moving chicken to new position...")
    
    target_x_chicken = -chicken_x - 0.2
    target_y_chicken = chicken_y
    target_z_chicken = chicken_z + 0.05
    
    T_target_chicken = SE3(target_x_chicken, target_y_chicken, target_z_chicken) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix_move = compute_ik_trajectory(ur3, T_target_chicken, q_current, interpolation, steps, "UR3")
    animate_robot_with_object(ur3, q_matrix_move, chicken, T_offset_chicken, safety, env, "UR3")
    
    print("[UR3] Chicken movement complete!")
    
    # Release chicken
    print("\n[UR3] Releasing chicken...")
    T_chicken_final = chicken.T @ SE3.Rx(math.pi).A
    
    # -------------------------
    # UR3 Movement to Pepper (on TABLE2)
    # -------------------------
    print("\n[UR3] Moving to pepper grinder...")
    
    pepper_x, pepper_y = POSITIONS["PEPPER"]
    pepper_z = TABLE2_HEIGHT + 0.01 # ← Uses TABLE2_HEIGHT
    
    T_pepper = SE3(pepper_x, pepper_y, pepper_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix = compute_ik_trajectory(ur3, T_pepper, q_current, interpolation, steps, "UR3")
    animate_robot_movement(ur3, q_matrix, safety, env, obstacles_list, "UR3")
    
    print("[UR3] Reached pepper grinder!")
    
    # Attach pepper
    PEPPER_OFFSET_X = -0.3
    PEPPER_OFFSET_Y = -0.09
    PEPPER_OFFSET_Z = 0
    T_offset_pepper = SE3(PEPPER_OFFSET_X, PEPPER_OFFSET_Y, PEPPER_OFFSET_Z)
    
    # === After you arrive at the pepper (right before "Attach pepper") ===
    T_ee = ur3.fkine(ur3.q)          # SE3 of the gripper at grasp
    T_pep_world = SE3(pepper_grinder.T)   # current SE3 of the pepper in world

    # Capture the true grasp offset (pepper expressed in the EE frame)
    T_ee_to_pepper = T_ee.inv() @ T_pep_world

    # (Optional) snap to eliminate any micro gap at the moment of attach
    pepper_grinder.T = (T_ee @ T_ee_to_pepper).A    
    env.step(DT)
    
    # Move pepper to new position
    print("\n[UR3] Moving pepper grinder to new position...")
    
    target_x_pepper = -pepper_x - 0.05
    target_y_pepper = pepper_y
    target_z_pepper = pepper_z + 0.05
    
    T_target_pepper = SE3(target_x_pepper, target_y_pepper, target_z_pepper) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix_move = compute_ik_trajectory(ur3, T_target_pepper, q_current, interpolation, steps, "UR3")
    animate_robot_with_object(ur3, q_matrix_move, pepper_grinder, T_offset_pepper, safety, env, "UR3")
    
    print("[UR3] Pepper grinder movement complete!")
    
    # Release pepper
    print("\n[UR3] Releasing pepper grinder...")
    T_pepper_final = pepper_grinder.T @ SE3.Rx(math.pi).A
    
    # -------------------------
    # UR3 Movement to Jug (on TABLE2)
    # -------------------------
    print("\n[UR3] Moving to jug...")
    
    jug_x, jug_y = POSITIONS["JUG"]
    jug_z = TABLE2_HEIGHT + 0.15  # ← Uses TABLE2_HEIGHT
    
    T_jug = SE3(jug_x, jug_y, jug_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix = compute_ik_trajectory(ur3, T_jug, q_current, interpolation, steps, "UR3")
    animate_robot_movement(ur3, q_matrix, safety, env, obstacles_list, "UR3")
    
    print("[UR3] Reached jug!")
    
    # Attach jug
    JUG_OFFSET_X = -0.3
    JUG_OFFSET_Y = -0.09
    JUG_OFFSET_Z = 0
    T_offset_jug = SE3(JUG_OFFSET_X, JUG_OFFSET_Y, JUG_OFFSET_Z)
    
    T_ee = ur3.fkine(ur3.q)
    jug.T = T_ee @ T_offset_jug
    env.step(DT)
    
    # Move jug to new position
    print("\n[UR3] Moving jug to new position...")
    
    target_x_jug = -jug_x - 0.2
    target_y_jug = jug_y
    target_z_jug = jug_z + 0.05
    
    T_target_jug = SE3(target_x_jug, target_y_jug, target_z_jug) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix_move = compute_ik_trajectory(ur3, T_target_jug, q_current, interpolation, steps, "UR3")
    animate_robot_with_object(ur3, q_matrix_move, jug, T_offset_jug, safety, env, "UR3")
    
    print("[UR3] Jug movement complete!")
    
    # Release jug
    print("\n[UR3] Releasing jug...")
    T_jug_final = jug.T @ SE3.Rx(math.pi).A
    
    # -------------------------
    # CR3 Movement to Beef
    # -------------------------
    print("\n[CR3] Starting ingredient collection...")
    print("[CR3] Moving to beef...")
    
    beef_pickup_x = -beef_x - 0.2
    beef_pickup_y = beef_y
    beef_pickup_z = beef_z + 0.05 + 0.15
    
    T_beef_pickup = SE3(beef_pickup_x, beef_pickup_y, beef_pickup_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    
    q_matrix_cr3 = compute_ik_trajectory(cr3, T_beef_pickup, q_current_cr3, interpolation, steps, "CR3")
    animate_robot_movement(cr3, q_matrix_cr3, safety, env, robot_name="CR3")
    
    print("[CR3] Reached beef!")
    
    # Attach beef to CR3
    CR3_OFFSET_X = 0.0
    CR3_OFFSET_Y = 0.0
    CR3_OFFSET_Z = -0.05
    T_offset_cr3 = SE3(CR3_OFFSET_X, CR3_OFFSET_Y, CR3_OFFSET_Z)
    
    # Move beef to pot (on TABLE1)
    print("[CR3] Moving beef to pot...")
    
    pot_x, pot_y = POSITIONS["POT"]
    pot_z = TABLE1_HEIGHT + 0.2  # ← Uses TABLE1_HEIGHT
    
    T_pot = SE3(pot_x, pot_y, pot_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    
    q_matrix_to_pot = compute_ik_trajectory(cr3, T_pot, q_current_cr3, interpolation, steps, "CR3")
    animate_robot_with_object(cr3, q_matrix_to_pot, beef, T_offset_cr3 @ SE3.Rx(math.pi), safety, env, "CR3")
    
    print("[CR3] Beef placed in pot!")
    beef.T = SE3(pot_x, pot_y, pot_z - 0.1).A
    
    # -------------------------
    # CR3 Movement to Chicken
    # -------------------------
    print("[CR3] Moving to chicken...")
    
    chicken_pickup_x = -chicken_x - 0.2
    chicken_pickup_y = chicken_y
    chicken_pickup_z = chicken_z + 0.05 + 0.15
    
    T_chicken_pickup = SE3(chicken_pickup_x, chicken_pickup_y, chicken_pickup_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    
    q_matrix_cr3 = compute_ik_trajectory(cr3, T_chicken_pickup, q_current_cr3, interpolation, steps, "CR3")
    animate_robot_movement(cr3, q_matrix_cr3, safety, env, robot_name="CR3")
    
    print("[CR3] Moving chicken to pot...")
    
    q_current_cr3 = cr3.q.copy()
    q_matrix_to_pot = compute_ik_trajectory(cr3, T_pot, q_current_cr3, interpolation, steps, "CR3")
    animate_robot_with_object(cr3, q_matrix_to_pot, chicken, T_offset_cr3 @ SE3.Rx(math.pi), safety, env, "CR3")
    
    print("[CR3] Chicken placed in pot!")
    chicken.T = SE3(pot_x, pot_y, pot_z - 0.1).A
    
    # -------------------------
    # CR3 Movement to Pepper
    # -------------------------
    print("[CR3] Moving to pepper...")
    
    pepper_pickup_x = -pepper_x - 0.05
    pepper_pickup_y = pepper_y
    pepper_pickup_z = pepper_z + 0.05
    
    T_pepper_pickup = SE3(pepper_pickup_x, pepper_pickup_y, pepper_pickup_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    
    q_matrix_cr3 = compute_ik_trajectory(cr3, T_pepper_pickup, q_current_cr3, interpolation, steps, "CR3")
    animate_robot_movement(cr3, q_matrix_cr3, safety, env, robot_name="CR3")
    
    print("[CR3] Moving pepper to pot...")
    
    q_current_cr3 = cr3.q.copy()
    q_matrix_to_pot = compute_ik_trajectory(cr3, T_pot, q_current_cr3, interpolation, steps, "CR3")
    animate_robot_with_object(cr3, q_matrix_to_pot, pepper_grinder, T_offset_cr3 @ SE3.Rx(math.pi), safety, env, "CR3")
    
    print("[CR3] Pepper placed in pot!")
    pepper_grinder.T = SE3(pot_x, pot_y, pot_z - 0.01).A
    
    # -------------------------
    # CR3 Movement to Jug
    # -------------------------
    print("[CR3] Moving to jug...")
    
    jug_pickup_x = -jug_x - 0.2
    jug_pickup_y = jug_y
    jug_pickup_z = jug_z + 0.05 + 0.15
    
    T_jug_pickup = SE3(jug_pickup_x, jug_pickup_y, jug_pickup_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    
    q_matrix_cr3 = compute_ik_trajectory(cr3, T_jug_pickup, q_current_cr3, interpolation, steps, "CR3")
    animate_robot_movement(cr3, q_matrix_cr3, safety, env, robot_name="CR3")
    
    print("[CR3] Moving jug to pot...")
    
    q_current_cr3 = cr3.q.copy()
    q_matrix_to_pot = compute_ik_trajectory(cr3, T_pot, q_current_cr3, interpolation, steps, "CR3")
    animate_robot_with_object(cr3, q_matrix_to_pot, jug, T_offset_cr3 @ SE3.Rx(math.pi), safety, env, "CR3")
    
    print("[CR3] Jug placed in pot!")
    jug.T = SE3(pot_x, pot_y, pot_z - 0.1).A
    
    print("[CR3] All ingredients placed in pot - cooking complete!")
    
    # -------------------------
    # CR16 Movement to Pot (on TABLE1)
    # -------------------------
    print("\n[CR16] Starting pot transfer to stove...")
    print("[CR16] Moving to pot...")
    
    pot_x, pot_y = POSITIONS["POT"]
    pot_z = TABLE1_HEIGHT + 0.2  # ← Uses TABLE1_HEIGHT
    
    T_pot_pickup = SE3(pot_x, pot_y, pot_z) @ SE3.Rx(math.pi)
    q_current_cr16 = cr16.q.copy()
    
    print("[CR16] Calculating IK for pot pickup...")
    q_matrix_cr16 = compute_ik_trajectory(cr16, T_pot_pickup, q_current_cr16, interpolation, steps, "CR16")
    animate_robot_movement(cr16, q_matrix_cr16, safety, env, robot_name="CR16")
    
    print("[CR16] Reached pot position!")
    
    # Attach pot to CR16 with offset
    CR16_OFFSET_X = 0.0
    CR16_OFFSET_Y = 0.0
    CR16_OFFSET_Z = -0.05
    T_offset_cr16 = SE3(CR16_OFFSET_X, CR16_OFFSET_Y, CR16_OFFSET_Z)
    
    # Get stove position for placement
    stove_x, stove_y = POSITIONS["STOVE"]
    stove_z = FLOOR_TOP + 0.9
    
    # Move pot to stove
    print("[CR16] Moving pot to stove...")
    T_stove = SE3(stove_x, stove_y, stove_z) @ SE3.Rx(math.pi)
    
    q_current_cr16 = cr16.q.copy()
    q_matrix_to_stove = compute_ik_trajectory(cr16, T_stove, q_current_cr16, interpolation, steps, "CR16")
    
    # Animate with pot and all ingredients following
    objects_with_offsets = [
        (pot, T_offset_cr16),
        (beef, T_offset_cr16 @ SE3(0, 0, -0.1)),
        (chicken, T_offset_cr16 @ SE3(0, 0, -0.1)),
        (pepper_grinder, T_offset_cr16 @ SE3(0, 0, -0.1)),
        (jug, T_offset_cr16 @ SE3(0, 0, -0.1))
    ]
    
    animate_robot_with_multiple_objects(cr16, q_matrix_to_stove, objects_with_offsets, safety, env, "CR16")
    
    print("[CR16] Pot placed on stove!")
    
    # Release pot on stove
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