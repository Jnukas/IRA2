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

# --- asset helpers -----------------------------------------------------------
try:
    import trimesh
except Exception:
    trimesh = None

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
    "TABLE1": (-1.3, -2),         # First table
    "TABLE2": (1.3, 1),          # Work table (second table = grocery table)
    "POT": (-1.15, -1),           # On table 1
    "JUG": (-0.65, 0),             # On table 2 (1.50 - 0.3)
    "PEPPER": (0.35, -0.7),         # On table 2 (1.50 - 0.3)
    "BEEF": (0.2, -1),            # On table 2
    "FRUIT_VEG": (0.3, -0.5),      # On table 2
    "CHICKEN": (0.25, -1),         # On table 2 (1.50 + 0.3)
    "UR3": (0, -0.5),             # LinearUR3 rail position
    "CR3": (-0.6, 0),            # CR3 robot
    "CR16": (-0.6, -2),           # CR16 robot
}

# HEIGHT ADJUSTMENTS (in meters)
# Negative = lower, Positive = raise
# Start with 0 for all objects, then adjust if they float or sink
HEIGHT_OFFSETS = {
    # Robots
    "CR3": -0.02,      # On table 1
    "CR16": -0.02,     # On table 1
    "UR3": 0,      # On floor
    
    # Tables
    "STOVE": 0.0,    # On floor
    "TABLE1": 0.0,   # On floor
    "TABLE2": 0.0,   # On floor
    
    # Items on tables
    "POT": -0.02,      # On table 1
    "JUG": -0.02,      # On table 2
    "PEPPER": -0.02,   # On table 2
    "BEEF": -0.02,     # On table 2
    "FRUIT_VEG": -0.02,  # On table 2
    "CHICKEN": -0.02,  # On table 2
}

# ===== Motion toggles =====
RUN_WIGGLE_CR3  = True
RUN_WIGGLE_CR16 = True
RUN_RAIL_SLIDE  = True
FPS = 60
DT = 1.0 / FPS

# ===== Safety Controller (unchanged) =====
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


def main():
    # -------------------------
    # Launch Swift
    # -------------------------
    apply_swift_browser_fix()
    env = swift.Swift()
    env.launch(realtime=True, browser=None, host="127.0.0.1", port=52100, ws_port=53100)

    safety = SafetyController()
    threading.Thread(target=launch_safety_gui, args=(safety,), daemon=True).start()

    # -------------------------
    # Room Constants
    # -------------------------
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

    # -------------------------
    # Helper: Add mesh with consistent Z positioning
    # -------------------------
    def add_mesh(obj_name: str, filename: str, rotation_z: float = 0.0,
                 z_base: float = FLOOR_TOP, color=None, extra_rotation=None):
        """
        Add a mesh with automatic Z positioning and named height offset.
        
        Args:
            obj_name: Name key for POSITIONS and HEIGHT_OFFSETS lookups
            filename: STL filename (looked up in SCALES dict)
            rotation_z: Rotation around Z axis (radians)
            z_base: Base height (default: FLOOR_TOP for floor-level objects)
            color: RGBA color list
            extra_rotation: Additional SE3 rotation to apply after Rz
        
        Returns:
            (mesh_object, top_z) where top_z is the height of the top surface
        """
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
        
        # Build transformation: translate to position, then rotate
        T = SE3(x, y, z_base + z_lift + height_offset) @ SE3.Rz(rotation_z)
        if extra_rotation is not None:
            T = T @ extra_rotation
        
        mesh.T = T
        env.add(mesh)
        
        return mesh, top_z

    # -------------------------
    # Add all objects with named positioning
    # -------------------------
    
    SMALL_GAP = 0.003  # Small gap to prevent Z-fighting
    TABLE_HEIGHT = 0.45  # Hardcoded table surface height (same as CR3)
    
    # Floor objects
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
    print(f"✓ Table 1 top surface at Z = {table_top_z:.4f} m")

    work_table, table2_top_z = add_mesh(
        "TABLE2", "rightwayup.stl",
        rotation_z=0.0,
        extra_rotation=SE3.RPY([-90, 0, 0], order='xyz', unit='deg'),
        color=[0.50, 0.50, 0.50, 1.0]
    )
    print(f"✓ Table 2 top surface at Z = {table2_top_z:.4f} m")

    # Objects on table 1 - using hardcoded height
    pot, _ = add_mesh(
        "POT", "Potwithoutthelid.stl",
        z_base=TABLE_HEIGHT + SMALL_GAP,
        color=[1.0, 0.0, 0.0, 1.0]
    )

    # Objects on table 2 - using hardcoded height
    jug, _ = add_mesh(
        "JUG", "jugfixed.stl",
        z_base=TABLE_HEIGHT + SMALL_GAP,
        color=[0.8, 0.9, 1.0, 1.0]
    )

    pepper_grinder, _ = add_mesh(
        "PEPPER", "pepper_grinder.stl",
        z_base=TABLE_HEIGHT + SMALL_GAP,
        color=[0.2, 0.2, 0.2, 1.0]
    )

    beef, _ = add_mesh(
        "BEEF", "beef.stl",
        z_base=TABLE_HEIGHT + SMALL_GAP,
        color=[0.8, 0.3, 0.3, 1.0]
    )

    fruit_veg_tray, _ = add_mesh(
        "FRUIT_VEG", "Fruit_and_Vegetables_Tray.stl",
        z_base=TABLE_HEIGHT + SMALL_GAP,
        color=[0.4, 0.7, 0.3, 1.0]
    )

    chicken, _ = add_mesh(
        "CHICKEN", "chicken.stl",
        z_base=TABLE_HEIGHT + SMALL_GAP,
        color=[1.0, 0.9, 0.7, 1.0]
    )

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
    # CR3 Robot
    # -------------------------
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

    # -------------------------
    # CR16 Robot
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

    # -------------------------
    # Scene Summary
    # -------------------------
    env.step(0.02)
    print("\n" + "="*70)
    print("[Scene] All objects positioned with full control")
    print(f"  FLOOR_TOP = {FLOOR_TOP:.4f} m")
    print(f"  TABLE_HEIGHT (hardcoded) = {TABLE_HEIGHT:.4f} m")
    print(f"  Table 1 top = {table_top_z:.4f} m")
    print(f"  Table 2 top = {table2_top_z:.4f} m")
    print("\n💡 To reposition objects:")
    print("   • Edit POSITIONS dictionary for X,Y coordinates")
    print("   • Edit HEIGHT_OFFSETS dictionary for Z adjustments")
    print("   • Edit TABLE_HEIGHT constant to change table surface height")
    print("="*70 + "\n")

    # -------------------------
    # UR3 Movement to Tray
    # -------------------------
    interpolation = 2           # 1 = Quintic Polynomial, 2 = Trapezoidal Velocity
    steps = 50                  # Specify no. of steps
    
    # Get tray position
    tray_x, tray_y = POSITIONS["FRUIT_VEG"]
    tray_z = TABLE_HEIGHT + 0.15  # Approach from above (15cm above table)
    
    # Define end-effector pose as a 4x4 Homogeneous Transformation Matrix
    # Use SE3 to create translation and rotation (vertical gripper pointing down)
    T_tray = SE3(tray_x, tray_y, tray_z) @ SE3.Rx(math.pi)
    
    # Get current joint configuration
    q_current = ur3.q.copy()
    
    # Solve inverse kinematics to get required joint angles
    print("\n[UR3] Calculating inverse kinematics for tray position...")
    sol = ur3.ikine_LM(T_tray, q0=q_current)
    q_tray = sol.q
    
    # Check if solution is within joint limits
    q_tray_in_limits = not ur3.islimit(q_tray)
    print(f"q_tray within joint limits: {q_tray_in_limits}")
    if not q_tray_in_limits:
        print(f"Warning: q_tray may be outside limits: {q_tray}")
    
    # Generate a matrix of interpolated joint angles using chosen method
    if interpolation == 1:
        # Quintic Polynomial
        q_matrix = rtb.jtraj(q_current, q_tray, steps).q
    elif interpolation == 2:
        # Trapezoidal Velocity
        from roboticstoolbox import trapezoidal
        s = trapezoidal(0, 1, steps).q                          # Create the scalar function
        q_matrix = np.empty((steps, ur3.n))                     # Create memory allocation
        for i in range(steps):
            q_matrix[i, :] = (1 - s[i]) * q_current + s[i] * q_tray  # Generate interpolated joint angles
    else:
        raise ValueError("interpolation = 1 for Quintic Polynomial, or 2 for Trapezoidal Velocity")
    
    print(f"\n[UR3] Moving to tray at ({tray_x:.2f}, {tray_y:.2f}, {tray_z:.2f})...")
    print("Press Ctrl+C to stop\n")
    
    # Animate the trajectory
    for i, q in enumerate(q_matrix):
        safety.block_until_allowed(env, DT)
        ur3.q = q
        env.step(DT)
        time.sleep(DT)
    
    print("[UR3] Reached tray position!")

    env.set_camera_pose([1.8, 3.4, 1.6], [0.0, -0.5, 0.8])
    env.hold()


if __name__ == "__main__":
    main()