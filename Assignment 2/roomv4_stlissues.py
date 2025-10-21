#!/usr/bin/env python3
# room_only_fixed.py — Z-positioning consistency fixes
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
    "Stove.stl": [1.0, 1.0, 1.0],  # ✅ YOUR file is already in meters
    "table.stl": [1.0, 1.0, 1.0],  # ✅ YOUR file is already in meters
    "pepper_grinder.stl": [1.0, 1.0, 1.0],  # ✅ YOUR file is already in meters
    
    # In millimeters - convert to meters
    "rightwayup.stl": [0.001, 0.001, 0.001],
    "Potwithoutthelid.stl": [0.001, 0.001, 0.001],  # ✅ FIXED from [0.002, 0.002, 0.002]
    "jugfixed.stl": [0.001, 0.001, 0.001],
    "Fruit_and_Vegetables_Tray.stl": [0.001, 0.001, 0.001],
    
    # In centimeters - convert to meters
    "beef.stl": [0.01, 0.01, 0.01],
    "chicken.stl": [0.01, 0.01, 0.01],
}

# ===== HEIGHT ADJUSTMENTS =====
# If objects are floating or buried, adjust these offsets (in meters)
# Negative values = lower the object, Positive values = raise the object
# Your table is now 1.05m tall, so offsets should be near 0
HEIGHT_OFFSETS = {
    "CR3": -0.14,   # Adjust if CR3 floats or sinks into table (start with 0)
    "CR16": -0.14,  # Adjust if CR16 floats or sinks into table (start with 0)
    "POT": -0.14,   # Adjust if pot floats or sinks into table (start with 0)
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
    def add_mesh(filename: str, x: float, y: float, rotation_z: float = 0.0,
                 z_base: float = FLOOR_TOP, color=None, extra_rotation=None):
        """
        Add a mesh with automatic Z positioning.
        
        Args:
            filename: STL filename (looked up in SCALES dict)
            x, y: Horizontal position
            rotation_z: Rotation around Z axis (radians)
            z_base: Base height (default: FLOOR_TOP for floor-level objects)
            color: RGBA color list
            extra_rotation: Additional SE3 rotation to apply after Rz
        
        Returns:
            (mesh_object, top_z) where top_z is the height of the top surface
        """
        path = Path(__file__).parent / "assets" / filename
        scale = SCALES.get(filename, [1.0, 1.0, 1.0])
        
        z_lift, size_m = mesh_bounds_info(path, scale)
        
        if size_m is not None:
            print(f"{filename:30s} size (m): X={size_m[0]:.3f}  Y={size_m[1]:.3f}  Z={size_m[2]:.3f}")
            top_z = z_base + z_lift + float(size_m[2])
        else:
            print(f"{filename:30s} [trimesh unavailable, using z_lift only]")
            top_z = z_base + z_lift
        
        mesh = sg.Mesh(
            filename=str(path),
            scale=scale,
            color=color or [0.7, 0.7, 0.7, 1.0],
        )
        
        # Build transformation: translate to position, then rotate
        T = SE3(x, y, z_base + z_lift) @ SE3.Rz(rotation_z)
        if extra_rotation is not None:
            T = T @ extra_rotation
        
        mesh.T = T
        env.add(mesh)
        
        return mesh, top_z

    # -------------------------
    # Add all objects with consistent positioning
    # -------------------------
    
    # Stove (back wall)
    stove, _ = add_mesh(
        "Stove.stl",
        x=0.0,
        y=-ROOM_D / 2 + 0.60,
        rotation_z=math.pi,
        color=[0.70, 0.70, 0.70, 1.0]
    )

    # First table
    table, table_top_z = add_mesh(
        "table.stl",
        x=-1.5,
        y=-0.5,
        rotation_z=math.pi / 2,
        color=[0.50, 0.50, 0.50, 1.0]
    )
    print(f"✓ Table 1 top surface at Z = {table_top_z:.4f} m")

    # Pot on first table
    SMALL_GAP = 0.003  # Small gap to prevent Z-fighting
    pot, _ = add_mesh(
        "Potwithoutthelid.stl",
        x=-1.75,
        y=-0.5,
        z_base=table_top_z + SMALL_GAP + HEIGHT_OFFSETS["POT"],  # With manual offset
        color=[1.0, 0.0, 0.0, 1.0]
    )
    print(f"✓ Pot positioned at Z = {table_top_z + SMALL_GAP + HEIGHT_OFFSETS['POT']:.4f} m")

    # Second table (work table)
    work_table, table2_top_z = add_mesh(
        "rightwayup.stl",
        x=1.50,
        y=0.0,
        rotation_z=0.0,
        extra_rotation=SE3.RPY([-90, 0, 0], order='xyz', unit='deg'),
        color=[0.50, 0.50, 0.50, 1.0]
    )
    print(f"✓ Table 2 top surface at Z = {table2_top_z:.4f} m")

    # Items on second table
    def add_table2_item(filename, x_offset, y_offset, color, rotation=0.0):
        return add_mesh(
            filename,
            x=1.50 + x_offset,
            y=y_offset,
            rotation_z=rotation,
            z_base=table2_top_z + SMALL_GAP,
            color=color
        )

    jug, _ = add_table2_item(
        "jugfixed.stl",
        x_offset=-0.3, y_offset=0.3,
        color=[0.8, 0.9, 1.0, 1.0]
    )

    pepper_grinder, _ = add_table2_item(
        "pepper_grinder.stl",
        x_offset=-0.3, y_offset=-0.3,
        color=[0.2, 0.2, 0.2, 1.0]
    )

    beef, _ = add_table2_item(
        "beef.stl",
        x_offset=0.0, y_offset=0.3,
        color=[0.8, 0.3, 0.3, 1.0]
    )

    fruit_veg_tray, _ = add_table2_item(
        "Fruit_and_Vegetables_Tray.stl",
        x_offset=0.0, y_offset=-0.3,
        color=[0.4, 0.7, 0.3, 1.0]
    )

    chicken, _ = add_table2_item(
        "chicken.stl",
        x_offset=0.3, y_offset=0.0,
        color=[1.0, 0.9, 0.7, 1.0]
    )

    # -------------------------
    # Linear UR3 on rail
    # -------------------------
    ur3 = LinearUR3()
    RAIL_X0 = 0.4
    RAIL_Y  = -1
    RAIL_Z  = FLOOR_TOP + SMALL_GAP
    YAW     = math.pi / 90

    ur3.base = SE3(RAIL_X0, RAIL_Y, RAIL_Z) @ SE3.Rz(YAW) @ ur3.base
    ur3.add_to_env(env)

    # -------------------------
    # CR3 — WITH ADJUSTABLE HEIGHT OFFSET
    # -------------------------
    CR3_FILE = Path(__file__).parent / "Cr3UR3editon.py"
    CR3Class = _load_robot_class(CR3_FILE, ("CR3", "Cr3UR3editon", "DobotCR3", "RobotCR3"))
    cr3 = CR3Class()

    CR3_X, CR3_Y = -1.2, 0.45
    CR3_Z = table_top_z + SMALL_GAP + HEIGHT_OFFSETS["CR3"]  # With manual offset
    CR3_YAW = -math.pi / 2
    
    print(f"✓ CR3 base at Z = {CR3_Z:.4f} m (table_top={table_top_z:.4f}, offset={HEIGHT_OFFSETS['CR3']})")

    base0_cr3 = getattr(cr3, "base", SE3())
    cr3.base = SE3(CR3_X, CR3_Y, CR3_Z) @ SE3.Rz(CR3_YAW) @ base0_cr3

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
    # CR16 — WITH ADJUSTABLE HEIGHT OFFSET
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
        CR16_X, CR16_Y = -1.2, -1.0
        CR16_Z = table_top_z + SMALL_GAP + HEIGHT_OFFSETS["CR16"]  # With manual offset
        CR16_YAW = +math.pi / 2
        
        print(f"✓ CR16 base at Z = {CR16_Z:.4f} m (table_top={table_top_z:.4f}, offset={HEIGHT_OFFSETS['CR16']})")

        base0_cr16 = getattr(cr16, "base", SE3())
        cr16.base = SE3(CR16_X, CR16_Y, CR16_Z) @ SE3.Rz(CR16_YAW) @ base0_cr16

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
    # Phase 1 complete
    # -------------------------
    env.step(0.02)
    print("\n" + "="*60)
    print("[Scene] All objects positioned with adjustable offsets")
    print(f"  FLOOR_TOP = {FLOOR_TOP:.4f} m")
    print(f"  Table 1 top = {table_top_z:.4f} m")
    print(f"  Table 2 top = {table2_top_z:.4f} m")
    print(f"  CR3 base = {CR3_Z:.4f} m (offset: {HEIGHT_OFFSETS['CR3']} m)")
    if cr16:
        print(f"  CR16 base = {CR16_Z:.4f} m (offset: {HEIGHT_OFFSETS['CR16']} m)")
    print(f"  Pot base = {table_top_z + SMALL_GAP + HEIGHT_OFFSETS['POT']:.4f} m (offset: {HEIGHT_OFFSETS['POT']} m)")
    print("\n💡 TIP: Adjust HEIGHT_OFFSETS at top of file if objects float/sink")
    print("="*60 + "\n")

    # -------------------------
    # Phase 2: Motion (unchanged logic)
    # -------------------------
    q = ur3.q.copy()
    q[0] = 0.0
    ur3.q = q
    env.step(0.02)

    if RUN_WIGGLE_CR3 and cr3 is not None:
        try:
            T = 2.0
            t = np.arange(0, T + DT, DT)
            qs = cr3.q.copy()
            qg = cr3.q.copy()
            j = min(2, qs.size - 1)
            qg[j] += np.deg2rad(15)
            traj = rtb.jtraj(qs, qg, t)
            for qk in traj.q:
                safety.block_until_allowed(env, DT)
                cr3.q = qk
                env.step(DT)
                time.sleep(DT)
        except Exception as _e:
            print("[CR3] Wiggle skipped:", _e)

    if RUN_WIGGLE_CR16 and cr16 is not None:
        try:
            T = 2.0
            t = np.arange(0, T + DT, DT)
            qs = cr16.q.copy()
            qg = cr16.q.copy()
            j = min(2, qs.size - 1)
            qg[j] += np.deg2rad(12)
            traj = rtb.jtraj(qs, qg, t)
            for qk in traj.q:
                safety.block_until_allowed(env, DT)
                cr16.q = qk
                env.step(DT)
                time.sleep(DT)
        except Exception as _e:
            print("[CR16] Wiggle skipped:", _e)

    if RUN_RAIL_SLIDE:
        q_start = ur3.q.copy()
        q_goal  = q_start.copy()
        q_goal[0] = -0.8
        T = 3.0
        t = np.arange(0, T + DT, DT)
        traj = rtb.jtraj(q_start, q_goal, t)
        for qk in traj.q:
            safety.block_until_allowed(env, DT)
            ur3.q = qk
            env.step(DT)
            time.sleep(DT)

    env.set_camera_pose([1.8, 3.4, 1.6], [0.0, -0.5, 0.8])
    #print("Open Swift at http://localhost:52100")
    env.hold()


if __name__ == "__main__":
    main()