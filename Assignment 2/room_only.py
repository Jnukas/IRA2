# room_only.py  — cleaned
from pathlib import Path
import math
import swift
import spatialgeometry as sg
from spatialmath import SE3
from room_utils import apply_swift_browser_fix, make_room
import roboticstoolbox as rtb
import numpy as np
import time
from ir_support.robots.LinearUR3 import LinearUR3

# -------------------------
# Launch Swift
# -------------------------
apply_swift_browser_fix()
env = swift.Swift()
env.launch(realtime=True, browser=None, host="127.0.0.1", port=52100, ws_port=53100)

# -------------------------
# Room (constants + build)
# -------------------------
ROOM_W = 6.0
ROOM_D = 6.0
FLOOR_TOP = 0.005  # keep floor a few mm above z=0 to avoid z-fighting

make_room(
    env,
    room_w=ROOM_W,
    room_d=ROOM_D,
    floor_t=0.10,        # 10 cm slab
    open_side="+Y",
    floor_top=FLOOR_TOP,
)

# -------------------------
# Oven (STL)
# -------------------------
oven_path = Path(__file__).parent / "assets" / "oven2.stl"

# NOTE: this scale matches what you had working; keep unless you re-export units.
oven = sg.Mesh(
    filename=str(oven_path),
    scale=[0.05, 0.05, 0.05],
    color=[0.70, 0.70, 0.70, 1.0],
)

# Auto-lift so the base sits on the slab; print size (if trimesh available)
z_lift = 0.0
try:
    import trimesh
    tm = trimesh.load_mesh(str(oven_path), process=False)
    zmin = float(tm.bounds[0, 2])
    z_lift = -zmin * 0.001
    size_m = (tm.bounds[1] - tm.bounds[0]) * 0.001
    print(f"Oven size (m): X={size_m[0]:.3f}  Y={size_m[1]:.3f}  Z={size_m[2]:.3f}")
except Exception:
    pass

# Place: centered in X, 0.60 m off the back wall, facing +Y
oven.T = SE3(0.0, -ROOM_D / 2 + 0.60, FLOOR_TOP + z_lift) @ SE3.Rz(math.pi)
env.add(oven)

# -------------------------
# -------------------------
# Table (STL)
# -------------------------
table_path = Path(__file__).parent / "assets" / "table.stl"

TABLE_SCALE = [1, 1, 1]   # use [0.001]*3 if STL is in mm
SCALE_MM = 1.0            # set 0.001 if STL is in mm

table = sg.Mesh(
    filename=str(table_path),
    scale=TABLE_SCALE,
    color=[0.82, 0.82, 0.82, 1.0],
)

z_lift_table = 0.0
try:
    import trimesh
    tm_tbl = trimesh.load_mesh(str(table_path), process=False)
    zmin_tbl = float(tm_tbl.bounds[0, 2])
    z_lift_table = -zmin_tbl * SCALE_MM
    size_tbl_m = (tm_tbl.bounds[1] - tm_tbl.bounds[0]) * SCALE_MM
    print(f"Table size (m): X={size_tbl_m[0]:.3f}  Y={size_tbl_m[1]:.3f}  Z={size_tbl_m[2]:.3f}")
except Exception:
    size_tbl_m = None
    pass

# >>> ADD THIS: world Z of the tabletop
try:
    table_top_z = FLOOR_TOP + z_lift_table + float(size_tbl_m[2])
except Exception:
    table_top_z = FLOOR_TOP + z_lift_table + 0.75  # fallback if bounds unknown

# Place: left side of room, rotated 90° about Z
table.T = SE3(-1.5, -0.5, FLOOR_TOP + z_lift_table) @ SE3.Rz(math.pi / 2)
env.add(table)


# --- Linear UR3 (with prismatic rail as joint 0) ---
ur3 = LinearUR3()              # uses the .dae/.stl files in the same folder as the class
# Place the whole robot (rail) in your room.
# IMPORTANT: multiply your placement on the LEFT so you keep the class's internal orientation.
# PICK THE SPOT (edit these numbers)
RAIL_X0 = 0.4         # left/right  (more + = right, more − = left)
RAIL_Y  = -1            # front/back  (increase to move toward the camera)
RAIL_Z  = FLOOR_TOP + 0.003      # height (usually the floor)
YAW     = math.pi/90        # 0, ±pi/2, or pi — try pi/2 to face the table

# Place the whole rail/robot in the room
ur3.base = SE3(RAIL_X0, RAIL_Y, RAIL_Z) @ SE3.Rz(YAW) @ ur3.base

# Add to Swift
ur3.add_to_env(env)

# Add CR3 (from Cr3UR3editon.py in this same folder)
# -------------------------
import importlib.util, types, inspect
from pathlib import Path as _Path

CR3_FILE = _Path(__file__).parent / "Cr3UR3editon.py"

def _load_robot_class_from_file(pyfile: _Path):
    if not pyfile.exists():
        raise FileNotFoundError(f"CR3 file not found: {pyfile}")

    spec = importlib.util.spec_from_file_location("cr3_custom_module", str(pyfile))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # module.__file__ is set correctly

    # 1) Try common names first
    for name in ("CR3", "Cr3UR3editon", "DobotCR3", "RobotCR3"):
        if hasattr(mod, name) and inspect.isclass(getattr(mod, name)):
            return getattr(mod, name)

    # 2) Otherwise pick any RTB robot subclass (prefer names containing 'CR3')
    candidates = []
    for name, obj in vars(mod).items():
        if inspect.isclass(obj):
            try:
                if issubclass(obj, (rtb.ERobot, rtb.DHRobot, rtb.Robot)):  # type: ignore
                    candidates.append((name, obj))
            except Exception:
                pass

    if candidates:
        for name, cls in candidates:
            if "CR3" in name.upper():
                print(f"[CR3 loader] Using {name} from {pyfile.name}")
                return cls
        name, cls = candidates[0]
        print(f"[CR3 loader] Using {name} from {pyfile.name}")
        return cls

    raise ImportError(f"No robot class found in {pyfile.name}. "
                      f"Export a class like `class CR3(DHRobot/ERobot): ...`.")

# Load, place, and add CR3
CR3Class = _load_robot_class_from_file(CR3_FILE)
cr3 = CR3Class()

# Base pose — right side of room, facing the table
CR3_X, CR3_Y = -1.2, 0.45
CR3_Z        = 0.945 + 0.003           # or use table_top_z + 0.003 if mounting on table
CR3_YAW      = -math.pi / 2

base0 = getattr(cr3, "base", SE3())
cr3.base = SE3(CR3_X, CR3_Y, CR3_Z) @ SE3.Rz(CR3_YAW) @ base0

# Spawn pose BEFORE adding to env (prevents flicker/teleport)
try:
    q_spawn = cr3.q.copy()
except Exception:
    q_spawn = np.zeros(getattr(cr3, "n", 6))
cr3.q = q_spawn
if hasattr(cr3, "qtest"):
    cr3.qtest = q_spawn

# Add to Swift
if hasattr(cr3, "add_to_env"):
    cr3.add_to_env(env)
else:
    env.add(cr3)
env.step(0.02)

# (Optional) quick wiggle to confirm it's alive
try:
    T, dt = 2.0, 1/60
    t = np.arange(0, T+dt, dt)
    qs = cr3.q.copy(); qg = cr3.q.copy()
    j = min(2, qs.size-1)            # wiggle joint 2 (or last valid)
    qg[j] += np.deg2rad(15)
    traj = rtb.jtraj(qs, qg, t)
    for qk in traj.q:
        cr3.q = qk
        env.step(dt)
        time.sleep(dt)
except Exception as _e:
    print("[CR3] Wiggle skipped:", _e)

# Comfortable starting posture (the class sets qtest; you can keep it or change it)
# Example: keep the rail at -0.4 m and leave arm joints as they are
q = ur3.q.copy()
q[0] = 0                    # prismatic rail joint (limits in the class: [-0.8, 0])
ur3.q = q
env.step(0.02)

# (Optional) quick demo: slide the rail
q_start = ur3.q.copy()
q_goal  = q_start.copy(); q_goal[0] = -0.8     # full travel toward negative X
T  = 3.0                 # seconds of motion
dt = 1.0 / 60.0          # 60 FPS
t  = np.arange(0, T+dt, dt)

traj = rtb.jtraj(q_start, q_goal, t)  # LSPB profile with zero vel at ends

for qk in traj.q:
    ur3.q = qk
    env.step(dt)          # <-- step INSIDE the loop
    time.sleep(dt)

# -------------------------
# -------------------------


# -------------------------

# -------------------------
# Camera + hold
# -------------------------
env.set_camera_pose([1.8, 3.4, 1.6], [0.0, -0.5, 0.8])
print("Open Swift at http://localhost:52100")
env.hold()
