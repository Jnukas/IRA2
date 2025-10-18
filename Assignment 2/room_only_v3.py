#!/usr/bin/env python3
# room_only.py — generate everything first, then move (placements preserved)
from __future__ import annotations

from pathlib import Path
import math
import time
import importlib.util
import inspect

import numpy as np
import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
from spatialmath import SE3

from room_utils import apply_swift_browser_fix, make_room
from ir_support.robots.LinearUR3 import LinearUR3


# ===== Motion toggles (no placement changes) =====
RUN_WIGGLE_CR3  = True
RUN_WIGGLE_CR16 = True
RUN_RAIL_SLIDE  = True
FPS = 60
DT = 1.0 / FPS


def _load_robot_class(pyfile: Path, prefer_names: tuple[str, ...]) -> type:
    """Dynamic loader that returns the first matching RTB robot class."""
    if not pyfile.exists():
        raise FileNotFoundError(f"Robot file not found: {pyfile}")

    spec = importlib.util.spec_from_file_location("user_robot_module", str(pyfile))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # 1) try preferred names first
    for name in prefer_names:
        if hasattr(mod, name) and inspect.isclass(getattr(mod, name)):
            return getattr(mod, name)

    # 2) else scan for any RTB robot subclass
    candidates = []
    RobotBase = getattr(rtb, "Robot", tuple())  # RTB may not export 'Robot'
    for name, obj in vars(mod).items():
        if inspect.isclass(obj):
            try:
                if issubclass(obj, (rtb.ERobot, rtb.DHRobot, RobotBase)):  # type: ignore[arg-type]
                    candidates.append((name, obj))
            except Exception:
                pass

    if candidates:
        print(f"[loader] Using {candidates[0][0]} from {pyfile.name}")
        return candidates[0][1]

    raise ImportError(f"No robot class found in {pyfile.name}. "
                      f"Export a class like `class MyBot(DHRobot/ERobot): ...`.")


def main():
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
        floor_t=0.10,      # 10 cm slab
        open_side="+Y",
        floor_top=FLOOR_TOP,
    )

    # -------------------------
    # Oven (STL) — placements unchanged
    # -------------------------
    oven_path = Path(__file__).parent / "assets" / "oven2.stl"
    oven = sg.Mesh(
        filename=str(oven_path),
        scale=[0.05, 0.05, 0.05],  # keep as-is
        color=[0.70, 0.70, 0.70, 1.0],
    )

    # Auto-lift so base sits on slab (assumes STL in mm)
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

    oven.T = SE3(0.0, -ROOM_D / 2 + 0.60, FLOOR_TOP + z_lift) @ SE3.Rz(math.pi)
    env.add(oven)

    # -------------------------
    # Table (STL) — placements unchanged
    # -------------------------
    table_path = Path(__file__).parent / "assets" / "table.stl"
    TABLE_SCALE = [1, 1, 1]   # keep as-is (use [0.001]*3 if your STL is mm)
    SCALE_MM = 1.0            # keep as-is (set 0.001 if STL is mm)

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

    try:
        table_top_z = FLOOR_TOP + z_lift_table + float(size_tbl_m[2])  # noqa: F823
    except Exception:
        table_top_z = FLOOR_TOP + z_lift_table + 0.75  # fallback if bounds unknown

    table.T = SE3(-1.5, -0.5, FLOOR_TOP + z_lift_table) @ SE3.Rz(math.pi / 2)
    env.add(table)

    # Potwithoutthelid(STL) — placements unchanged
    # -------------------------
    Potwithoutthelid_path = Path(__file__).parent / "assets" / "Potwithoutthelid.stl"
    Potwithoutthelid = sg.Mesh(
        filename=str(Potwithoutthelid_path),
        scale=[0.002, 0.002, 0.002],  # keep as-is
        color=[0.70, 0.70, 0.70, 1.0],
    )

    # Auto-lift so base sits on slab (assumes STL in mm)
    z_lift = 0.0
    try:
        import trimesh
        tm = trimesh.load_mesh(str(Potwithoutthelid_path), process=False)
        zmin = float(tm.bounds[0, 2])
        z_lift = -zmin * 0.001
        size_m = (tm.bounds[1] - tm.bounds[0]) * 0.001
        print(f"Pot size (m): X={size_m[0]:.3f}  Y={size_m[1]:.3f}  Z={size_m[2]:.3f}")
    except Exception:
        pass

    Potwithoutthelid.T = SE3(-2, -0.5, 0.945 + 0.003) @ SE3.Rz(0.0) #manual positioning of the pot
    env.add(Potwithoutthelid)

    # -------------------------
    # Linear UR3 — placements unchanged
    # -------------------------
    ur3 = LinearUR3()  # uses class assets
    RAIL_X0 = 0.4
    RAIL_Y  = -1
    RAIL_Z  = FLOOR_TOP + 0.003
    YAW     = math.pi / 90  # keep as-is

    ur3.base = SE3(RAIL_X0, RAIL_Y, RAIL_Z) @ SE3.Rz(YAW) @ ur3.base
    ur3.add_to_env(env)

    # -------------------------
    # CR3 — placements unchanged (no motion yet)
    # -------------------------
    CR3_FILE = Path(__file__).parent / "Cr3UR3editon.py"
    CR3Class = _load_robot_class(CR3_FILE, ("CR3", "Cr3UR3editon", "DobotCR3", "RobotCR3"))
    cr3 = CR3Class()

    CR3_X, CR3_Y = -1.2, 0.45
    CR3_Z        = 0.945 + 0.003  # keep as-is (or switch to table_top_z + 0.003 later)
    CR3_YAW      = -math.pi / 2

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
    # CR16 — placements unchanged (no motion yet)
    # -------------------------
    CR16_FILE = Path(__file__).parent / "CR16Creator.py"  # change if different
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
        # NOTE: keep your manual Z (as requested)
        CR16_X, CR16_Y = -1.2, -1
        CR16_Z = 0.945 + 0.003  # keep as-is
        CR16_YAW = +math.pi / 2

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
    # Phase 1 complete: everything generated & visible, no movement yet
    # -------------------------
    env.step(0.02)
    print("[Scene] All robots added. Starting motion phase...")

    # -------------------------
    # Phase 2: Motion (after everything exists)
    # -------------------------

    # Comfortable starting posture (rail) — unchanged logic, not animated
    q = ur3.q.copy()
    q[0] = 0.0  # prismatic rail joint (limits in class: [-0.8, 0])
    ur3.q = q
    env.step(0.02)

    # Optional wiggle: CR3
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
                cr3.q = qk
                env.step(DT)
                time.sleep(DT)
        except Exception as _e:
            print("[CR3] Wiggle skipped:", _e)

    # Optional wiggle: CR16
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
                cr16.q = qk
                env.step(DT)
                time.sleep(DT)
        except Exception as _e:
            print("[CR16] Wiggle skipped:", _e)

    # Optional demo: slide the UR3 rail
    if RUN_RAIL_SLIDE:
        q_start = ur3.q.copy()
        q_goal  = q_start.copy()
        q_goal[0] = -0.8  # full travel toward negative X
        T = 3.0
        t = np.arange(0, T + DT, DT)
        traj = rtb.jtraj(q_start, q_goal, t)  # LSPB with zero vel at ends
        for qk in traj.q:
            ur3.q = qk
            env.step(DT)
            time.sleep(DT)

    # Camera + hold — unchanged
    env.set_camera_pose([1.8, 3.4, 1.6], [0.0, -0.5, 0.8])
    print("Open Swift at http://localhost:52100")
    env.hold()


if __name__ == "__main__":
    main()
