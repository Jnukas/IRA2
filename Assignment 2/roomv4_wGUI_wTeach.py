#!/usr/bin/env python3
# room_only.py — generate everything first, then move (placements preserved)
from __future__ import annotations

from pathlib import Path
import math
import time
import importlib.util
import inspect
import threading                         ### SAFETY: NEW
import tkinter as tk                     ### SAFETY: NEW

import numpy as np
import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
from spatialmath import SE3


import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import time
import roboticstoolbox as rtb

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
        sv = np.repeat(float(sv), 3)       # e.g., 0.001 -> [0.001, 0.001, 0.001]
    elif sv.size != 3:
        raise ValueError("scale_vec must be a scalar or a 3-vector")

    size_m = (bmax - bmin) * sv            # per-axis scaled size in metres
    z_lift = -bmin[2] * sv[2]              # lift so bottom rests on z=0

    return float(z_lift), size_m


# ===== Motion toggles (no placement changes) =====
RUN_WIGGLE_CR3  = True
RUN_WIGGLE_CR16 = True
RUN_RAIL_SLIDE  = True  # set True to demo the E-STOP easily
FPS = 60
DT = 1.0 / FPS

### SAFETY: NEW — small, latched safety controller + GUI
class SafetyController:
    """
    Latched e-stop:
      - engage_e_stop(): immediately blocks motion (clears run event)
      - disengage_e_stop(): remains blocked (READY state), requires resume()
      - resume(): only works if e-stop not engaged; sets run event
    No busy waiting: block_until_allowed() uses Event.wait(timeout=dt)
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._run_evt = threading.Event()
        self._run_evt.set()              # allowed at start
        self.e_stop_engaged = False      # latched flag

    def engage_e_stop(self):
        with self._lock:
            self.e_stop_engaged = True
            self._run_evt.clear()       # stop immediately

    def disengage_e_stop(self):
        with self._lock:
            self.e_stop_engaged = False
            # still paused; requires resume() explicitly

    def resume(self):
        with self._lock:
            if not self.e_stop_engaged:
                self._run_evt.set()      # allow motion again

    def is_running(self) -> bool:
        return self._run_evt.is_set() and not self.e_stop_engaged

    def block_until_allowed(self, env, dt: float):
        """
        Gate for motion loops: while not allowed, we wait with timeout and
        step the env so UI stays responsive (no busy loop).
        """
        while not self._run_evt.wait(timeout=dt):
            # During pause, keep UI responsive without advancing robot states
            try:
                env.step(dt)
            except Exception:
                time.sleep(dt)
            # If re-engaged while READY, we remain blocked; resume will set the event

def launch_safety_gui(safety: SafetyController):
    """Tiny Tk window with E-STOP + RESUME buttons (latched behaviour)."""
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
        # Toggle: engage if currently running/ready; else disengage to READY
        if not safety.e_stop_engaged:
            safety.engage_e_stop()
        else:
            safety.disengage_e_stop()  # stays paused (READY)
        refresh_label()

    def on_resume():
        # Only resumes if not engaged
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
# --- end SAFETY block ---


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


# ===== Multi-robot Teach GUI (Tk) ============================================
import tkinter as tk
from tkinter import ttk

def _qlim_list(robot):
    """Return per-joint (qmin, qmax) as a list of tuples, robust to 2×n or n×2."""
    if not hasattr(robot, "qlim") or robot.qlim is None:
        return None
    ql = np.asarray(robot.qlim, dtype=float)
    if ql.shape == (2, robot.n):
        mins, maxs = ql[0, :], ql[1, :]
    elif ql.shape == (robot.n, 2):
        mins, maxs = ql[:, 0], ql[:, 1]
    else:
        return None
    return [(float(a), float(b)) for a, b in zip(mins, maxs)]

def _joint_types(robot):
    """
    Return list like ['R','R','P',...] using .links[*].isrevolute/isprismatic when available.
    Fallback: assume revolute.
    """
    js = []
    links = getattr(robot, "links", [])
    if links and len(links) >= robot.n:
        for L in links[:robot.n]:
            if getattr(L, "isprismatic", False): js.append("P")
            else: js.append("R")
    else:
        js = ["R"] * int(getattr(robot, "n", 6))
    return js

import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import time
import roboticstoolbox as rtb

class MultiRobotTeachPanel:
    """
    Fixed version with:
    - Non-blocking motion (separate thread)
    - Optional real-time slider tracking
    - Proper thread safety
    """
    def __init__(self, root, env, safety, robots: dict[str, object], dt=1/60, base_T=2.0):
        self.root = root
        self.env = env
        self.safety = safety
        self.dt = float(dt)
        self.base_T = float(base_T)
        
        # Thread safety
        self.motion_lock = threading.Lock()
        self.motion_thread = None
        self.stop_motion_flag = False
        
        self.root.title("Safety + Teach")
        try: 
            self.root.wm_attributes("-topmost", True)
        except: 
            pass

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)

        # Safety tab
        self._build_safety_tab()

        # Robot tabs
        for name, rob in robots.items():
            if rob is None: 
                continue
            try:
                self._build_robot_tab(name, rob)
            except Exception as e:
                print(f"[TeachGUI] Skipped tab for {name}: {e}")

    def _build_safety_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Safety")

        status = tk.StringVar(value="RUNNING")
        
        def refresh():
            if self.safety.e_stop_engaged:
                status.set("E-STOP ENGAGED")
            elif self.safety.is_running():
                status.set("RUNNING")
            else:
                status.set("READY (disengaged, press RESUME)")

        def on_estop():
            if not self.safety.e_stop_engaged:
                self.safety.engage_e_stop()
            else:
                self.safety.disengage_e_stop()
            refresh()

        def on_resume():
            self.safety.resume()
            refresh()

        big = ("Segoe UI", 16, "bold")
        btnE = tk.Button(tab, text="E-STOP", command=on_estop, bg="#b30000", fg="white", font=big, height=2)
        btnR = tk.Button(tab, text="RESUME", command=on_resume, bg="#006400", fg="white", font=big, height=2)
        lbl = ttk.Label(tab, textvariable=status, font=("Segoe UI", 12))

        btnE.pack(fill="x", padx=12, pady=(12,6))
        btnR.pack(fill="x", padx=12, pady=(6,6))
        lbl.pack(padx=12, pady=(6,10))
        refresh()

    def _build_robot_tab(self, name, robot):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=name)

        n = int(getattr(robot, "n", 6))
        types = self._joint_types(robot)
        qlim = self._qlim_list(robot)
        q_now = np.array(robot.q, dtype=float).copy()

        # Fallback limits
        mins, maxs = [], []
        for i in range(n):
            t = types[i] if i < len(types) else "R"
            if qlim is None or not np.isfinite(qlim[i][0]) or not np.isfinite(qlim[i][1]):
                if t == "P": 
                    a, b = (-0.8, 0.8)
                else: 
                    a, b = (-np.pi, np.pi)
            else:
                a, b = qlim[i]
            mins.append(a)
            maxs.append(b)

        # Control variables
        speed_var = tk.DoubleVar(value=1.0)
        realtime_var = tk.BooleanVar(value=False)  # NEW: real-time tracking mode
        
        # Slider variables
        vars_ = []
        for i in range(n):
            v = tk.DoubleVar()
            if types[i] == "R": 
                v.set(np.degrees(q_now[i]))
            else: 
                v.set(q_now[i])
            vars_.append(v)

        # Build UI
        frmTop = ttk.Frame(tab)
        frmTop.pack(fill="x", padx=10, pady=(10,6))
        ttk.Label(frmTop, text=f"{name} (n={n})").pack(side="left")
        
        # Speed control
        ttk.Label(frmTop, text="Speed").pack(side="left", padx=(12,4))
        tk.Scale(frmTop, from_=0.2, to=2.0, resolution=0.1, orient="horizontal",
                 variable=speed_var, length=120).pack(side="left")
        
        # NEW: Real-time tracking checkbox
        ttk.Checkbutton(frmTop, text="Real-time", variable=realtime_var,
                       command=lambda: self._toggle_realtime(robot, types, vars_, mins, maxs, realtime_var)
                       ).pack(side="left", padx=(12,0))

        frm = ttk.Frame(tab)
        frm.pack(fill="both", expand=True, padx=10, pady=6)

        # Helper functions
        def get_target_q():
            tg = []
            for i in range(n):
                if types[i] == "R":
                    tg.append(np.radians(vars_[i].get()))
                else:
                    tg.append(vars_[i].get())
            return np.array(tg, dtype=float)

        def clamp_to_qlim(q):
            qt = q.copy()
            for i in range(n):
                qt[i] = float(np.clip(qt[i], mins[i], maxs[i]))
            return qt

        def read_from_robot():
            """Update sliders to match current robot position"""
            q = np.array(robot.q, dtype=float).copy()
            for i in range(n):
                if types[i] == "R": 
                    vars_[i].set(np.degrees(q[i]))
                else: 
                    vars_[i].set(q[i])

        def do_move_threaded():
            """NON-BLOCKING: Run motion on separate thread"""
            # Stop any existing motion
            self.stop_motion_flag = True
            if self.motion_thread and self.motion_thread.is_alive():
                self.motion_thread.join(timeout=0.5)
            
            self.stop_motion_flag = False
            
            # Start new motion thread
            self.motion_thread = threading.Thread(
                target=self._execute_motion,
                args=(robot, get_target_q(), clamp_to_qlim, speed_var.get()),
                daemon=True
            )
            self.motion_thread.start()

        def set_zero():
            for i in range(n):
                vars_[i].set(0.0)
            do_move_threaded()

        def go_home():
            qh = None
            if hasattr(robot, "q_home"): 
                qh = np.array(robot.q_home, dtype=float)
            elif hasattr(robot, "qtest"): 
                qh = np.array(robot.qtest, dtype=float)
            if qh is None: 
                return
            
            for i in range(n):
                if types[i] == "R": 
                    vars_[i].set(np.degrees(qh[i]))
                else: 
                    vars_[i].set(qh[i])
            do_move_threaded()

        # Build joint sliders
        for i in range(n):
            row = ttk.Frame(frm)
            row.pack(fill="x", pady=4)
            jt = types[i]
            pretty = f"J{i+1} ({'°' if jt=='R' else 'm'})"
            ttk.Label(row, text=pretty, width=12).pack(side="left")

            lo = np.degrees(mins[i]) if jt == "R" else mins[i]
            hi = np.degrees(maxs[i]) if jt == "R" else maxs[i]
            res = 0.5 if jt == "R" else 0.001

            s = tk.Scale(row, from_=lo, to=hi, resolution=res, length=360,
                        orient="horizontal", variable=vars_[i])
            s.pack(side="left", padx=8)

            # Jog buttons
            def make_jog(di, sign):
                def jog_action():
                    step = 3.0 if types[di] == "R" else 0.01
                    val = vars_[di].get() + sign * step
                    lo_jog = np.degrees(mins[di]) if types[di] == "R" else mins[di]
                    hi_jog = np.degrees(maxs[di]) if types[di] == "R" else maxs[di]
                    val = float(np.clip(val, lo_jog, hi_jog))
                    vars_[di].set(val)
                    # NEW: Auto-move if in real-time mode
                    if realtime_var.get():
                        self._instant_move(robot, get_target_q(), clamp_to_qlim)
                return jog_action
            
            tk.Button(row, text="−", width=3, command=make_jog(i, -1)).pack(side="left", padx=(6,2))
            tk.Button(row, text="+", width=3, command=make_jog(i, +1)).pack(side="left", padx=(2,6))

        # Bottom buttons
        frmB = ttk.Frame(tab)
        frmB.pack(fill="x", padx=10, pady=(6,10))
        ttk.Button(frmB, text="Read", command=read_from_robot).pack(side="left")
        ttk.Button(frmB, text="Move", command=do_move_threaded).pack(side="left", padx=(8,0))
        ttk.Button(frmB, text="Zero", command=set_zero).pack(side="left", padx=(8,0))
        ttk.Button(frmB, text="Home", command=go_home).pack(side="left", padx=(8,0))
        ttk.Button(frmB, text="Stop", command=self._stop_motion, 
                  ).pack(side="left", padx=(8,0))

    def _execute_motion(self, robot, target_q, clamp_fn, speed):
        """Execute smooth trajectory (runs on separate thread)"""
        with self.motion_lock:
            try:
                q0 = np.array(robot.q, dtype=float).copy()
                q1 = clamp_fn(target_q)
                
                if np.allclose(q0, q1, atol=1e-6):
                    return
                
                T = self.base_T / max(0.2, float(speed))
                t = np.arange(0.0, T + self.dt, self.dt)
                traj = rtb.jtraj(q0, q1, t)
                
                for qk in traj.q:
                    if self.stop_motion_flag:
                        break
                    
                    self.safety.block_until_allowed(self.env, self.dt)
                    robot.q = qk
                    
                    try:
                        self.env.step(self.dt)
                    except:
                        pass
                    
                    time.sleep(self.dt)
            except Exception as e:
                print(f"[Motion Error] {e}")

    def _instant_move(self, robot, target_q, clamp_fn):
        """Instant move to position (no trajectory, for real-time tracking)"""
        with self.motion_lock:
            try:
                q1 = clamp_fn(target_q)
                robot.q = q1
                self.env.step(self.dt)
            except Exception as e:
                print(f"[Instant Move Error] {e}")

    def _toggle_realtime(self, robot, types, vars_, mins, maxs, realtime_var):
        """Enable/disable real-time slider tracking"""
        if realtime_var.get():
            print("[Real-time] ENABLED - robot follows sliders directly (no trajectory)")
            # Bind slider changes to instant motion
            for i, v in enumerate(vars_):
                v.trace_add('write', lambda *args, i=i: self._on_slider_change(
                    robot, types, vars_, mins, maxs
                ))
        else:
            print("[Real-time] DISABLED - use Move button for smooth trajectories")

    def _on_slider_change(self, robot, types, vars_, mins, maxs):
        """Callback when slider moves in real-time mode"""
        def get_target():
            tg = []
            for i in range(len(vars_)):
                if types[i] == "R":
                    tg.append(np.radians(vars_[i].get()))
                else:
                    tg.append(vars_[i].get())
            return np.array(tg, dtype=float)
        
        def clamp(q):
            qt = q.copy()
            for i in range(len(vars_)):
                qt[i] = float(np.clip(qt[i], mins[i], maxs[i]))
            return qt
        
        self._instant_move(robot, get_target(), clamp)

    def _stop_motion(self):
        """Stop any ongoing motion"""
        self.stop_motion_flag = True

    @staticmethod
    def _qlim_list(robot):
        if not hasattr(robot, "qlim") or robot.qlim is None:
            return None
        ql = np.asarray(robot.qlim, dtype=float)
        if ql.shape == (2, robot.n):
            mins, maxs = ql[0, :], ql[1, :]
        elif ql.shape == (robot.n, 2):
            mins, maxs = ql[:, 0], ql[:, 1]
        else:
            return None
        return [(float(a), float(b)) for a, b in zip(mins, maxs)]

    @staticmethod
    def _joint_types(robot):
        js = []
        links = getattr(robot, "links", [])
        if links and len(links) >= robot.n:
            for L in links[:robot.n]:
                if getattr(L, "isprismatic", False): 
                    js.append("P")
                else: 
                    js.append("R")
        else:
            js = ["R"] * int(getattr(robot, "n", 6))
        return js

def launch_control_gui(env, safety, robots: dict[str, object], dt=1/60, base_T=2.0):
    """
    Single window that includes Safety and Teach tabs.
    IMPORTANT: Call this AFTER robots are created (so tabs can be built).
    Launch on a daemon thread as you did before.
    """
    root = tk.Tk()
    panel = MultiRobotTeachPanel(root, env, safety, robots, dt=dt, base_T=base_T)
    root.mainloop()
# ============================================================================ 


def main():
    # -------------------------
    # Launch Swift
    # -------------------------
    apply_swift_browser_fix()
    env = swift.Swift()
    env.launch(realtime=True, browser=None, host="127.0.0.1", port=52100, ws_port=53100)

    ### SAFETY: NEW — start safety controller + GUI on a daemon thread
    safety = SafetyController()
    threading.Thread(target=launch_safety_gui, args=(safety,), daemon=True).start()

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
    # Stove (STL)
    # -------------------------
    stove_path  = Path(__file__).parent / "assets" / "Stove.stl"
    STOVE_SCALE = [1.0, 1.0, 1.0]    # If the file is in mm, change to [0.001, 0.001, 0.001]

    z_lift_stove, stove_size_m = mesh_bounds_info(stove_path, STOVE_SCALE)
    if stove_size_m is not None:
        print(f"Stove size (m): X={stove_size_m[0]:.3f}  Y={stove_size_m[1]:.3f}  Z={stove_size_m[2]:.3f}")

    stove = sg.Mesh(
        filename=str(stove_path),
        scale=STOVE_SCALE,
        color=[0.70, 0.70, 0.70, 1.0],
    )
    stove.T = SE3(0.0, -ROOM_D / 2 + 0.60, FLOOR_TOP + z_lift_stove) @ SE3.Rz(math.pi)
    env.add(stove)

    # -------------------------
    # Table (STL)
    # -------------------------
    table_path  = Path(__file__).parent / "assets" / "table.stl"
    TABLE_SCALE = [1.0, 1.0, 1.0]    # If the file is in mm, use [0.001, 0.001, 0.001]

    z_lift_table, table_size_m = mesh_bounds_info(table_path, TABLE_SCALE)
    if table_size_m is not None:
        print(f"Table size (m): X={table_size_m[0]:.3f}  Y={table_size_m[1]:.3f}  Z={table_size_m[2]:.3f}")

    table = sg.Mesh(
        filename=str(table_path),
        scale=TABLE_SCALE,
        color=[0.50, 0.50, 0.50, 1.0],
    )
    table.T = SE3(-1.5, -0.5, FLOOR_TOP + z_lift_table) @ SE3.Rz(math.pi / 2)
    env.add(table)

    # Compute table top Z for placing items on this table
    if table_size_m is not None:
        TABLE_TOP_Z = FLOOR_TOP + z_lift_table + float(table_size_m[2])
    else:
        TABLE_TOP_Z = FLOOR_TOP + z_lift_table + 0.75  # fallback

    # -------------------------
    # Pot without lid (STL) — place ON the first table
    # -------------------------
    pot_path  = Path(__file__).parent / "assets" / "Potwithoutthelid.stl"
    POT_SCALE = [0.002, 0.002, 0.002]  # keep as-is; whatever you use here will be honored

    z_lift_pot, pot_size_m = mesh_bounds_info(pot_path, POT_SCALE)
    if pot_size_m is not None:
        print(f"Pot size (m): X={pot_size_m[0]:.3f}  Y={pot_size_m[1]:.3f}  Z={pot_size_m[2]:.3f}")

    pot = sg.Mesh(
        filename=str(pot_path),
        scale=POT_SCALE,
        color=[1.0, 0.0, 0.0, 1.0],
    )
    # Position pot on the table surface (no magic 0.945): table top + pot base lift
    pot.T = SE3(-1.75, -0.5, TABLE_TOP_Z + z_lift_pot)  # keep your XY, correct Z
    env.add(pot)

    # -------------------------
    # Second table WORK_TABLE_BS2 (STL)
    # -------------------------
    tbl2_path  = Path(__file__).parent / "assets" / "rightwayup.stl"
    TBL2_SCALE = [0.001, 0.001, 0.001]  # mm → m

    z_lift_tbl2, tbl2_size_m = mesh_bounds_info(tbl2_path, TBL2_SCALE)
    if tbl2_size_m is not None:
        print(f"Table2 size (m): X={tbl2_size_m[0]:.3f}  Y={tbl2_size_m[1]:.3f}  Z={tbl2_size_m[2]:.3f}")

    WORK_TABLE_BS2 = sg.Mesh(
        filename=str(tbl2_path),
        scale=TBL2_SCALE,
        color=[0.50, 0.50, 0.50, 1.0],
    )
    # Use the SAME FLOOR_TOP everywhere (don’t reset to 0.0)
    WORK_TABLE_BS2.T = SE3(1.50, 0.0, FLOOR_TOP + z_lift_tbl2) @ SE3.RPY([-90, 0, 0], order='xyz', unit='deg')
    env.add(WORK_TABLE_BS2)

    # Top surface Z for table 2
    if tbl2_size_m is not None:
        TABLE2_TOP_Z = FLOOR_TOP + z_lift_tbl2 + float(tbl2_size_m[2])
    else:
        TABLE2_TOP_Z = FLOOR_TOP + z_lift_tbl2 + 0.80  # fallback

    # -------------------------
    # Helper to add items onto table 2 (respects full scale vector)
    # -------------------------
    def add_table2_item(filename, scale_vec, color, x_offset, y_offset, z_extra=0.0, rotation=0.0):
        p = Path(__file__).parent / "assets" / filename
        z_lift_item, item_size_m = mesh_bounds_info(p, scale_vec)

        item = sg.Mesh(str(p), scale=scale_vec, color=color)
        item.T = (
            SE3(1.50 + x_offset, y_offset, TABLE2_TOP_Z + z_lift_item + z_extra)
            @ SE3.Rz(rotation)
        )
        env.add(item)
        if item_size_m is not None:
            print(f"{filename} size (m): X={item_size_m[0]:.3f}  Y={item_size_m[1]:.3f}  Z={item_size_m[2]:.3f}")
        return item

    # -------------------------
    # Items on table 2 (unchanged layout, now with correct Z)
    # -------------------------
    jug = add_table2_item(
        "jugfixed.stl",
        scale_vec=[0.001, 0.001, 0.001],
        color=[0.8, 0.9, 1.0, 1.0],
        x_offset=-0.3, y_offset=0.3, rotation=0.0
    )

    pepper_grinder = add_table2_item(
        "pepper_grinder.stl",
        scale_vec=[1.0, 1.0, 1.0],  # if it’s actually mm, change to [0.001]*3
        color=[0.2, 0.2, 0.2, 1.0],
        x_offset=-0.3, y_offset=-0.3, rotation=0.0
    )

    beef = add_table2_item(
        "beef.stl",
        scale_vec=[0.01, 0.01, 0.01],  # cm → m (assuming it was modelled in cm)
        color=[0.8, 0.3, 0.3, 1.0],
        x_offset=0.0, y_offset=0.3, rotation=0.0
    )

    fruit_veg_tray = add_table2_item(
        "Fruit_and_Vegetables_Tray.stl",
        scale_vec=[0.001, 0.001, 0.001],
        color=[0.4, 0.7, 0.3, 1.0],
        x_offset=0.0, y_offset=-0.3, rotation=0.0
    )

    chicken = add_table2_item(
        "chicken.stl",
        scale_vec=[0.01, 0.01, 0.01],
        color=[1.0, 0.9, 0.7, 1.0],
        x_offset=0.3, y_offset=0.0, rotation=0.0
    )

    #-------------------------
    #Linear UR3 — placements unchanged
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
        # -------------------------
    # Phase 1 complete: everything generated & visible, no movement yet
    # -------------------------
    env.step(0.02)
    print("[Scene] All robots and objects added. Starting motion phase...")

    # NEW: start the combined Safety + Teach GUI (tabs for UR3 / CR3 / CR16)
    robots = {
        "LinearUR3": ur3,
        "CR3":       cr3,
        "CR16":      cr16,
    }
    threading.Thread(
        target=launch_control_gui,
        args=(env, safety, robots),
        kwargs={"dt": DT, "base_T": 2.0},
        daemon=True
    ).start()

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
                safety.block_until_allowed(env, DT)    ### SAFETY: gate each step
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
                safety.block_until_allowed(env, DT)    ### SAFETY: gate each step
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
            safety.block_until_allowed(env, DT)        ### SAFETY: gate each step
            ur3.q = qk
            env.step(DT)
            time.sleep(DT)

    # Camera + hold — unchanged
    env.set_camera_pose([1.8, 3.4, 1.6], [0.0, -0.5, 0.8])
    print("Open Swift at http://localhost:52100")
    env.hold()


if __name__ == "__main__":
    main()
