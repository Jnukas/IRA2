from __future__ import annotations

from pathlib import Path
import math
import time
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
from Cr3UR3editon import CR3
from CR16Creator import CR16
from roboticstoolbox import trapezoidal


from spatialmath.base import tr2rpy
from spatialmath import UnitQuaternion

# asset helpers
from itertools import combinations


# Room Constants
ROOM_W = 6.0
ROOM_D = 6.0
FLOOR_TOP = 0.005
TABLE1_HEIGHT = 0.32 
TABLE2_HEIGHT = 0.32


# Scales
SCALES = {
    "Stove.stl": [1.0, 1.0, 1.0],
    "pepper_grinder.stl": [1.0, 1.0, 1.0],
    "barrier.stl" : [1.0, 1.0, 1.0],
    "table1.stl": [1.0, 1.0, 1.0],
    "table2.stl": [1.0, 1.0, 1.0],
    "Potwithoutthelid.stl": [1.0, 1.0, 1.0],
    "jugfixed.stl": [1.0, 1.0, 1.0],
    "Fruit_and_Vegetables_Tray.stl": [1, 1, 1],
    "beef.stl": [1.0, 1.0, 1.0],
    "chicken.stl": [1.0, 1.0, 1.0],
}

# POSITIONS
POSITIONS = {
    "STOVE": (0.0, -2.4),
    "TABLE2": (0.80, -0.85),    
    "JUG": (0.35, -1.3),
    "PEPPER": (0.35, -0.7),
    "BEEF": (0.35, -0.9),
    "FRUIT_VEG": (0.4, -0.5),
    "CHICKEN": (0.35, -1),
    "TABLE1": (-0.80, -0.85),
    "CR3": (-0.5, -0.4),
    "CR16": (-0.5, -1.5),
    "POT": (-1.15, -1),
    "UR3": (0, -0.5),
    "BARRIER" : (-0.75, 1.25),
    "HW_STOP": (0.7, 0),
}

HEIGHT_OFFSETS = {
    "CR3": -0.02,
    "CR16": -0.02,
    "UR3": 0,
    "STOVE": 0.0 + 0.000059,
    "TABLE1": 0.0 + 0.287252,
    "TABLE2": 0.0 + 0.287252,
    "POT": -0.02 + 0.136847 + 0.32,
    "JUG": -0.02 + 0.003000 + 0.32,
    "PEPPER": -0.02 + -0.000127 + 0.32,
    "BEEF": -0.02 + 0.040339 + 0.32,
    "FRUIT_VEG": -0.02 + 0.020911 + 0.32,
    "CHICKEN": -0.02 + 0.035977 + 0.32,
    "BARRIER" : 0.0 + 1.054546,
    "HW_STOP": 0.283,

}



# referesh rate
DT = 0.015

#Arduinoi e-stop
# hardware
import serial

def start_arduino_estop_listener(safety, port="COM3", baud=115200):
    
    def _worker():
        try:
            ser = serial.Serial(port, baudrate=baud, timeout=0.05)
            time.sleep(2.0)
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            print(f"Listening on {port} @ {baud}")
        except Exception as e:
            print(f" Serial open failed ({port}): {e}")
            return

        while True:
            try:
                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue
                print(f"[E-STOP] RX: {line}")
                if line.startswith("E,"):
                    parts = line.split(",", 1)
                    if len(parts) == 2 and parts[1].strip() == "1":
                        if not safety.e_stop_engaged:
                            print("Arduino button pressed → E-STOP engaged")
                        safety.engage_e_stop()
            except Exception as e:
                print(f"[E-STOP] Listener error: {e}")
                break

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t

# Safety Controller enforces the e-stop across all motion loops
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

#line place intersection
def is_intersection_point_inside_triangle(intersect_p, triangle_verts):
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
                            print(f"[{robot_name}] COLLISION DETECTED!")
                            print(f"   Link {i} collided with obstacle {obs_idx+1} at position: ({intersect_p[0]:.3f}, {intersect_p[1]:.3f}, {intersect_p[2]:.3f})")
    return False


#GUI
def teach_multi_swift(robots: dict, env, safety, default="CR16", dt=0.02):
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

    header = swift.Label(f"Teach Panel - ACTIVE: {active['name']}")
    env.add(header)

    def set_active(name):
        active["name"] = name
        header.desc = f"Teach Panel - ACTIVE: {name}"
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

#Jacobian
def warn_if_singular(robot, q, robot_name, tol=1e-3):
        J = robot.jacob0(q)
        Jpos = J[:3, :]                  # translational part
        manip = np.sqrt(np.linalg.det(Jpos @ Jpos.T))
        if manip < tol:
            print(f"[{robot_name}] Jacobian near-singular (manip={manip:.2e})")

#RMRC_trial
def rmrc_follow_path(robot, poses, safety, env, robot_name="Robot", dt=DT, gain=1.0):
    """
    Resolved-motion-rate control: follow a list of SE3 targets by
    incrementally adjusting joints using the Jacobian pseudoinverse.
    """
    q = robot.q.copy()

    for step, T_goal in enumerate(poses):
        safety.block_until_allowed(env, dt)

        # Current pose
        T_now = robot.fkine(q)
        p_err = T_goal.t - T_now.t                       # 3D position error
        R_err = T_now.R.T @ T_goal.R
        rotvec = rtb.AngleAxis(R_err).v                  # minimal orientation error

        # Desired spatial velocity (stack translation + rotation)
        v = np.hstack((gain * p_err, gain * rotvec))

        # Jacobian and damped pseudoinverse
        J = robot.jacob0(q)
        lam = 1e-3
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))

        dq = J_pinv @ v
        q = q + dq * dt

        robot.q = q
        env.step(dt)
        time.sleep(dt)

#  Movement FUNCTIONS

def compute_ik_trajectory(robot, T_target, q_current, steps, robot_name="Robot"):
      
    sol = robot.ikine_LM(T_target, q0=q_current)
    #Solve inverse kinematics numerically to reach T_target, 
    #starting the search from my current joint angles
    q_target = sol.q
    q_matrix = rtb.jtraj(q_current, q_target, steps).q 
    return q_matrix

def animate_robot_movement(robot, q_matrix, safety, env, obstacles_list=None, robot_name="Robot"):
    for i, q in enumerate(q_matrix):
        safety.block_until_allowed(env, DT)
        
        if obstacles_list is not None:
            if check_collision(robot, q, obstacles_list, robot_name=robot_name):
                pass  
        
        robot.q = q
        warn_if_singular(robot, robot.q, robot_name)

        #singularity check
        J = robot.jacob0(robot.q)
        trans_jac = J[:3, :]  # translational part
        w = np.sqrt(np.linalg.det(trans_jac @ trans_jac.T))
        if w < 1e-3:
            print(f"[{robot_name}] Near singularity (manipulability={w:.2e}) at step {i}")
        
        env.step(DT)
        time.sleep(DT)


def animate_robot_with_object(robot, q_matrix, obj, T_offset, safety, env, robot_name="Robot"):
    for i, q in enumerate(q_matrix):
        safety.block_until_allowed(env, DT)
        robot.q = q
        
        T_ee = robot.fkine(robot.q)
        obj.T = (T_ee @ T_offset).A
        
        env.step(DT)
        time.sleep(DT)


def animate_robot_with_multiple_objects(robot, q_matrix, objects_with_offsets, safety, env, robot_name="Robot"):
    
    for i, q in enumerate(q_matrix):
        safety.block_until_allowed(env, DT)
        robot.q = q
        
        T_ee = robot.fkine(robot.q)
        for obj, T_offset in objects_with_offsets:
            obj.T = (T_ee @ T_offset).A
        
        env.step(DT)
        time.sleep(DT)

#how the code works
def main():
    apply_swift_browser_fix()
    env = swift.Swift()
    env.launch(realtime=True, browser=None, host="127.0.0.1", port=52100, ws_port=53100)

    safety = SafetyController()
    threading.Thread(target=launch_safety_gui, args=(safety,), daemon=True).start()
    start_arduino_estop_listener(safety, port="COM9", baud=115200)

    make_room(
        env,
        room_w=ROOM_W,
        room_d=ROOM_D,
        floor_t=0.10,
        open_side="+Y",
        floor_top=FLOOR_TOP,
    )

    # Add all objects
    
    SMALL_GAP = 0.003
    obstacles_list = []
    
# Stove
  
    stove = sg.Mesh(
        filename=str(Path(__file__).parent / "assets" / "Stove.stl"),
        scale=[1.0, 1.0, 1.0],
        color=[0.70, 0.70, 0.70, 1.0]
        )

    stove_x, stove_y = POSITIONS["STOVE"]
    stove_z = HEIGHT_OFFSETS["STOVE"]
    stoveT = SE3(stove_x, stove_y, FLOOR_TOP + stove_z) @ SE3.Rz(math.pi)

    stove.T = stoveT
    env.add(stove)

 # Table1

    table1 = sg.Mesh(
        filename=str(Path(__file__).parent / "assets" / "table.stl"),
        scale=[1.0, 1.0, 1.0],
        color=[0.50, 0.50, 0.50, 1.0]
        )
    
    table1_x, table1_y = POSITIONS["TABLE1"]
    table1_z = HEIGHT_OFFSETS["TABLE1"]
    table1_T = SE3(table1_x, table1_y, FLOOR_TOP + table1_z) @ SE3.Rz(math.pi / 2)

    table1.T = table1_T
    env.add(table1)

    # Add table 1 collision geometry
    table1_x, table1_y = POSITIONS["TABLE1"]
    table1_lwh = [1.5, 1.0, 0.8]
    table1_center = [table1_x, table1_y, table1_z / 2]
    vertices_t1, faces_t1, normals_t1 = RectangularPrism(
        table1_lwh[0], table1_lwh[1], table1_lwh[2], center=table1_center
    ).get_data()
    obstacles_list.append((faces_t1, vertices_t1, normals_t1))

# table 2

    table2 = sg.Mesh(
        filename=str(Path(__file__).parent / "assets" / "table2.stl"),
        scale=[1.0, 1.0, 1.0],
        color=[0.50, 0.50, 0.50, 1.0]
        )

    table2_x, table2_y = POSITIONS["TABLE2"]
    table2_z = HEIGHT_OFFSETS["TABLE2"]
    table2_T = SE3(table2_x, table2_y, FLOOR_TOP + table2_z) @ SE3.RPY([-90, 0, 0], order='xyz', unit='deg')

    table2.T = table2_T
    env.add(table2)

    # Add table 2 collision geometry
    table2_x, table2_y = POSITIONS["TABLE2"]
    table2_lwh = [1.5, 1.0, 0.8]
    table2_center = [table2_x, table2_y, table2_z / 2]
    vertices_t2, faces_t2, normals_t2 = RectangularPrism(
        table2_lwh[0], table2_lwh[1], table2_lwh[2], center=table2_center
    ).get_data()
    obstacles_list.append((faces_t2, vertices_t2, normals_t2))

# pot

    pot = sg.Mesh(
        filename=str(Path(__file__).parent / "assets" / "Potwithoutthelid.stl"),
        scale=[1.0, 1.0, 1.0],
        color=[1.0, 0.0, 0.0, 1.0]
        )

    pot_x, pot_y = POSITIONS["POT"]
    pot_z = HEIGHT_OFFSETS["POT"]
    pot_T = SE3(pot_x, pot_y, FLOOR_TOP + pot_z)
    pot.T = pot_T
    env.add(pot)

    # jug


    jug = sg.Mesh(
        filename=str(Path(__file__).parent / "assets" / "jugfixed.stl"),
        scale=[1.0, 1.0, 1.0],
        color=[0.8, 0.9, 1.0, 1.0]
        )

    jug_x, jug_y = POSITIONS["JUG"]
    jug_z = HEIGHT_OFFSETS["JUG"]
    jug_T = SE3(jug_x, jug_y, FLOOR_TOP + jug_z)

    jug.T = jug_T
    env.add(jug)

#pepper

    pepper_grinder = sg.Mesh(
        filename=str(Path(__file__).parent / "assets" / "pepper_grinder.stl"),
        scale=[1.0, 1.0, 1.0],
        color= [0.2, 0.2, 0.2, 1.0]
        )

    pepper_grinder_x, pepper_grinder_y = POSITIONS["PEPPER"]
    pepper_grinder_z = HEIGHT_OFFSETS["PEPPER"]
    pepper_grinder_T = SE3(pepper_grinder_x, pepper_grinder_y, FLOOR_TOP + pepper_grinder_z)

    pepper_grinder.T = pepper_grinder_T
    env.add(pepper_grinder)

#beef

    beef = sg.Mesh(
        filename=str(Path(__file__).parent / "assets" / "beef.stl"),
        scale=[1.0, 1.0, 1.0],
        color= [0.8, 0.3, 0.3, 1.0]
        )

    beef_x, beef_y = POSITIONS["BEEF"]
    beef_z = HEIGHT_OFFSETS["BEEF"]
    beef_T = SE3(beef_x, beef_y, FLOOR_TOP + beef_z)

    beef.T = beef_T
    env.add(beef)

# fruit and veg tray

    fruit_veg_tray = sg.Mesh(
        filename=str(Path(__file__).parent / "assets" / "Fruit_and_Vegetables_Tray.stl"),
        scale=[1.0, 1.0, 1.0],
        color= [0.4, 0.7, 0.3, 1.0])

    fruit_veg_tray_x, fruit_veg_tray_y = POSITIONS["FRUIT_VEG"]
    fruit_veg_tray_z = HEIGHT_OFFSETS["FRUIT_VEG"]
    fruit_veg_tray_T = SE3(fruit_veg_tray_x, fruit_veg_tray_y, FLOOR_TOP + fruit_veg_tray_z)

    fruit_veg_tray.T = fruit_veg_tray_T
    env.add(fruit_veg_tray)

# chicken

    chicken = sg.Mesh(
        filename=str(Path(__file__).parent / "assets" / "chicken.stl"),
        scale=[1.0, 1.0, 1.0],
        color= [1.0, 0.9, 0.7, 1.0])

    chicken_x, chicken_y = POSITIONS["CHICKEN"]
    chicken_z = HEIGHT_OFFSETS["CHICKEN"]
    chicken_T = SE3(chicken_x, chicken_y, FLOOR_TOP + chicken_z)

    chicken.T = chicken_T
    env.add(chicken)

# hardware stop

    hw_stop = sg.Mesh(
        filename=str(Path(__file__).parent / "assets" / "hardwarestop.stl"),
        scale=[1.0, 1.0, 1.0],
        color= [1.0, 0.2, 0.2, 1.0])

    hw_stop_x, hw_stop_y = POSITIONS["HW_STOP"]
    hw_stop_z = HEIGHT_OFFSETS["HW_STOP"]
    hw_stop_T = SE3(hw_stop_x, hw_stop_y, FLOOR_TOP + hw_stop_z)

    hw_stop.T = hw_stop_T
    env.add(hw_stop)


# barrier

    barrier = sg.Mesh(
        filename=str(Path(__file__).parent / "assets" / "barrier.stl"),
        scale=[1.0, 1.0, 1.0],
        color= [0.90, 0.20, 0.20, 1.0])
    barrier_x, barrier_y = POSITIONS["BARRIER"]
    barrier_z = HEIGHT_OFFSETS["BARRIER"]
    barrier_T = SE3(barrier_x, barrier_y, FLOOR_TOP + barrier_z)

    barrier.T = barrier_T
    env.add(barrier)

    # Linear UR3 on rail
    ur3 = LinearUR3()
    ur3_x, ur3_y = POSITIONS["UR3"]
    RAIL_Z = FLOOR_TOP + SMALL_GAP + HEIGHT_OFFSETS["UR3"]
    YAW = math.pi / 2
    ur3.base = SE3(ur3_x, ur3_y, RAIL_Z) @ SE3.Rz(YAW) @ ur3.base
    ur3.add_to_env(env)

    # CR3 Robot (on TABLE1)
    cr3 = CR3()
    cr3_x, cr3_y = POSITIONS["CR3"]
    CR3_Z = TABLE1_HEIGHT + SMALL_GAP + HEIGHT_OFFSETS["CR3"]  
    CR3_YAW = -math.pi / 2
    base0_cr3 = getattr(cr3, "base", SE3())
    cr3.base = SE3(cr3_x, cr3_y, CR3_Z) @ SE3.Rz(CR3_YAW) @ base0_cr3
    cr3.add_to_env(env)
    env.step(0.02)

    # CR16 
    cr16 = CR16()
    cr16_x, cr16_y = POSITIONS["CR16"]
    CR16_Z = TABLE1_HEIGHT + SMALL_GAP + HEIGHT_OFFSETS["CR16"] 
    CR16_YAW = +math.pi / 2
    base0_cr16 = getattr(cr16, "base", SE3())
    cr16.base = SE3(cr16_x, cr16_y, CR16_Z) @ SE3.Rz(CR16_YAW) @ base0_cr16
 
    cr16.add_to_env(env)


    # Teach panel for CR16 + CR3
    robots = {}
    robots["CR16"] = cr16
    robots["CR3"] = cr3

    teach_multi_swift(robots, env, safety, default="CR16", dt=DT)

    env.step(0.02)


    # Movement Configuration
    steps = 50
    
    # ROBOT MOVEMENTS
    
    # UR3 Movement to Beef
    beef_x, beef_y = POSITIONS["BEEF"]
    beef_z = TABLE2_HEIGHT + 0.15  
    T_beef = SE3(beef_x, beef_y, beef_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix = compute_ik_trajectory(ur3, T_beef, q_current, steps, "UR3")
    
    animate_robot_movement(ur3, q_matrix, safety, env, obstacles_list, "UR3")
    
    
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
    target_x = -beef_x - 0.2
    target_y = beef_y
    target_z = beef_z + 0.05
    
    T_target = SE3(target_x, target_y, target_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix_move = compute_ik_trajectory(ur3, T_target, q_current, steps, "UR3")
    animate_robot_with_object(ur3, q_matrix_move, beef, T_offset, safety, env, "UR3")
    
    
    # Release beef at target position
    T_beef_final = beef.T @ SE3.Rx(math.pi).A
    
    # UR3 Movement to Chicken (on TABLE2)
    
    chicken_x, chicken_y = POSITIONS["CHICKEN"]
    chicken_z = TABLE2_HEIGHT + 0.15
    
    T_chicken = SE3(chicken_x, chicken_y, chicken_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix = compute_ik_trajectory(ur3, T_chicken, q_current, steps, "UR3")
    animate_robot_movement(ur3, q_matrix, safety, env, obstacles_list, "UR3")
        
    # Attach chicken with offset

    CHICKEN_OFFSET_X = -0.3
    CHICKEN_OFFSET_Y = -0.09
    CHICKEN_OFFSET_Z = 0
    T_offset_chicken = SE3(CHICKEN_OFFSET_X, CHICKEN_OFFSET_Y, CHICKEN_OFFSET_Z)
    
    T_ee = ur3.fkine(ur3.q)
    chicken.T = T_ee @ T_offset_chicken
    env.step(DT)
    
    # Move chicken to new position
    
    target_x_chicken = -chicken_x - 0.2
    target_y_chicken = chicken_y
    target_z_chicken = chicken_z + 0.05
    
    T_target_chicken = SE3(target_x_chicken, target_y_chicken, target_z_chicken) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix_move = compute_ik_trajectory(ur3, T_target_chicken, q_current, steps, "UR3")
    animate_robot_with_object(ur3, q_matrix_move, chicken, T_offset_chicken, safety, env, "UR3")
    
    
    # Release chicken
    T_chicken_final = chicken.T @ SE3.Rx(math.pi).A
    
    # UR3 Movement to Pepper 
    
    pepper_x, pepper_y = POSITIONS["PEPPER"]
    pepper_z = TABLE2_HEIGHT + 0.01
    
    T_pepper = SE3(pepper_x, pepper_y, pepper_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix = compute_ik_trajectory(ur3, T_pepper, q_current, steps, "UR3")
    animate_robot_movement(ur3, q_matrix, safety, env, obstacles_list, "UR3")
    

    
    # Attach pepper
    PEPPER_OFFSET_X = -0.3
    PEPPER_OFFSET_Y = -0.09
    PEPPER_OFFSET_Z = 0
    T_offset_pepper = SE3(PEPPER_OFFSET_X, PEPPER_OFFSET_Y, PEPPER_OFFSET_Z)
    
    T_ee = ur3.fkine(ur3.q)          
    pepper_grinder.T = T_ee @ T_offset_pepper
    env.step(DT)
    
    # Move pepper to new position
    
    target_x_pepper = -pepper_x - 0.05
    target_y_pepper = pepper_y
    target_z_pepper = pepper_z + 0.05
    
    T_target_pepper = SE3(target_x_pepper, target_y_pepper, target_z_pepper) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix_move = compute_ik_trajectory(ur3, T_target_pepper, q_current, steps, "UR3")
    animate_robot_with_object(ur3, q_matrix_move, pepper_grinder, T_offset_pepper, safety, env, "UR3")
    
    
    # Release pepper
    T_pepper_final = pepper_grinder.T @ SE3.Rx(math.pi).A
    
    # UR3 Movement to Jug
    
    jug_x, jug_y = POSITIONS["JUG"]
    jug_z = TABLE2_HEIGHT + 0.15  
    
    T_jug = SE3(jug_x, jug_y, jug_z) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix = compute_ik_trajectory(ur3, T_jug, q_current, steps, "UR3")
    animate_robot_movement(ur3, q_matrix, safety, env, obstacles_list, "UR3")
    
    
    # Attach jug
    JUG_OFFSET_X = -0.3
    JUG_OFFSET_Y = -0.09
    JUG_OFFSET_Z = 0
    T_offset_jug = SE3(JUG_OFFSET_X, JUG_OFFSET_Y, JUG_OFFSET_Z)
    
    T_ee = ur3.fkine(ur3.q)
    jug.T = T_ee @ T_offset_jug
    env.step(DT)
    
    # Move jug to new position    
    target_x_jug = -jug_x - 0.2
    target_y_jug = jug_y
    target_z_jug = jug_z + 0.05
    
    T_target_jug = SE3(target_x_jug, target_y_jug, target_z_jug) @ SE3.Rx(math.pi)
    q_current = ur3.q.copy()
    
    q_matrix_move = compute_ik_trajectory(ur3, T_target_jug, q_current, steps, "UR3")
    animate_robot_with_object(ur3, q_matrix_move, jug, T_offset_jug, safety, env, "UR3")
    
    
    # Release jug
    T_jug_final = jug.T @ SE3.Rx(math.pi).A
    
    # CR3 Movement to Beef

    
    beef_pickup_x = -beef_x - 0.2
    beef_pickup_y = beef_y
    beef_pickup_z = beef_z + 0.05 + 0.15
    
    T_beef_pickup = SE3(beef_pickup_x, beef_pickup_y, beef_pickup_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    
    q_matrix_cr3 = compute_ik_trajectory(cr3, T_beef_pickup, q_current_cr3, steps, "CR3")
    animate_robot_movement(cr3, q_matrix_cr3, safety, env, robot_name="CR3")
    
    
    # Attach beef to CR3
    CR3_OFFSET_X = 0.0
    CR3_OFFSET_Y = 0.0
    CR3_OFFSET_Z = -0.05
    T_offset_cr3 = SE3(CR3_OFFSET_X, CR3_OFFSET_Y, CR3_OFFSET_Z)
    
    # Move beef to pot
    
    pot_x, pot_y = POSITIONS["POT"]
    pot_z = TABLE1_HEIGHT + 0.2
    
    T_pot = SE3(pot_x, pot_y, pot_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    
    q_matrix_to_pot = compute_ik_trajectory(cr3, T_pot, q_current_cr3, steps, "CR3")
    animate_robot_with_object(cr3, q_matrix_to_pot, beef, T_offset_cr3 @ SE3.Rx(math.pi), safety, env, "CR3")
    
    beef.T = SE3(pot_x, pot_y, pot_z - 0.1).A
    
    # CR3 Movement to Chicken
    
    chicken_pickup_x = -chicken_x - 0.2
    chicken_pickup_y = chicken_y
    chicken_pickup_z = chicken_z + 0.05 + 0.15
    
    T_chicken_pickup = SE3(chicken_pickup_x, chicken_pickup_y, chicken_pickup_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    
    q_matrix_cr3 = compute_ik_trajectory(cr3, T_chicken_pickup, q_current_cr3, steps, "CR3")
    animate_robot_movement(cr3, q_matrix_cr3, safety, env, robot_name="CR3")
        
    q_current_cr3 = cr3.q.copy()
    q_matrix_to_pot = compute_ik_trajectory(cr3, T_pot, q_current_cr3, steps, "CR3")
    animate_robot_with_object(cr3, q_matrix_to_pot, chicken, T_offset_cr3 @ SE3.Rx(math.pi), safety, env, "CR3")
    
    chicken.T = SE3(pot_x, pot_y, pot_z - 0.1).A
    
    # CR3 Movement to Pepper
    
    pepper_pickup_x = -pepper_x - 0.05
    pepper_pickup_y = pepper_y
    pepper_pickup_z = pepper_z + 0.05
    
    T_pepper_pickup = SE3(pepper_pickup_x, pepper_pickup_y, pepper_pickup_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    
    q_matrix_cr3 = compute_ik_trajectory(cr3, T_pepper_pickup, q_current_cr3, steps, "CR3")
    animate_robot_movement(cr3, q_matrix_cr3, safety, env, robot_name="CR3")
    
    
    q_current_cr3 = cr3.q.copy()
    q_matrix_to_pot = compute_ik_trajectory(cr3, T_pot, q_current_cr3, steps, "CR3")
    animate_robot_with_object(cr3, q_matrix_to_pot, pepper_grinder, T_offset_cr3 @ SE3.Rx(math.pi), safety, env, "CR3")
    
    pepper_grinder.T = SE3(pot_x, pot_y, pot_z - 0.01).A
    
    # CR3 Movement to Jug
    #
    
    jug_pickup_x = -jug_x - 0.2
    jug_pickup_y = jug_y
    jug_pickup_z = jug_z + 0.05 + 0.15
    
    T_jug_pickup = SE3(jug_pickup_x, jug_pickup_y, jug_pickup_z) @ SE3.Rx(math.pi)
    q_current_cr3 = cr3.q.copy()
    
    q_matrix_cr3 = compute_ik_trajectory(cr3, T_jug_pickup, q_current_cr3, steps, "CR3")
    animate_robot_movement(cr3, q_matrix_cr3, safety, env, robot_name="CR3")
    
    
    q_current_cr3 = cr3.q.copy()
    q_matrix_to_pot = compute_ik_trajectory(cr3, T_pot, q_current_cr3, steps, "CR3")
    animate_robot_with_object(cr3, q_matrix_to_pot, jug, T_offset_cr3 @ SE3.Rx(math.pi), safety, env, "CR3")
    
    jug.T = SE3(pot_x, pot_y, pot_z - 0.1).A
    
    
    # CR16 Movement to Pot
    
    pot_x, pot_y = POSITIONS["POT"]
    pot_z = TABLE1_HEIGHT + 0.2 
    
    T_pot_pickup = SE3(pot_x, pot_y, pot_z) @ SE3.Rx(math.pi)
    q_current_cr16 = cr16.q.copy()
    
    q_matrix_cr16 = compute_ik_trajectory(cr16, T_pot_pickup, q_current_cr16, steps, "CR16")
    animate_robot_movement(cr16, q_matrix_cr16, safety, env, robot_name="CR16")
    
    
    # Attach pot to CR16 with offset
    CR16_OFFSET_X = 0.0
    CR16_OFFSET_Y = 0.0
    CR16_OFFSET_Z = -0.05
    T_offset_cr16 = SE3(CR16_OFFSET_X, CR16_OFFSET_Y, CR16_OFFSET_Z)
    
    # Get stove position for placement
    stove_x, stove_y = POSITIONS["STOVE"]
    stove_z = FLOOR_TOP + 0.9
    
    # Move pot to stove
    T_stove = SE3(stove_x, stove_y, stove_z) @ SE3.Rx(math.pi)
    
    q_current_cr16 = cr16.q.copy()
    q_matrix_to_stove = compute_ik_trajectory(cr16, T_stove, q_current_cr16, steps, "CR16")
    
    # Animate with pot and all ingredients following
    objects_with_offsets = [
        (pot, T_offset_cr16),
        (beef, T_offset_cr16 @ SE3(0, 0, -0.1)),
        (chicken, T_offset_cr16 @ SE3(0, 0, -0.1)),
        (pepper_grinder, T_offset_cr16 @ SE3(0, 0, -0.1)),
        (jug, T_offset_cr16 @ SE3(0, 0, -0.1))
    ]
    
    animate_robot_with_multiple_objects(cr16, q_matrix_to_stove, objects_with_offsets, safety, env, "CR16")
    
    
    # Release pot on stove
    pot.T = SE3(stove_x, stove_y, stove_z - 0.05).A
    beef.T = SE3(stove_x, stove_y, stove_z - 0.15).A
    chicken.T = SE3(stove_x, stove_y, stove_z - 0.15).A
    pepper_grinder.T = SE3(stove_x, stove_y, stove_z - 0.15).A
    jug.T = SE3(stove_x, stove_y, stove_z - 0.15).A
    
    env.hold()

if __name__ == "__main__":
    main()