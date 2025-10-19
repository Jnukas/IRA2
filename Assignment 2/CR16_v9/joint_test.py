##  @file
#   @brief UR3 Robot defined by standard DH parameters with 3D model
#   @author Ho Minh Quang Ngo
#   @date Jul 20, 2023

import time, os, threading
from math import pi
import numpy as np
import tkinter as tk

import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D


# -----------------------------------------------------------------------------------#
class CR16(DHRobot3D):
    def __init__(self):
        """
        UR3/CR16-style robot using DHRobot3D + Swift.
        """
        # DH links
        links = self._create_DH()

         # Names of the robot link files in the directory
        link3D_names = dict(link0 = 'CR16_link_1',
                            link1 = 'CR16_link_2',
                            link2 = 'CR16_link_3',
                            link3 = 'CR16_link_4',
                            link4 = 'CR16_link_5',
                            link5 = 'CR16_link_6',
                            link6 = 'CR16_link_7'
        )

        # A joint config and the 3D object transforms to match that config
        qtest = [0, -pi/2, 0, 0, 0, 0]
        qtest_transforms = [spb.transl(0, 0, 0) for _ in range(7)]

        current_path = os.path.abspath(os.path.dirname(__file__))
        super().__init__(
            links,
            link3D_names,
            name='UR3',
            link3d_dir=current_path,
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )
        self.q = qtest

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        """
        Create robot's standard DH model
        """
        a     = [0,      -0.512, -0.363, 0,       0,     0]
        d     = [0.1785,  0,      0,      0.191,  0.125, 0.1084]
        alpha = [pi/2,    0.0,    0.0,    pi/2,  -pi/2,  0.0]
        qlim  = [[-2*pi, 2*pi] for _ in range(6)]

        links = []
        for i in range(6):
            link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i])
            links.append(link)
        return links

    # -----------------------------------------------------------------------------------#
    def teach_swift(self, q0=None, use_degrees=True, step_hz=50):
        """
        Open a Swift window + Tk sliders to move each joint interactively.
        - q0: optional initial configuration (len==self.n)
        - use_degrees: sliders show degrees; values converted to radians
        - step_hz: Swift update rate
        """
        # ---- Swift setup ----
        env = swift.Swift()
        env.launch(realtime=True)

        if q0 is None:
            q0 = np.zeros(self.n)
        else:
            q0 = np.asarray(q0, dtype=float)

        self.q = q0
        self.base = SE3(0.5, 0.5, 0.0)  # move off origin if you like
        self.add_to_env(env)

        # ---- Tk slider panel ----
        root = tk.Tk()
        root.title("CR16 Swift Teach — Joint Sliders")

        qlim = np.array(self.qlim)  # shape (2, n)
        sliders = []

        def on_change(i, val):
            # Update joint i from slider, keep others as-is
            q = self.q.copy()
            v = float(val)
            q[i] = np.deg2rad(v) if use_degrees else v
            self.q = q  # DHRobot3D/RTB will redraw on next env.step()

        for i in range(self.n):
            if use_degrees:
                lo, hi = np.degrees(qlim[:, i])
                res = 1.0
                start = np.degrees(q0[i])
            else:
                lo, hi = qlim[:, i]
                res = 0.01
                start = q0[i]

            s = tk.Scale(
                root, from_=lo, to=hi, resolution=res, orient="horizontal",
                length=420, label=f"q{i+1} ({'deg' if use_degrees else 'rad'})",
                command=lambda v, i=i: on_change(i, v)
            )
            s.set(start)
            s.pack(padx=8, pady=6)
            sliders.append(s)

        # ---- Swift stepping thread ----
        stop = threading.Event()

        def on_close():
            stop.set()
            try:
                root.destroy()
            except Exception:
                pass

        root.protocol("WM_DELETE_WINDOW", on_close)

        def stepper():
            dt = 1.0 / max(1, step_hz)
            while not stop.is_set():
                env.step(dt)
                time.sleep(dt)

        threading.Thread(target=stepper, daemon=True).start()
        root.mainloop()

        # Optional: hold Swift open after sliders close
        # env.hold()  # uncomment if you want Swift to persist

    # -----------------------------------------------------------------------------------#
    # Keeping your original demo as-is (optional)
    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)
        self.q = self._qtest
        self.base = SE3(0.5, 0.5, 0)
        self.add_to_env(env)

        q_goal = [self.q[i]-pi/3 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q
        for q in qtraj:
            self.q = q
            env.step(0.02)
        time.sleep(3)
        env.hold()


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    r = CR16()
    # Launch Swift + sliders to move joints individually
    r.teach_swift(q0=[0, -pi/2, 0, 0, 0, 0], use_degrees=True, step_hz=60)
    # Or, to run your original scripted motion:
    # r.test()
