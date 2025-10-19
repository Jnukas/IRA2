#!/usr/bin/env python3
"""
CR16 with PLY meshes (converted from DAE) using DHRobot3D + Swift + Tk sliders.

Files expected in this folder:
  cr16_link1.ply ... cr16_link7.ply   (base + 6 links)
"""

import os, threading, time
from math import pi, radians
from pathlib import Path
import numpy as np

import swift
import spatialmath.base as spb
from spatialmath import SE3
import roboticstoolbox as rtb
from ir_support.robots.DHRobot3D import DHRobot3D
import tkinter as tk


class CR16(DHRobot3D):
    def __init__(self):
        # ---- DH: replace with your verified numbers when you have them ----
        a     = [0.0,   -0.512,  -0.363,  0.0,    0.0,    0.0]
        d     = [0.1765, 0.0,     0.0,     0.191,  0.125,  0.1084]
        alpha = [pi/2,   0.0,     0.0,     pi/2,  -pi/2,   0.0]
        qlim  = np.array([
            [-2*pi,  2*pi],
            [-2*pi,  2*pi],
            [radians(-160), radians(160)],
            [-2*pi,  2*pi],
            [-2*pi,  2*pi],
            [-2*pi,  2*pi],
        ]).T  # shape (2,6)

        links = [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[:, i]) for i in range(6)]

        # ---- Mesh filenames (PLY) — base + 6 links ----
        # Use your own basenames if different; no extension here (class adds .ply internally)
        link3D_names = dict(
            link0="cr16_link1",
            link1="cr16_link2",
            link2="cr16_link3",
            link3="cr16_link4",
            link4="cr16_link5",
            link5="cr16_link6",
            link6="cr16_link7",
        )

        # A neutral pose and per-link model transforms that match that pose
        # Start with identity transforms; tweak if your meshes have different local axes
        qtest = [0, -pi/2, 0, 0, 0, 0]
        qtest_transforms = [spb.trotx(0.0) for _ in range(7)]

        current_path = os.path.abspath(os.path.dirname(__file__))
        super().__init__(
            links,
            link3D_names,
            name="CR16",
            link3d_dir=current_path,
            qtest=qtest,
            qtest_transforms=qtest_transforms,
        )

        self.q = np.zeros(6)
        self.qlim = qlim  # ensure qlim is numpy array shape (2,6)


def launch_with_sliders():
    robot = CR16()

    env = swift.Swift()
    env.launch(realtime=True, browser=None)   # set browser=None to avoid Windows webbrowser quirks
    robot.base = SE3(0.5, 0.0, 0.0)
    robot.add_to_env(env)

    # --- Tk sliders (degrees) ---
    root = tk.Tk()
    root.title("CR16 (mesh) — Joint Sliders")

    lo_deg = np.degrees(robot.qlim[0, :])
    hi_deg = np.degrees(robot.qlim[1, :])

    sliders = []
    for i in range(6):
        frame = tk.Frame(root)
        frame.pack(fill="x", padx=8, pady=4)

        tk.Label(frame, text=f"J{i+1}", width=4).pack(side="left")

        var = tk.DoubleVar(value=0.0)
        sld = tk.Scale(
            frame, variable=var,
            from_=hi_deg[i], to=lo_deg[i],  # up = +
            orient=tk.HORIZONTAL, resolution=1.0, length=360
        )
        sld.pack(side="left", expand=True, fill="x")
        val_lbl = tk.Label(frame, text="0°", width=6)
        val_lbl.pack(side="right")

        def on_move(idx=i, v=var, readout=val_lbl):
            deg = v.get()
            readout.config(text=f"{deg:.0f}°")
            q_deg = [sliders[j][0].get() for j in range(6)]
            robot.q = np.radians(q_deg)

        sld.configure(command=lambda _evt=None, f=on_move: f())
        sliders.append((sld, var))

    ee_lbl = tk.Label(root, text="EE pose: updating...", anchor="w", justify="left")
    ee_lbl.pack(fill="x", padx=8, pady=6)

    stop = {"flag": False}

    def loop():
        while not stop["flag"]:
            env.step(0.02)
            try:
                T = robot.fkine(robot.q)
                p = T.t
                rpy = np.degrees(T.rpy(order="zyx"))
                ee_lbl.configure(
                    text=f"EE XYZ (m): [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]   RPY zyx (deg): [{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}]"
                )
            except Exception:
                pass
            time.sleep(0.02)

    th = threading.Thread(target=loop, daemon=True)
    th.start()

    def on_close():
        stop["flag"] = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    time.sleep(0.1)


if __name__ == "__main__":
    launch_with_sliders()
