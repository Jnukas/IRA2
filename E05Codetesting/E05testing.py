#!/usr/bin/env python
from __future__ import annotations
import time
import numpy as np
from math import pi
from pathlib import Path

import swift
from spatialmath import SE3
from spatialgeometry import Sphere
from spatialmath.base import transl
from roboticstoolbox import jtraj
from roboticstoolbox.robot.Robot import Robot

ABS_URDF = Path(__file__).resolve().parent / "E05_robot_ABS.urdf"

class E05ABS(Robot):
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"URDF not found: {path}\n"
                                    f"Build it with: python build_e05_absolute_urdf.py")
        links, name, urdf_string, urdf_filepath = self.URDF_read(str(path))
        super().__init__(links, name=name or "E05", urdf_string=urdf_string, urdf_filepath=str(path))
        self.qz = np.zeros(self.n); self.addconfiguration("qz", self.qz)
        self.qr = np.array([0, pi/2, -pi/2, 0, 0, 0])[: self.n]; self.addconfiguration("qr", self.qr)
        print(f"[E05] Loaded: {self.name}")
        print(f"[E05] URDF source: {self.urdf_filepath}")

def main():
    robot = E05ABS(ABS_URDF)

    env = swift.Swift()
    env.launch(realtime=True, browser="windows-default")  # <- no 'False' here

    # smoke test
    env.set_camera_pose([1.3, 1.3, 1.1], [0, 0, 0])

    for _ in range(20):
        env.step(0.02); time.sleep(0.02)

    # add robot
    env.add(robot, readonly=False)
    for _ in range(20):
        env.step(0.02); time.sleep(0.02)

    # short IK -> traj
    robot.q = robot.qz.copy()
    T_goal = transl(0.35, -0.10, 0.35)
    try:
        sol = robot.ikine_LM(T_goal, q0=robot.q)
        q_goal = sol.q if hasattr(sol, "q") else np.asarray(sol)
    except Exception as e:
        print("[WARN] IK failed; using qr. Reason:", e)
        q_goal = robot.qr

    for q in jtraj(robot.q, q_goal, 60).q:
        robot.q = q; env.step(0.03)

    input("✅ Done. Press Enter to close…")
    env.close()

if __name__ == "__main__":
    main()
