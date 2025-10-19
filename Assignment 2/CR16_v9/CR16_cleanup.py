#   @brief CR16 Robot defined by standard DH parameters with 3D model

import os, time
from math import pi

import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk



class CR16(DHRobot3D):
    def __init__(self):
        links = self._create_DH()

        # 7 visuals = base + 6 links (order must match your exported files)
        link3D_names = dict(
            link0='CR16_link_1',
            link1='CR16_link_2',
            link2='CR16_link_3',
            link3='CR16_link_4',
            link4='CR16_link_5',
            link5='CR16_link_6',
            link6='CR16_link_7',
        )

        # Home pose for testing visuals (elbow-down typical UR pose)
        self.qhome = [0, -pi/2, 0, 0, 0, 0]

        # Per-visual fixed offsets (identity → assume Blender pivots already match DH)
        qtest_transforms = [spb.transl(0, 0, 0) for _ in range(7)]

        link3d_dir = os.path.abspath(os.path.dirname(__file__))

        # Quick shape sanity checks before constructing
        assert len(links) == 6, "Expecting 6 RevoluteDH links"
        assert len(link3D_names) == 7, "Expecting 7 visual parts: base + 6"
        assert len(qtest_transforms) == 7, "qtest_transforms must match visual parts"

        super().__init__(
            links=links,
            link3D_names=link3D_names,
            name='CR16',
            link3d_dir=link3d_dir,
            qtest=self.qhome,
            qtest_transforms=qtest_transforms,
        )
        self.q = self.qhome

    def _create_DH(self):
        a     = [0.0,   -0.512, -0.363, 0.0,   0.0,   0.0]
        d     = [0.1785, 0.0,    0.0,    0.191, 0.125, 0.1084]
        alpha = [pi/2,   0.0,    0.0,    pi/2, -pi/2, 0.0]
        qlim  = [[-2*pi, 2*pi] for _ in range(6)]

        links = []
        for i in range(6):
            links.append(rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i]))
        return links

    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)

        self.q = self.qhome
        self.base = SE3(0.5, 0.5, 0.0)
        self.add_to_env(env)

        q_goal = [self.q[i] - pi/3 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q
        for q in qtraj:
            self.q = q
            env.step(0.02)
        time.sleep(1.0)
        env.hold()



if __name__ == "__main__":
    CR16().test()