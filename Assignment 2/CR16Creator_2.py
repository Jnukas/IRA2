import os, time
import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
from math import pi

class CR16(DHRobot3D):
    def __init__(self):
        # 6-DOF standard DH (meters, radians)
        links = self._create_DH()

        # Mesh file names (7 meshes: base + 6 links)
        link3D_names = dict(
            link0='CR16_link_1',  # base
            link1='CR16_link_2',
            link2='CR16_link_3',
            link3='CR16_link_4',
            link4='CR16_link_5',
            link5='CR16_link_6',
            link6='CR16_link_7'   # flange / tool-side housing
        )

        # Test pose and transforms (identity assumes meshes are DH-aligned)
        qtest = [0, 0, 0, 0, 0, 0]  # start at "all zeros" now that offsets are encoded
        qtest_transforms = [spb.transl(0, 0, 0) for _ in range(7)]

        current_path = os.path.abspath(os.path.dirname(__file__))
        super().__init__(
            links,
            link3D_names,
            name='CR16',
            link3d_dir=current_path,
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )
        self.q = qtest  # set starting pose

import numpy as np
import roboticstoolbox as rtb
from math import pi

def d2r(x): return np.deg2rad(x)

def _internal_lim(mech_min_deg, mech_max_deg, offset_deg):
    # internal q = mechanical - offset
    return (d2r(mech_min_deg - offset_deg), d2r(mech_max_deg - offset_deg))

def _create_DH(self):
    # --- your DH numbers ---
    a     = [0.0,   -0.512, -0.363,  0.0,   0.0,   0.0]
    d     = [0.1785, 0.0,    0.0,    0.191, 0.125, 0.1084]
    alpha = [pi/2,   0.0,    0.0,    pi/2, -pi/2, 0.0]

    # DH joint offsets (you had J2 = -90°)
    offsets_deg = [0.0, -90.0, 0.0, 0.0, 0.0, 0.0]

    # Mechanical limits from the datasheet image
    mech_limits_deg = [
        (-360.0,  360.0),  # J1
        (-360.0,  360.0),  # J2
        (-160.0,  160.0),  # J3
        (-360.0,  360.0),  # J4
        (-360.0,  360.0),  # J5
        (-360.0,  360.0),  # J6
    ]

    # Convert mechanical → internal using the offsets
    qlims = [_internal_lim(mn, mx, off)
             for (mn, mx), off in zip(mech_limits_deg, offsets_deg)]

    links = []
    for i in range(6):
        links.append(
            rtb.RevoluteDH(
                d=d[i], a=a[i], alpha=alpha[i],
                offset=d2r(offsets_deg[i]),
                qlim=qlims[i]
            )
        )
    return links


    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)

        # show it somewhere sensible
        self.base = SE3(0.5, 0.5, 0.0)
        self.add_to_env(env)

        # small joint wiggle to verify axes
        q_start = self.q
        q_goal  = [q_start[i] + (pi/6 if i % 2 == 0 else -pi/6) for i in range(self.n)]
        qtraj = rtb.jtraj(q_start, q_goal, 60).q

        for q in qtraj:
            self.q = q
            env.step(0.02)

        env.hold()

if __name__ == "__main__":
    r = CR16()
    r.test()
