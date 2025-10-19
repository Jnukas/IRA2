##  @file
#   @brief UR3 Robot defined by standard DH parameters with 3D model
#   @author Ho Minh Quang Ngo
#   @date Jul 20, 2023

import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import time
import os

# Useful variables
from math import pi

# -----------------------------------------------------------------------------------#
class CR16(DHRobot3D):
    def __init__(self):
        """
            UR3 Robot by DHRobot3D class

            Example usage:
            >>> from ir-support import UR3
            >>> import swift

            >>> r = UR3()
            >>> q = [0,-pi/2,pi/4,0,0,0]r
            >>> r.q = q
            >>> q_goal = [r.q[i]-pi/4 for i in range(r.n)]
            >>> env = swift.Swift()
            >>> env.launch(realtime= True)
            >>> r.add_to_env(env)
            >>> qtraj = rtb.jtraj(r.q, q_goal, 50).q
            >>> for q in qtraj:r
            >>>    r.q = q
            >>>    env.step(0.02)
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
                            link6 = 'CR16_link_7')

        # A joint config and the 3D object transforms to match that config
        qtest = [0,-pi/2,0,0,0,0]
        qtest_transforms = [spb.transl(0, 0, 0) for _ in range(7)]


        current_path = os.path.abspath(os.path.dirname(__file__))
        super().__init__(links, link3D_names, name = 'UR3', link3d_dir = current_path, qtest = qtest, qtest_transforms = qtest_transforms)
        self.q = qtest

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        """
        Create robot's standard DH model
        """
        a = [0,      -0.512, -0.363, 0,       0, 0]
        d = [0.1785, 0,         0,       0.191, 0.125, 0.1084]
        alpha = [pi/2, 0.0,    0.0,    pi/2, -pi/2, 0]
        qlim = [[-2*pi, 2*pi] for _ in range(6)]
        links = []
        for i in range(6):
            link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim= qlim[i])
            links.append(link)
        return links

    # -----------------------------------------------------------------------------------#
    def test(self):
        """
        Test the class by adding 3d objects into a new Swift window and do a simple movement
        """
        env = swift.Swift()
        env.launch(realtime= True)
        self.q = self._qtest
        self.base = SE3(0.5,0.5,0)
        self.add_to_env(env)

        q_goal = [self.q[i]-pi/3 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q
        # fig = self.plot(self.q)
        for q in qtraj:
            self.q = q
            env.step(0.02)
            # fig.step(0.01)
        time.sleep(3)
        env.hold()

# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    r = CR16()
    r.test()

