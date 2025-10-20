#!/usr/bin/env python3
"""
CR16 pick-from-ground demo:
- Pick target at (x, y, 0), height h
- Pre-grasp above (x, y, z_above), descend, attach, lift
"""

import time
import numpy as np
import swift
from spatialmath import SE3
import spatialgeometry as sg
from roboticstoolbox import jtraj

# Use the one you actually have:
from CR16_cleanup import CR16
# from CR16 import CR16


def ik_solve(robot: CR16, Tep: SE3, qseed=None, vertical=True):
    """
    Try IK with a sensible mask.
    vertical=True  → constrain roll & pitch (tool Z ~ world -Z), free yaw
    """
    if qseed is None:
        qseed = robot.q
    # Constrain position + roll + pitch; free yaw
    mask = np.array([1, 1, 1, 1, 1, 0], dtype=float) if vertical else np.ones(6)
    sol = robot.ikine_LM(Tep, q0=qseed, mask=mask, ilimit=200, slimit=200, tol=1e-9)
    return sol

def go(robot: CR16, env: swift.Swift, q_target, steps=120, hold_attached=None):
    """
    Smoothly move robot from current q to q_target.
    If hold_attached is a spatialgeometry object, keep it attached to EE during the move.
    """
    qtraj = jtraj(robot.q, q_target, steps).q
    for q in qtraj:
        robot.q = q
        if hold_attached is not None:
            hold_attached.T = robot.fkine(robot.q)
        env.step(0.02)


def main():
    # ---- Scene setup ----
    env = swift.Swift()
    env.launch(realtime=True)

    r = CR16()
    r.base = SE3(0.0, 0.0, 0.0)   # world origin; adjust if needed
    # Make the tool face downward for top-down picking
    # (Assumes the default tool +Z points "forward"; Rx(pi) flips it down)
    r.tool = SE3.Rx(np.pi)
    r.add_to_env(env)

    # ---- Pick target config ----
    x, y = 0.55, 0.55       # set your ground XY here
    cube_size = 0.04        # 4 cm cube
    half = cube_size / 2.0
    z_ground = 0.0
    z_obj = z_ground + half

    # Visual target: a little box on the floor
    box = sg.Cuboid(
        scale=[cube_size, cube_size, cube_size],
        pose=SE3(x, y, z_obj),
        color=[0.9, 0.4, 0.2, 1.0],
    )
    env.add(box)

    # Waypoint heights
    approach = 0.20         # height above the object for pre-grasp
    clearance = 0.015       # how close to the top face to "grasp"
    z_above = z_obj + approach
    z_grasp = z_obj + clearance

    # Optional: set a desired yaw for the tool while vertical (free to change)
    yaw = 0.0               # radians
    R_yaw = SE3.Rz(yaw)

    # ---- IK: pre-grasp (above) ----
    T_above = SE3(x, y, z_above) * R_yaw
    sol_above = ik_solve(r, T_above, qseed=r.q, vertical=True)
    if not sol_above.success:
        print("[warn] IK failed for pre-grasp; trying higher approach...")
        # try to raise approach once
        T_above = SE3(x, y, z_above + 0.10) * R_yaw
        sol_above = ik_solve(r, T_above, qseed=r.q, vertical=True)
        if not sol_above.success:
            print("[error] Could not reach pre-grasp. Adjust x,y or approach height and retry.")
            env.hold()
            return

    # ---- IK: grasp height ----
    T_grasp = SE3(x, y, z_grasp) * R_yaw
    sol_grasp = ik_solve(r, T_grasp, qseed=sol_above.q, vertical=True)
    if not sol_grasp.success:
        print("[warn] IK failed at grasp height; trying slightly higher grasp...")
        T_grasp = SE3(x, y, z_grasp + 0.01) * R_yaw
        sol_grasp = ik_solve(r, T_grasp, qseed=sol_above.q, vertical=True)
        if not sol_grasp.success:
            print("[error] Could not reach grasp height. Consider moving the object or relaxing constraints.")
            env.hold()
            return

    # ---- Animate: home → above → down(grasp) → attach → up(lift) ----
    # Start from current pose (likely r.qhome)
    go(r, env, sol_above.q, steps=100)
    go(r, env, sol_grasp.q, steps=120)

    # "Attach" object (simple sim): from now on, keep box pose = EE pose
    # You could also offset by a tool transform if your gripper shape requires it.
    box.T = r.fkine(r.q)
    attached = True

    # Lift back up with the box attached
    go(r, env, sol_above.q, steps=120, hold_attached=box if attached else None)

    print("Pick complete. Close the window to exit.")
    env.hold()


if __name__ == "__main__":
    main()
