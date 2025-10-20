#!/usr/bin/env python3
"""
CR16 elbow-UP pick & place (3×3 grid → mirrored placements).
- Wrist kept vertical; IK biased to an elbow-up posture.
- Safe corridor: above -> down -> up -> traverse -> down -> up.
"""

import numpy as np
import swift
from spatialmath import SE3
import spatialgeometry as sg
from roboticstoolbox import jtraj
from CR16_cleanup import CR16   # or: from CR16 import CR16

# Contact clearance (meters)
EPS_CLEAR = 0.003  # 3 mm hover above the top face
TOUCH = 0.0        # meters; try -0.0003 if you want a barely-visible press into the top

# ---------------- helpers ----------------
def unwrap_to_near(q_target, q_current):
    """Map each target joint to the nearest equivalent angle to current (prevents 360° spins)."""
    q_target = np.asarray(q_target, float).copy()
    q_current = np.asarray(q_current, float).copy()
    for i in range(len(q_target)):
        d = (q_target[i] - q_current[i] + np.pi) % (2*np.pi) - np.pi
        q_target[i] = q_current[i] + d
    return q_target

def qlim_2xn(robot):
    """Return qmin, qmax as (n,) each, regardless of how robot.qlim is shaped."""
    qlim = np.array(robot.qlim, float)
    if qlim.shape == (robot.n, 2):  # (n,2) -> (2,n)
        qlim = qlim.T
    return qlim[0], qlim[1]

def within_limits(robot, q):
    q = np.asarray(q).ravel()
    qmin, qmax = qlim_2xn(robot)
    return np.all(q >= qmin) and np.all(q <= qmax)

def ik_vertical_biased(robot, T, q_bias):
    """
    Solve IK with vertical wrist (x,y,z + roll,pitch; free yaw) and
    choose the solution closest to q_bias to keep the elbow-up branch.
    """
    mask = np.array([1, 1, 1, 1, 1, 0], float)
    seeds = [q_bias.copy()] + [np.array(q_bias) + np.r_[0,0,0,0,0,np.radians(d)] for d in (15,-15,30,-30)]

    best = None; best_cost = 1e9
    for q0 in seeds:
        sol = robot.ikine_LM(T, q0=q0, mask=mask, ilimit=200, slimit=200, tol=1e-9)
        if sol.success and within_limits(robot, sol.q):
            cost = np.linalg.norm(np.asarray(sol.q) - q_bias)
            if cost < best_cost:
                best, best_cost = sol, cost
    # fallback: try from current pose
    if best is None:
        best = robot.ikine_LM(T, q0=robot.q, mask=mask, ilimit=200, slimit=200, tol=1e-9)
    return best

def go(robot, env, q_target, steps=120, carry=None, carry_offset=SE3()):
    """Smoothly move to q_target; if `carry` is set, keep it attached to the EE."""
    q_traj = jtraj(robot.q, q_target, steps).q
    for q in q_traj:
        robot.q = q
        if carry is not None:
            carry.T = robot.fkine(robot.q) * carry_offset
        env.step(0.02)


# ---------------- main ----------------
def main():
    # Scene + robot
    env = swift.Swift(); env.launch(realtime=True)
    r = CR16(); 
    r.base = SE3(0, 0, 0); 
    r.tool = SE3(0, 0, 0.03) * SE3.Rx(np.pi)   # tool Z downward
    r.add_to_env(env)

    # Elbow-UP safe pose (paste from your teach panel if you like)
    q_bias_up_deg = [19, -145, -113, -15, 91, 98]
    q_bias_up = np.radians(q_bias_up_deg)

    # Animate into the safe pose (no teleport)
    env.step(0.02)
    go(r, env, unwrap_to_near(q_bias_up, r.q), steps=150)

    # Narrow base rotation during picking to avoid big pirouettes
    qmin_full, qmax_full = qlim_2xn(r)
    q1_now, band = float(r.q[0]), np.radians(45)
    qmin_t, qmax_t = qmin_full.copy(), qmax_full.copy()
    qmin_t[0] = max(qmin_t[0], q1_now - band)
    qmax_t[0] = min(qmax_t[0], q1_now + band)
    r.qlim = np.vstack([qmin_t, qmax_t])

    # Object/corridor params  (object frame is at its center)
    cube_size = 0.06
    z_obj     = cube_size / 2                 # object center when resting on floor
    z_top     = z_obj + cube_size / 2         # top face height (== cube_size)
    z_above   = z_top + 0.12                  # come in 12 cm above the top
    z_grasp   = z_top + TOUCH                 # exactly on top (or tiny push if TOUCH<0)

    R_yaw     = SE3.Rz(0.0)
    grasp_offset = SE3(0, 0, cube_size/2)     # attach so TCP sits on the TOP FACE


    # 3×3 pick grid (further out) and mirrored place grid
    pick_center = (0.60, 0.55); spacing = 0.10
    xs = [pick_center[0] - spacing, pick_center[0], pick_center[0] + spacing]
    ys = [pick_center[1] - spacing, pick_center[1], pick_center[1] + spacing]
    pick_xy  = [(x, y) for y in ys for x in xs]
    place_xy = [(x, -y) for (x, y) in pick_xy]

    # Spawn cubes
    cubes = []
    for (x, y) in pick_xy:
        box = sg.Cuboid(scale=[cube_size]*3, pose=SE3(x, y, z_obj), color=[0.9, 0.4, 0.2, 1.0])
        env.add(box); cubes.append(box)
    env.step(0.02)

    # Run the picks
    for idx, (box, (px, py), (qx, qy)) in enumerate(zip(cubes, pick_xy, place_xy), 1):
        print(f"\n[{idx}/9] pick ({px:.2f},{py:.2f}) → place ({qx:.2f},{qy:.2f})")

        # pre-grasp above pick
        sol_above = ik_vertical_biased(r, SE3(px, py, z_above) * R_yaw, q_bias=r.q)
        if not sol_above.success:
            sol_above = ik_vertical_biased(r, SE3(px, py, z_above + 0.10) * R_yaw, q_bias=r.q)
            if not sol_above.success: print("  skip: pre-grasp unreachable"); continue
        sol_above.q = unwrap_to_near(sol_above.q, r.q);      go(r, env, sol_above.q, steps=90)

        # grasp height
        sol_grasp = ik_vertical_biased(r, SE3(px, py, z_grasp) * R_yaw, q_bias=sol_above.q)
        if not sol_grasp.success:
            sol_grasp = ik_vertical_biased(r, SE3(px, py, z_grasp + 0.01) * R_yaw, q_bias=sol_above.q)
            if not sol_grasp.success: print("  skip: grasp unreachable"); continue
        sol_grasp.q = unwrap_to_near(sol_grasp.q, sol_above.q); go(r, env, sol_grasp.q, steps=100)

        # attach and lift
        box.T = r.fkine(r.q) * grasp_offset
        go(r, env, sol_above.q, steps=100, carry=box, carry_offset=grasp_offset)

        # pre-place above
        sol_above_p = ik_vertical_biased(r, SE3(qx, qy, z_above) * R_yaw, q_bias=sol_above.q)
        if not sol_above_p.success:
            sol_above_p = ik_vertical_biased(r, SE3(qx, qy, z_above + 0.10) * R_yaw, q_bias=sol_above.q)
            if not sol_above_p.success: print("  warn: pre-place unreachable; dropping."); box.T = SE3(qx, qy, z_obj); continue
        sol_above_p.q = unwrap_to_near(sol_above_p.q, sol_above.q); go(r, env, sol_above_p.q, steps=120, carry=box, carry_offset=grasp_offset)

        # place height
        sol_place = ik_vertical_biased(r, SE3(qx, qy, z_grasp) * R_yaw, q_bias=sol_above_p.q)
        if not sol_place.success:
            sol_place = ik_vertical_biased(r, SE3(qx, qy, z_grasp + 0.01) * R_yaw, q_bias=sol_above_p.q)
            if not sol_place.success: print("  warn: cannot descend to place; dropping from above"); box.T = SE3(qx, qy, z_obj); continue
        sol_place.q = unwrap_to_near(sol_place.q, sol_above_p.q); go(r, env, sol_place.q, steps=100, carry=box, carry_offset=grasp_offset)

        # release and lift away
        box.T = SE3(qx, qy, z_obj)
        go(r, env, sol_above_p.q, steps=100)

    # restore full base limits
    r.qlim = np.vstack(qlim_2xn(r))

    print("\nDone. Close the window to exit.")
    env.hold()


if __name__ == "__main__":
    main()
