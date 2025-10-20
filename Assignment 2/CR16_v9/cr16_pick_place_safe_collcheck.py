#!/usr/bin/env python3
"""
CR16 elbow-UP pick & place (3×3 grid → mirrored placements) — with Lab5 Q2/Q3 safety.

- Vertical-wrist IK (free yaw) biased to elbow-up.
- Safety layer:
    • ≤ 1° per-joint sampling for every motion segment
    • Explicit geometric collisions: link-segment vs cuboid faces (triangles)
      for *all* cubes except the currently carried one (to avoid false positives while carrying).
- Corridor: high aisle y=0 @ z_above (no dense IK stepping).
"""

import numpy as np
import swift
from spatialmath import SE3
import spatialgeometry as sg
from CR16_cleanup import CR16   # or: from CR16 import CR16

# ---------------- helpers: joint limits & IK ----------------
def unwrap_to_near(q_target, q_current):
    q_target = np.asarray(q_target, float).copy()
    q_current = np.asarray(q_current, float).copy()
    for i in range(len(q_target)):
        d = (q_target[i] - q_current[i] + np.pi) % (2*np.pi) - np.pi
        q_target[i] = q_current[i] + d
    return q_target

def qlim_2xn(robot):
    qlim = np.array(robot.qlim, float)
    if qlim.shape == (robot.n, 2):
        qlim = qlim.T
    return qlim[0], qlim[1]

def within_limits(robot, q):
    q = np.asarray(q).ravel()
    qmin, qmax = qlim_2xn(robot)
    return np.all(q >= qmin) and np.all(q <= qmax)

def ik_vertical_biased(robot, T, q_bias):
    mask = np.array([1, 1, 1, 1, 1, 0], float)   # free yaw
    seeds = [q_bias.copy()] + [np.array(q_bias) + np.r_[0,0,0,0,0,np.radians(d)] for d in (15,-15,30,-30)]
    best = None; best_cost = 1e9
    for q0 in seeds:
        sol = robot.ikine_LM(T, q0=q0, mask=mask, ilimit=120, slimit=120, tol=1e-8)
        if sol.success and within_limits(robot, sol.q):
            cost = np.linalg.norm(np.asarray(sol.q) - q_bias)
            if cost < best_cost:
                best, best_cost = sol, cost
    if best is None:
        best = robot.ikine_LM(T, q0=robot.q, mask=mask, ilimit=120, slimit=120, tol=1e-8)
    return best

# ---------------- Lab5 Q2/Q3: ≤1° sampling + geometric collisions ----------------
def fine_interpolation(q1, q2, max_step_rad=np.deg2rad(1.0)):
    q1 = np.asarray(q1, float).ravel()
    q2 = np.asarray(q2, float).ravel()
    dq = np.abs(q2 - q1)
    steps = int(np.ceil(float(np.max(dq) / max_step_rad)))
    steps = max(steps, 1)
    return [q1 + (k/steps) * (q2 - q1) for k in range(1, steps+1)]

def _joint_positions(robot, q):
    T = robot.base
    pts = [T.t]
    for i, link in enumerate(robot.links):
        T = T * link.A(float(q[i]))
        pts.append(T.t)
    return pts

def _segment_triangle_intersect(p, q, tri, eps=2e-6):
    p = np.asarray(p,float); q = np.asarray(q,float)
    v0, v1, v2 = np.asarray(tri,float)
    u = v1 - v0; v = v2 - v0
    n = np.cross(u, v)
    n2 = float(np.dot(n, n))
    if n2 < eps**2:
        return False
    dir = q - p
    denom = float(np.dot(n, dir))
    if abs(denom) < eps:
        return False
    t = float(np.dot(n, v0 - p) / denom)
    if t < -eps or t > 1.0 + eps:
        return False
    I = p + t * dir
    # barycentric
    w = I - v0
    uu = float(np.dot(u,u)); vv = float(np.dot(v,v)); uv = float(np.dot(u,v))
    wu = float(np.dot(w,u)); wv = float(np.dot(w,v))
    D = uv*uv - uu*vv
    if abs(D) < eps:
        return False
    s = (uv*wv - vv*wu) / D
    t2 = (uv*wu - uu*wv) / D
    return (s >= -eps) and (t2 >= -eps) and (s + t2 <= 1.0 + eps)

# ---- Cuboid geometry (robust to .T being SE3 or ndarray) ----
def _as_T4x4(obj):
    T = getattr(obj, "T", None)
    if T is None:
        T = getattr(obj, "pose", None)
    if T is None:
        raise AttributeError("Cuboid has neither .T nor .pose")
    M = T.A if hasattr(T, "A") else np.asarray(T, dtype=float)
    if M.shape != (4, 4):
        raise TypeError(f"Pose must be 4x4, got shape {M.shape}")
    return M

def _cuboid_vertices_world(cuboid):
    sx, sy, sz = map(float, cuboid.scale)
    Vloc = np.array([
        [-sx/2,-sy/2,-sz/2],[ -sx/2,-sy/2,+sz/2],
        [-sx/2,+sy/2,-sz/2],[ -sx/2,+sy/2,+sz/2],
        [+sx/2,-sy/2,-sz/2],[ +sx/2,-sy/2,+sz/2],
        [+sx/2,+sy/2,-sz/2],[ +sx/2,+sy/2,+sz/2],
    ], dtype=float)
    TW = _as_T4x4(cuboid)
    VW = (TW @ np.c_[Vloc, np.ones(8)].T).T[:, :3]
    return VW

def _cuboid_triangles(cuboid):
    V = _cuboid_vertices_world(cuboid)
    faces = [
        (0,1,3),(0,3,2), (4,5,7),(4,7,6),
        (0,1,5),(0,5,4), (2,3,7),(2,7,6),
        (0,2,6),(0,6,4), (1,3,7),(1,7,5),
    ]
    return [V[list(f)] for f in faces]

def _cuboid_aabb(cuboid):
    V = _cuboid_vertices_world(cuboid)
    return V.min(axis=0), V.max(axis=0)

def build_obstacles(cuboids):
    return {obj: {"tris": _cuboid_triangles(obj), "aabb": _cuboid_aabb(obj)} for obj in cuboids}

def _aabb_overlap(a_min, a_max, b_min, b_max):
    return (a_min[0] <= b_max[0] and a_max[0] >= b_min[0] and
            a_min[1] <= b_max[1] and a_max[1] >= b_min[1] and
            a_min[2] <= b_max[2] and a_max[2] >= b_min[2])

def path_collides(robot, q_path, obstacles, ignore_obj=None, eps=2e-6, inflate=2e-3):
    for q in q_path:
        pts = _joint_positions(robot, q)
        for i in range(len(pts)-1):
            p, r = pts[i], pts[i+1]
            seg_min = np.minimum(p, r)
            seg_max = np.maximum(p, r)
            for obj, rec in obstacles.items():
                if obj is ignore_obj:
                    continue
                bmin, bmax = rec["aabb"]
                bmin_i = bmin - inflate; bmax_i = bmax + inflate
                if not _aabb_overlap(seg_min, seg_max, bmin_i, bmax_i):
                    continue
                for tri in rec["tris"]:
                    if _segment_triangle_intersect(p, r, tri, eps=eps):
                        return True
    return False

def go_safe(robot, env, q_target, obstacles, carry=None, carry_offset=SE3(),
            max_step_deg=1.0, dt=0.02, ignore_obj=None):
    q_start  = np.asarray(robot.q, float).ravel()
    q_target = np.asarray(q_target, float).ravel()
    q_path   = fine_interpolation(q_start, q_target, max_step_rad=np.deg2rad(max_step_deg))
    if path_collides(robot, q_path, obstacles, ignore_obj=ignore_obj):
        print("  ABORT: predicted collision on the way to target — segment skipped.")
        return False
    for q in q_path:
        robot.q = q
        if carry is not None:
            carry.T = robot.fkine(robot.q) * carry_offset
        env.step(dt)
    return True

# ----- high-aisle corridor (no dense IK) -----
def go_corridor_to_above(robot, env, target_xy, z_above, yaw_SE3, obstacles,
                         ignore_obj=None, max_step_deg=1.0):
    tx, ty = target_xy
    cx, cy, cz = robot.fkine(robot.q).t
    waypoints = [
        SE3(float(cx), 0.0, z_above) * yaw_SE3,
        SE3(float(tx), 0.0, z_above) * yaw_SE3,
        SE3(float(tx), float(ty), z_above) * yaw_SE3,
    ]
    for T in waypoints:
        sol = ik_vertical_biased(robot, T, q_bias=robot.q)
        if not sol.success:
            return False
        qn = unwrap_to_near(sol.q, robot.q)
        if not go_safe(robot, env, qn, obstacles, ignore_obj=ignore_obj, max_step_deg=max_step_deg):
            return False
    return True

# ----- choose TCP translation sign automatically so TCP plane matches fingertip -----
def calibrate_tool_sign(robot, R_yaw, sample_xyz, z_grasp, cube_size, q_bias):
    candidates = [SE3.Rx(np.pi) * SE3(0,0, +cube_size/2),
                  SE3.Rx(np.pi) * SE3(0,0, -cube_size/2)]
    best = None; best_err = 1e9
    for tool in candidates:
        robot.tool = tool
        sol = ik_vertical_biased(robot, SE3(*sample_xyz) * R_yaw, q_bias=q_bias)
        if not sol.success:
            continue
        z = float(robot.fkine(sol.q).t[2])
        err = abs(z - z_grasp)
        if err < best_err:
            best, best_err = tool, err
    robot.tool = best if best is not None else candidates[0]

# ---------------- main ----------------
def main():
    # Scene + robot
    env = swift.Swift(); env.launch(realtime=True)
    r = CR16(); r.base = SE3(0, 0, 0)
    r.add_to_env(env)
    env.step(0.02)

    # Elbow-UP safe pose
    q_bias_up_deg = [19, -145, -113, -15, 91, 98]
    q_bias_up = np.radians(q_bias_up_deg)

    # ---- cubes & approach heights ----
    cube_size = 0.04
    z_obj     = cube_size / 2
    z_top     = z_obj + cube_size / 2
    touch_pad = 0.0015            # 1.5 mm to be extra safe
    z_above   = z_top + 0.35
    z_grasp   = z_top + touch_pad
    R_yaw     = SE3.Rz(0.0)

    # 3×3 pick grid and mirrored place grid
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

    # Obstacles dict
    obstacles = build_obstacles(cubes)

    # Auto-calibrate tool TCP plane (+/- half-cube)
    sample_px, sample_py = pick_xy[0]
    calibrate_tool_sign(r, R_yaw, (sample_px, sample_py, z_grasp), z_grasp, cube_size, q_bias_up)

    # Start from a known high, clear pose
    go_safe(r, env, unwrap_to_near(q_bias_up, r.q), obstacles, max_step_deg=1.0, dt=0.02)

    # Narrow base rotation during picking to avoid pirouettes
    qmin_full, qmax_full = qlim_2xn(r)
    q1_now, band = float(r.q[0]), np.radians(45)
    qmin_t, qmax_t = qmin_full.copy(), qmax_full.copy()
    qmin_t[0] = max(qmin_t[0], q1_now - band)
    qmax_t[0] = min(qmax_t[0], q1_now + band)
    r.qlim = np.vstack([qmin_t, qmax_t])

    # Run the picks
    for idx, (box, (px, py), (qx, qy)) in enumerate(zip(cubes, pick_xy, place_xy), 1):
        print(f"\n[{idx}/9] pick ({px:.2f},{py:.2f}) → place ({qx:.2f},{qy:.2f})")

        # Always return to a high, safe joint pose first
        if not go_safe(r, env, unwrap_to_near(q_bias_up, r.q), obstacles, max_step_deg=1.0):
            print("  skip: cannot reach safe pose"); continue

        # -- to above pick: high aisle corridor (do NOT ignore target here)
        if not go_corridor_to_above(r, env, (px, py), z_above, R_yaw, obstacles,
                                    ignore_obj=None, max_step_deg=1.0):
            print("  skip: blocked en route to pre-grasp"); continue

        # -- descend to grasp height (check vs target)
        sol_grasp = ik_vertical_biased(r, SE3(px, py, z_grasp) * R_yaw, q_bias=r.q)
        if not sol_grasp.success:
            sol_grasp = ik_vertical_biased(r, SE3(px, py, z_grasp + 0.003) * R_yaw, q_bias=r.q)
            if not sol_grasp.success:
                print("  skip: grasp surface unreachable"); continue
        sol_grasp.q = unwrap_to_near(sol_grasp.q, r.q)
        if not go_safe(r, env, sol_grasp.q, obstacles, ignore_obj=None, max_step_deg=0.5):
            print("  skip: blocked when descending to grasp"); continue

        # -- attach (dynamic offset) and lift back to above (ignore carried cube during checks)
        T_ee = r.fkine(r.q)
        carry_offset = T_ee.inv() * box.T
        box.T = T_ee * carry_offset

        sol_back_above = ik_vertical_biased(r, SE3(px, py, z_above) * R_yaw, q_bias=r.q)
        if sol_back_above.success:
            sol_back_above.q = unwrap_to_near(sol_back_above.q, r.q)
            if not go_safe(r, env, sol_back_above.q, obstacles,
                           carry=box, carry_offset=carry_offset,
                           ignore_obj=box, max_step_deg=0.5):
                print("  warn: blocked when lifting; dropping in place")
                box.T = SE3(px, py, z_obj); continue
        else:
            print("  warn: cannot lift to above; dropping in place")
            box.T = SE3(px, py, z_obj); continue

        # -- traverse to above place (carry & ignore that cube)
        if not go_corridor_to_above(r, env, (qx, qy), z_above, R_yaw, obstacles,
                                    ignore_obj=box, max_step_deg=1.0):
            print("  warn: blocked on traverse; dropping mid-air")
            box.T = SE3(r.fkine(r.q).t[0], r.fkine(r.q).t[1], z_obj); continue

        # -- descend to place (carry & ignore)
        sol_place = ik_vertical_biased(r, SE3(qx, qy, z_grasp) * R_yaw, q_bias=r.q)
        if not sol_place.success:
            sol_place = ik_vertical_biased(r, SE3(qx, qy, z_grasp + 0.003) * R_yaw, q_bias=r.q)
            if not sol_place.success:
                print("  warn: cannot descend to place; dropping from above")
                box.T = SE3(qx, qy, z_obj); continue
        sol_place.q = unwrap_to_near(sol_place.q, r.q)
        if not go_safe(r, env, sol_place.q, obstacles,
                       carry=box, carry_offset=carry_offset,
                       ignore_obj=box, max_step_deg=0.5):
            print("  warn: blocked when descending to place; dropping from above")
            box.T = SE3(qx, qy, z_obj); continue

        # -- release and retreat
        box.T = SE3(qx, qy, z_obj)
        obstacles = build_obstacles(cubes)       # cube moved → rebuild
        go_safe(r, env, unwrap_to_near(q_bias_up, r.q), obstacles, max_step_deg=0.5)

    # restore full base limits
    r.qlim = np.vstack(qlim_2xn(r))
    print("\nDone. Close the window to exit.")
    env.hold()

if __name__ == "__main__":
    main()
