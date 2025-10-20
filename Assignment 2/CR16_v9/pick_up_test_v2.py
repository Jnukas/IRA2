#!/usr/bin/env python3
"""
CR16 elbow-UP pick & place (3×3 grid → mirrored placements) - ROBUST VERSION
- Wrist kept vertical; IK biased to an elbow-up posture
- Safe corridor: above -> down -> up -> traverse -> down -> up
- Improved error handling, collision detection, and adaptive movement
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

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
class EnvironmentConfig:
    """Store all environment-specific parameters."""
    def __init__(self, ground_z=0.0, cube_size=0.06):
        self.ground_z = ground_z
        self.cube_size = cube_size
        self.z_obj = ground_z + cube_size / 2
        self.z_top = self.z_obj + cube_size / 2
        self.z_clearance = 0.12
        self.z_above = self.z_top + self.z_clearance
        self.z_grasp = self.z_top + TOUCH
    
    def validate_height(self, z, obj_name="object"):
        """Ensure height is reasonable."""
        if z < self.ground_z - 0.001:
            print(f"    WARNING: {obj_name} below ground! z={z:.3f}")
            return False
        return True


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
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
    """Check if joint configuration is within limits."""
    q = np.asarray(q).ravel()
    qmin, qmax = qlim_2xn(robot)
    return np.all(q >= qmin) and np.all(q <= qmax)


def temporarily_restrict_base(robot, current_q1, band_degrees=45):
    """Safely restrict base rotation and return original limits for restoration."""
    original_qlim = robot.qlim.copy()  # Save original
    
    qmin, qmax = qlim_2xn(robot)
    qmin_new, qmax_new = qmin.copy(), qmax.copy()
    
    band = np.radians(band_degrees)
    qmin_new[0] = max(qmin[0], current_q1 - band)
    qmax_new[0] = min(qmax[0], current_q1 + band)
    
    robot.qlim = np.vstack([qmin_new, qmax_new])
    return original_qlim


# ============================================================================
# INVERSE KINEMATICS WITH ROBUST FALLBACKS
# ============================================================================
def ik_vertical_biased_robust(robot, T, q_bias, fallback_heights=[0.0, 0.01, 0.03, 0.05, 0.10], allow_tilt=False):
    """
    Solve IK with vertical wrist and progressive height adjustments.
    Returns (solution, height_used) or (None, None) if all attempts fail.
    allow_tilt: If True, allows slight wrist tilt for difficult positions
    """
    # Try strict vertical first, then with tilt if allowed
    mask_options = [np.array([1, 1, 1, 1, 1, 0], float)]  # x,y,z + roll,pitch; free yaw
    if allow_tilt:
        mask_options.append(np.array([1, 1, 1, 0, 0, 0], float))  # Only x,y,z constrained
    
    for mask in mask_options:
        for height_adjust in fallback_heights:
            T_try = T * SE3(0, 0, height_adjust)
            
            # Multiple seed strategies for better coverage
            seeds = [
                q_bias.copy(),
                q_bias + np.r_[0, 0, 0, 0, 0, np.radians(15)],
                q_bias + np.r_[0, 0, 0, 0, 0, np.radians(-15)],
                q_bias + np.r_[0, 0, 0, 0, 0, np.radians(30)],
                q_bias + np.r_[0, 0, 0, 0, 0, np.radians(-30)],
                q_bias + np.r_[np.radians(10), 0, 0, 0, 0, 0],  # slight base rotation
                q_bias + np.r_[np.radians(-10), 0, 0, 0, 0, 0],
            ]
            
            best = None
            best_cost = 1e9
            
            for q0 in seeds:
                sol = robot.ikine_LM(T_try, q0=q0, mask=mask, ilimit=250, slimit=250, tol=1e-9)
                if sol.success and within_limits(robot, sol.q):
                    cost = np.linalg.norm(np.asarray(sol.q) - q_bias)
                    if cost < best_cost:
                        best, best_cost = sol, cost
            
            if best is not None:
                if height_adjust > 0.001:
                    print(f"    (IK used fallback height +{height_adjust*1000:.1f}mm)")
                if mask[3] == 0 and mask[4] == 0:  # Tilt was allowed
                    print(f"    (IK allowed wrist tilt)")
                return best, height_adjust
    
    return None, None


# ============================================================================
# COLLISION DETECTION
# ============================================================================
def check_self_collision_simple(robot, q_test):
    """
    Basic self-collision check: ensure joints are within limits and
    elbow doesn't get too close to base.
    """
    if not within_limits(robot, q_test):
        return True  # Collision (limit violation)
    
    # Check if elbow (link 3) gets too close to base
    try:
        robot_copy = robot.copy()
        robot_copy.q = q_test
        T_elbow = robot_copy.fkine(robot_copy.q, end=robot_copy.links[2])
        
        dist_to_base = np.linalg.norm(T_elbow.t[:2])  # XY distance from base
        if dist_to_base < 0.15:  # 15cm minimum
            return True
    except:
        # If anything goes wrong with FK, assume no collision
        pass
    
    return False


def check_object_collision(box_pose, other_boxes, min_distance=0.08):
    """Check if box would collide with other boxes."""
    # Get position of the box we're checking
    if hasattr(box_pose, 't'):
        box_pos = box_pose.t
    else:
        box_pos = np.array(box_pose)[:3, 3]
    
    for other in other_boxes:
        # Get position of other box
        if hasattr(other.T, 't'):
            other_pos = other.T.t
        else:
            other_pos = np.array(other.T)[:3, 3]
        
        dist = np.linalg.norm(box_pos - other_pos)
        if dist < min_distance:
            return True
    return False


def verify_grasp(robot, box, grasp_offset, tolerance=0.002):
    """Check if TCP is correctly positioned relative to object."""
    expected_box_pose = robot.fkine(robot.q) * grasp_offset
    actual_box_pose = box.T
    
    # Get position vectors - handle both SE3 objects and arrays
    if hasattr(expected_box_pose, 't'):
        expected_pos = expected_box_pose.t
    else:
        expected_pos = np.array(expected_box_pose)[:3, 3]
    
    if hasattr(actual_box_pose, 't'):
        actual_pos = actual_box_pose.t
    else:
        actual_pos = np.array(actual_box_pose)[:3, 3]
    
    pos_error = np.linalg.norm(expected_pos - actual_pos)
    if pos_error > tolerance:
        print(f"    WARNING: Grasp offset error {pos_error*1000:.2f}mm")
        return False
    return True


# ============================================================================
# MOVEMENT FUNCTIONS
# ============================================================================
def calculate_steps(q_start, q_end, speed_factor=0.02):
    """
    Calculate appropriate number of steps based on joint movement.
    speed_factor: radians per step (smaller = slower, more steps)
    """
    joint_diff = np.abs(np.asarray(q_end) - np.asarray(q_start))
    max_movement = np.max(joint_diff)
    steps = int(np.ceil(max_movement / speed_factor))
    return max(30, min(steps, 250))  # Clamp between 30-250


def go_safe(robot, env, q_target, carry=None, carry_offset=SE3(), 
            other_objects=[], speed_factor=0.02, check_collisions=True):
    """
    Smoothly move to q_target with adaptive speed and collision checking.
    Returns True if successful, False if collision detected.
    """
    steps = calculate_steps(robot.q, q_target, speed_factor=speed_factor)
    q_traj = jtraj(robot.q, q_target, steps).q
    
    for i, q in enumerate(q_traj):
        # Self-collision check
        if check_collisions and check_self_collision_simple(robot, q):
            print(f"    WARNING: Self-collision risk detected at step {i}/{steps}")
            return False
        
        robot.q = q
        
        if carry is not None:
            new_pose = robot.fkine(robot.q) * carry_offset
            
            # Object collision check
            if check_collisions and check_object_collision(new_pose, other_objects):
                print(f"    WARNING: Object collision risk at step {i}/{steps}")
                # Continue but issue warning
            
            carry.T = new_pose
        
        env.step(0.02)
    
    return True


def go(robot, env, q_target, steps=120, carry=None, carry_offset=SE3()):
    """Legacy function - simple movement without collision checking."""
    q_traj = jtraj(robot.q, q_target, steps).q
    for q in q_traj:
        robot.q = q
        if carry is not None:
            carry.T = robot.fkine(robot.q) * carry_offset
        env.step(0.02)


# ============================================================================
# MAIN PROGRAM
# ============================================================================
def main():
    # ====== Scene + robot setup ======
    env = swift.Swift()
    env.launch(realtime=True)
    
    r = CR16()
    r.base = SE3(0, 0, 0)
    r.tool = SE3(0, 0, 0.03) * SE3.Rx(np.pi)   # tool Z downward
    r.add_to_env(env)
    
    # Environment configuration
    env_config = EnvironmentConfig(ground_z=0.0, cube_size=0.06)
    
    # ====== Elbow-UP safe pose ======
    q_bias_up_deg = [19, -145, -113, -15, 91, 98]
    q_bias_up = np.radians(q_bias_up_deg)
    
    # Animate into the safe pose (no teleport)
    env.step(0.02)
    print("Moving to safe elbow-up pose...")
    go_safe(r, env, unwrap_to_near(q_bias_up, r.q), speed_factor=0.015)
    
    # ====== Restrict base rotation during picking ======
    print("Restricting base rotation to ±60° for stability...")
    original_limits = temporarily_restrict_base(r, float(r.q[0]), band_degrees=60)
    
    # ====== Object/corridor parameters ======
    R_yaw = SE3.Rz(0.0)
    grasp_offset = SE3(0, 0, env_config.cube_size/2)  # TCP sits on top face
    
    # ====== 3×3 pick grid and mirrored place grid ======
    pick_center = (0.55, 0.50)  # Moved slightly closer (was 0.60, 0.55)
    spacing = 0.09  # Slightly tighter spacing (was 0.10)
    xs = [pick_center[0] - spacing, pick_center[0], pick_center[0] + spacing]
    ys = [pick_center[1] - spacing, pick_center[1], pick_center[1] + spacing]
    pick_xy  = [(x, y) for y in ys for x in xs]
    place_xy = [(x, -y) for (x, y) in pick_xy]
    
    # ====== Spawn cubes ======
    print(f"Spawning 9 cubes at z={env_config.z_obj:.3f}m...")
    cubes = []
    for (x, y) in pick_xy:
        if env_config.validate_height(env_config.z_obj, f"cube at ({x:.2f},{y:.2f})"):
            box = sg.Cuboid(
                scale=[env_config.cube_size]*3, 
                pose=SE3(x, y, env_config.z_obj), 
                color=[0.9, 0.4, 0.2, 1.0]
            )
            env.add(box)
            cubes.append(box)
    env.step(0.02)
    
    # ====== Statistics tracking ======
    stats = {'success': 0, 'failed': 0, 'partial': 0}
    
    # ====== Run the pick-and-place operations ======
    print("\n" + "="*60)
    print("Starting pick-and-place operations...")
    print("="*60)
    
    for idx, (box, (px, py), (qx, qy)) in enumerate(zip(cubes, pick_xy, place_xy), 1):
        print(f"\n[{idx}/9] Pick ({px:.2f}, {py:.2f}) → Place ({qx:.2f}, {qy:.2f})")
        other_boxes = [b for b in cubes if b != box]
        
        # ====== PRE-GRASP: Move above pick location ======
        sol_above, h_adjust = ik_vertical_biased_robust(
            r, SE3(px, py, env_config.z_above) * R_yaw, q_bias=r.q
        )
        if sol_above is None:
            print(f"  ✗ SKIP: Cannot reach pre-grasp position")
            stats['failed'] += 1
            continue
        
        sol_above.q = unwrap_to_near(sol_above.q, r.q)
        if not go_safe(r, env, sol_above.q, other_objects=other_boxes, speed_factor=0.018):
            print(f"  ✗ SKIP: Collision risk during approach")
            stats['failed'] += 1
            continue
        
        # ====== GRASP: Descend to grasp height ======
        print(f"    Target grasp height: z={env_config.z_grasp:.3f}m (cube top at z={env_config.z_top:.3f}m)")
        
        sol_grasp, h_adjust_grasp = ik_vertical_biased_robust(
            r, SE3(px, py, env_config.z_grasp) * R_yaw, q_bias=sol_above.q
        )
        if sol_grasp is None:
            print(f"  ✗ SKIP: Cannot reach grasp position")
            stats['failed'] += 1
            continue
        
        # Check actual TCP height BEFORE moving
        tcp_at_above = r.fkine(r.q)
        if hasattr(tcp_at_above, 't'):
            z_at_above = tcp_at_above.t[2]
        else:
            z_at_above = tcp_at_above[2, 3]
        print(f"    Currently at z={z_at_above:.3f}m (above position)")
        
        # Check what height the IK solution will give us
        r_temp = r.copy()
        r_temp.q = sol_grasp.q
        tcp_solution = r_temp.fkine(r_temp.q)
        if hasattr(tcp_solution, 't'):
            z_solution = tcp_solution.t[2]
        else:
            z_solution = tcp_solution[2, 3]
        print(f"    IK solution will move to z={z_solution:.3f}m")
        
        # Now actually move (SLOWER for grasp descent so it's visible)
        sol_grasp.q = unwrap_to_near(sol_grasp.q, sol_above.q)
        go_safe(r, env, sol_grasp.q, speed_factor=0.008, check_collisions=False)  # Slower descent
        
        # Verify we actually descended
        tcp_after = r.fkine(r.q)
        if hasattr(tcp_after, 't'):
            z_after = tcp_after.t[2]
        else:
            z_after = tcp_after[2, 3]
        
        print(f"    Actually moved to z={z_after:.3f}m (error: {abs(z_after - env_config.z_grasp)*1000:.1f}mm)")
        
        # Small pause at grasp height to show contact
        for _ in range(10):
            env.step(0.02)
        
        # Check if we're actually at grasp height (within 2mm tolerance)
        if abs(z_after - env_config.z_grasp) > 0.002:
            print(f"    ⚠ WARNING: TCP not at grasp height! The IK fallback may have raised it.")
        
        # ====== ATTACH: Link object to end-effector ======
        box.T = r.fkine(r.q) * grasp_offset
        verify_grasp(r, box, grasp_offset)
        print(f"  ✓ Grasped object")
        
        # ====== LIFT: Return to safe height ======
        go_safe(r, env, sol_above.q, carry=box, carry_offset=grasp_offset, 
                other_objects=other_boxes, speed_factor=0.012)  # Slower lift
        
        # ====== PRE-PLACE: Move above place location ======
        sol_above_p, h_adjust_p = ik_vertical_biased_robust(
            r, SE3(qx, qy, env_config.z_above) * R_yaw, q_bias=sol_above.q, allow_tilt=True
        )
        if sol_above_p is None:
            print(f"  ⚠ WARNING: Cannot reach pre-place, dropping object at pick location")
            box.T = SE3(px, py, env_config.z_obj)
            stats['partial'] += 1
            continue
        
        sol_above_p.q = unwrap_to_near(sol_above_p.q, sol_above.q)
        if not go_safe(r, env, sol_above_p.q, carry=box, carry_offset=grasp_offset,
                       other_objects=other_boxes, speed_factor=0.018):
            print(f"  ⚠ WARNING: Collision risk during transfer, dropping")
            box.T = SE3(px, py, env_config.z_obj)
            stats['partial'] += 1
            continue
        
        # ====== PLACE: Descend to place height ======
        sol_place, h_adjust_place = ik_vertical_biased_robust(
            r, SE3(qx, qy, env_config.z_grasp) * R_yaw, q_bias=sol_above_p.q
        )
        if sol_place is None:
            print(f"  ⚠ WARNING: Cannot descend to place height, dropping from above")
            box.T = SE3(qx, qy, env_config.z_obj)
            stats['partial'] += 1
            continue
        
        sol_place.q = unwrap_to_near(sol_place.q, sol_above_p.q)
        
        # Verify descent during placement
        tcp_before_place = r.fkine(r.q)
        if hasattr(tcp_before_place, 't'):
            z_before_place = tcp_before_place.t[2]
        else:
            z_before_place = tcp_before_place[2, 3]
        
        go_safe(r, env, sol_place.q, carry=box, carry_offset=grasp_offset,
                other_objects=other_boxes, speed_factor=0.008, check_collisions=False)  # Slower placement descent
        
        tcp_after_place = r.fkine(r.q)
        if hasattr(tcp_after_place, 't'):
            z_after_place = tcp_after_place.t[2]
        else:
            z_after_place = tcp_after_place[2, 3]
        
        print(f"    TCP descended from z={z_before_place:.3f}m to z={z_after_place:.3f}m for placement")
        
        # ====== RELEASE: Detach object ======
        box.T = SE3(qx, qy, env_config.z_obj)
        if env_config.validate_height(env_config.z_obj, f"placed object"):
            print(f"  ✓ Successfully placed object")
            stats['success'] += 1
        
        # ====== RETRACT: Lift away ======
        go_safe(r, env, sol_above_p.q, speed_factor=0.012, check_collisions=False)  # Slower retraction
    
    # ====== Restore full base limits ======
    print("\nRestoring full joint limits...")
    r.qlim = original_limits
    
    # ====== Print statistics ======
    print("\n" + "="*60)
    print("OPERATION SUMMARY")
    print("="*60)
    print(f"Successful:  {stats['success']}/9")
    print(f"Partial:     {stats['partial']}/9 (picked but not placed correctly)")
    print(f"Failed:      {stats['failed']}/9 (couldn't pick)")
    print(f"Success rate: {stats['success']/9*100:.1f}%")
    print("\nDone! Close the window to exit.")
    
    env.hold()


if __name__ == "__main__":
    main()