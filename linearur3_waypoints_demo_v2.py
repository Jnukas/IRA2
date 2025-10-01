# linearur3_waypoints_demo_v2.py
"""
Swift demo for the LinearUR3 that:
  • launches a Swift scene
  • auto-detects the rail (prismatic) joint
  • defines several safe waypoints (rail + small arm motions)
  • animates smooth joint-space motion between waypoints
  • prints helpful diagnostics (joint names, rail index, clamping warnings)

Tested with:
  roboticstoolbox-python >= 1.1
  swift-sim
  spatialmath-python
"""

import time
from typing import List, Optional

import numpy as np
import roboticstoolbox as rtb
import swift

# OPTIONAL: only needed if you want to nudge/rotate the robot base later
from spatialmath import SE3  # noqa: F401  (not used by default)


# ---------------- helpers: robust env/robot setup ---------------- #

def set_camera_pose_safe(env: swift.Swift,
                         pos=(1.8, 1.8, 1.4),
                         target=(0.0, 0.0, 0.0)) -> None:
    """
    Some Swift versions differ in API; try the common signatures.
    """
    try:
        env.set_camera_pose(list(pos), list(target))
    except TypeError:
        # Older/newer signature with keywords
        try:
            env.set_camera_pose(camera=list(pos), target=list(target))
        except Exception as e:
            print(f"[!] Could not set camera pose automatically: {e}")


def add_robot_to_env(env: swift.Swift, robot) -> None:
    """
    Course packages sometimes add a convenience method; fall back to env.add(robot).
    """
    try:
        # Some course wrappers expose this:
        robot.add_to_env(env)  # type: ignore[attr-defined]
    except Exception:
        env.add(robot)


# ---------------- helpers: joints & limits ---------------- #

def find_rail_idx(robot) -> int:
    """
    Return index of the first prismatic joint (sigma == 1).
    LinearUR3 should have exactly one (the rail).
    """
    for i, link in enumerate(robot.links):
        if getattr(link, "sigma", 0) == 1:  # 1 => prismatic, 0 => revolute
            return i
    raise RuntimeError("No prismatic (rail) joint found in this robot model.")


def show_joint_names(robot) -> None:
    try:
        print("Joint names:", list(robot.qname))
    except Exception:
        print("(No joint-name metadata available for this model.)")


def _normalize_name(s: str) -> str:
    return s.replace(" ", "").lower()


def find_joint_by_name(robot, name_candidates: List[str]) -> Optional[int]:
    """
    Try to locate a joint by (partial) name match. Returns index or None.
    """
    try:
        names = list(robot.qname)
    except Exception:
        return None

    norm_names = [_normalize_name(n) for n in names]
    cand_norm = [_normalize_name(c) for c in name_candidates]

    # Prefer exact (normalized) matches, then substring matches
    for i, n in enumerate(norm_names):
        if n in cand_norm:
            return i
    for i, n in enumerate(norm_names):
        if any(c in n for c in cand_norm):
            return i
    return None


def clamp_to_limits(q: np.ndarray, qlim: Optional[np.ndarray]) -> np.ndarray:
    """
    Clamp a joint vector to the robot's joint limits (qlim shape (2, n)).
    Prints a one-line warning if anything changed.
    """
    q = np.asarray(q).astype(float).copy()
    if qlim is None:
        return q
    q_clamped = np.clip(q, qlim[0, :], qlim[1, :])
    if not np.allclose(q, q_clamped, atol=1e-12):
        changed = np.where(np.abs(q - q_clamped) > 1e-12)[0]
        print(f" [!] Waypoint clamped at joints: {changed.tolist()}")
    return q_clamped


# ---------------- helpers: motion ---------------- #

def animate_q_sequence(q_seq: np.ndarray, robot, env, dt: float = 0.01) -> None:
    """
    Step through an M x n sequence of joint vectors and advance Swift.
    """
    for q in np.asarray(q_seq):
        robot.q = np.asarray(q).ravel()
        env.step(dt)


def move_to(robot, env, q_target: np.ndarray, steps: int = 180, dt: float = 0.01) -> None:
    """
    Smoothly move from the robot's current q to q_target using a quintic profile.
    """
    q_start = robot.q.copy().ravel()
    q_goal = np.asarray(q_target).ravel()
    traj = rtb.jtraj(q_start, q_goal, steps)  # returns Trajectory with .q
    animate_q_sequence(traj.q, robot, env, dt=dt)


def run_waypoints(robot,
                  env,
                  waypoints: List[np.ndarray],
                  steps_each: int = 240,
                  dt: float = 0.008,
                  pause_s: float = 0.5) -> None:
    """
    Visit each waypoint in order, moving with jtraj between them.
    """
    for i, q_wp in enumerate(waypoints):
        print(f"→ Moving to waypoint {i + 1}/{len(waypoints)}: {np.round(q_wp, 3)}")
        move_to(robot, env, q_wp, steps=steps_each, dt=dt)
        time.sleep(pause_s)


# ---------------- main ---------------- #

def main():
    # 1) Build robot
    try:
        from ir_support.robots import LinearUR3  # imported inside to fail fast with a clear message
    except Exception as e:
        raise SystemExit(
            "Couldn't import LinearUR3 from ir_support.robots. "
            "Activate the correct virtual environment and ensure ir_support is installed.\n"
            + str(e)
        )

    robot = LinearUR3()
    print(f"Loaded robot: {robot.name} | DoF = {robot.n}")

    # Basic shape checks (helps catch model/env mismatches early)
    try:
        assert robot.q.shape[0] == robot.n, "robot.q length doesn't match DoF"
    except Exception:
        # Some models expose q as Python list initially; standardize to zeros
        robot.q = np.zeros(robot.n)

    if robot.qlim is not None:
        assert robot.qlim.shape == (2, robot.n), "qlim must be 2 x n"
        print("Joint limits [min; max] (rad/metre):")
        print(np.round(robot.qlim, 3))

    show_joint_names(robot)

    # 2) Launch Swift
    env = swift.Swift()
    env.launch(realtime=True)
    set_camera_pose_safe(env, pos=(1.8, 1.8, 1.4), target=(0.0, 0.0, 0.0))

    # 3) Zero pose and add robot to scene
    robot.q = np.zeros(robot.n)
    add_robot_to_env(env, robot)

    # 4) Detect rail joint and prepare safe rail positions
    rail_idx = find_rail_idx(robot)
    print(f"Prismatic rail joint index: {rail_idx}")

    if robot.qlim is not None:
        rail_min, rail_max = float(robot.qlim[0, rail_idx]), float(robot.qlim[1, rail_idx])
    else:
        rail_min, rail_max = -0.5, 0.5  # conservative fallback in metres

    # Choose a few rail positions well inside the limits
    rail_mid = 0.5 * (rail_min + rail_max)
    rail_span = 0.5 * (rail_max - rail_min)  # half-span
    r0 = np.clip(rail_mid - 0.30 * rail_span, rail_min, rail_max)
    r1 = np.clip(rail_mid + 0.30 * rail_span, rail_min, rail_max)
    r2 = np.clip(rail_mid - 0.10 * rail_span, rail_min, rail_max)

    print(f"Rail range: [{rail_min:.3f}, {rail_max:.3f}] m | picks: r0={r0:.3f}, r1={r1:.3f}, r2={r2:.3f}")

    # 5) Define waypoints (rail + small arm poses)
    q0 = robot.q.copy()

    # Helper: waypoint with specific rail position
    def with_rail(q_base: np.ndarray, rail_val: float) -> np.ndarray:
        q = np.asarray(q_base).copy()
        q[rail_idx] = float(rail_val)
        return q

    # Prefer name-based indices if available; otherwise fall back to offsets
    j_shoulder = find_joint_by_name(
        robot, ["shoulder", "shoulderlift", "shoulder_lift_joint", "shoulder_lift"])
    j_elbow = find_joint_by_name(
        robot, ["elbow", "elbow_joint"])
    j_wrist1 = find_joint_by_name(
        robot, ["wrist1", "wrist_1", "wrist_1_joint"])

    if j_shoulder is None:
        j_shoulder = rail_idx + 1
    if j_elbow is None:
        j_elbow = rail_idx + 2
    if j_wrist1 is None:
        j_wrist1 = rail_idx + 4

    # Waypoint 1: move along the rail, keep arm neutral
    q1 = with_rail(q0, r1)

    # Waypoint 2: slight arm bend (shoulder & elbow) while at rail r1
    q2 = q1.copy()
    if j_elbow < robot.n:
        q2[j_shoulder] += np.deg2rad(20.0)   # +20°
        q2[j_elbow]    -= np.deg2rad(30.0)   # -30°

    # Waypoint 3: slide back near the middle and change wrist a touch
    q3 = with_rail(q2, r2)
    if j_wrist1 < robot.n:
        q3[j_wrist1] += np.deg2rad(25.0)     # +25° (slightly larger so it's obvious)

    # Waypoint 4: return to starting rail and neutral arm
    q4 = with_rail(np.zeros(robot.n), r0)

    # 6) Clamp all waypoints to limits (safety), then run
    raw_waypoints = [q1, q2, q3, q4, q0]

    waypoints = [clamp_to_limits(q, robot.qlim) for q in raw_waypoints]

    print("\nPlanned waypoints (post-clamp):")
    for i, q in enumerate(waypoints, 1):
        print(f"  WP{i}: {np.round(q, 3)}")

    # 7) Animate waypoint traversal (slower/smoother defaults)
    run_waypoints(robot, env, waypoints, steps_each=240, dt=0.008, pause_s=0.5)

    print("\nDone. Scene is now holding. Close the Swift tab/window to exit.")
    env.hold()


if __name__ == "__main__":
    main()
