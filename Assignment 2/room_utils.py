# room_utils.py
import webbrowser as _wb
from spatialmath import SE3
import spatialgeometry as sg

def apply_swift_browser_fix():
    """Treat bools passed to webbrowser.get(...) as 'use default'."""
    _orig = _wb.get
    def _safe_get(using=None):
        if isinstance(using, bool):
            using = None
        return _orig(using)
    _wb.get = _safe_get

def make_room(env, 
              room_w=4.0, 
              room_d=6.0,
              floor_t=0.10, 
              wall_t=0.05, 
              wall_h=2.0,
              open_side="+Y", 
              floor_top=0.005):   # <-- new: floor_top
    # floor: center at (floor_top - floor_t/2) so the top face is at floor_top
    floor = sg.Cuboid(
        scale=[room_w, room_d, floor_t],
        pose=SE3(0, 0, floor_top - floor_t/2),
        color=[0.92, 0.92, 0.92, 1.0]
    )
    env.add(floor)

    S = open_side.upper().strip()
    leave_xp = S == "+X"; leave_xn = S == "-X"
    leave_yp = S == "+Y"; leave_yn = S == "-Y"

    walls = []
    # Walls now sit ON the floor: their centers are at floor_top + wall_h/2
    if not leave_yn:  # back
        walls.append(sg.Cuboid(
            scale=[room_w, wall_t, wall_h],
            pose=SE3(0, -room_d/2 - wall_t/2, floor_top + wall_h/2),
            color=[0.85, 0.85, 0.85, 0.9]
        ))
    if not leave_yp:  # front
        walls.append(sg.Cuboid(
            scale=[room_w, wall_t, wall_h],
            pose=SE3(0,  room_d/2 + wall_t/2, floor_top + wall_h/2),
            color=[0.85, 0.85, 0.85, 0.9]
        ))
    if not leave_xn:  # left
        walls.append(sg.Cuboid(
            scale=[wall_t, room_d, wall_h],
            pose=SE3(-room_w/2 - wall_t/2, 0, floor_top + wall_h/2),
            color=[0.85, 0.85, 0.85, 0.9]
        ))
    if not leave_xp:  # right
        walls.append(sg.Cuboid(
            scale=[wall_t, room_d, wall_h],
            pose=SE3(room_w/2 + wall_t/2, 0, floor_top + wall_h/2),
            color=[0.85, 0.85, 0.85, 0.9]
        ))
    for w in walls:
        env.add(w)

    return dict(floor=floor, walls=walls)