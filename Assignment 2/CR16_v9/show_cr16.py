#!/usr/bin/env python3

import swift
from spatialmath import SE3

# If your file is named CR16_cleanup.py and the class is CR16:
from CR16_cleanup import CR16
# If you renamed it to CR16.py instead, use:
# from CR16 import CR16

def main():
    # Launch Swift
    env = swift.Swift()
    env.launch(realtime=True)

    # Make robot and place it (adjust base if you like)
    r = CR16()
    r.base = SE3(0.0, 0.0, 0.0)

    # Add to scene using meshes (default style)
    r.add_to_env(env)

    # Park it here; window stays open
    env.hold()

if __name__ == "__main__":
    main()
