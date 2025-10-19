#!/usr/bin/env python3

import time
import numpy as np
import swift
from spatialmath import SE3
from roboticstoolbox import jtraj
from CR16_cleanup import CR16  # or: from CR16 import CR16

def main():
    env = swift.Swift(); env.launch(realtime=True)

    r = CR16()
    r.base = SE3(0.0, 0.0, 0.0)
    r.add_to_env(env)

    # ---- Random pose (smooth) ----
    qlim = np.array(r.qlim, dtype=float)
    if qlim.shape == (r.n, 2):
        qmin, qmax = qlim[:, 0], qlim[:, 1]
    else:
        qmin, qmax = qlim[0, :], qlim[1, :]
    qrand = np.random.uniform(qmin, qmax)

    qtraj = jtraj(r.q, qrand, 100).q
    for q in qtraj:
        r.q = q
        env.step(0.02)

    env.hold()

if __name__ == "__main__":
    main()
