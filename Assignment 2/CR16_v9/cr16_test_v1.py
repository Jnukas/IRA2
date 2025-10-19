"""
CR16 Robot — Standard DH + 3D meshes (DHRobot3D)
- Teach panel (Tk sliders, degrees, auto-ranged from qlim)
- Mesh alignment verifier at home pose

Notes:
- qtest_transforms are identity on purpose. Align pivots/axes in Blender so that
  each link mesh matches its DH frame at qhome; then deltas print near zero.
"""

import os
import time
import threading
from math import pi


import numpy as np
import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
from spatialmath.base import tr2rpy

# GUI (Tk) for teach
import tkinter as tk
from tkinter import ttk


class CR16(DHRobot3D):
    def __init__(self):
        links = self._create_DH()

        # 7 visuals = base + 6 links (order must match your exported files)
        link3D_names = dict(
            link0="CR16_link_1",  # base
            link1="CR16_link_2",  # link1
            link2="CR16_link_3",
            link3="CR16_link_4",
            link4="CR16_link_5",
            link5="CR16_link_6",
            link6="CR16_link_7",  # link6 / flange
        )

        # Home pose (typical elbow-down UR style)
        self.qhome = [0.0, -pi / 2, 0.0, 0.0, 0.0, 0.0]

        # Per-visual fixed offsets: identity → assume Blender pivots already match DH
        qtest_transforms = [spb.transl(0, 0, 0) for _ in range(7)]

        link3d_dir = os.path.abspath(os.path.dirname(__file__))

        # Safety checks
        assert len(links) == 6, "Expecting 6 RevoluteDH links"
        assert len(link3D_names) == 7, "Expecting 7 visuals (base + 6 links)"
        assert len(qtest_transforms) == 7, "qtest_transforms must be length 7"

        super().__init__(
            links=links,
            link3D_names=link3D_names,
            name="CR16",
            link3d_dir=link3d_dir,
            qtest=self.qhome,
            qtest_transforms=qtest_transforms,
        )
        self.q = self.qhome

    # ---------- DH model ----------
    def _create_DH(self):
        # TODO: keep these consistent with your chosen CR16 spec
        a = [0.0, -0.512, -0.363, 0.0, 0.0, 0.0]
        d = [0.1785, 0.0, 0.0, 0.191, 0.125, 0.1084]
        alpha = [pi / 2, 0.0, 0.0, pi / 2, -pi / 2, 0.0]
        qlim = [[-2 * pi, 2 * pi] for _ in range(6)]

        links = []
        for i in range(6):
            links.append(rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i]))
        return links

    # ---------- Teach panel ----------
    def teach(self, env=None, block=True, title="CR16 Teach"):
        """
        Tk slider panel (degrees) auto-ranged from self.qlim. Updates Swift live.
        """
        created_env = False
        if env is None:
            env = swift.Swift()
            env.launch(realtime=True)
            self.add_to_env(env)
            created_env = True

        root = tk.Tk()
        root.title(title)
        root.geometry("560x360")

        def rad2deg(x):
            return float(np.degrees(x))

        def deg2rad(x):
            return float(np.radians(x))

        # Make qlim robust to either shape (n,2) or (2,n)
        qlim_arr = np.array(self.qlim, dtype=float)
        if qlim_arr.shape == (self.n, 2):
            qlim_deg = np.degrees(qlim_arr).T  # (2, n)
        else:
            qlim_deg = np.degrees(qlim_arr)    # assume already (2, n)

        sliders = []
        current_deg = [rad2deg(float(q)) for q in self.q]

        def on_change(j, val_str):
            val_deg = float(val_str)
            q_new = list(self.q)
            q_new[j] = deg2rad(val_deg)
            self.q = q_new
            env.step(0.01)

        frm = ttk.Frame(root, padding=8)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="CR16 Teach Panel (deg)").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )

        for j in range(self.n):
            mn, mx = float(qlim_deg[0, j]), float(qlim_deg[1, j])
            ttk.Label(frm, text=f"q{j + 1}").grid(
                row=j + 1, column=0, sticky="e", padx=(0, 6)
            )
            s = tk.Scale(
                frm,
                from_=mn,
                to=mx,
                resolution=1.0,
                orient=tk.HORIZONTAL,
                length=380,
                command=lambda v, jj=j: on_change(jj, v),
            )
            s.set(current_deg[j])
            s.grid(row=j + 1, column=1, sticky="we")
            sliders.append(s)
            val = ttk.Label(frm, text=f"{current_deg[j]:.1f}°")
            val.grid(row=j + 1, column=2, sticky="w")

            def bind_value_label(scale=s, label=val):
                def tick(*_):
                    label.config(text=f"{scale.get():.1f}°")
                    root.after(100, tick)

                tick()

            bind_value_label()

        btns = ttk.Frame(frm)
        btns.grid(row=self.n + 2, column=0, columnspan=3, pady=(10, 0))

        def go_home():
            self.q = list(self.qhome)
            env.step(0.01)
            for k in range(self.n):
                sliders[k].set(rad2deg(self.qhome[k]))

        def go_zero():
            self.q = [0.0] * self.n
            env.step(0.01)
            for k in range(self.n):
                sliders[k].set(0.0)

        ttk.Button(btns, text="Home", command=go_home).pack(side="left", padx=6)
        ttk.Button(btns, text="Zero", command=go_zero).pack(side="left", padx=6)
        ttk.Button(btns, text="Close", command=root.destroy).pack(side="left", padx=6)

        # Keep Swift ticking smoothly in background
        running = True

        def sim_loop():
            while running:
                env.step(0.02)

        t = threading.Thread(target=sim_loop, daemon=True)
        t.start()

        def on_close():
            nonlocal running
            running = False
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)

        if block:
            root.mainloop()
            if created_env:
                env.hold()
        else:
            threading.Thread(target=root.mainloop, daemon=True).start()

        return env

    # ---------- Mesh alignment printout ----------
    def verify_mesh_alignment(self, env=None):
        """
        Print per-link delta between expected DH frame and mesh pose at qhome.
        Call AFTER add_to_env(env). If env is provided, does a step() to refresh.

        Output:
            link0..link6: Δpos=[dx dy dz], Δrpy(deg)=[roll pitch yaw]
            All near zero ⇒ pivots/axes are aligned to DH at qhome.
        """

        # ensure known pose for a fair check
        if hasattr(self, "qhome"):
            self.q = list(self.qhome)
        if env is not None:
            env.step(0.01)

        # Build DH frames in world: index 0 = base frame, 1..6 = after each joint
        Ts_dh = []
        T = self.base.A if hasattr(self.base, "A") else np.array(self.base)
        Ts_dh.append(T)
        for i, link in enumerate(self.links):
            Ai = link.A(self.q[i])
            Ai = Ai.A if hasattr(Ai, "A") else np.array(Ai)
            T = T @ Ai
            Ts_dh.append(T)

        # Try to access the internal visual objects list (depends on DHRobot3D version)
        visuals = None
        for attr in ("_link3d", "link3d", "_sg_links", "_link3D_objs"):
            if hasattr(self, attr):
                try:
                    objs = getattr(self, attr)
                    if objs and len(objs) == 7:
                        visuals = objs
                        break
                except Exception:
                    pass

        print("\n=== Mesh alignment @ qhome ===")
        if visuals is None:
            print(
                "(!) Mesh object list not accessible. Make sure r.add_to_env(env) ran.\n"
                "    Printing DH frames only (world pose):"
            )
            for i, Td in enumerate(Ts_dh):
                p = spb.transl(Td)          # was: tr2transl(Td)
                rpy = tr2rpy(Td, unit="deg")
                print(f"Frame {i}: p={p}, rpy(deg)={rpy}")
            return

        # Δ = inv(T_DH) * T_mesh for base + 6 links
        for i in range(7):
            Td = Ts_dh[i]
            Tm = visuals[i].T.A if hasattr(visuals[i].T, "A") else np.array(visuals[i].T)
            Delta = np.linalg.inv(Td) @ Tm
            p = spb.transl(Delta)
            rpy = tr2rpy(Delta, unit="deg")
            print(f"link{i}:  Δpos={p},  Δrpy(deg)={rpy}")
        print("If all Δpos≈[0,0,0] and Δrpy≈[0,0,0], pivots/axes are aligned.\n")

    # ---------- Simple jtraj demo ----------
    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)

        self.q = self.qhome
        self.base = SE3(0.5, 0.5, 0.0)
        self.add_to_env(env)

        q_goal = [self.q[i] - pi / 3 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q
        for q in qtraj:
            self.q = q
            env.step(0.02)
        time.sleep(1.0)
        env.hold()


# ---------- Run directly ----------
if __name__ == "__main__":
    r = CR16()
    env = swift.Swift()
    env.launch(realtime=True)

    # Place robot and load meshes
    r.base = SE3(0.5, 0.5, 0.0)
    r.add_to_env(env)

    # 1) Print per-link mesh alignment deltas at qhome
    r.verify_mesh_alignment(env)

    # 2) Open teach sliders (auto-ranged from qlim); close the window to stop
    r.teach(env, block=True)
