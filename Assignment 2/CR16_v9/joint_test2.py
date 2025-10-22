from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import List, Callable, Tuple, Optional

import numpy as np
import swift
import roboticstoolbox as rtb
from spatialmath.base import tr2rpy
from spatialmath import UnitQuaternion

# ---- User knobs --------------------------------------------------------------
DT: float = 0.02          # ~50 Hz UI update and Swift step
AUTOSYNC_FROM_ROBOT = False  # if True, sliders reflect external changes to robot.q
WINDOW_TITLE = "CR16 Teach Panel"
WINDOW_GEOM = "560x580"


# ---- Utilities ---------------------------------------------------------------
def is_prismatic(link) -> bool:
    """Return True if link is prismatic across RTB versions."""
    if hasattr(link, "isprismatic"):
        return bool(link.isprismatic)
    # Fallback for older link classes
    return getattr(link, "sigma", 0) == 1


def qlim_pair(link) -> Tuple[Optional[float], Optional[float]]:
    """Return (min, max) for the link's qlim or (None, None) if absent."""
    ql = getattr(link, "qlim", None)
    if ql is None:
        return None, None
    arr = np.asarray(ql).flatten()
    if arr.size < 2:
        return None, None
    return float(arr[0]), float(arr[1])


def quat_wxyz_from_T(T: np.ndarray) -> Tuple[float, float, float, float]:
    """Robust [w,x,y,z] from 4x4 transform; works across spatialmath versions."""
    R = T[:3, :3]
    qobj = UnitQuaternion(R)
    # Prefer scalar/vector accessors (present in all versions)
    w = float(getattr(qobj, "s", 1.0))
    v = getattr(qobj, "v", np.array([0.0, 0.0, 0.0], dtype=float))
    x, y, z = [float(c) for c in np.ravel(v)]
    return w, x, y, z


class TeachPanel:
    """Offset-aware Tkinter teach panel for an RTB robot inside a Swift env."""

    def __init__(self, root: tk.Tk, robot: rtb.Robot, env: swift.Swift):
        self.root = root
        self.robot = robot
        self.env = env

        self.paused = False
        self._updating_ui = False  # guard to avoid feedback loops on slider set()
        self._closed = False       # stops the tick loop on close

        # Cache offsets once (revolute links)
        self.offsets: List[float] = [float(getattr(L, "offset", 0.0)) for L in self.robot.links]

        # ---- Window ---------------------------------------------------------
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOM)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        # ---- Pose block -----------------------------------------------------
        pose_box = ttk.LabelFrame(outer, text="End-Effector Pose", padding=(8, 6))
        pose_box.pack(fill="x", pady=(0, 8))

        self.pose_vars = {
            "X": tk.StringVar(),
            "Y": tk.StringVar(),
            "Z": tk.StringVar(),
            "Roll": tk.StringVar(),
            "Pitch": tk.StringVar(),
            "Yaw": tk.StringVar(),
            "Quat": tk.StringVar(),
        }

        grid_items = [("X", 0), ("Y", 1), ("Z", 2), ("Roll", 3), ("Pitch", 4), ("Yaw", 5)]
        for name, row in grid_items:
            ttk.Label(pose_box, text=f"{name}:").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Label(pose_box, textvariable=self.pose_vars[name]).grid(row=row, column=1, sticky="w")

        ttk.Label(pose_box, text="Quat [w, x, y, z]:").grid(row=6, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Label(pose_box, textvariable=self.pose_vars["Quat"]).grid(row=6, column=1, sticky="w")

        # ---- Controls -------------------------------------------------------
        ctrl_box = ttk.LabelFrame(outer, text="Controls", padding=(8, 6))
        ctrl_box.pack(fill="x")

        self.estop_state = tk.StringVar(value="DISENGAGED")
        ttk.Label(ctrl_box, text="E-STOP:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(ctrl_box, textvariable=self.estop_state).grid(row=0, column=1, sticky="w")
        ttk.Button(ctrl_box, text="ENGAGE", command=lambda: self.set_estop(True)).grid(row=0, column=2, padx=6)
        ttk.Button(ctrl_box, text="RESUME", command=lambda: self.set_estop(False)).grid(row=0, column=3, padx=6)

        # ---- Joint sliders --------------------------------------------------
        self.slider_rows = ttk.LabelFrame(outer, text="Joint Sliders", padding=(8, 6))
        self.slider_rows.pack(fill="both", expand=True, pady=(8, 0))

        self.vars: List[tk.DoubleVar] = []         # one tk.DoubleVar per slider
        self.value_labels: List[ttk.Label] = []    # numeric readout per joint
        self.configs: List[dict] = []              # per-joint mapping config

        for j in range(self.robot.n):
            self._add_joint_row(j)

        # Initial update
        self._refresh_pose()
        self._sync_sliders_from_robot()

        # Start periodic tick
        self._tick()

    # ---- UI construction ----------------------------------------------------
    def _add_joint_row(self, j: int) -> None:
        link = self.robot.links[j]
        is_pris = is_prismatic(link)
        off = self.offsets[j]
        ql0, ql1 = qlim_pair(link)

        row = ttk.Frame(self.slider_rows)
        row.pack(fill="x", pady=3)

        ttk.Label(row, text=f"J{j+1}").pack(side="left", padx=(0, 8))

        var = tk.DoubleVar()
        self.vars.append(var)

        # Mapping + UI ranges
        if is_pris:
            # metres
            ui_min = ql0 if ql0 is not None else -1.0
            ui_max = ql1 if ql1 is not None else 1.0
            ui_min, ui_max = sorted((ui_min, ui_max))
            ui_val = float(self.robot.q[j])
            unit = " m"

            def ui_to_internal(v: float) -> float:
                q = float(v)
                if ql0 is not None:
                    q = float(np.clip(q, ui_min, ui_max))
                return q

            def internal_to_ui(q: float) -> float:
                return float(q)
        else:
            # degrees, mechanical angle = q + offset
            if ql0 is not None and ql1 is not None:
                ui_min = float(np.rad2deg(ql0 + off))
                ui_max = float(np.rad2deg(ql1 + off))
            else:
                ui_min, ui_max = -180.0, 180.0
            ui_min, ui_max = sorted((ui_min, ui_max))
            ui_val = float(np.rad2deg(self.robot.q[j] + off))
            unit = "°"

            def ui_to_internal(v: float) -> float:
                q_int = np.deg2rad(float(v)) - off
                if ql0 is not None and ql1 is not None:
                    q_int = float(np.clip(q_int, ql0, ql1))
                return float(q_int)

            def internal_to_ui(q: float) -> float:
                return float(np.rad2deg(q + off))

        # Save config so we can resync sliders from robot state
        self.configs.append({
            "is_prismatic": is_pris,
            "ui_min": ui_min,
            "ui_max": ui_max,
            "ui_to_internal": ui_to_internal,
            "internal_to_ui": internal_to_ui,
            "unit": unit,
        })

        # ttk.Scale (no resolution knob), format value in side label instead
        scale = ttk.Scale(
            row,
            from_=ui_min, to=ui_max,
            variable=var,
            command=lambda _v, jj=j: self._on_slider(jj)
        )
        scale.pack(side="left", fill="x", expand=True)

        # numeric readout
        val_label = ttk.Label(row, width=12, anchor="e")
        val_label.pack(side="left", padx=(8, 0))
        self.value_labels.append(val_label)

        # set initial value and label
        self._updating_ui = True
        var.set(float(np.clip(ui_val, ui_min, ui_max)))
        self._updating_ui = False
        self._update_value_label(j)

    # ---- Events & actions ---------------------------------------------------
    def set_estop(self, active: bool) -> None:
        self.paused = bool(active)
        self.estop_state.set("ENGAGED" if active else "DISENGAGED")

    def _on_slider(self, j: int) -> None:
        if self.paused or self._updating_ui:
            return
        cfg = self.configs[j]
        # UI → internal q and write via the safety gate
        qnew = cfg["ui_to_internal"](self.vars[j].get())
        self._set_q_component_if_safe(j, qnew)
        self._update_value_label(j)
        self._refresh_pose()

    def _set_q_component_if_safe(self, j: int, qval: float) -> None:
        if self.paused:
            return
        q = np.asarray(self.robot.q, dtype=float)
        q[j] = qval
        self.robot.q = q.tolist()

    # ---- UI updates ---------------------------------------------------------
    def _update_value_label(self, j: int) -> None:
        cfg = self.configs[j]
        val = self.vars[j].get()
        # Pretty formatting: metres to 3 dp, degrees to 1 dp
        if cfg["unit"] == " m":
            s = f"{val:.3f}{cfg['unit']}"
        else:
            s = f"{val:.1f}{cfg['unit']}"
        self.value_labels[j].configure(text=s)

    def _refresh_pose(self) -> None:
        T = self.robot.fkine(self.robot.q)
        T = T.A if hasattr(T, "A") else np.asarray(T)
        xyz = T[:3, 3]
        rpy = tr2rpy(T, unit="deg")   # ZYX
        w, x, y, z = quat_wxyz_from_T(T)

        self.pose_vars["X"].set(f"{xyz[0]:.3f} m")
        self.pose_vars["Y"].set(f"{xyz[1]:.3f} m")
        self.pose_vars["Z"].set(f"{xyz[2]:.3f} m")
        self.pose_vars["Roll"].set(f"{rpy[0]:.2f}°")
        self.pose_vars["Pitch"].set(f"{rpy[1]:.2f}°")
        self.pose_vars["Yaw"].set(f"{rpy[2]:.2f}°")
        self.pose_vars["Quat"].set(f"{w:.3f}, {x:.3f}, {y:.3f}, {z:.3f}")

    def _sync_sliders_from_robot(self) -> None:
        """If robot.q changed elsewhere, reflect in sliders."""
        self._updating_ui = True
        try:
            for j in range(self.robot.n):
                cfg = self.configs[j]
                ui_val = cfg["internal_to_ui"](self.robot.q[j])
                ui_val = float(np.clip(ui_val, cfg["ui_min"], cfg["ui_max"]))
                self.vars[j].set(ui_val)
                self._update_value_label(j)
        finally:
            self._updating_ui = False

    # ---- Main periodic tick -------------------------------------------------
    def _tick(self) -> None:
        if self._closed:
            return
        try:
            self.env.step(DT)
        except Exception:
            # Ignore transient browser disconnects
            pass
        self._refresh_pose()
        if AUTOSYNC_FROM_ROBOT:
            self._sync_sliders_from_robot()
        self.root.after(int(DT * 1000), self._tick)

    # ---- Clean shutdown -----------------------------------------------------
    def _on_close(self) -> None:
        self._closed = True
        try:
            self.env.close()
        except Exception:
            pass
        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


# ---- Standalone entrypoint --------------------------------------------------
if __name__ == "__main__":
    # Your robot: import kept local to avoid hard dependency when imported
    from CR16Creator import CR16

    env = swift.Swift()
    env.launch(realtime=True)

    robot = CR16()
    robot.add_to_env(env)

    root = tk.Tk()
    TeachPanel(root, robot, env)
    root.mainloop()
