#!/usr/bin/env python
from __future__ import annotations

import numpy as np
from math import pi
from pathlib import Path
from roboticstoolbox.robot.Robot import Robot


class E05(Robot):
    """
    E05 robot loader that ALWAYS uses YOUR local URDF/Xacro next to this file.
    - Tries E05_robot.urdf first, then E05_robot.urdf.xacro
    - Calls self.URDF_read(str(abs_path)) so RTB doesn't fall back to rtbdata
    - Adds simple named poses (qz/qr)
    """

    def __init__(self):
        here = Path(__file__).resolve().parent

        # Prefer plain URDF (no xacro dependency); fall back to xacro if needed.
        urdf = here / "E05_robot.urdf"
        xacro = here / "E05_robot.urdf.xacro"

        if urdf.exists():
            model = urdf
        elif xacro.exists():
            # If you use Xacro, ensure `pip install xacro` is installed
            try:
                import xacro  # noqa: F401
            except Exception as e:
                raise ImportError(
                    "Found E05_robot.urdf.xacro but the 'xacro' package is missing.\n"
                    "Install it with:  pip install xacro"
                ) from e
            model = xacro
        else:
            raise FileNotFoundError(
                f"No model found next to E05bot.py.\n"
                f"Expected one of:\n  {urdf}\n  {xacro}"
            )

        # IMPORTANT: call via self.URDF_read on the ABSOLUTE PATH
        links, name, urdf_string, urdf_filepath = self.URDF_read(str(model))

        super().__init__(
            links,
            name=name or "E05",
            urdf_string=urdf_string,
            urdf_filepath=str(model),
        )

        # --- Sanity prints / guards (helpful while wiring things up)
        print(f"[E05] Loaded model from: {model}")
        print(f"[E05] Robot name: {self.name}")

        # If for any reason you're still loading a packaged model, bail out
        if "rtbdata" in str(model).lower():
            raise RuntimeError(
                "Loaded a packaged RTB model unexpectedly; expected your local E05 URDF/Xacro"
            )

        # Quick named configurations
        self.qz = np.zeros(self.n)
        self.addconfiguration("qz", self.qz)

        self.qr = np.array([0, pi/2, -pi/2, 0, 0, 0])[: self.n]
        self.addconfiguration("qr", self.qr)
