"""OpenRoboxing — a fighting game on GR00T-WholeBodyControl.

Nothing is exported here. The one thing this module does is choose MuJoCo's GL backend, and it has
to happen at package import: MuJoCo binds its GL platform library the first time ``mujoco`` is
imported, and ``runtime.conventions`` imports it at module scope to derive the joint mappings. Any
later assignment is silently ignored, and rendering then fails with "an OpenGL platform library has
not been loaded" — which reads like a driver problem and is not one.

``setdefault``, so a developer at a real display keeps their own backend.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
