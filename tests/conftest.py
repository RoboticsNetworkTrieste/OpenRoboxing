"""Session-wide test setup.

The only thing here is the MuJoCo GL backend, and it has to be here rather than in a fixture:
MuJoCo binds its GL platform library the first time ``mujoco`` is imported, and several test modules
import it at collection time via ``pytest.importorskip``. Setting the variable afterwards silently
produces a renderer that cannot open a context. ``conftest.py`` is imported before any test module,
which makes it the only place early enough.

``setdefault``, so a developer at a real display keeps their own backend.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
