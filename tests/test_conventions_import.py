"""gear_sonic must be importable after the OpenRoboxing/upstream split.

`conventions.py::_load_isaaclab_joint_names` imports `gear_sonic.envs.env_utils.joint_utils`, an
upstream package living at `GR00T_ROOT/gear_sonic/`. Before the extraction this worked by accident:
OpenRoboxing lived *inside* GR00T-WholeBodyControl, `gear_sonic/` sat next to the code as a sibling
top-level package, and running from the repository root put it on `sys.path` for free. The split
broke that silently — nothing put `GR00T_ROOT` on `sys.path` unless `_load_isaaclab_joint_names`
does it itself, and until it did, every match raised `ConventionError` on startup.

Kept separate from `test_conventions.py`: that file imports `conventions` at module scope, which
also builds the G1 mapping at import time and so requires real MuJoCo meshes — a different,
unrelated dependency this file does not want collection to depend on. The import here is lazy
(inside the test, after the asset check below), so this file still collects cleanly when only the
submodule's LFS pointers are present.
"""

from __future__ import annotations

import pytest

from openroboxing import paths
from openroboxing.spec.constants import NUM_JOINTS


def test_load_isaaclab_joint_names_returns_the_29_joints() -> None:
    """The regression this test exists for: gear_sonic must resolve via GR00T_ROOT, not by
    accident. Run with OPENROBOXING_GR00T_ROOT set to a checkout that actually has gear_sonic/ and
    real G1 meshes for this to do anything but skip.
    """
    if not paths.POLICY_ENCODER_ONNX.exists():
        pytest.skip("GR00T_ROOT has no real checkout here (submodule LFS pointers only)")

    from openroboxing.runtime.conventions import _load_isaaclab_joint_names

    names = _load_isaaclab_joint_names()
    assert len(names) == NUM_JOINTS
    assert len(set(names)) == NUM_JOINTS, "duplicate joint names"
    assert all(isinstance(n, str) and n for n in names)
