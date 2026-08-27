"""CSV corpus ingest: units, ordering, and the invertibility invariant."""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.paths import MOTIONS_DIR
from openroboxing.runtime.conventions import G1
from openroboxing.studio import motion_import

TAKE = MOTIONS_DIR / "ib_dodge_up_R_001__A437.csv"


def test_reads_header_and_frames():
    take = motion_import.read_take(TAKE)
    assert take.joint_names == tuple(n[: -len("_dof")] for n in take.raw_joint_columns)
    assert len(take.joint_names) == G1.num_joints
    # 6 root columns + 29 joints. The `Frame` column is dropped, so this is 35, not 36 —
    # the 36 of a qpos is 3 position + 4 quaternion + 29, which `load_take` produces.
    assert take.frames.shape == (591, 35)


def test_joint_permutation_is_invertible():
    take = motion_import.read_take(TAKE)
    perm = motion_import.joint_permutation(take.joint_names)
    x = np.arange(G1.num_joints, dtype=float)
    assert np.array_equal(x[perm][motion_import.invert(perm)], x)


def test_joint_permutation_rejects_a_missing_name():
    names = ("not_a_joint",) + tuple(G1.mujoco_joint_names[1:])
    with pytest.raises(motion_import.MotionImportError, match="not_a_joint"):
        motion_import.joint_permutation(names)


def test_joint_permutation_rejects_a_duplicate():
    names = (G1.mujoco_joint_names[0],) + tuple(G1.mujoco_joint_names[:-1])
    with pytest.raises(motion_import.MotionImportError):
        motion_import.joint_permutation(names)
