"""Segmentation: salient speed, keyframe selection, and combination runs."""

from __future__ import annotations

import numpy as np

from openroboxing.paths import MOTIONS_DIR
from openroboxing.spec.constants import MAX_LEG_FRAMES, MIN_KEYFRAME_GAP_FRAMES, QPOS_DIM
from openroboxing.studio import motion_import, segment

TAKE = MOTIONS_DIR / "ib_dodge_up_R_001__A437.csv"


def test_salient_speed_ignores_the_root():
    qpos = np.zeros((10, QPOS_DIM))
    qpos[:, 0] = np.arange(10) * 10.0  # the root sprints; no joint moves
    assert np.allclose(segment.salient_speed(qpos), 0.0)


def test_salient_speed_sees_an_arm():
    from openroboxing.runtime.conventions import G1

    elbow = 7 + G1.mujoco_joint_names.index("right_elbow_joint")
    qpos = np.zeros((10, QPOS_DIM))
    qpos[:, elbow] = np.arange(10) * 0.1
    assert np.allclose(segment.salient_speed(qpos), 0.1)


def test_keyframes_respect_the_minimum_gap():
    qpos = motion_import.load_take(TAKE)
    indices = segment.keyframe_indices(qpos)
    assert len(indices) >= 3
    assert np.all(np.diff(indices) >= MIN_KEYFRAME_GAP_FRAMES)
    assert np.all(indices >= 0) and np.all(indices < len(qpos))


def test_every_leg_is_plannable():
    """Densification's contract: no gap exceeds what MotionBricks can plan in one go."""
    for path in sorted(MOTIONS_DIR.glob("*.csv")):
        gaps = np.diff(segment.keyframe_indices(motion_import.load_take(path)))
        assert np.all(gaps >= MIN_KEYFRAME_GAP_FRAMES), path.name
        assert np.all(gaps <= MAX_LEG_FRAMES), path.name


def test_keyframes_are_deterministic():
    qpos = motion_import.load_take(TAKE)
    assert np.array_equal(segment.keyframe_indices(qpos), segment.keyframe_indices(qpos))


def test_every_take_yields_at_least_one_combination():
    for path in sorted(MOTIONS_DIR.glob("*.csv")):
        qpos = motion_import.load_take(path)
        indices = segment.keyframe_indices(qpos)
        runs = segment.combination_runs(indices)
        assert runs, f"{path.name} yielded no combination from {len(indices)} keyframes"


def test_combination_runs_are_bounded_and_ordered():
    indices = np.arange(0, 14 * MIN_KEYFRAME_GAP_FRAMES, MIN_KEYFRAME_GAP_FRAMES)
    runs = segment.combination_runs(indices)
    for run in runs:
        assert 3 <= len(run) <= 6
        assert list(run) == sorted(run)
    assert sum(len(r) for r in runs) <= len(indices)
