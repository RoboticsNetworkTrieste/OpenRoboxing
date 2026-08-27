"""Segmentation: salient speed, keyframe selection, and combination runs."""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.paths import MOTIONS_DIR
from openroboxing.spec.constants import (
    MAX_LEG_FRAMES,
    MAX_TOKENS,
    MIN_KEYFRAME_GAP_FRAMES,
    MIN_TOKENS,
    NUM_FRAMES_PER_TOKEN,
    QPOS_DIM,
)
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


def test_leg_tokens_are_within_the_planner_bounds():
    tokens = segment.leg_tokens([24, 40, 64, 30])
    assert all(MIN_TOKENS <= n <= MAX_TOKENS for n in tokens)


def test_leg_tokens_hold_total_duration_within_one_token():
    gaps = [26, 27, 26, 27, 26, 27, 26]  # each 6.5 tokens: independent rounding drifts
    tokens = segment.leg_tokens(gaps)
    error_frames = abs(sum(tokens) * NUM_FRAMES_PER_TOKEN - sum(gaps))
    assert error_frames <= NUM_FRAMES_PER_TOKEN


def test_leg_tokens_rejects_a_gap_below_the_minimum():
    with pytest.raises(segment.SegmentError, match="shorter than"):
        segment.leg_tokens([10])


def test_leg_tokens_rejects_a_gap_above_the_maximum():
    with pytest.raises(segment.SegmentError, match="longer than"):
        segment.leg_tokens([MAX_LEG_FRAMES + 1])


def test_every_take_tokenises():
    """Densification's payoff: every recorded gap tokenises with no special cases."""
    import itertools

    for path in sorted(MOTIONS_DIR.glob("*.csv")):
        qpos = motion_import.load_take(path)
        for run in segment.combination_runs(segment.keyframe_indices(qpos)):
            gaps = [b - a for a, b in itertools.pairwise(run)]
            assert len(segment.leg_tokens(gaps)) == len(gaps)


def test_whole_library_duration_error_stays_within_one_token():
    """The owner's requirement, measured across every combination rather than asserted once."""
    import itertools

    for path in sorted(MOTIONS_DIR.glob("*.csv")):
        qpos = motion_import.load_take(path)
        for run in segment.combination_runs(segment.keyframe_indices(qpos)):
            gaps = [b - a for a, b in itertools.pairwise(run)]
            planned = sum(segment.leg_tokens(gaps)) * NUM_FRAMES_PER_TOKEN
            assert abs(planned - sum(gaps)) <= NUM_FRAMES_PER_TOKEN, path.name
