"""Segmentation: Cartesian body signals, turning-point selection, and combination runs.

The central regression test here (``test_keyframes_capture_most_punches``) is the one that catches
the bug reported by the project owner: a segmenter that samples mid-swing frames instead of the
poses either side of them produces "not punches but statuary positioning". See
``src/openroboxing/studio/segment.py``'s module docstring for the full argument.
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.paths import MOTIONS_DIR
from openroboxing.spec.constants import (
    MAX_TARGET_LEG_FRAMES,
    MAX_TOKENS,
    MIN_KEYFRAME_GAP_FRAMES,
    MIN_TOKENS,
    NUM_FRAMES_PER_TOKEN,
    QPOS_DIM,
)
from openroboxing.studio import motion_import, segment

TAKE = MOTIONS_DIR / "ib_dodge_up_R_001__A437.csv"

#: The four takes measured in the design's D1 correction
#: (``docs/superpowers/specs/2026-08-27-motion-combinations-design.md``), with the punch counts
#: found there: prominent local maxima of ``reach`` (wrist-to-pelvis distance), prominence
#: >= `segment.REACH_TURNING_PROMINENCE_M`. Reproduced here as a fixed reference independent of
#: `segment.keyframe_indices` itself, so the capture test below cannot become circular.
EVIDENCE_TAKES: tuple[tuple[str, int], ...] = (
    ("shadow_boxing_R_001__A359", 13),
    ("shadow_boxing_R_003__A360", 11),
    ("shadow_boxing_R_002__A361", 12),
    ("ib_dodge_up_R_001__A437", 3),
)

#: How many frames a chosen keyframe may sit from a reference punch and still count as having
#: captured it. `keyframe_indices` in fact lands exactly on the punch frame when it captures one at
#: all (both are the same reach-signal local maximum), so this is slack for the comparison, not an
#: expectation that it needs it.
CAPTURE_TOLERANCE_FRAMES = 3

#: The floor the new segmenter must clear and the old one (quantile of joint-space speed) does not.
#: Measured 2026-08-28 over `EVIDENCE_TAKES`: the new rule captures 30/39 = 76.9 % comfortably
#: above this floor; the quantile-of-speed rule it replaces captures 4/39 = 10.3 %, nowhere close.
PUNCH_CAPTURE_FLOOR = 0.70


def _reference_punches(qpos: np.ndarray) -> np.ndarray:
    """Prominent local maxima of ``reach`` — full extension, the moment a punch actually lands.

    Uses `segment.body_signals` and `segment.turning_points` (shared machinery with the segmenter
    under test) but neither the greedy selection, the min-gap exclusion nor the fill step, so a
    change to how keyframes are *chosen* cannot also change what counts as a punch.
    """
    reach, _level, _shift = segment.body_signals(qpos)
    maxima = [
        frame
        for frame, prominence in segment.turning_points(reach, segment.REACH_TURNING_PROMINENCE_M)
        if reach[frame] == max(reach[max(0, frame - 1) : frame + 2])
    ]
    return np.array(sorted(maxima), dtype=int)


def test_reference_punch_counts_match_the_measured_evidence():
    """Sanity check on the fixture itself: it must reproduce the design doc's own numbers."""
    for take, expected in EVIDENCE_TAKES:
        qpos = motion_import.load_take(MOTIONS_DIR / f"{take}.csv")
        assert len(_reference_punches(qpos)) == expected, take


def test_keyframes_capture_most_punches():
    """The regression test for "not punches but statuary positioning".

    A keyframe must land at (or within `CAPTURE_TOLERANCE_FRAMES` of) most of a take's punches. The
    old segmenter (quantile of joint-space speed) samples the fast mid-swing frame between two poses
    and essentially never lands on the punch itself — measured 4/39 = 10.3 % over exactly these
    takes. The new one (turning points of `reach`, Cartesian body space) lands on the reversal
    itself by construction.
    """
    total_punches = 0
    total_captured = 0
    per_take: dict[str, tuple[int, int]] = {}
    for take, _expected in EVIDENCE_TAKES:
        qpos = motion_import.load_take(MOTIONS_DIR / f"{take}.csv")
        punches = _reference_punches(qpos)
        indices = segment.keyframe_indices(qpos)
        captured = sum(
            any(abs(int(p) - int(k)) <= CAPTURE_TOLERANCE_FRAMES for k in indices) for p in punches
        )
        per_take[take] = (captured, len(punches))
        total_punches += len(punches)
        total_captured += captured

    rate = total_captured / total_punches
    assert rate >= PUNCH_CAPTURE_FLOOR, (
        f"captured {total_captured}/{total_punches} = {rate:.1%} of punches across "
        f"{EVIDENCE_TAKES}; per-take breakdown {per_take}"
    )
    # No single take may collapse even if the aggregate clears the floor.
    for take, (captured, punches) in per_take.items():
        assert captured / punches >= 0.5, f"{take}: only {captured}/{punches} punches captured"


def test_turning_points_finds_a_local_maximum():
    signal = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    points = segment.turning_points(signal, min_prominence=0.5)
    assert points == [(2, 1.0)]


def test_turning_points_finds_a_local_minimum():
    signal = np.array([0.0, 0.0, -1.0, 0.0, 0.0])
    points = segment.turning_points(signal, min_prominence=0.5)
    assert points == [(2, 1.0)]


def test_turning_points_respects_the_prominence_floor():
    signal = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0])
    points = segment.turning_points(signal, min_prominence=0.5)
    assert [frame for frame, _ in points] == [5]


def test_turning_points_are_sorted_strongest_first():
    signal = np.array([0.0, 0.3, 0.0, 1.0, 0.0, 0.6, 0.0])
    points = segment.turning_points(signal, min_prominence=0.1)
    prominences = [prominence for _, prominence in points]
    assert prominences == sorted(prominences, reverse=True)
    assert points[0][0] == 3  # the tallest peak leads


def test_densify_inserts_at_the_strongest_turning_point_in_the_gap():
    turning = [(50, 0.9), (20, 0.1)]  # strongest first, as `turning_points` returns them
    out = segment.densify([0, 100], turning, min_gap=10, max_gap=60)
    assert out == [0, 50, 100]


def test_densify_falls_back_to_the_midpoint_with_no_turning_point_in_the_gap():
    out = segment.densify([0, 100], [], min_gap=10, max_gap=60)
    assert out == [0, 50, 100]


def test_densify_never_reuses_an_already_chosen_frame():
    import itertools

    turning = [(50, 0.9)]
    out = segment.densify([0, 50, 100], turning, min_gap=10, max_gap=40)
    assert 50 in out and out.count(50) == 1
    assert all(b - a <= 40 for a, b in itertools.pairwise(out)), out


def test_body_signals_shapes_match_the_qpos_length():
    qpos = motion_import.load_take(TAKE)
    reach, level, shift = segment.body_signals(qpos)
    assert reach.shape == level.shape == shift.shape == (len(qpos),)
    assert np.all(reach >= 0.0)


def test_keyframes_respect_the_minimum_gap():
    qpos = motion_import.load_take(TAKE)
    indices = segment.keyframe_indices(qpos)
    assert len(indices) >= 3
    assert np.all(np.diff(indices) >= MIN_KEYFRAME_GAP_FRAMES)
    assert np.all(indices >= 0) and np.all(indices < len(qpos))


def test_every_leg_is_reachable():
    """Densification's contract. Since `spec/intent.md` 3.2 a leg is no longer one plan - a long one
    runs an untargeted phase and then a landing in-between - so the bound is the maximum *leg*,
    `MAX_TARGET_LEG_FRAMES`, not the maximum plan."""
    for path in sorted(MOTIONS_DIR.glob("*.csv")):
        gaps = np.diff(segment.keyframe_indices(motion_import.load_take(path)))
        assert np.all(gaps >= MIN_KEYFRAME_GAP_FRAMES), path.name
        assert np.all(gaps <= MAX_TARGET_LEG_FRAMES), path.name


def test_keyframes_are_deterministic():
    qpos = motion_import.load_take(TAKE)
    assert np.array_equal(segment.keyframe_indices(qpos), segment.keyframe_indices(qpos))


def test_a_take_with_too_few_turning_points_raises():
    """A near-static clip has nothing to key off of and must not be padded into a fake combination."""
    qpos = np.zeros((40, QPOS_DIM))
    qpos[:, 2] = 0.8  # a plausible standing pelvis height
    qpos[:, 3] = 1.0  # identity quaternion (wxyz); everything else at the origin
    with pytest.raises(segment.SegmentError, match="turning points"):
        segment.keyframe_indices(qpos)


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
        segment.leg_tokens([MAX_TARGET_LEG_FRAMES + 1])


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
