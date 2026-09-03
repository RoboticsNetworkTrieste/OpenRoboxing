"""Thinning detected keyframes down to sparse targets, keeping the first and last punch.

`spec/intent.md` 3.2: a leg carries twice the motion, so half the detected poses stop being hard
targets and MotionBricks in-fills between the survivors instead. Which poses survive is the owner's
decision (2026-09-03): a combination's signature opening and closing strike stay recorded, interior
ones become model-improvised.

Detection is untouched — `MIN_KEYFRAME_GAP_FRAMES` still governs which turning points are *found*,
and the measured 39/48 punch-capture rate with it. These tests are about **selection**.
"""

from __future__ import annotations

import numpy as np

from openroboxing.spec.constants import MIN_TARGET_GAP_FRAMES
from openroboxing.studio import segment


def test_the_first_and_last_keyframe_always_survive():
    kept = segment.thin_targets([0, 10, 20, 30, 200], punch_frames=set())
    assert kept[0] == 0
    assert kept[-1] == 200


def test_crowded_non_punches_are_dropped():
    kept = segment.thin_targets([0, 10, 20, 30, 200], punch_frames=set())
    assert kept == [0, 200], kept


def test_well_spaced_keyframes_all_survive():
    frames = [0, MIN_TARGET_GAP_FRAMES, 2 * MIN_TARGET_GAP_FRAMES]
    assert segment.thin_targets(frames, punch_frames=set()) == frames


def test_the_first_and_last_punch_survive_however_crowded():
    kept = segment.thin_targets([0, 10, 20, 30, 200], punch_frames={10, 30})
    assert 10 in kept and 30 in kept, kept
    assert 20 not in kept, "an interior non-punch must still be dropped"


def test_a_single_punch_is_both_first_and_last():
    kept = segment.thin_targets([0, 10, 20, 200], punch_frames={10})
    assert 10 in kept


def test_punches_outrank_fill_for_the_remaining_space():
    """Both sit in the same slot; the punch must take it."""
    kept = segment.thin_targets([0, 50, 52, 200], punch_frames={52})
    assert 52 in kept and 50 not in kept, kept


def test_thinning_is_idempotent():
    once = segment.thin_targets([0, 10, 20, 30, 200], punch_frames={10, 30})
    assert segment.thin_targets(once, punch_frames={10, 30}) == once


def test_a_two_keyframe_input_is_returned_unchanged():
    """Nothing to thin: both are mandatory by definition."""
    assert segment.thin_targets([0, 10], punch_frames=set()) == [0, 10]


def test_the_result_is_sorted():
    kept = segment.thin_targets([200, 0, 30, 10, 20], punch_frames={30})
    assert kept == sorted(kept)


def _swinging_qpos(frames: int = 400) -> np.ndarray:
    """A qpos stream whose arms actually reverse, so turning points exist to detect."""
    qpos = np.zeros((frames, 36))
    qpos[:, 3] = 1.0
    t = np.arange(frames)
    qpos[:, 7] = 0.6 * np.sin(2 * np.pi * t / 60.0)
    qpos[:, 10] = 0.4 * np.sin(2 * np.pi * t / 90.0)
    return qpos


def test_keyframe_indices_reports_which_frames_were_punches():
    """Provenance is lost once a CombinationRecord is written — a Keyframe stores angles and timing,
    not how it was chosen — so thinning has to happen while this is still known."""
    indices, punches = segment.keyframe_indices_with_provenance(_swinging_qpos())
    assert isinstance(punches, set)
    assert punches.issubset({int(i) for i in indices})


def test_provenance_agrees_with_the_plain_indices():
    """The two entry points must not disagree about which frames are keyframes."""
    qpos = _swinging_qpos()
    plain = segment.keyframe_indices(qpos)
    indices, _ = segment.keyframe_indices_with_provenance(qpos)
    assert list(plain) == list(indices)
