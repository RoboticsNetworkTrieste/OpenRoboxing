"""The warp: recorded footwork at true size, leftover travel ramped, heading from the recording."""

from __future__ import annotations

import math

import numpy as np
import pytest

from openroboxing.runtime import warp
from openroboxing.runtime.conventions import G1
from openroboxing.studio import combination_record as cr

ANGLES = {name: 0.0 for name in G1.mujoco_joint_names}


def record(offsets, headings, tokens):
    keyframes = [cr.Keyframe(dict(ANGLES), None, (0.0, 0.0), 0.0)]
    for offset, heading, token in zip(offsets, headings, tokens, strict=True):
        keyframes.append(cr.Keyframe(dict(ANGLES), token, offset, heading))
    return cr.CombinationRecord(
        name="c", library_version="v0.2",
        source=cr.CombinationSource("t", 0, 100, False), keyframes=keyframes,
    )


def test_last_leg_lands_exactly_on_the_ghost():
    """The commanded target overshoots the ghost by 1/DRIFT_GAIN, so the *generator's* undershoot
    is what actually lands the fighter on the ghost (M6-T2)."""
    from openroboxing.spec.constants import DRIFT_GAIN

    # 12 tokens is 1.6 s, so the ghost must sit inside 0.83 * 1.6 = 1.33 m of drift (raw, pre-gain).
    rec = record([(0.1, 0.0), (0.2, 0.0)], [0.0, 0.0], [6, 6])
    legs = warp.warp(rec, (1.0, 2.0), 0.0, (2.0, 2.2))
    raw_residual = (0.8, 0.2)  # ghost - anchor - recorded end, before the gain
    overshoot = 1.0 / DRIFT_GAIN - 1.0
    expected = (2.0 + raw_residual[0] * overshoot, 2.2 + raw_residual[1] * overshoot)
    assert np.allclose(legs[-1].target_position, expected)


def test_zero_recorded_travel_still_reaches_the_ghost():
    """The degenerate case proportional scaling could not express (design D4).

    With no recorded travel the whole displacement is residual, so the commanded target is the
    ghost scaled by 1/DRIFT_GAIN (M6-T2) - the generator's shortfall is what lands it on the ghost.
    """
    from openroboxing.spec.constants import DRIFT_GAIN

    rec = record([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (1.0, 0.0))
    assert np.allclose(legs[-1].target_position, (1.0 / DRIFT_GAIN, 0.0))
    assert np.allclose(legs[0].target_position, (0.5 / DRIFT_GAIN, 0.0))


def test_ghost_at_the_recorded_end_leaves_the_recording_untouched():
    rec = record([(0.1, 0.05), (0.2, 0.0)], [0.0, 0.0], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (0.2, 0.0))
    assert np.allclose(legs[0].target_position, (0.1, 0.05))


def test_footwork_keeps_its_recorded_size():
    """A 2 cm shift stays 2 cm however far away the ghost is - the whole point of D4.

    Only the residual half of ``far[0]`` is gained (M6-T2); the recorded 2 cm is untouched.
    """
    from openroboxing.spec.constants import DRIFT_GAIN

    # 32 tokens is 4.27 s, which affords 3.5 m of drift; a 2 m ghost is comfortably inside it.
    rec = record([(0.02, 0.0), (0.02, 0.0)], [0.0, 0.0], [16, 16])
    near = warp.warp(rec, (0.0, 0.0), 0.0, (0.02, 0.0))
    far = warp.warp(rec, (0.0, 0.0), 0.0, (2.0, 0.0))
    assert np.allclose(near[0].target_position, (0.02, 0.0))
    assert np.allclose(far[0].target_position, (0.02 + 0.99 / DRIFT_GAIN, 0.0))


def test_ramp_is_on_time_not_index():
    """Legs of unequal length must drift at a constant speed, not per keyframe."""
    from openroboxing.spec.constants import DRIFT_GAIN

    rec = record([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], [6, 12])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (1.8, 0.0))
    # 6 of 18 tokens elapsed at keyframe 1, so a third of the way - not half, which is where
    # a ramp on keyframe index would have put it. The residual (1.8 m) is gained (M6-T2).
    assert np.allclose(legs[0].target_position, (0.6 / DRIFT_GAIN, 0.0))
    assert np.allclose(legs[1].target_position, (1.8 / DRIFT_GAIN, 0.0))


def test_heading_comes_from_the_recording_not_the_ghost():
    rec = record([(0.1, 0.0), (0.2, 0.0)], [math.pi / 4, math.pi / 2], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), 1.0, (0.5, 0.5))
    assert legs[0].facing_angle == pytest.approx(1.0 + math.pi / 4)
    assert legs[1].facing_angle == pytest.approx(1.0 + math.pi / 2)
    assert legs[-1].target_heading == pytest.approx(1.0 + math.pi / 2)


def test_recorded_offsets_rotate_with_the_fighter():
    rec = record([(1.0, 0.0), (1.0, 0.0)], [0.0, 0.0], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), math.pi / 2, (0.0, 1.0))
    # Facing +y, a recorded +x step becomes a +y step.
    assert np.allclose(legs[0].target_position, (0.0, 1.0), atol=1e-9)


def test_movement_and_facing_are_different_signals():
    """CLAUDE.md's named trap: leaving movement at its default says 'straight ahead, always'."""
    rec = record([(0.0, 0.0), (0.0, 0.0)], [math.pi, math.pi], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (1.0, 0.0))
    assert legs[0].movement_angle == pytest.approx(0.0)  # travelling +x
    assert legs[0].facing_angle == pytest.approx(math.pi)  # looking -x


def test_a_still_leg_inherits_facing_rather_than_defaulting_to_zero():
    rec = record([(0.0, 0.0), (0.0, 0.0)], [math.pi, math.pi], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (0.0, 0.0))
    assert legs[0].movement_angle == pytest.approx(legs[0].facing_angle)


def test_an_unreachable_ghost_raises_with_the_number():
    rec = record([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], [6, 6])
    with pytest.raises(warp.WarpError, match="m/s"):
        warp.warp(rec, (0.0, 0.0), 0.0, (50.0, 0.0))


def test_legs_carry_their_recorded_pose_and_length():
    rec = record([(0.1, 0.0), (0.2, 0.0)], [0.0, 0.0], [7, 9])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (0.2, 0.0))
    assert [leg.horizon_tokens for leg in legs] == [7, 9]
    assert legs[0].joint_angles == ANGLES


def test_ghost_heading_is_the_recorded_turn():
    rec = record([(0.1, 0.0), (0.2, 0.0)], [0.3, 0.7], [6, 6])
    assert warp.ghost_heading(rec, 1.0) == pytest.approx(1.7)


def test_every_library_combination_places_at_a_reachable_ghost():
    """The real library, not a synthetic record: each one placed along its own recorded direction."""
    from openroboxing.paths import COMBINATION_DIR

    records = [cr.load(p) for p in sorted(COMBINATION_DIR.glob("*.json"))]
    assert records
    for rec in records:
        legs = warp.warp(rec, (0.0, 0.0), 0.0, rec.recorded_displacement)
        assert len(legs) == len(rec.keyframes) - 1
        assert np.allclose(legs[-1].target_position, rec.recorded_displacement, atol=1e-9)


def test_the_residual_is_divided_by_the_drift_gain():
    """The generator covers DRIFT_GAIN of what it is asked for, so ask for more (M6-T1)."""
    from openroboxing.spec.constants import DRIFT_GAIN

    rec = record([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], [16, 16])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (1.0, 0.0))
    assert legs[-1].target_position[0] == pytest.approx(1.0 / DRIFT_GAIN)


def test_the_recorded_footwork_is_not_gained():
    """Only the residual is corrected; the recording is already the right size (design D4)."""
    rec = record([(0.5, 0.0), (0.5, 0.0)], [0.0, 0.0], [16, 16])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (0.5, 0.0))
    assert legs[-1].target_position[0] == pytest.approx(0.5)


def test_the_gain_ramps_with_the_residual():
    """An intermediate keyframe carries its share of the gained residual, not of the raw one."""
    from openroboxing.spec.constants import DRIFT_GAIN

    rec = record([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], [16, 16])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (1.0, 0.0))
    assert legs[0].target_position[0] == pytest.approx(0.5 / DRIFT_GAIN)


def test_a_none_ceiling_allows_any_drift():
    """Execution re-warps from where the fighter actually is and never refuses (owner, 2026-08-28)."""
    rec = record([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (50.0, 0.0), speed_ceiling=None)
    assert legs[-1].target_position[0] > 0.0


def test_the_ceiling_is_checked_before_the_gain():
    """The gain corrects the generator, it is not distance the player asked for.

    3.4 m over 32 tokens (4.27 s) is 0.80 m/s - under the 0.83 ceiling raw, over it once gained.
    It must not raise.
    """
    rec = record([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], [16, 16])
    warp.warp(rec, (0.0, 0.0), 0.0, (3.4, 0.0))
