"""CombinationRunner: which leg is live at a tick, and the intent it produces."""

from __future__ import annotations

import pytest

from openroboxing.runtime import sequence, warp
from openroboxing.runtime.conventions import G1
from openroboxing.spec.constants import SECONDS_PER_TOKEN, TICK_HZ
from openroboxing.studio import combination_record as cr

ANGLES = {name: 0.0 for name in G1.mujoco_joint_names}


def record(tokens):
    keyframes = [cr.Keyframe(dict(ANGLES), None, (0.0, 0.0), 0.0)]
    for i, token in enumerate(tokens, start=1):
        keyframes.append(cr.Keyframe(dict(ANGLES), token, (0.1 * i, 0.0), 0.1 * i))
    return cr.CombinationRecord(
        name="c", library_version="v0.2",
        source=cr.CombinationSource("t", 0, 100, False), keyframes=keyframes,
    )


def runner(tokens, commit_at=0):
    rec = record(tokens)
    legs = warp.warp(rec, (0.0, 0.0), 0.0, rec.recorded_displacement)
    return sequence.CombinationRunner(rec, legs, commit_at=commit_at), rec


def test_boundaries_match_the_records_duration():
    run, rec = runner([6, 8, 6])
    assert run.end_tick == rec.duration_ticks


def test_end_tick_is_offset_by_the_commit_tick():
    run, rec = runner([6, 8, 6], commit_at=500)
    assert run.end_tick == 500 + rec.duration_ticks


def test_the_first_leg_is_live_at_the_commit_tick():
    run, _ = runner([6, 8, 6])
    assert run.leg_index(0) == 0


def test_legs_advance_on_time():
    run, _ = runner([6, 8, 6])
    first = round(6 * SECONDS_PER_TOKEN * TICK_HZ)
    assert run.leg_index(first - 1) == 0
    assert run.leg_index(first) == 1


def test_the_last_leg_holds_past_the_end():
    """Holding a pose is the same target re-armed - existing runtime behaviour, unchanged."""
    run, _ = runner([6, 8, 6])
    assert run.leg_index(run.end_tick) == 2
    assert run.leg_index(run.end_tick + 10_000) == 2


def test_is_finished_flips_at_the_end_tick():
    run, _ = runner([6, 8, 6])
    assert not run.is_finished(run.end_tick - 1)
    assert run.is_finished(run.end_tick)


def test_intent_carries_the_legs_pose_target_and_the_hole_to_its_keyframe():
    """At a leg's first tick the hole *is* the leg, so the horizon equals its recorded length.

    That coincidence is the only place the two agree. Everywhere else the horizon is the distance
    still to run to a keyframe pinned at its boundary, which is what `tests/test_sequence_pinned.py`
    covers — see `spec/intent.md` 3.2.
    """
    run, _ = runner([6, 8, 6])
    intent = run.intent_for(0)
    assert intent.style == sequence.COMBINATION_CONTEXT
    assert intent.horizon_tokens == 6
    assert intent.replan is True
    assert intent.target_position is not None
    assert intent.target_heading is not None


def test_intent_pose_is_a_pose_record_the_override_can_read():
    from openroboxing.studio.pose_record import PoseRecord

    run, _ = runner([6, 8, 6])
    pose = run.intent_for(0).pose
    assert isinstance(pose, PoseRecord)
    assert set(pose.joint_angles) == set(G1.mujoco_joint_names)


def test_movement_and_facing_survive_into_the_intent():
    """CLAUDE.md's named trap - they are different signals and both must be carried."""
    run, _ = runner([6, 8, 6])
    intent = run.intent_for(0)
    leg = run.legs[0]
    assert intent.movement_angle == pytest.approx(leg.movement_angle)
    assert intent.facing_angle == pytest.approx(leg.facing_angle)


def test_a_tick_before_the_commit_raises():
    run, _ = runner([6, 8, 6], commit_at=100)
    with pytest.raises(sequence.SequenceError):
        run.leg_index(99)


def test_every_library_combination_visits_every_leg():
    """No leg is skipped by the token-to-tick rounding, across the whole library."""
    from openroboxing.paths import COMBINATION_DIR

    for path in sorted(COMBINATION_DIR.glob("*.json")):
        rec = cr.load(path)
        legs = warp.warp(rec, (0.0, 0.0), 0.0, rec.recorded_displacement)
        run = sequence.CombinationRunner(rec, legs, commit_at=0)
        seen = {run.leg_index(t) for t in range(run.end_tick)}
        assert seen == set(range(len(legs))), f"{path.name} skipped a leg: {sorted(seen)}"


def test_every_library_combination_ends_where_its_record_says():
    from openroboxing.paths import COMBINATION_DIR

    for path in sorted(COMBINATION_DIR.glob("*.json")):
        rec = cr.load(path)
        legs = warp.warp(rec, (0.0, 0.0), 0.0, rec.recorded_displacement)
        run = sequence.CombinationRunner(rec, legs, commit_at=17)
        assert run.end_tick == 17 + rec.duration_ticks


# --- the live bearing (owner, 2026-09-03: a fighter always faces its opponent) --------------------
def test_a_supplied_bearing_overrides_the_recorded_heading():
    """The reversal of design D5. Both signals the generator aims with move: the *target frame's*
    heading and the facing direction. The recorded turn no longer decides where a fighter looks."""
    run, _ = runner([6, 8, 6])
    recorded = run.legs[0].facing_angle
    intent = run.intent_for(0, facing_angle=1.25)

    assert recorded != pytest.approx(1.25), "the fixture must not accidentally record the bearing"
    assert intent.facing_angle == pytest.approx(1.25)
    assert intent.target_heading == pytest.approx(1.25)


def test_a_still_legs_travel_follows_the_supplied_bearing():
    """A leg that does not travel has no direction of its own, so it inherits the one the fighter
    is actually facing - which is now the bearing, not the recording (`warp.STILL_LEG_M`)."""
    keyframes = [cr.Keyframe(dict(ANGLES), None, (0.0, 0.0), 0.0)]
    for token in (6, 6):
        keyframes.append(cr.Keyframe(dict(ANGLES), token, (0.0, 0.0), 0.9))
    rec = cr.CombinationRecord(
        name="still", library_version="v0.2",
        source=cr.CombinationSource("t", 0, 100, False), keyframes=keyframes,
    )
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (0.0, 0.0))
    assert all(leg.is_still for leg in legs), "the fixture must stand still to mean anything"

    intent = sequence.CombinationRunner(rec, legs, commit_at=0).intent_for(0, facing_angle=1.25)
    assert intent.movement_angle == pytest.approx(1.25)


def test_without_a_bearing_the_recording_still_decides():
    """Off the bench there is no opponent - the Studio's rehearsal and the warp tools drive a lone
    fighter - so the recorded heading is what a runner falls back to, and nothing silently zeroes."""
    run, _ = runner([6, 8, 6])
    intent = run.intent_for(0)
    assert intent.facing_angle == pytest.approx(run.legs[0].facing_angle)
    assert intent.target_heading == pytest.approx(run.legs[0].target_heading)
