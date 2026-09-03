"""The keyframe is pinned in absolute time; the requested horizon shrinks toward it.

`spec/intent.md` 3.2, from the owner's framing (2026-09-03): time in MotionBricks is a continuous
array that has to be filled where there are holes, and the keyframes you put in it stay in place
while the array moves forward. The defect these tests lock down is what happens when that is *not*
honoured — asking for the leg's full length on every replan re-aims the keyframe
``REPLAN_DT * GENERATOR_HZ`` frames further out each time, so it recedes and never lands on its
boundary.

Pure arithmetic: no generator, no physics, no GPU. `CombinationRunner` is a function of ticks.

Every record here has **three keyframes (two legs)**, which is valid both under the 3-6 bounds the
shipped library uses and under the 2-3 bounds it takes on once rebuilt on sparse targets — so these
tests do not move when that constant does.
"""

from __future__ import annotations

import math

from openroboxing.runtime import sequence, warp
from openroboxing.runtime.conventions import G1
from openroboxing.spec.constants import MAX_TOKENS, MIN_TOKENS, SECONDS_PER_TOKEN, TICK_HZ
from openroboxing.studio import combination_record as cr

ANGLES = {name: 0.0 for name in G1.mujoco_joint_names}
TICKS_PER_TOKEN = SECONDS_PER_TOKEN * TICK_HZ


def record(tokens):
    keyframes = [cr.Keyframe(dict(ANGLES), None, (0.0, 0.0), 0.0)]
    for i, token in enumerate(tokens, start=1):
        keyframes.append(cr.Keyframe(dict(ANGLES), token, (0.1 * i, 0.0), 0.1 * i))
    return cr.CombinationRecord(
        name="c",
        library_version="v0.2",
        source=cr.CombinationSource("t", 0, 100, False),
        keyframes=keyframes,
    )


def runner(tokens, commit_at=0):
    rec = record(tokens)
    legs = warp.warp(rec, (0.0, 0.0), 0.0, rec.recorded_displacement)
    return sequence.CombinationRunner(rec, legs, commit_at=commit_at)


def boundaries(tokens, commit_at=0):
    """Each leg's boundary tick, the same cumulative rounding the runner uses."""
    out, cumulative = [], 0
    for token in tokens:
        cumulative += token
        out.append(commit_at + round(cumulative * TICKS_PER_TOKEN))
    return out


def test_the_horizon_shrinks_as_the_keyframe_is_approached():
    """The defect this fixes: the old code asked for the leg's full length on every replan, so the
    keyframe was re-aimed further out each time and never arrived."""
    tokens = [6, 12]
    run = runner(tokens)
    first, end = boundaries(tokens)
    horizons = [
        run.intent_for(tick).horizon_tokens
        for tick in range(first, end)
        if run.intent_for(tick).replan
    ]
    assert horizons == sorted(horizons, reverse=True), horizons
    assert horizons[0] <= MAX_TOKENS
    assert horizons[0] > horizons[-1], "a constant horizon means the keyframe is still receding"


def test_the_implied_landing_never_falls_short_of_the_boundary():
    """Pinning, stated directly: request R tokens at tick T and the plan ends at
    T + R * ticks-per-token. That sum must never land *before* the leg's boundary, and never more
    than one token after it.

    Never short is the half that matters. A plan ending before its boundary leaves the play cursor
    clamped on its last frame, and the generator's context collapses to four copies of that frame —
    see `sequence._ticks_to_tokens`. Overshoot is harmless: the next leg's replan writes over it.
    """
    tokens = [6, 12]
    run = runner(tokens)
    bounds = boundaries(tokens)
    for tick in range(bounds[-1]):
        intent = run.intent_for(tick)
        if not intent.replan or intent.pose is None:
            continue
        boundary = bounds[run.leg_index(tick)]
        landing = tick + intent.horizon_tokens * TICKS_PER_TOKEN
        assert 0 <= landing - boundary < TICKS_PER_TOKEN, (
            tick,
            intent.horizon_tokens,
            landing,
            boundary,
        )


def test_no_replan_in_the_legs_tail():
    """Below the shortest plan the model can produce there is nothing left to fill.

    The tail is ``MIN_TOKENS - 1`` tokens, not ``MIN_TOKENS``: `_ticks_to_tokens` rounds up, so a
    hole of 5.85 tokens still asks for a 6-token plan. Replanning must stop strictly below that.
    """
    tokens = [6, 12]
    run = runner(tokens)
    end = boundaries(tokens)[-1]
    tail_start = math.ceil(end - (MIN_TOKENS - 1) * TICKS_PER_TOKEN)
    for tick in range(tail_start, end):
        assert run.intent_for(tick).replan is False, tick
    # ...and it must still be replanning before the tail, or nothing is being aimed at all.
    assert run.intent_for(tail_start - 3).replan is True


def test_a_long_leg_has_no_pose_target_until_the_keyframe_is_reachable():
    """55% of the rebuilt library's legs exceed MAX_TOKENS, so this is the majority path."""
    tokens = [6, 24]
    run = runner(tokens)
    first, end = boundaries(tokens)

    opening = run.intent_for(first)
    assert opening.pose is None, "a keyframe more than one plan away must not be aimed at"
    assert opening.horizon_tokens == MAX_TOKENS

    landed = run.intent_for(math.ceil(end - MAX_TOKENS * TICKS_PER_TOKEN) + 1)
    assert landed.pose is not None
    assert landed.horizon_tokens <= MAX_TOKENS


def test_the_hold_re_aims_at_the_final_pose_at_min_tokens():
    """Unchanged in behaviour from today: past the end the runner converges on the last pose."""
    run = runner([6, 8])
    held = run.intent_for(run.end_tick + 500)
    assert held.replan is True
    assert held.pose is not None
    assert held.horizon_tokens == MIN_TOKENS


def test_every_horizon_is_a_length_the_model_can_be_asked_for():
    """`narrow_allowed_tokens` raises outside [MIN_TOKENS, MAX_TOKENS], so an out-of-range horizon
    would be a crash at generation time rather than a wrong motion."""
    run = runner([6, 24])
    for tick in range(run.end_tick + 200):
        intent = run.intent_for(tick)
        if intent.replan:
            assert MIN_TOKENS <= intent.horizon_tokens <= MAX_TOKENS, (tick, intent.horizon_tokens)


def test_each_leg_pins_its_own_keyframe():
    """Every leg's boundary is a landing, not just the combination's last."""
    tokens = [8, 10]
    run = runner(tokens)
    for boundary in boundaries(tokens):
        probe = boundary - math.ceil(MIN_TOKENS * TICKS_PER_TOKEN)
        intent = run.intent_for(probe)
        assert intent.replan and intent.pose is not None, (probe, boundary)
        landing = probe + intent.horizon_tokens * TICKS_PER_TOKEN
        assert 0 <= landing - boundary < TICKS_PER_TOKEN, (probe, landing, boundary)
