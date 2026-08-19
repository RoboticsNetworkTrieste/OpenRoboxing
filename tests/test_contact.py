"""M3-T3 acceptance: contact sensing and hit attribution.

Acceptance criterion from WORKPLAN.md M3-T3:
  a scripted scenario (A throws a hook, B stands still) yields exactly one attributed hit event with
  a plausible impulse; a scenario with a blocked strike attributes contact to the guarding arm, not
  the head.

Scenarios are driven kinematically — the glove is placed where a punch would put it and physics is
stepped — so the test exercises *attribution* rather than the policy's aim, and runs without a GPU.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_contact.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.runtime.arena import (
    ArenaConfig,
    build_arena,
    fighter_qpos_slice,
    reset_to_stance,
)
from openroboxing.runtime.contact import (
    DOWN_HEIGHT_FRACTION,
    STANDING_TORSO_HEIGHT_M,
    ContactError,
    ContactTracker,
    FightTrace,
    HitEvent,
    fighter_of,
    is_glove,
    region_of,
    strip_fighter,
)

pytest.importorskip("mujoco")


#: How close the fighters stand in the scripted scenarios. The G1's hand reaches 0.33 m forward of
#: its own pelvis (measured, `studio/pose_ik.py`), so at the ring's 2.4 m starting separation nothing
#: can land however well aimed. Attribution is what is under test, not distance management.
#:
#: Scanned rather than picked. Re-scanned 2026-08-08 when the arena moved to the simulation-ready
#: robot model, whose hands carry no mesh geoms of their own and so reach ~2 cm less far:
#:
#: ===== ===================== =========================
#: sep    unguarded             guarded
#: ===== ===================== =========================
#: 0.14   two events            one, on the elbow
#: 0.15   **one, on the torso** **two: torso and glove**
#: 0.16   one, on the torso     nothing intercepts
#: 0.17+  nothing touches       nothing touches
#: ===== ===================== =========================
#:
#: 0.15 m is the only value where a single body pair is touched *and* a raised guard actually gets in
#: the way, which is what makes "exactly one event" a statement about the tracker rather than about
#: the staging.
SCENARIO_SEPARATION = 0.15


@pytest.fixture(scope="module")
def ring():
    """A ring with gravity off.

    Nothing drives the fighters in these scenarios, so under gravity they collapse before a punch
    arrives and the test would measure falling over. Zero-g holds the scripted pose still; the glove
    overlapping the opponent produces real contact forces either way, which is what is being read.
    """
    import mujoco

    config = ArenaConfig()
    model = build_arena(config)
    model.opt.gravity[:] = 0.0
    return mujoco, model, config


def _set_arm(model, data, fighter: str, side: str, shoulder: float, elbow: float) -> None:
    data.qpos[model.joint(f"{fighter}_{side}_shoulder_pitch_joint").qposadr[0]] = shoulder
    data.qpos[model.joint(f"{fighter}_{side}_elbow_joint").qposadr[0]] = elbow


def _stage(mujoco, model, config, guard_blue: bool = False, ticks: int = 40):
    """Red's left glove is put on blue; blue optionally holds a high guard in the way."""
    data = mujoco.MjData(model)
    reset_to_stance(model, data, config)

    data.qpos[fighter_qpos_slice(0)][0] = -SCENARIO_SEPARATION
    data.qpos[fighter_qpos_slice(1)][0] = SCENARIO_SEPARATION
    _set_arm(model, data, "red", "left", shoulder=-1.35, elbow=-0.1)
    if guard_blue:
        for side in ("left", "right"):
            _set_arm(model, data, "blue", side, shoulder=-1.30, elbow=-0.55)
    mujoco.mj_forward(model, data)

    tracker = ContactTracker()
    for tick in range(ticks):
        for _ in range(10):
            mujoco.mj_step(model, data)
        tracker.observe(model, data, tick)
    tracker.flush()
    return tracker, data


# --- the acceptance criterion ---------------------------------------------------------------------
def test_a_thrown_punch_yields_one_attributed_event(ring) -> None:
    mujoco, model, config = ring
    tracker, _ = _stage(mujoco, model, config)

    hits = [e for e in tracker.events if e.attacker == "red"]
    assert hits, "the punch was not detected at all"
    assert len(hits) == 1, (
        f"one punch produced {len(hits)} events: "
        f"{[(h.defender_body, h.start_tick, h.end_tick) for h in hits]}"
    )

    hit = hits[0]
    assert hit.attacker == "red" and hit.defender == "blue"
    assert "wrist" in hit.attacker_body, "a hit must come from a hand"
    assert hit.duration_ticks > 1, "a punch lasts longer than a single tick"
    assert hit.peak_force_n > 0.0
    assert hit.impulse_ns > 0.0
    assert np.isfinite(hit.position).all()


def test_the_impulse_is_plausible(ring) -> None:
    """Not a number check for its own sake: an impulse off by orders of magnitude means the force
    was read from the wrong column of `mj_contactForce`."""
    mujoco, model, config = ring
    tracker, _ = _stage(mujoco, model, config)
    hit = next(e for e in tracker.events if e.attacker == "red")

    # A G1 arm is a few kg moving at a few m/s, so an impulse of order 0.01-100 N*s. Outside that
    # range the units are wrong, not the boxing.
    assert 1e-3 < hit.impulse_ns < 1e3, f"implausible impulse {hit.impulse_ns} N*s"
    assert hit.peak_force_n < 1e5, f"implausible peak force {hit.peak_force_n} N"


def test_a_blocked_strike_is_attributed_to_the_arm_not_the_head(ring) -> None:
    """The scenario the criterion names: contact lands on what actually intercepted it.

    Asserted positively — the region must *be* ``arm`` — rather than merely "not head", because
    "not head" also passes when nothing was hit at all. Note what this staging does not cover: it
    never produces a clean head hit to contrast against, so it demonstrates that a guarded punch is
    attributed to the guard, not that an unguarded one reaches the head. Producing the latter needs
    a driven fighter rather than a placed one, which is M3-T4's business.

    A partly-blocked punch also reaching the body is not a failure — it is boxing, and the criterion
    is about *where contact is attributed*, not about the guard being impenetrable.
    """
    mujoco, model, config = ring
    tracker, _ = _stage(mujoco, model, config, guard_blue=True)

    landed = [e for e in tracker.events if e.attacker == "red"]
    assert landed, "the blocked punch registered no contact at all"

    regions = {e.region for e in landed}
    assert "arm" in regions, (
        f"the guard did not intercept: {[(e.defender_body, e.region) for e in landed]}"
    )
    assert "head" not in regions, (
        f"a guarded head was still hit: {[(e.defender_body, e.region) for e in landed]}"
    )


# --- attribution ------------------------------------------------------------------------------------
def test_geoms_are_attributed_by_prefix_not_position() -> None:
    assert fighter_of("red_glove_left") == "red"
    assert fighter_of("blue_torso_link") == "blue"
    assert fighter_of("rope_0_1") is None, "ring furniture must not belong to a fighter"
    assert strip_fighter("red_glove_left") == "glove_left"
    assert strip_fighter("canvas") == "canvas"
    assert is_glove("red_glove_left") and not is_glove("red_torso_link")


@pytest.mark.parametrize(
    "body, expected",
    [
        ("head_link", "head"),
        ("red_torso_link", "body"),
        ("left_wrist_yaw_link", "arm"),
        ("right_elbow_link", "arm"),
        ("left_knee_link", "leg"),
        ("pelvis", "body"),
        ("something_unexpected", "other"),
    ],
)
def test_bodies_map_to_scoring_regions(body, expected) -> None:
    assert region_of(body) == expected


# --- episodes ---------------------------------------------------------------------------------------
class _FakeModel:
    class _Opt:
        timestep = 0.001

    opt = _Opt()


def _fake_tick(
    tracker, tick, force=100.0, attacker="red_left_wrist_yaw_link", defender="blue_head_link"
):
    """A tick with contact. Drives the tracker directly, so episodes are testable without physics."""
    tracker._accumulate(attacker, defender, force, (0.0, 0.0, 1.0), tick, 0.001)
    tracker._close_stale(tick)


def _idle_tick(tracker, tick):
    """A tick with no contact. `observe` closes stale episodes every tick, contact or not — without
    this the clock only advances when something is touching and nothing ever goes stale."""
    tracker._close_stale(tick)


def test_contiguous_contact_is_one_event() -> None:
    tracker = ContactTracker()
    for tick in range(20):
        _fake_tick(tracker, tick)
    tracker.flush()
    assert len(tracker.events) == 1
    assert tracker.events[0].duration_ticks == 20


def test_two_separated_punches_are_two_events() -> None:
    tracker = ContactTracker(gap_ticks=3)
    for tick in range(5):
        _fake_tick(tracker, tick)
    for tick in range(5, 40):
        _idle_tick(tracker, tick)
    for tick in range(40, 45):
        _fake_tick(tracker, tick)
    tracker.flush()
    assert len(tracker.events) == 2


def test_a_short_bounce_does_not_split_a_punch() -> None:
    """A glove skipping off a guard and landing again is one exchange."""
    tracker = ContactTracker(gap_ticks=3)
    for tick in range(6):
        if tick == 3:
            _idle_tick(tracker, tick)  # one tick of separation
        else:
            _fake_tick(tracker, tick)
    tracker.flush()
    assert len(tracker.events) == 1


def test_the_reported_position_is_the_hardest_moment() -> None:
    tracker = ContactTracker()
    hand, head = "red_left_wrist_yaw_link", "blue_head_link"
    tracker._accumulate(hand, head, 10.0, (0.0, 0.0, 1.0), 0, 0.001)
    tracker._accumulate(hand, head, 90.0, (1.0, 2.0, 3.0), 1, 0.001)
    tracker._accumulate(hand, head, 20.0, (9.0, 9.0, 9.0), 2, 0.001)
    tracker.flush()

    event = tracker.events[0]
    assert event.position == (1.0, 2.0, 3.0)
    assert event.peak_force_n == 90.0
    assert event.impulse_ns == pytest.approx((10.0 + 90.0 + 20.0) * 0.001)


def test_each_body_pair_gets_its_own_episode() -> None:
    tracker = ContactTracker()
    for tick in range(6):
        _fake_tick(tracker, tick, defender="blue_head_link")
        _fake_tick(tracker, tick, attacker="red_right_wrist_yaw_link", defender="blue_torso_link")
    tracker.flush()
    assert len(tracker.events) == 2
    assert {e.region for e in tracker.events} == {"head", "body"}


def test_a_negative_gap_raises() -> None:
    with pytest.raises(ContactError, match="must not be negative"):
        ContactTracker(gap_ticks=-1)


# --- what is not a hit --------------------------------------------------------------------------------
def test_only_a_glove_on_an_opponent_counts() -> None:
    """Contacts are described as ``(body name, is_glove)``; the robot's geoms are unnamed."""
    tracker = ContactTracker()
    red_glove = ("red_left_wrist_yaw_link", True)
    blue_glove = ("blue_left_wrist_yaw_link", True)
    blue_head = ("blue_head_link", False)
    red_torso = ("red_torso_link", False)
    blue_torso = ("blue_torso_link", False)
    ring = ("world", False)

    assert tracker._as_punch(red_glove, blue_head) is not None
    assert tracker._as_punch(blue_head, red_glove) is not None
    assert tracker._as_punch(red_glove, red_torso) is None, "own fighter"
    assert tracker._as_punch(red_glove, blue_glove) is None, "a parry, not a hit"
    assert tracker._as_punch(red_torso, blue_torso) is None, "a clinch, not a hit"
    assert tracker._as_punch(red_glove, ring) is None, "the ring is not a fighter"
    assert tracker._as_punch(ring, ring) is None


def test_a_fighter_touching_the_ropes_is_not_a_hit(ring) -> None:
    mujoco, model, config = ring
    data = mujoco.MjData(model)
    reset_to_stance(model, data, config)
    # Push red into the ropes.
    data.qpos[fighter_qpos_slice(0)][0] = -(config.ring_size / 2.0) + 0.15
    mujoco.mj_forward(model, data)

    tracker = ContactTracker()
    for tick in range(30):
        for _ in range(10):
            mujoco.mj_step(model, data)
        tracker.observe(model, data, tick)
    tracker.flush()
    assert not tracker.events, f"the ropes scored a hit: {tracker.events}"


# --- the trace ------------------------------------------------------------------------------------------
def test_the_trace_records_separation_and_ring_position(ring) -> None:
    mujoco, model, config = ring
    data = mujoco.MjData(model)
    reset_to_stance(model, data, config)

    trace = FightTrace()
    for tick in range(10):
        for _ in range(10):
            mujoco.mj_step(model, data)
        trace.observe(model, data, tick)

    summary = trace.summary()
    assert summary["ticks"] == 10
    assert summary["min_separation_m"] == pytest.approx(2 * config.start_separation, abs=0.2)
    assert "red_mean_centre_distance_m" in summary
    assert len(trace.positions["red"]) == 10


def test_an_empty_trace_summarises_without_crashing() -> None:
    assert FightTrace().summary() == {"ticks": 0.0}


def test_the_trace_records_torso_height_and_orientation(ring) -> None:
    """The two signals a knockdown is read from."""
    mujoco, model, config = ring
    data = mujoco.MjData(model)
    reset_to_stance(model, data, config)

    trace = FightTrace()
    trace.observe(model, data, 0)

    for fighter in ("red", "blue"):
        assert trace.torso_height_m[fighter][0] == pytest.approx(
            STANDING_TORSO_HEIGHT_M, abs=0.02
        )
        assert trace.torso_upright[fighter][0] == pytest.approx(1.0, abs=0.02)
        assert trace.torso_quat[fighter][0].shape == (4,)
        assert np.linalg.norm(trace.torso_quat[fighter][0]) == pytest.approx(1.0, abs=1e-6)


def test_a_standing_fighter_is_not_down(ring) -> None:
    mujoco, model, config = ring
    data = mujoco.MjData(model)
    reset_to_stance(model, data, config)

    trace = FightTrace()
    trace.observe(model, data, 0)
    assert not trace.is_down("red", 0)
    assert trace.down_ticks("red") == []


def test_a_collapsed_fighter_is_down() -> None:
    """Measured on a real collapse: a fallen G1's torso sits at 0.058 m and 0.05 upright."""
    import mujoco

    config = ArenaConfig()
    model = build_arena(config)  # gravity ON: the fighters must actually fall
    data = mujoco.MjData(model)
    reset_to_stance(model, data, config)

    trace = FightTrace()
    for tick in range(300):
        for _ in range(10):
            mujoco.mj_step(model, data)
        trace.observe(model, data, tick)

    assert trace.is_down("red", -1), (
        f"a collapsed fighter was not detected: torso at "
        f"{trace.torso_height_m['red'][-1]:.3f} m, upright {trace.torso_upright['red'][-1]:.3f}"
    )
    assert trace.down_ticks("red"), "no tick was recorded as down"
    assert trace.summary()["red_down_ticks"] > 0


def test_low_and_tilted_are_each_enough_to_be_down() -> None:
    """A fighter can be folded with its torso high, or flat with it barely below stance."""
    trace = FightTrace()
    trace.tick = [0, 1, 2]
    trace.torso_height_m["red"] = [STANDING_TORSO_HEIGHT_M, 0.05, STANDING_TORSO_HEIGHT_M]
    trace.torso_upright["red"] = [1.0, 1.0, 0.1]

    assert not trace.is_down("red", 0), "standing upright"
    assert trace.is_down("red", 1), "low torso alone must count"
    assert trace.is_down("red", 2), "tilted torso alone must count"
    assert trace.down_ticks("red") == [1, 2]


def test_the_down_threshold_sits_between_standing_and_collapsed() -> None:
    """Derived, not invented: 0.42 m is far below a crouch and far above the canvas."""
    threshold = DOWN_HEIGHT_FRACTION * STANDING_TORSO_HEIGHT_M
    assert 0.058 < threshold < STANDING_TORSO_HEIGHT_M
    assert threshold == pytest.approx(0.42, abs=0.01)


def test_an_untraced_fighter_raises() -> None:
    with pytest.raises(ContactError, match="no trace for"):
        FightTrace().is_down("green", 0)


def test_hit_events_are_frozen() -> None:
    event = HitEvent(
        attacker="red",
        defender="blue",
        attacker_body="glove_left",
        defender_body="head_link",
        region="head",
        start_tick=0,
        end_tick=4,
        peak_force_n=100.0,
        impulse_ns=0.4,
        position=(0.0, 0.0, 1.0),
    )
    assert event.duration_ticks == 5
    with pytest.raises(Exception):
        event.peak_force_n = 1.0  # type: ignore[misc]
