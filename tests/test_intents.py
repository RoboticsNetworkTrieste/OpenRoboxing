"""M2-T4 acceptance: the intent timeline and the commit queue.

Acceptance criterion from WORKPLAN.md M2-T4:
  a scripted intent sequence drives a single fighter end to end; a commit that breaks the queue rule
  is rejected with a specific error; `spec/intent.md` is versioned.

At `spec/intent.md` 1.0 the rule that criterion refers to *changed*: "one active commit" became "up
to MAX_OUTSTANDING_COMMITS, back to back, no cancellation". The tests below check the new rule with
the same rigour, because it is still the rule the game is made of.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_intents.py -v
"""

from __future__ import annotations

import math
import re

import pytest

from openroboxing.runtime.conventions import G1
from openroboxing.runtime.intents import (
    SPEC_VERSION,
    Commit,
    IntentError,
    IntentTimeline,
    Loadout,
    Placement,
    StagedIntent,
    apply_adjustment,
)
from openroboxing.spec.constants import (
    COMMIT_HORIZON_TICKS,
    MAX_OUTSTANDING_COMMITS,
    MAX_TOKENS,
    MIN_TOKENS,
    POSE_DWELL_TICKS,
)
from openroboxing.studio.pose_record import PoseRecord


def _pose(name: str, *, horizon_tokens: int = 8, admitted: bool = True, **overrides) -> PoseRecord:
    from openroboxing.runtime.obs import default_angles

    base = {
        "name": name,
        "joint_angles": dict(zip(G1.mujoco_joint_names, default_angles(G1, "mujoco"))),
        "horizon_tokens": horizon_tokens,
        "library_version": "v0.1",
        "adjustment_envelope": {"left_shoulder_pitch_joint": 0.2},
    }
    if admitted:
        base |= {"admission": "admitted", "telegraph_ms": 180.0, "generator_error_rad": 0.1}
    base.update(overrides)
    return PoseRecord(**base)


def _loadout(**overrides) -> Loadout:
    slots = {"1": _pose("jab-left"), "2": _pose("hook-right", horizon_tokens=12)}
    slots.update(overrides)
    return Loadout(name="orthodox", version="v0.1", slots=slots)


def _timeline(**kwargs) -> IntentTimeline:
    return IntentTimeline(_loadout(), **kwargs)


def _drive(timeline, through: int, *, arrived=None, facing_angle: float = 0.0, start: int = 0):
    """Tick the timeline from ``start`` to ``through``, returning one intent per tick.

    Since `spec/intent.md` 1.1 a commit's span is not computed when it is issued — it is settled as
    the move runs, and `generator_intent` is what runs it. So a test that wants to know when a move
    started has to drive the timeline there, exactly as the reference stream does.

    ``arrived`` defaults to "the moment it starts walking, it is there", which collapses the approach
    to nothing and isolates whatever else the test is about. Tests that care about the walk pass
    their own.
    """
    if arrived is None:
        def arrived(_commit):
            return True
    return [
        timeline.generator_intent(tick, facing_angle=facing_angle, has_arrived=arrived)
        for tick in range(start, through)
    ]


# --- the acceptance criterion -----------------------------------------------------------------------
def test_a_commit_past_the_queue_bound_is_rejected() -> None:
    """The 1.0 form of M2-T4's acceptance: the queue is bounded and says when it will not be."""
    timeline = _timeline()
    for _ in range(MAX_OUTSTANDING_COMMITS):
        timeline.stage(pose_slot="1")
        timeline.commit(now=100)

    with pytest.raises(IntentError) as caught:
        timeline.commit(now=100)

    message = str(caught.value)
    assert str(MAX_OUTSTANDING_COMMITS) in message, "the error must name the limit"
    assert "No cancellation" in message
    # It must *not* name a tick. Since 1.1 the move in front ends when the fighter gets where it is
    # going, so a number here would be a promise nothing can keep.
    assert "arrives" in message, "the error must say what frees a slot, not when"


def test_the_queue_holds_exactly_the_bound() -> None:
    timeline = _timeline()
    for index in range(MAX_OUTSTANDING_COMMITS):
        timeline.stage(pose_slot="1")
        timeline.commit(now=0)
        assert len(timeline.scheduled(0)) == index + 1
    assert len(timeline.commits) == MAX_OUTSTANDING_COMMITS


def test_a_scripted_sequence_drives_a_fighter_end_to_end() -> None:
    """Tick a timeline through two committed moves and check what the generator is told."""
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement(position=(1.5, 0.0), heading=0.3))

    armed_ticks, unarmed_ticks, styles = [], [], set()
    first = timeline.commit(now=10)

    timeline.stage(pose_slot="2", context="walk_boxing")
    second = timeline.commit(now=20)  # queued while the first is still scheduled

    for tick, intent in enumerate(_drive(timeline, 400)):
        styles.add(intent.style)
        (armed_ticks if intent.pose is not None else unarmed_ticks).append(tick)

    from openroboxing.runtime.intents import OPENING_STANCE_CONTEXT, TRAVEL_CONTEXT

    assert len(timeline.commits) == 2
    assert armed_ticks and unarmed_ticks, "the fighter was never both commanded and waiting"
    # Every move is generated in its own context, and since 2.0 so is the hold that follows it: the
    # first move travels in the default, the second in the one it was staged with, and the opening
    # stance is what comes before either.
    assert styles == {TRAVEL_CONTEXT, "walk_boxing", OPENING_STANCE_CONTEXT}

    # Both arrive instantly here, so each move is its dwell alone and the two run back to back: the
    # second starts on the tick the first ends, with no gap to stutter in.
    assert second.commit_at == first.end_tick
    # Armed from the tick the first move starts and *never disarmed*: past the queue's end the last
    # commit's intent stays armed, which is how the fighter holds the pose (`spec/intent.md` 2.0).
    assert armed_ticks == list(range(first.commit_at, 400))
    assert unarmed_ticks == list(range(0, first.commit_at)), "only the opening stance is unarmed"
    assert first.issued_at in unarmed_ticks, "the readable window holds, it does not move"


def test_the_spec_is_versioned() -> None:
    from pathlib import Path

    from openroboxing.paths import OPENROBOXING_ROOT

    assert SPEC_VERSION == "2.2"
    spec = Path(OPENROBOXING_ROOT / "spec/intent.md").read_text()
    assert re.search(rf"Version \*\*{re.escape(SPEC_VERSION)}\*\*", spec), (
        "spec/intent.md does not declare the version runtime/intents.py implements"
    )


# --- the commit rule ------------------------------------------------------------------------------------
def test_committing_with_nothing_staged_is_rejected() -> None:
    with pytest.raises(IntentError, match="no pose is staged"):
        _timeline().commit(now=0)


def test_a_commit_is_scheduled_from_the_tick_it_is_issued() -> None:
    """Scheduled from issue, executing from commit_at. The queue bound counts the first."""
    timeline = _timeline()
    timeline.stage(pose_slot="1")
    commit = timeline.commit(now=50)

    assert commit.issued_at == 50
    assert commit.commit_at is None, "a commit's span is settled as it runs, not when it is issued"
    assert timeline.scheduled(50) == (commit,)
    assert timeline.scheduled(49) == ()

    _drive(timeline, 500, start=50)
    assert commit.commit_at == 50 + COMMIT_HORIZON_TICKS
    assert timeline.scheduled(commit.end_tick - 1) == (commit,)
    assert timeline.scheduled(commit.end_tick) == ()


def test_an_unfinished_commit_counts_against_the_queue_forever() -> None:
    """`end_tick` is None while a move is still walking, and None means *later than any tick you can
    name* — not zero and not "already over". Read the other way, a fighter walking across the ring
    would look idle and the queue bound would stop bounding anything."""
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement((3.0, 0.0), 0.0))
    commit = timeline.commit(now=0)
    _drive(timeline, 200, arrived=lambda _c: False)

    assert commit.end_tick is None and commit.strike_at is None
    assert commit.is_scheduled(10_000), "an unfinished commit is outstanding at every later tick"
    assert timeline.scheduled(10_000) == (commit,)


def test_execution_starts_only_after_the_readable_window() -> None:
    timeline = _timeline()
    timeline.stage(pose_slot="1")
    commit = timeline.commit(now=0)

    assert commit.is_scheduled(0) and not commit.is_executing(0)
    assert timeline.executing(0) is None, "issued is not the same as under way"
    assert timeline.current(0) is None, "nothing is current inside the readable window"

    _drive(timeline, 400)
    assert commit.commit_at == COMMIT_HORIZON_TICKS
    assert commit.is_executing(COMMIT_HORIZON_TICKS)
    assert not commit.is_executing(commit.end_tick)
    assert timeline.executing(COMMIT_HORIZON_TICKS) is commit


# --- the queue (spec/intent.md 1.0) -------------------------------------------------------------
def test_the_horizon_is_a_floor_not_a_gap_between_queued_moves() -> None:
    """A commit into an empty queue pays the full 0.6 s. One queued behind a running move pays
    nothing extra, because the readable window elapsed while the player watched the first move.
    Inserting a fixed pause between queued moves would stutter every combination."""
    timeline = _timeline()
    timeline.stage(pose_slot="1")
    first = timeline.commit(now=0)
    timeline.stage(pose_slot="2")
    second = timeline.commit(now=1)
    _drive(timeline, 400)

    assert first.commit_at == COMMIT_HORIZON_TICKS
    assert second.commit_at == first.end_tick, "queued moves run back to back"
    assert second.commit_at > second.issued_at + COMMIT_HORIZON_TICKS


def test_a_commit_into_a_drained_queue_still_pays_the_horizon() -> None:
    timeline = _timeline()
    timeline.stage(pose_slot="1")
    first = timeline.commit(now=0)
    _drive(timeline, 200)

    timeline.stage(pose_slot="2")
    later = timeline.commit(now=first.end_tick + 500)
    _drive(timeline, later.issued_at + 400, start=200)
    assert later.commit_at == later.issued_at + COMMIT_HORIZON_TICKS


def test_queued_moves_never_overlap() -> None:
    timeline = _timeline()
    for slot in ("1", "2", "1", "2"):
        timeline.stage(pose_slot=slot)
        timeline.commit(now=0)
    _drive(timeline, 800)

    spans = [(c.commit_at, c.end_tick) for c in timeline.commits]
    assert all(None not in span for span in spans), "every move should have finished"
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start == end, "a gap or an overlap between queued moves"


def test_at_most_one_commit_executes_at_a_time() -> None:
    timeline = _timeline()
    for slot in ("1", "2", "1"):
        timeline.stage(pose_slot=slot)
        timeline.commit(now=0)
    _drive(timeline, 800)

    last = max(c.end_tick for c in timeline.commits)
    for tick in range(0, last + 5):
        executing = [c for c in timeline.commits if c.is_executing(tick)]
        assert len(executing) <= 1, f"{len(executing)} moves executing at tick {tick}"


def test_a_queued_commit_cannot_be_taken_back() -> None:
    """No cancellation, extended to a move that has not started. That is what makes a deep queue a
    risk rather than a free lookahead."""
    timeline = _timeline()
    timeline.stage(pose_slot="1")
    timeline.commit(now=0)
    timeline.stage(pose_slot="2")
    queued = timeline.commit(now=1)

    timeline.clear_pose()
    timeline.stage(pose_slot="1", placement=Placement((9.0, 9.0), 0.0))

    assert queued in timeline.scheduled(1), "staging must not remove a queued commit"
    assert queued.placement != Placement((9.0, 9.0), 0.0), "nor re-aim it"

    _drive(timeline, 400)
    assert timeline.executing(queued.commit_at) is queued, "it ran anyway, as promised"


def test_the_anchor_is_where_the_queue_leaves_the_fighter() -> None:
    timeline = _timeline()
    assert timeline.anchor_placement(0) is None, "a drained queue has no anchor of its own"

    timeline.stage(pose_slot="1", placement=Placement((1.0, 0.0), 0.0))
    timeline.commit(now=0)
    timeline.stage(pose_slot="2", placement=Placement((2.0, 1.0), 0.5))
    last = timeline.commit(now=1)

    # Asked at tick 1, not 0: the anchor is where the queue *as it stands* leaves the fighter, and
    # at tick 0 the second commit had not been issued yet.
    assert timeline.anchor_placement(1) == Placement((2.0, 1.0), 0.5)

    _drive(timeline, 400)
    assert timeline.anchor_placement(last.end_tick) is None, "spent commits stop anchoring"


def test_tail_end_is_gone() -> None:
    """It answered "when does everything queued finish", which since 1.1 nothing can know. Deleted
    rather than left returning a plausible number (`CLAUDE.md` prefers deleting to disabling)."""
    assert not hasattr(_timeline(), "tail_end")


def test_current_is_the_oldest_unfinished_commit() -> None:
    """Order is the one thing a player controls about a queue they cannot cancel, so a commit still
    inside its readable window blocks the queue rather than being stepped over."""
    timeline = _timeline()
    timeline.stage(pose_slot="1")
    first = timeline.commit(now=0)
    timeline.stage(pose_slot="2")
    second = timeline.commit(now=0)

    assert timeline.current(0) is None, "the window has not elapsed"
    assert timeline.current(COMMIT_HORIZON_TICKS) is first
    assert timeline.current(COMMIT_HORIZON_TICKS) is not second, "it did not jump the queue"

    _drive(timeline, 400)
    assert timeline.current(first.end_tick) is second


def test_the_bound_is_configurable_and_must_be_at_least_one() -> None:
    timeline = _timeline(max_outstanding=1)
    timeline.stage(pose_slot="1")
    timeline.commit(now=0)
    with pytest.raises(IntentError, match="already queued"):
        timeline.commit(now=1)

    with pytest.raises(IntentError, match="at least 1"):
        _timeline(max_outstanding=0)


def test_a_new_commit_is_allowed_once_the_last_one_ends() -> None:
    timeline = _timeline()
    timeline.stage(pose_slot="1")
    first = timeline.commit(now=0)
    _drive(timeline, 400)

    timeline.stage(pose_slot="2")
    second = timeline.commit(now=first.end_tick)
    assert second.pose.name == "hook-right"
    assert len(timeline.commits) == 2


def test_the_timeline_does_not_run_backwards() -> None:
    timeline = _timeline()
    timeline.stage(pose_slot="1")
    timeline.commit(now=200)
    with pytest.raises(IntentError, match="does not run backwards"):
        timeline.commit(now=100)


def test_a_negative_tick_is_rejected() -> None:
    timeline = _timeline()
    timeline.stage(pose_slot="1")
    with pytest.raises(IntentError, match="must not be negative"):
        timeline.commit(now=-1)


# --- staging is always allowed ---------------------------------------------------------------------------
def test_staging_continues_while_a_commit_executes() -> None:
    """Continuous staging: the player keeps aiming the next move while this one plays out."""
    timeline = _timeline()
    timeline.stage(pose_slot="1")
    commit = timeline.commit(now=0)

    staged = timeline.stage(pose_slot="2", placement=Placement((2.0, 1.0), 1.2))
    assert staged.pose_slot == "2"
    assert timeline.scheduled(commit.issued_at + 1) == (commit,), "staging must not disturb it"

    _drive(timeline, 400)
    assert timeline.executing(commit.commit_at).pose.name == "jab-left", "the commit is frozen"


def test_staging_only_changes_the_channels_it_is_given() -> None:
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement((1.0, 2.0), 0.5), context="walk")
    staged = timeline.stage(pose_slot="2")

    assert staged.pose_slot == "2"
    assert staged.context == "walk"
    assert staged.placement == Placement((1.0, 2.0), 0.5)


def test_staging_an_unknown_slot_fails_at_staging_time() -> None:
    with pytest.raises(IntentError, match="slot '9' is not in loadout"):
        _timeline().stage(pose_slot="9")


def test_clearing_the_pose_leaves_the_other_channels() -> None:
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement((1.0, 0.0), 0.0), context="walk")
    staged = timeline.clear_pose()

    assert staged.pose_slot is None
    assert not staged.is_committable()
    assert staged.context == "walk" and staged.placement is not None


# --- adjustment stays inside the admitted envelope -----------------------------------------------------------
def test_an_adjustment_inside_the_envelope_is_applied() -> None:
    pose = _pose("jab")
    adjusted = apply_adjustment(pose, {"left_shoulder_pitch_joint": 0.15})
    delta = (
        adjusted.joint_angles["left_shoulder_pitch_joint"]
        - pose.joint_angles["left_shoulder_pitch_joint"]
    )
    assert delta == pytest.approx(0.15)
    assert adjusted.name == pose.name


def test_an_adjustment_outside_the_envelope_is_rejected_not_clamped() -> None:
    with pytest.raises(IntentError, match="leaves pose 'jab''s envelope"):
        apply_adjustment(_pose("jab"), {"left_shoulder_pitch_joint": 0.5})


def test_an_adjustment_on_an_uncovered_joint_is_rejected() -> None:
    with pytest.raises(IntentError, match="no adjustment envelope for 'left_knee_joint'"):
        apply_adjustment(_pose("jab"), {"left_knee_joint": 0.01})


def test_the_commit_carries_the_adjusted_pose() -> None:
    timeline = _timeline()
    timeline.stage(pose_slot="1", adjustment={"left_shoulder_pitch_joint": 0.1})
    commit = timeline.commit(now=0)

    base = timeline.loadout.resolve("1")
    assert commit.pose.joint_angles["left_shoulder_pitch_joint"] == pytest.approx(
        base.joint_angles["left_shoulder_pitch_joint"] + 0.1
    )


# --- the loadout ----------------------------------------------------------------------------------------------
def test_a_match_refuses_an_unadmitted_pose() -> None:
    loadout = Loadout(name="dev", version="v0", slots={"1": _pose("draft", admitted=False)})
    with pytest.raises(IntentError, match="which is 'draft'"):
        IntentTimeline(loadout)


def test_the_studio_may_use_an_unadmitted_pose() -> None:
    loadout = Loadout(name="dev", version="v0", slots={"1": _pose("draft", admitted=False)})
    timeline = IntentTimeline(loadout, require_admitted=False)
    timeline.stage(pose_slot="1")
    assert timeline.commit(now=0).pose.name == "draft"


def test_an_empty_loadout_is_rejected() -> None:
    with pytest.raises(IntentError, match="has no slots"):
        IntentTimeline(Loadout(name="empty", version="v0", slots={}))


def test_an_invalid_pose_names_its_slot() -> None:
    broken = _pose("broken", horizon_tokens=MAX_TOKENS + 5)
    loadout = Loadout(name="bad", version="v0", slots={"3": broken})
    with pytest.raises(IntentError, match=r"slot '3': horizon_tokens"):
        IntentTimeline(loadout)


# --- a commit has no length of its own -----------------------------------------------------------
def test_the_commit_length_is_the_dwell_not_the_poses_horizon() -> None:
    """Since 2.0 MotionBricks chooses how long the motion runs, so a commit cannot be scheduled
    around a token count. Two poses with different horizons get the same span from the same walk."""
    spans = []
    for slot in ("1", "2"):  # jab-left is 8 tokens, hook-right 12
        timeline = _timeline()
        timeline.stage(pose_slot=slot)
        commit = timeline.commit(now=0)
        _drive(timeline, 400)
        spans.append(commit.end_tick - commit.strike_at)

    assert not hasattr(timeline.commits[0], "duration_ticks")
    assert spans == [POSE_DWELL_TICKS, POSE_DWELL_TICKS]


# --- what the generator is told ------------------------------------------------------------------
def test_the_opening_stance_does_not_travel() -> None:
    """`spec/intent.md` 0.2, and still true at 2.0. Before the first commit of a round there is no
    committed intent to hold, and the fighter must **stand**. It must not use the ambient context:
    in v0.1 this did, which for `walk_boxing` meant 2.0 m/s in a fixed direction — `ASSUMPTIONS.md`
    §A18, the reason fighters ended up at opposite ends of the ring.

    Not a duplicate of `test_a_drained_queue_holds_the_last_committed_pose`: that one covers *after*
    a commit, this one *before any*, and since 2.0 they are different code paths."""
    from openroboxing.runtime.intents import OPENING_STANCE_CONTEXT

    timeline = _timeline()
    timeline.stage(context="walk_boxing")
    intent = timeline.generator_intent(tick=0)

    assert intent.pose is None
    assert intent.style == OPENING_STANCE_CONTEXT
    assert intent.style != timeline.staged.context, "the opening stance is not the ambient clip"
    assert intent.target_position is None
    assert intent.movement_angle == 0.0, "standing is a clip that does not travel, not a direction"


def test_a_scheduled_but_unstarted_commit_still_holds() -> None:
    """The readable window is a hold, not a lean-in. Nothing moves until commit_at."""
    from openroboxing.runtime.intents import OPENING_STANCE_CONTEXT

    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement((1.0, 0.0), 0.0))
    commit = timeline.commit(now=0)

    window = _drive(timeline, COMMIT_HORIZON_TICKS)
    assert commit.commit_at is None, "nothing started"
    assert all(intent.pose is None for intent in window)
    assert all(intent.style == OPENING_STANCE_CONTEXT for intent in window)
    assert all(intent.target_position is None for intent in window)


def test_holding_still_faces_where_it_is_told() -> None:
    timeline = _timeline()
    intent = timeline.generator_intent(tick=0, facing_angle=1.3)
    assert intent.facing_angle == pytest.approx(1.3)


def test_the_movement_channel_is_gone() -> None:
    """Retired at 1.0 rather than deprecated: `CLAUDE.md` prefers deleting to disabling, and two
    ways to move would be two things to learn and two things to balance."""
    import openroboxing.runtime.intents as intents

    assert not hasattr(intents, "MOVEMENTS")
    assert not hasattr(intents, "MOVE_CONTEXT")
    assert "movement" not in StagedIntent.__dataclass_fields__
    with pytest.raises(TypeError):
        _timeline().stage(movement="in")


def test_an_executing_commit_arms_the_pose_and_its_placement() -> None:
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement(position=(1.5, -0.5), heading=0.9))
    commit = timeline.commit(now=0)

    intents = _drive(timeline, COMMIT_HORIZON_TICKS + 1)
    intent = intents[commit.commit_at]
    assert intent.pose is not None and intent.pose.name == "jab-left"
    assert intent.target_position == (1.5, -0.5)
    assert intent.target_heading == pytest.approx(0.9)
    assert intent.horizon_tokens is None, "the pose's horizon must not reach the generator"


def test_placement_is_optional() -> None:
    timeline = _timeline()
    timeline.stage(pose_slot="1")
    commit = timeline.commit(now=0)

    intents = _drive(timeline, COMMIT_HORIZON_TICKS + 1, facing_angle=0.4)
    intent = intents[commit.commit_at]
    assert intent.target_position is None and intent.target_heading is None
    assert intent.facing_angle == pytest.approx(0.4)


def test_placement_must_be_a_pair() -> None:
    with pytest.raises(IntentError, match=r"must be \(x, y\)"):
        Placement(position=(1.0, 2.0, 3.0), heading=0.0)  # type: ignore[arg-type]


def test_a_placement_must_be_finite() -> None:
    """A client sends these. NaN would propagate into the generator and out into physics as a
    fighter that simply vanishes, which is a very hard bug to read backwards."""
    with pytest.raises(IntentError, match="must be finite"):
        Placement(position=(float("nan"), 0.0), heading=0.0)
    with pytest.raises(IntentError, match="must be finite"):
        Placement(position=(0.0, 0.0), heading=float("inf"))


# --- a commit runs until it arrives (spec/intent.md 1.1) -----------------------------------------
def test_a_commit_walks_until_it_arrives_however_long_that_takes() -> None:
    """The 1.1 fix, stated as a test. At 1.0 a commit was one fixed-length plan, so a fighter walked
    for about a second and stopped wherever that left it — pointing further did not go further.

    A far placement therefore takes *longer* than a near one rather than falling short of it, and
    nothing about the pose says how long. That is the whole change.
    """
    near, far = 60, 400
    spans = []
    for arrive_at in (near, far):
        timeline = _timeline()
        timeline.stage(pose_slot="1", placement=Placement((3.0, 0.0), 0.0))
        commit = timeline.commit(now=0)
        _drive(timeline, far + 200, arrived=lambda _c, at=arrive_at: timeline._driven_to >= at)
        spans.append(commit.end_tick - commit.commit_at)

    assert spans[1] - spans[0] == far - near, "the walk is what makes one move longer than the other"
    assert spans[0] > POSE_DWELL_TICKS, "even the near one is longer than the dwell it ends with"


def test_arriving_ends_the_move_a_dwell_later() -> None:
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement((1.0, 0.0), 0.0))
    commit = timeline.commit(now=0)

    walked = 120
    _drive(timeline, 400, arrived=lambda _c: timeline._driven_to >= walked)

    assert commit.commit_at == COMMIT_HORIZON_TICKS
    assert commit.strike_at == walked
    assert commit.arrived is True
    assert commit.end_tick == walked + POSE_DWELL_TICKS, "it stands in the pose, then ends"


def test_a_stalled_approach_throws_where_it_stands_rather_than_stalling_the_queue() -> None:
    """A fighter wedged in a corner would otherwise hold everything behind it and the player would
    lose the round standing still. It throws, and the record says it never got there."""
    timeline = _timeline(approach_timeout_ticks=100)
    timeline.stage(pose_slot="1", placement=Placement((3.0, 0.0), 0.0))
    first = timeline.commit(now=0)
    timeline.stage(pose_slot="2", placement=Placement((3.0, 1.0), 0.0))
    second = timeline.commit(now=1)

    _drive(timeline, 900, arrived=lambda _c: False)

    assert first.arrived is False, "it never got there"
    assert first.strike_at == first.commit_at + 100, "and gave up exactly at the timeout"
    assert first.end_tick is not None, "a timed-out commit still completes"
    assert second.commit_at == first.end_tick, "the queue moved on rather than jamming"


def test_a_commit_with_no_placement_throws_immediately() -> None:
    """"Do this where you stand" is a complete instruction, not a missing one — it is what the
    Studio and the offline tools issue, and it must not need a world to resolve."""
    timeline = _timeline()
    timeline.stage(pose_slot="1")
    commit = timeline.commit(now=0)

    _drive(timeline, 200, arrived=lambda _c: pytest.fail("nothing to walk to"))
    assert commit.strike_at == commit.commit_at and commit.arrived is True


def test_a_placement_with_no_arrival_test_is_refused_loudly() -> None:
    """Defaulting to "arrived" would quietly restore the 1.0 bug, and defaulting to "not yet" would
    hang every queue until the timeout. Neither is a default worth having."""
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement((3.0, 0.0), 0.0))
    timeline.commit(now=0)

    with pytest.raises(IntentError, match="no arrival test"):
        for tick in range(200):
            timeline.generator_intent(tick)


def test_driving_the_queue_backwards_is_refused() -> None:
    """`generator_intent` advances the queue, so an out-of-order tick would rewrite a move's history
    rather than fail — the quietest possible way to get a fight wrong."""
    timeline = _timeline()
    timeline.stage(pose_slot="1")
    timeline.commit(now=0)
    _drive(timeline, 100)

    with pytest.raises(IntentError, match="went backwards"):
        timeline.generator_intent(50, has_arrived=lambda _c: True)


def test_the_plan_key_is_gone() -> None:
    """It existed only to bind a forced plan to the commit that forced it. Since 2.0 nothing is
    forced, so it is deleted rather than left carrying a value nobody reads (`CLAUDE.md` prefers
    deleting to disabling)."""
    from openroboxing.runtime.generator import GeneratorIntent

    assert "plan_key" not in GeneratorIntent.__dataclass_fields__
    with pytest.raises(TypeError):
        GeneratorIntent(plan_key=1)


def test_the_approach_timeout_is_derived_from_the_ring() -> None:
    """Derived, not chosen: the ring's diagonal at the measured approach speed. A match that shrinks
    the ring must shrink the patience a stalled approach is given with it."""
    from openroboxing.runtime.intents import DEFAULT_APPROACH_TIMEOUT_TICKS, approach_timeout_ticks
    from openroboxing.spec.constants import APPROACH_SPEED_M_S, RING_SIZE_M, TICK_HZ

    expected = math.ceil(RING_SIZE_M * math.sqrt(2.0) / APPROACH_SPEED_M_S * TICK_HZ)
    assert DEFAULT_APPROACH_TIMEOUT_TICKS == expected
    assert approach_timeout_ticks(RING_SIZE_M / 2) < DEFAULT_APPROACH_TIMEOUT_TICKS

    with pytest.raises(IntentError, match="positive length"):
        approach_timeout_ticks(0.0)


# --- the commit log ------------------------------------------------------------------------------
def test_every_commit_is_logged_in_order() -> None:
    timeline = _timeline()
    tick = 0
    for slot in ("1", "2", "1"):
        timeline.stage(pose_slot=slot)
        commit = timeline.commit(now=tick)
        _drive(timeline, commit.issued_at + 200, start=tick)
        tick = commit.issued_at + 200

    assert [c.slot for c in timeline.commits] == ["1", "2", "1"]
    assert [c.issued_at for c in timeline.commits] == sorted(c.issued_at for c in timeline.commits)


def test_the_log_keeps_queue_order() -> None:
    """Queued all at once, so issue order is the only thing that decides execution order."""
    timeline = _timeline()
    for slot in ("2", "1", "2"):
        timeline.stage(pose_slot=slot)
        timeline.commit(now=7)
    _drive(timeline, 800, start=7)

    assert [c.slot for c in timeline.commits] == ["2", "1", "2"]
    assert [c.commit_at for c in timeline.commits] == sorted(c.commit_at for c in timeline.commits)
    assert all(isinstance(c, Commit) for c in timeline.commits)


def test_the_staged_intent_is_immutable_once_read() -> None:
    timeline = _timeline()
    staged = timeline.stage(pose_slot="1")
    timeline.stage(pose_slot="2")
    assert staged.pose_slot == "1", "a previously returned StagedIntent must not change underfoot"
    assert isinstance(staged, StagedIntent)


# --- the real generator --------------------------------------------------------------------------
@pytest.mark.slow
def test_drives_the_real_generator() -> None:
    """The timeline's output must be something MotionBricks actually accepts."""
    import numpy as np

    from openroboxing.runtime.generator import GeneratorConfig, MotionBricksGenerator

    generator = MotionBricksGenerator(GeneratorConfig(random_seed=1234))
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement((1.0, 0.5), 0.2))
    timeline.commit(now=0)

    generator.reset(seed=1234)
    frames = []
    # Arrives half way, so the run covers both halves of one continuous intent: converging on the
    # placement, then standing in the pose it converged into. The intent itself does not change.
    for tick in range(0, 60):
        frames.append(generator.next_frame())
        intent = timeline.generator_intent(tick, has_arrived=lambda _commit: tick >= 30)
        generator.generate(intent, generator.context_qpos(), dt=0.5)

    qpos = np.asarray(frames)
    assert np.isfinite(qpos).all()
    assert qpos.shape[0] == 60
    assert timeline.commits[0].strike_at == 30, "the approach should have ended when it arrived"


# --- the horizon reaches the generator ------------------------------------------------------
def test_a_horizon_narrows_the_clip_mask_to_one_length() -> None:
    from openroboxing.runtime.generator import narrow_allowed_tokens
    from openroboxing.spec.constants import NUM_TIME_TOKENS

    permissive = [1] * NUM_TIME_TOKENS
    narrowed = narrow_allowed_tokens(permissive, 9, style="walk_boxing")

    assert sum(narrowed) == 1, "exactly one token length must be permitted"
    assert narrowed[9 - MIN_TOKENS] == 1
    assert len(narrowed) == NUM_TIME_TOKENS


def test_no_horizon_leaves_the_clip_mask_alone() -> None:
    from openroboxing.runtime.generator import narrow_allowed_tokens
    from openroboxing.spec.constants import NUM_TIME_TOKENS

    mask = [1] * 6 + [0] * (NUM_TIME_TOKENS - 6)
    assert narrow_allowed_tokens(mask, None) == mask


def test_a_horizon_the_clip_forbids_raises_and_lists_what_it_permits() -> None:
    from openroboxing.runtime.generator import GeneratorError, narrow_allowed_tokens
    from openroboxing.spec.constants import NUM_TIME_TOKENS

    mask = [1] * 6 + [0] * (NUM_TIME_TOKENS - 6)  # only 6..11 tokens
    with pytest.raises(GeneratorError, match=r"does not allow a 14-token move"):
        narrow_allowed_tokens(mask, 14, style="walk")


def test_a_horizon_outside_the_token_range_raises() -> None:
    from openroboxing.runtime.generator import GeneratorError, narrow_allowed_tokens
    from openroboxing.spec.constants import NUM_TIME_TOKENS

    with pytest.raises(GeneratorError, match="outside"):
        narrow_allowed_tokens([1] * NUM_TIME_TOKENS, MAX_TOKENS + 1)


def test_a_clip_mask_of_the_wrong_length_raises() -> None:
    from openroboxing.runtime.generator import GeneratorError, narrow_allowed_tokens

    with pytest.raises(GeneratorError, match="declares 3 token slots"):
        narrow_allowed_tokens([1, 1, 1], 8, style="broken")


# --- one continuous intent, and the hold (spec/intent.md 2.0) ------------------------------------
def test_a_walking_commit_already_carries_its_pose() -> None:
    """Intent 2.0: there is no poseless approach."""
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement(position=(3.0, 0.0), heading=0.0))
    timeline.commit(0)
    intents = [
        timeline.generator_intent(t, has_arrived=lambda _c: False)
        for t in range(COMMIT_HORIZON_TICKS, COMMIT_HORIZON_TICKS + 40)
    ]
    assert all(i.pose is not None for i in intents), "the pose is armed for the whole move"
    assert all(i.horizon_tokens is None for i in intents), "the model chooses the length"
    assert all(i.target_position == (3.0, 0.0) for i in intents)


def test_a_drained_queue_holds_the_last_committed_pose() -> None:
    """The owner's first correction: no idle clip at the end of a commit."""
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement(position=(0.5, 0.0), heading=0.0))
    commit = timeline.commit(0)
    intents = [
        timeline.generator_intent(t, has_arrived=lambda _c: True)
        for t in range(COMMIT_HORIZON_TICKS, COMMIT_HORIZON_TICKS + 200)
    ]
    after = intents[-1]
    assert commit.end_tick is not None
    assert after.pose is not None, "the fighter holds the pose it was commanded into"
    assert after.pose.name == commit.pose.name
    assert after.style == commit.context, "and not the idle clip"
    assert after.target_position == (0.5, 0.0), "it stays where it arrived"
    assert after.target_heading == 0.0, "and does not turn to face anyone"


def test_a_commit_with_no_settle_test_ends_on_the_counted_dwell() -> None:
    """A caller with no body — a Studio rehearsal — still gets the measured dwell.

    Since 2.2 the span is *stamped*, so it is None until the timeline is driven past the dwell:
    "not yet" and "74 ticks from now" are different claims and only one of them is knowable.
    """
    from openroboxing.spec.constants import POSE_DWELL_TICKS

    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement(position=(0.5, 0.0), heading=0.0))
    commit = timeline.commit(0)
    for t in range(COMMIT_HORIZON_TICKS, COMMIT_HORIZON_TICKS + 5):
        timeline.generator_intent(t, has_arrived=lambda _c: True)
    assert commit.strike_at == COMMIT_HORIZON_TICKS
    assert commit.end_tick is None, "the move is not over, and no tick can be promised for it"

    for t in range(COMMIT_HORIZON_TICKS + 5, COMMIT_HORIZON_TICKS + POSE_DWELL_TICKS + 2):
        timeline.generator_intent(t, has_arrived=lambda _c: True)
    assert commit.end_tick == commit.strike_at + POSE_DWELL_TICKS
    assert commit.completed_by == "dwell"


def test_a_settle_test_ends_the_move_when_the_body_says_so() -> None:
    """The 2.2 rule: the next target goes in when the move is over, not when a counter says so."""
    from openroboxing.spec.constants import POSE_DWELL_TICKS

    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement(position=(0.5, 0.0), heading=0.0))
    commit = timeline.commit(0)
    settled = {"yet": False}
    for t in range(COMMIT_HORIZON_TICKS, COMMIT_HORIZON_TICKS + 20):
        timeline.generator_intent(
            t, has_arrived=lambda _c: True, has_settled=lambda _c: settled["yet"]
        )
    assert commit.end_tick is None, "not settled, so not over — the dwell is not a countdown"

    settled["yet"] = True
    timeline.generator_intent(
        COMMIT_HORIZON_TICKS + 20, has_arrived=lambda _c: True, has_settled=lambda _c: True
    )
    assert commit.end_tick == COMMIT_HORIZON_TICKS + 20
    assert commit.completed_by == "settled"
    assert commit.end_tick < commit.strike_at + POSE_DWELL_TICKS, "and it ended early"


def test_a_pose_the_body_never_settles_into_does_not_hold_the_queue_forever() -> None:
    """The one failure the counted dwell did not have, and the guard that answers it."""
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement(position=(0.5, 0.0), heading=0.0))
    commit = timeline.commit(0)
    for t in range(COMMIT_HORIZON_TICKS, COMMIT_HORIZON_TICKS + timeline.max_dwell_ticks + 2):
        timeline.generator_intent(
            t, has_arrived=lambda _c: True, has_settled=lambda _c: False
        )
    assert commit.end_tick == commit.strike_at + timeline.max_dwell_ticks
    assert commit.completed_by == "timeout"


def test_the_next_commit_starts_when_the_dwell_ends() -> None:
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement(position=(0.5, 0.0), heading=0.0))
    first = timeline.commit(0)
    timeline.stage(pose_slot="2", placement=Placement(position=(1.0, 0.0), heading=0.0))
    second = timeline.commit(1)
    for t in range(COMMIT_HORIZON_TICKS, COMMIT_HORIZON_TICKS + 300):
        timeline.generator_intent(t, has_arrived=lambda _c: True)
    assert second.commit_at == first.end_tick, "queued moves still run back to back"
