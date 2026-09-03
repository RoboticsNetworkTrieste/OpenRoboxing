"""M6-T5 acceptance: the intent timeline against `spec/intent.md` 3.0 — a commit is a combination.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_intents_combinations.py -v

These tests are written against the target API before `runtime/intents.py` is rewritten, and are
expected to fail (mostly on import) until that rewrite lands.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from openroboxing.paths import OPENROBOXING_ROOT
from openroboxing.runtime.conventions import G1
from openroboxing.runtime.intents import (
    OPENING_STANCE_CONTEXT,
    SPEC_VERSION,
    IntentError,
    IntentTimeline,
)
from openroboxing.spec.constants import COMMIT_HORIZON_TICKS, MAX_OUTSTANDING_COMMITS
from openroboxing.studio.combination_record import (
    CombinationRecord,
    CombinationSource,
    Keyframe,
)

ANGLES = {name: 0.0 for name in G1.mujoco_joint_names}


def _record(
    name: str,
    *,
    offsets=((0.3, 0.0), (0.6, 0.0)),
    headings=(0.0, 0.0),
    tokens=(6, 6),
) -> CombinationRecord:
    """A small, admitted combination — two legs, cheap enough to drive a timeline through fully."""
    keyframes = [Keyframe(dict(ANGLES), None, (0.0, 0.0), 0.0)]
    for offset, heading, token in zip(offsets, headings, tokens, strict=True):
        keyframes.append(Keyframe(dict(ANGLES), token, offset, heading))
    return CombinationRecord(
        name=name,
        library_version="v0.2",
        source=CombinationSource("t", 0, 100, False),
        keyframes=keyframes,
        telegraph_ms=180.0,
        tracking_error_rad=0.1,
        admission="admitted",
    )


def _library(**overrides) -> dict[str, CombinationRecord]:
    library = {
        "combo-a": _record("combo-a"),
        "combo-b": _record("combo-b", offsets=((0.1, 0.0), (0.2, 0.0))),
    }
    library.update(overrides)
    return library


def _timeline(**kwargs) -> IntentTimeline:
    return IntentTimeline(_library(), **kwargs)


def _anchor(position=(0.0, 0.0), heading: float = 0.0):
    """A counting anchor: how many times it was called is the whole point of some tests below."""
    calls = {"n": 0}

    def anchor():
        calls["n"] += 1
        return position, heading

    anchor.calls = calls
    return anchor


# --- staging and committing by name ----------------------------------------------------------
def test_staging_and_committing_a_combination_by_name() -> None:
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    commit = timeline.commit(now=0)

    assert commit.record.name == "combo-a"
    assert commit.ghost == (1.0, 0.0)
    assert commit.issued_at == 0
    assert commit.commit_at is None, "a commit's span is settled when it starts, not when issued"
    assert commit.end_tick is None


def test_staging_an_unknown_combination_fails_at_staging_time() -> None:
    with pytest.raises(IntentError, match="not in the library"):
        _timeline().stage(combination="does-not-exist")


def test_a_commit_past_the_queue_bound_is_rejected() -> None:
    timeline = _timeline()
    for _ in range(MAX_OUTSTANDING_COMMITS):
        timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
        timeline.commit(now=0)

    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    with pytest.raises(IntentError) as caught:
        timeline.commit(now=0)

    message = str(caught.value)
    assert str(MAX_OUTSTANDING_COMMITS) in message
    assert "No cancellation" in message


def test_committing_with_nothing_staged_is_rejected() -> None:
    with pytest.raises(IntentError, match="no combination is staged"):
        _timeline().commit(now=0)


def test_committing_with_a_combination_but_no_ghost_is_rejected() -> None:
    timeline = _timeline()
    timeline.stage(combination="combo-a")
    with pytest.raises(IntentError, match="no ghost is staged"):
        timeline.commit(now=0)


# --- the spec pairing -------------------------------------------------------------------------
def test_the_spec_is_versioned_at_3_1() -> None:
    assert SPEC_VERSION == "3.1"
    spec = Path(OPENROBOXING_ROOT / "spec/intent.md").read_text()
    assert re.search(rf"Version \*\*{re.escape(SPEC_VERSION)}\*\*", spec), (
        "spec/intent.md does not declare the version runtime/intents.py implements"
    )


# --- a commit's span: known the instant it starts ---------------------------------------------
def test_commit_at_is_stamped_on_the_first_generator_intent_at_the_horizon_floor() -> None:
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    commit = timeline.commit(now=0)
    anchor = _anchor()

    for tick in range(COMMIT_HORIZON_TICKS):
        timeline.generator_intent(tick, anchor=anchor)
        assert commit.commit_at is None, "the readable window has not elapsed yet"

    timeline.generator_intent(COMMIT_HORIZON_TICKS, anchor=anchor)
    assert commit.commit_at == COMMIT_HORIZON_TICKS


def test_end_tick_is_commit_at_plus_duration_and_none_before_that() -> None:
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    commit = timeline.commit(now=0)
    assert commit.end_tick is None

    anchor = _anchor()
    for tick in range(COMMIT_HORIZON_TICKS + 1):
        timeline.generator_intent(tick, anchor=anchor)

    assert commit.commit_at == COMMIT_HORIZON_TICKS
    assert commit.end_tick == commit.commit_at + commit.record.duration_ticks


# --- the queue is a schedule again --------------------------------------------------------------
def test_the_queue_advances_to_the_next_commit_exactly_at_end_tick() -> None:
    """No tick at which neither commit is current: the second starts the instant the first ends."""
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    first = timeline.commit(now=0)
    timeline.stage(combination="combo-b", ghost=(0.5, 0.0))
    second = timeline.commit(now=1)

    anchor = _anchor()
    styles = [timeline.generator_intent(tick, anchor=anchor).style for tick in range(400)]

    assert first.commit_at is not None and first.end_tick is not None
    assert second.commit_at == first.end_tick, "the second starts exactly where the first ends"
    assert OPENING_STANCE_CONTEXT not in styles[first.commit_at :], (
        "once the first commit has started, nothing should ever fall back to the opening stance"
    )


def test_a_drained_queue_holds_the_last_commits_final_leg() -> None:
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    commit = timeline.commit(now=0)
    anchor = _anchor()
    for tick in range(400):
        timeline.generator_intent(tick, anchor=anchor)

    expected = commit.runner.intent_for(500)
    held = timeline.generator_intent(500, anchor=anchor)

    assert held.style == expected.style
    assert held.target_position == expected.target_position
    assert held.target_heading == expected.target_heading
    assert held.pose.joint_angles == expected.pose.joint_angles


def test_before_any_commit_the_style_is_the_opening_stance() -> None:
    timeline = _timeline()
    intent = timeline.generator_intent(0)
    assert intent.style == OPENING_STANCE_CONTEXT
    assert intent.pose is None


# --- generator_intent is the clock, not a query -------------------------------------------------
def test_generator_intent_raises_on_a_backwards_tick() -> None:
    timeline = _timeline()
    timeline.generator_intent(10)
    with pytest.raises(IntentError, match="went backwards"):
        timeline.generator_intent(5)


# --- a combination starting in place needs to know where that is --------------------------------
def test_a_commit_starting_with_no_anchor_raises() -> None:
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    timeline.commit(now=0)

    with pytest.raises(IntentError, match="anchor"):
        for tick in range(COMMIT_HORIZON_TICKS + 1):
            timeline.generator_intent(tick)


def test_anchor_is_called_exactly_once_per_commit_not_once_per_tick() -> None:
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    timeline.commit(now=0)
    timeline.stage(combination="combo-b", ghost=(0.5, 0.0))
    timeline.commit(now=1)

    anchor = _anchor()
    for tick in range(400):
        timeline.generator_intent(tick, anchor=anchor)

    assert anchor.calls["n"] == 2, "once to start each commit, never once per tick"


# --- staging: two channels, and nothing more --------------------------------------------------
def test_staging_only_changes_the_channel_it_is_given() -> None:
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 2.0))
    staged = timeline.stage(combination="combo-b")

    assert staged.combination == "combo-b"
    assert staged.ghost == (1.0, 2.0), "staging the combination must not disturb the ghost"


def test_a_ghost_must_be_a_pair() -> None:
    with pytest.raises(IntentError, match=r"must be \(x, y\)"):
        _timeline().stage(ghost=(1.0, 2.0, 3.0))  # type: ignore[arg-type]


def test_a_ghost_must_be_finite() -> None:
    """A client sends this. NaN would propagate into `warp()` and out into physics as a fighter
    that simply vanishes, which is a very hard bug to read backwards."""
    with pytest.raises(IntentError, match="must be finite"):
        _timeline().stage(ghost=(float("nan"), 0.0))


# --- no cancellation, of anything - unchanged since 1.0 -----------------------------------------
def test_staging_after_a_commit_does_not_reach_back_into_it() -> None:
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    queued = timeline.commit(now=0)

    timeline.stage(combination="combo-b", ghost=(9.0, 9.0))

    assert queued.record.name == "combo-a", "staging a new combination must not re-aim a queued one"
    assert queued.ghost == (1.0, 0.0)
    assert queued in timeline.scheduled(0)


# --- admission is enforced at construction (unchanged in rule, extended in scope at 3.0) ---------
def test_a_match_refuses_an_unadmitted_combination() -> None:
    draft = _record("combo-a")
    draft = CombinationRecord(
        name=draft.name,
        library_version=draft.library_version,
        source=draft.source,
        keyframes=draft.keyframes,
        admission="draft",
    )
    with pytest.raises(IntentError, match="is 'draft'"):
        IntentTimeline({"combo-a": draft})


def test_the_studio_may_use_an_unadmitted_combination() -> None:
    draft = CombinationRecord(
        name="combo-a",
        library_version="v0.2",
        source=CombinationSource("t", 0, 100, False),
        keyframes=_record("combo-a").keyframes,
        admission="draft",
    )
    timeline = IntentTimeline({"combo-a": draft}, require_admitted=False)
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    assert timeline.commit(now=0).record.name == "combo-a"


def test_an_empty_library_is_rejected() -> None:
    with pytest.raises(IntentError, match="empty"):
        IntentTimeline({})


def test_at_most_one_commit_executes_at_a_time() -> None:
    """Ported from 2.2. Two moves running at once would drive one fighter from two intents.

    The hand-over test above proves there is no *gap* between queued moves; this proves there is no
    *overlap*. They are different failures and the queue has to avoid both.
    """
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    first = timeline.commit(now=0)
    timeline.stage(combination="combo-b", ghost=(0.5, 0.0))
    second = timeline.commit(now=1)

    anchor = _anchor()
    for tick in range(400):
        timeline.generator_intent(tick, anchor=anchor)
        executing = [c for c in (first, second) if c.is_executing(tick)]
        assert len(executing) <= 1, f"tick {tick}: {len(executing)} commits executing at once"


def test_a_queued_commit_cannot_be_taken_back() -> None:
    """No cancellation, of anything, including a queued move. That rule is the game.

    Ported from 2.2: the timeline exposes no way to remove or replace a commit, and the only thing
    that frees a queue slot is a move finishing.
    """
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    timeline.commit(now=0)

    assert not any(
        hasattr(timeline, name) for name in ("cancel", "remove", "clear_commits", "replace")
    ), "the timeline grew a way to take a commit back"

    # Restaging cannot reach a queued commit either.
    timeline.stage(combination="combo-b", ghost=(9.0, 9.0))
    assert timeline.commits[0].record.name == "combo-a"
    assert timeline.commits[0].ghost == (1.0, 0.0)


# --- the live bearing (owner, 2026-09-03: a fighter always faces its opponent) --------------------
def test_the_bearing_reaches_an_executing_commit() -> None:
    """The timeline no longer swallows ``facing_angle`` once a commit is current: the world tells it
    where the opponent is every tick, and that is what the fighter must face (reversing D5)."""
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    timeline.commit(now=0)
    anchor = _anchor(heading=0.7)

    for tick in range(COMMIT_HORIZON_TICKS):
        timeline.generator_intent(tick, facing_angle=1.25, anchor=anchor)
    intent = timeline.generator_intent(COMMIT_HORIZON_TICKS, facing_angle=1.25, anchor=anchor)

    assert intent.facing_angle == pytest.approx(1.25)
    assert intent.target_heading == pytest.approx(1.25)


def test_the_bearing_reaches_a_held_final_leg() -> None:
    """Holding a pose is the same leg re-armed, so a fighter standing in its last pose keeps turning
    to face the opponent rather than freezing on the heading it happened to finish with."""
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    timeline.commit(now=0)
    anchor = _anchor(heading=0.7)
    for tick in range(400):
        timeline.generator_intent(tick, facing_angle=0.0, anchor=anchor)

    held = timeline.generator_intent(500, facing_angle=-0.9, anchor=anchor)
    assert held.facing_angle == pytest.approx(-0.9)
    assert held.target_heading == pytest.approx(-0.9)
