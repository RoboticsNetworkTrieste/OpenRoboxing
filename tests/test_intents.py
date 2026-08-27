"""M2-T4 acceptance, pruned for `spec/intent.md` 3.0 (M6-T5).

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_intents.py -v

At 3.0 a commit stopped being *a placement and a final pose walked to* and became *a recorded
combination played in place, landing on a ghost*. Everything this file tested about the approach —
arrival, the counted dwell, `Placement`, the live adjustment envelope, and pose-slot staging — is
gone with the model it tested, and is deleted below rather than left to bit-rot against an API that
no longer exists. What is kept is what still applies at 3.0: queue capacity, backwards-tick
rejection, and the horizon floor, ported to the new `combination` / `ghost` staging channels. Full
3.0 coverage — combinations, ghosts, `commit_at` / `end_tick` arithmetic, the hand-over rule, the two
hold states, and the `anchor` contract — lives in `tests/test_intents_combinations.py`.

A handful of tests below (the `narrow_allowed_tokens` and `plan_key` ones) exercise
`runtime/generator.py`, not this module's rewrite, and are untouched because nothing about them
changed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from openroboxing.paths import OPENROBOXING_ROOT
from openroboxing.runtime.conventions import G1
from openroboxing.runtime.intents import (
    SPEC_VERSION,
    IntentError,
    IntentTimeline,
    StagedIntent,
)
from openroboxing.spec.constants import COMMIT_HORIZON_TICKS, MAX_OUTSTANDING_COMMITS
from openroboxing.studio.combination_record import (
    CombinationRecord,
    CombinationSource,
    Keyframe,
)

ANGLES = {name: 0.0 for name in G1.mujoco_joint_names}


def _record(name: str, *, tokens=(6, 6)) -> CombinationRecord:
    keyframes = [Keyframe(dict(ANGLES), None, (0.0, 0.0), 0.0)]
    for i, token in enumerate(tokens, start=1):
        keyframes.append(Keyframe(dict(ANGLES), token, (0.1 * i, 0.0), 0.0))
    return CombinationRecord(
        name=name,
        library_version="v0.2",
        source=CombinationSource("t", 0, 100, False),
        keyframes=keyframes,
        telegraph_ms=180.0,
        tracking_error_rad=0.1,
        admission="admitted",
    )


def _library() -> dict[str, CombinationRecord]:
    return {"combo-a": _record("combo-a"), "combo-b": _record("combo-b", tokens=(7, 9))}


def _timeline(**kwargs) -> IntentTimeline:
    return IntentTimeline(_library(), **kwargs)


def _origin() -> tuple[tuple[float, float], float]:
    return (0.0, 0.0), 0.0


def _drive(timeline: IntentTimeline, through: int, *, start: int = 0) -> None:
    """Tick the timeline from ``start`` to ``through``, anchoring every commit at the origin."""
    for tick in range(start, through):
        timeline.generator_intent(tick, anchor=_origin)


# --- the spec pairing (kept: this is the test that catches a version bump without a rewrite) -----
def test_the_spec_is_versioned() -> None:
    assert SPEC_VERSION == "3.0"
    spec = Path(OPENROBOXING_ROOT / "spec/intent.md").read_text()
    assert re.search(rf"Version \*\*{re.escape(SPEC_VERSION)}\*\*", spec), (
        "spec/intent.md does not declare the version runtime/intents.py implements"
    )


# --- queue capacity (spec/intent.md 1.0, unchanged in rule at 3.0) -------------------------------
def test_the_queue_holds_exactly_the_bound() -> None:
    timeline = _timeline()
    for index in range(MAX_OUTSTANDING_COMMITS):
        timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
        timeline.commit(now=0)
        assert len(timeline.scheduled(0)) == index + 1
    assert len(timeline.commits) == MAX_OUTSTANDING_COMMITS


def test_the_bound_is_configurable_and_must_be_at_least_one() -> None:
    timeline = _timeline(max_outstanding=1)
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    timeline.commit(now=0)
    with pytest.raises(IntentError, match="already queued"):
        timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
        timeline.commit(now=1)

    with pytest.raises(IntentError, match="at least 1"):
        _timeline(max_outstanding=0)


def test_a_new_commit_is_allowed_once_the_last_one_ends() -> None:
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    first = timeline.commit(now=0)
    _drive(timeline, 400)

    timeline.stage(combination="combo-b", ghost=(1.0, 0.0))
    second = timeline.commit(now=first.end_tick)
    assert second.record.name == "combo-b"
    assert len(timeline.commits) == 2


# --- the horizon floor (spec/intent.md 1.0, unchanged in rule at 3.0) ----------------------------
def test_the_horizon_is_a_floor_not_a_gap_between_queued_moves() -> None:
    """A commit into an empty queue pays the full 0.6 s. One queued behind a running move pays
    nothing extra, because the readable window has already elapsed while the first move played.
    Inserting a fixed pause between queued moves would stutter every combination."""
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    first = timeline.commit(now=0)
    timeline.stage(combination="combo-b", ghost=(1.0, 0.0))
    second = timeline.commit(now=1)
    _drive(timeline, 400)

    assert first.commit_at == COMMIT_HORIZON_TICKS
    assert second.commit_at == first.end_tick, "queued moves run back to back"
    assert second.commit_at > second.issued_at + COMMIT_HORIZON_TICKS


def test_a_commit_into_a_drained_queue_still_pays_the_horizon() -> None:
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    first = timeline.commit(now=0)
    _drive(timeline, 200)

    timeline.stage(combination="combo-b", ghost=(1.0, 0.0))
    later = timeline.commit(now=first.end_tick + 500)
    _drive(timeline, later.issued_at + 400, start=200)
    assert later.commit_at == later.issued_at + COMMIT_HORIZON_TICKS


# --- backwards ticks (unchanged in rule at 3.0) ---------------------------------------------------
def test_the_timeline_does_not_run_backwards() -> None:
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    timeline.commit(now=200)
    with pytest.raises(IntentError, match="does not run backwards"):
        timeline.commit(now=100)


def test_a_negative_tick_is_rejected() -> None:
    timeline = _timeline()
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    with pytest.raises(IntentError, match="must not be negative"):
        timeline.commit(now=-1)


# --- staging: still frozen once read (unchanged in rule at 3.0) ----------------------------------
def test_the_staged_intent_is_immutable_once_read() -> None:
    timeline = _timeline()
    staged = timeline.stage(combination="combo-a")
    timeline.stage(combination="combo-b")
    assert staged.combination == "combo-a", "a previously returned StagedIntent must not change"
    assert isinstance(staged, StagedIntent)


# --- the horizon reaches the generator (runtime/generator.py; untouched by this rewrite) ----------
def test_a_horizon_narrows_the_clip_mask_to_one_length() -> None:
    from openroboxing.runtime.generator import narrow_allowed_tokens
    from openroboxing.spec.constants import MIN_TOKENS, NUM_TIME_TOKENS

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
    from openroboxing.spec.constants import MAX_TOKENS, NUM_TIME_TOKENS

    with pytest.raises(GeneratorError, match="outside"):
        narrow_allowed_tokens([1] * NUM_TIME_TOKENS, MAX_TOKENS + 1)


def test_a_clip_mask_of_the_wrong_length_raises() -> None:
    from openroboxing.runtime.generator import GeneratorError, narrow_allowed_tokens

    with pytest.raises(GeneratorError, match="declares 3 token slots"):
        narrow_allowed_tokens([1, 1, 1], 8, style="broken")


def test_the_plan_key_is_gone() -> None:
    """It existed only to bind a forced plan to the commit that forced it. Since 2.0 nothing at the
    commit level is forced, so it is deleted rather than left carrying a value nobody reads
    (`CLAUDE.md` prefers deleting to disabling). Unaffected by the 3.0 rewrite - `GeneratorIntent`
    lives in `generator.py`, not here."""
    from openroboxing.runtime.generator import GeneratorIntent

    assert "plan_key" not in GeneratorIntent.__dataclass_fields__
    with pytest.raises(TypeError):
        GeneratorIntent(plan_key=1)
