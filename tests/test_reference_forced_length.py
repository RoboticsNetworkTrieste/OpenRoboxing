"""Regression tests for the three defects forced plan lengths caused under `spec/intent.md` 1.1.

`runtime/reference.py`'s module docstring is the authority for what each defect is; this file's
docstrings paraphrase it. `spec/intent.md` 2.0 deleted the machinery that caused them
(`_committed_frame`, `_committed_plan_length`, `_plan_key`, `MAX_HELD_STRIKE_FRAMES`) by moving to
`horizon_tokens=None` — the model chooses its own plan length, so there is nothing to force and
nothing to lose count of. Phase 2 (`CombinationRunner.intent_for`, `runtime/sequence.py`)
reintroduces forced lengths so that a combination lasts exactly as long as its recording, which makes
all three defects live risks again. These tests do not hunt for a new bug; a phase-1 spike already
measured plan length honoured exactly on 30 of 30 real legs. The point is that these three must *stay*
fixed as the surrounding code keeps changing.

Mechanism, not a match
-----------------------
None of this needs the real generator or a GPU. `_MarkerGenerator` stands in for
`MotionBricksGenerator`: instead of producing real MuJoCo poses, it writes a scalar "which intent
produced this frame" marker into the qpos root-position x slot (`bridge._ROOT_POS`, index 0) and
holds the identity quaternion everywhere else so the frames stay valid input to
`bridge.resample_qpos`. Driving a real `ReferenceStream` with this stub exercises the exact frame
accounting and resampling `reference.py` and `bridge.py` do in production — the only thing that
differs from a real run is what the numbers in qpos mean.

`intent_at` callables here return real `GeneratorIntent` instances (matching production), but with the
marker riding on `target_position[0]` rather than a real `(x, z)` placement — `ReferenceStream` never
interprets `target_position` itself, it is opaque cargo forwarded straight to the generator.

Conventions
-----------
- Ticks are 50 Hz control ticks, matching `TICK_HZ`. Token-to-tick conversion mirrors
  `CombinationRunner`'s own rounding (`sequence._tokens_to_ticks`): ``round(tokens * SECONDS_PER_TOKEN
  * TICK_HZ)``, duplicated here rather than imported since it is one line and importing a private
  helper from another test's module under test would couple this file to `sequence.py`'s internals
  for no benefit.
- "Tick alignment" is swept by varying both the tick a leg starts at (`commit_at`) and its token count
  (`tokens`), since the intent-switch machinery (`ReferenceStream.tick_of_frame`,
  `frame_cursor`) is sensitive to the *absolute* generator-frame index, not the leg's own local
  frame count.
"""

from __future__ import annotations

import numpy as np

from openroboxing.runtime.generator import GeneratorIntent
from openroboxing.runtime.reference import ReferenceStream
from openroboxing.spec.constants import MAX_TOKENS, MIN_TOKENS, QPOS_DIM, SECONDS_PER_TOKEN, TICK_HZ

#: Root position x (`bridge._ROOT_POS` starts at 0) carries the marker; everything else is inert.
_MARKER_INDEX = 0

#: `commit_at` values swept for every token count, to drive the tick alignment across every residue
#: of the GENERATOR_HZ/TICK_HZ = 30/50 = 3/5 cycle (period 3 in frames), several times over.
_COMMIT_AT_OFFSETS = range(15)

#: The full range of plan lengths MotionBricks can be asked for (`spec/constants.py`), so the swept
#: alignments include the 8-token case `reference.py`'s docstring measured directly.
_TOKEN_COUNTS = range(MIN_TOKENS, MAX_TOKENS + 1)

#: Ticks after a leg boundary a real transition is allowed to take before `motion` must show the new
#: leg's marker and hold it. Measured worst case over the full sweep below is 3 ticks; this leaves
#: margin without being so loose it would forgive a real multi-tick stall.
_SETTLE_TOLERANCE_TICKS = 5


def _tokens_to_ticks(tokens: int) -> int:
    """Tokens to ticks, `CombinationRunner`'s own rounding (`sequence._tokens_to_ticks`)."""
    return round(tokens * SECONDS_PER_TOKEN * TICK_HZ)


class _MarkerGenerator:
    """Stand-in for `MotionBricksGenerator` that encodes "which intent produced this frame" as a
    number instead of producing a real pose.

    Mirrors `ReferenceStream.ensure`'s own call order: it asks `intent_at` for the tick the frame
    *about to be appended* is for, pops `next_frame()`, and only then calls `generate()` with that
    intent. So a `generate()` call's marker never affects the frame just popped — only the one after
    — exactly like a real generator, whose replan changes its *future* output, not a frame already
    queued. This is the most generous case for a real generator, since upstream's own cadence gate
    only replans every `dt` seconds; if a leak or a dropped frame shows up under this idealised
    one-frame lag, a real generator's slower cadence could not do better.
    """

    def __init__(self) -> None:
        self._next_marker = 0.0
        self.forced_calls: list[bool] = []
        self.dts: list[float] = []

    def next_frame(self) -> np.ndarray:
        qpos = np.zeros(QPOS_DIM)
        qpos[3] = 1.0  # identity quaternion (w=1), so `bridge._check_qpos`/slerp see valid input
        qpos[_MARKER_INDEX] = self._next_marker
        return qpos

    def generate(self, intent: GeneratorIntent, context_qpos: np.ndarray, dt: float, *, force: bool = False) -> None:
        self.forced_calls.append(force)
        self.dts.append(dt)
        self._next_marker = intent.target_position[0]

    def context_qpos(self) -> np.ndarray:
        return np.zeros((1, QPOS_DIM))


def _marker_intent(marker: float) -> GeneratorIntent:
    """A `GeneratorIntent` carrying `marker` as opaque cargo on `target_position[0]`."""
    return GeneratorIntent(target_position=(marker, 0.0), target_heading=0.0)


def test_defect1_final_frame_is_not_lost_at_any_tick_alignment() -> None:
    """Defect 1 — `reference.py`'s docstring: "an 8-token pose losing its final frame (the authored
    pose) on 20 % of tick alignments". Caused by the forced-plan machinery `spec/intent.md` 2.0
    deleted (`_committed_frame`, `_committed_plan_length`); phase 2's `CombinationRunner.intent_for`
    reintroduces forced `horizon_tokens`, so a leg's final generator frame — the frame the resampled
    tick stream must show at the leg's last tick — can drop again if `bridge.resample_qpos` or the
    frame/tick accounting in `ReferenceStream.ensure` regresses.

    For every leg length MotionBricks can produce (6..16 tokens) and every starting tick offset in a
    sweep well past the 3-frame/5-tick alignment cycle, the tick immediately before the leg's boundary
    must show *exactly* the leg's own marker — not a blend with whatever comes next.
    """
    failures = []
    for commit_at in _COMMIT_AT_OFFSETS:
        for tokens_a in _TOKEN_COUNTS:
            boundary = commit_at + _tokens_to_ticks(tokens_a)

            def intent_at(tick: int, boundary: int = boundary) -> GeneratorIntent:
                return _marker_intent(1.0 if tick < boundary else 2.0)

            generator = _MarkerGenerator()
            stream = ReferenceStream(generator)
            stream.ensure(intent_at, tick=boundary + _tokens_to_ticks(8) + 20, ticks_ahead=10)

            last_of_leg = stream.motion[boundary - 1, _MARKER_INDEX]
            if last_of_leg != 1.0:
                failures.append((commit_at, tokens_a, boundary, last_of_leg))

    assert not failures, (
        f"the leg's own final tick was contaminated on {len(failures)}/"
        f"{len(_COMMIT_AT_OFFSETS) * len(_TOKEN_COUNTS)} alignments "
        f"(commit_at, tokens, boundary, contaminated value): {failures[:10]}"
    )


def test_defect2_no_frame_leaks_into_the_next_commits_first_tick() -> None:
    """Defect 2 — `reference.py`'s docstring: "that lost frame then playing into the *next* commit's
    approach". Under the deleted 1.1 machinery, the frame defect 1 dropped from one commit's plan
    was not discarded — it was still sitting in the generator's queue, and got played back as if it
    belonged to the *following* commit's opening frames, contaminating the next move's start with the
    previous move's end. `spec/intent.md` 2.0 deleted the bookkeeping that made this possible
    (`_plan_key` et al.); phase 2's forced lengths reintroduce the forced plan that could leave such a
    straggler frame behind.

    For the same sweep as defect 1: once the tick stream reaches the *next* leg's marker, it must
    reach it quickly (within `_SETTLE_TOLERANCE_TICKS`) and never fall back towards the previous leg's
    marker afterwards - a reversion would mean a stale frame from the old leg was played late.
    """
    failures = []
    for commit_at in _COMMIT_AT_OFFSETS:
        for tokens_a in _TOKEN_COUNTS:
            boundary = commit_at + _tokens_to_ticks(tokens_a)

            def intent_at(tick: int, boundary: int = boundary) -> GeneratorIntent:
                return _marker_intent(1.0 if tick < boundary else 2.0)

            generator = _MarkerGenerator()
            stream = ReferenceStream(generator)
            tail = 40
            stream.ensure(intent_at, tick=boundary + tail, ticks_ahead=10)
            column = stream.motion[:, _MARKER_INDEX]

            settle_window_end = min(boundary + _SETTLE_TOLERANCE_TICKS, len(column))
            settled_at = next(
                (t for t in range(boundary, settle_window_end) if column[t] == 2.0), None
            )
            if settled_at is None:
                failures.append(("never settled", commit_at, tokens_a, boundary))
                continue

            # Exclude the last few ticks: they sit in resample_qpos's tail, where fewer frames are
            # available ahead and the values are not final yet - not the "next commit" this defect
            # is about.
            after = column[settled_at : len(column) - 5]
            if not np.all(after == 2.0):
                failures.append(("reverted after settling", commit_at, tokens_a, boundary))

    assert not failures, f"a leg leaked into or reverted within the next leg's window: {failures[:10]}"


def test_defect3_end_of_move_replan_uses_the_ambient_cadence() -> None:
    """Defect 3 — `reference.py`'s docstring: "the end-of-strike replan bypassing the ambient
    cadence". Under the deleted 1.1 machinery, reaching a plan's end forced an immediate replan
    instead of waiting for `REPLAN_DT`, so the end of a move got a differently-timed plan from its
    middle. `spec/intent.md` 2.0 deleted that special case: `ReferenceStream.ensure` now calls
    `_plan` with `force=False` unconditionally, on every frame, regardless of whether the intent just
    changed. Phase 2's forced-length legs reintroduce a natural place to want "finish this leg now" —
    which is exactly the temptation defect 3 is a regression test against.

    Drives a `CombinationRunner`-shaped chain of six legs (varied token counts, so several different
    tick alignments occur across the boundaries) through one continuous `ensure` call and asserts
    every `generate()` call - including the ones immediately after a leg boundary is crossed - was
    made with `force=False` and the ambient `replan_dt`.
    """
    tokens = [6, 11, 8, 16, 6, 9]
    boundaries: list[int] = []
    cumulative = 0
    for leg_tokens in tokens:
        cumulative += _tokens_to_ticks(leg_tokens)
        boundaries.append(cumulative)
    markers = [float(i + 1) for i in range(len(tokens))]

    def intent_at(tick: int) -> GeneratorIntent:
        for boundary, marker in zip(boundaries, markers):
            if tick < boundary:
                return _marker_intent(marker)
        return _marker_intent(markers[-1])

    generator = _MarkerGenerator()
    replan_dt = 0.5
    stream = ReferenceStream(generator, replan_dt=replan_dt)
    stream.ensure(intent_at, tick=boundaries[-1] + 20, ticks_ahead=10)

    assert generator.forced_calls, "the sweep must actually drive some generate() calls"
    assert not any(generator.forced_calls), (
        f"a replan was forced (bypassing the {replan_dt}s ambient cadence) on "
        f"{sum(generator.forced_calls)}/{len(generator.forced_calls)} calls"
    )
    assert all(dt == replan_dt for dt in generator.dts), (
        "a replan used a dt other than the ambient replan_dt - the end-of-move case must use the "
        "same cadence as every other replan"
    )
