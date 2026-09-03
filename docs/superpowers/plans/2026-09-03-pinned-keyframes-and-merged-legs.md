# Pinned Keyframes and Merged Legs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a recorded keyframe land on its recorded tick by pinning it in absolute time and shrinking the requested plan horizon toward it, then halve the library's keyframes so legs run 2.27 s instead of 1.20 s.

**Architecture:** Two halves, sequenced. Half 1 changes `CombinationRunner.intent_for` to request the *remaining* hole (`ceil`, clamped to `[MIN_TOKENS, MAX_TOKENS]`) instead of the leg's full length, drops the pose target while the hole is bigger than any plan, and stops replanning once the hole is smaller than the shortest plan. `GeneratorIntent` carries a `replan` flag that `ReferenceStream` honours. Half 2 thins the segmenter's keyframes to sparse targets keeping the first and last punch, and moves three bounds that still assume "a leg is one plan" onto a new `MAX_TARGET_LEG_TOKENS`. A `DRIFT_GAIN` re-measurement sits between them so the schedule change and the library change are not confounded.

**Tech Stack:** Python 3.10+, numpy, pytest, MuJoCo, MotionBricks (upstream, unmodified). Interpreter is `.venv_mb/bin/python`.

**Spec:** `docs/superpowers/specs/2026-09-03-pinned-keyframes-and-merged-legs-design.md`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/openroboxing/spec/constants.py` | canonical constants | add `MIN_TARGET_GAP_FRAMES`, `MAX_TARGET_LEG_FRAMES`, `MAX_TARGET_LEG_TOKENS`; change `COMBINATION_MIN/MAX_KEYFRAMES`; rewrite `MAX_LEG_FRAMES` docstring |
| `src/openroboxing/runtime/generator.py` | `GeneratorIntent` | add `replan: bool = True` |
| `src/openroboxing/runtime/reference.py` | pull/resample/stay ahead | honour `intent.replan` |
| `src/openroboxing/runtime/sequence.py` | which leg is live, and its intent | pinned-keyframe horizon arithmetic |
| `src/openroboxing/studio/segment.py` | take → keyframes | return punch provenance; add `thin_targets`; move leg cap |
| `src/openroboxing/studio/combination_record.py` | record schema + build | validate against `MAX_TARGET_LEG_TOKENS`; thin in `build_from_take` |
| `tests/test_sequence_pinned.py` | **new** — Half 1 runner arithmetic | created Task 3 |
| `tests/test_reference_replan_flag.py` | **new** — stream honours the flag, context integrity | created Tasks 2, 4 |
| `tests/test_segment_thinning.py` | **new** — Half 2 thinning rule | created Task 7 |

`sequence.py` is 158 lines and stays focused; no split needed. `segment.py` gains one function and one return-value change, staying inside its single responsibility (turning a take into keyframes).

---

## Phase 1 — Runtime: pin the keyframe

### Task 1: `GeneratorIntent` carries the replan decision

**Files:**
- Modify: `src/openroboxing/runtime/generator.py:260-285` (the `GeneratorIntent` dataclass)

- [ ] **Step 1: Add the field**

In `src/openroboxing/runtime/generator.py`, add to the `GeneratorIntent` docstring's `Attributes:`
block, immediately after the `horizon_tokens:` entry:

```
        replan: whether the reference stream should ask the generator to plan on this frame at all.
            ``False`` means the hole between the context and the pinned keyframe is shorter than
            ``MIN_TOKENS``, the shortest plan the model can produce — there is nothing left to fill,
            and re-filling would only push the keyframe past its boundary. See
            :meth:`~openroboxing.runtime.sequence.CombinationRunner.intent_for`.
```

and add the field itself, after `horizon_tokens`:

```python
    horizon_tokens: int | None = None
    replan: bool = True
```

- [ ] **Step 2: Verify nothing broke**

Run: `.venv_mb/bin/python -m pytest tests/ -q --no-header 2>&1 | tail -3`
Expected: same pass count as before the change (the field is additive with a default).

- [ ] **Step 3: Commit**

```bash
git add src/openroboxing/runtime/generator.py
git commit -m "M7-T1: GeneratorIntent carries the replan decision"
```

---

### Task 2: `ReferenceStream` honours the flag

**Files:**
- Modify: `src/openroboxing/runtime/reference.py:132-134`
- Create: `tests/test_reference_replan_flag.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reference_replan_flag.py`:

```python
"""The reference stream must not plan on a frame whose intent says there is no hole to fill."""

from __future__ import annotations

import numpy as np

from openroboxing.runtime.generator import GeneratorIntent
from openroboxing.runtime.reference import ReferenceStream
from openroboxing.spec.constants import QPOS_DIM


class _CountingGenerator:
    """Counts generate() calls; frames are inert but valid input to `bridge.resample_qpos`."""

    def __init__(self) -> None:
        self.calls = 0

    def next_frame(self) -> np.ndarray:
        qpos = np.zeros(QPOS_DIM)
        qpos[3] = 1.0  # identity quaternion, so the resampler's slerp sees valid input
        return qpos

    def generate(self, intent, context_qpos, dt, *, force: bool = False) -> None:
        self.calls += 1

    def context_qpos(self) -> np.ndarray:
        return np.zeros((1, QPOS_DIM))


def test_replan_false_suppresses_every_generate_call() -> None:
    generator = _CountingGenerator()
    stream = ReferenceStream(generator)
    stream.ensure(lambda _tick: GeneratorIntent(replan=False), tick=100, ticks_ahead=10)
    assert generator.calls == 0


def test_replan_true_still_plans() -> None:
    generator = _CountingGenerator()
    stream = ReferenceStream(generator)
    stream.ensure(lambda _tick: GeneratorIntent(replan=True), tick=100, ticks_ahead=10)
    assert generator.calls > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv_mb/bin/python -m pytest tests/test_reference_replan_flag.py -q --no-header 2>&1 | tail -5`
Expected: `test_replan_false_suppresses_every_generate_call` FAILS (`assert 150 == 0` or similar);
`test_replan_true_still_plans` passes.

- [ ] **Step 3: Implement**

In `src/openroboxing/runtime/reference.py`, replace the body of the fill loop's tail. Find:

```python
            intent = intent_at(self.tick_of_frame(len(self._frames)))
            self._frames.append(self.generator.next_frame())
            self._plan(intent, force=False)
```

Replace with:

```python
            intent = intent_at(self.tick_of_frame(len(self._frames)))
            self._frames.append(self.generator.next_frame())
            # A leg whose remaining hole is shorter than the shortest plan the model can produce has
            # nothing left to fill; replanning would only re-aim its keyframe past its own boundary.
            # The stream still knows nothing about legs — the intent carries the decision.
            if intent.replan:
                self._plan(intent, force=False)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv_mb/bin/python -m pytest tests/test_reference_replan_flag.py tests/test_reference_forced_length.py -q --no-header 2>&1 | tail -3`
Expected: all PASS. The three forced-length regressions still pass — skipping a call does not violate
an assertion about the calls that are made.

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/runtime/reference.py tests/test_reference_replan_flag.py
git commit -m "M7-T2: the reference stream honours the replan decision"
```

---

### Task 3: `CombinationRunner` pins the keyframe

**Files:**
- Modify: `src/openroboxing/runtime/sequence.py`
- Create: `tests/test_sequence_pinned.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sequence_pinned.py`:

```python
"""The keyframe is pinned in absolute time; the requested horizon shrinks toward it.

`spec/intent.md`: time in MotionBricks is a continuous array filled where there are holes, and a
keyframe placed in it stays put while the array moves forward. These tests assert the three regimes
that follow — no target while the hole is bigger than any plan, an exact in-between while it fits,
and no replan at all once it is smaller than the shortest plan.
"""

from __future__ import annotations

import math

from openroboxing.runtime import sequence, warp
from openroboxing.runtime.conventions import G1
from openroboxing.spec.constants import (
    MAX_TOKENS,
    MIN_TOKENS,
    SECONDS_PER_TOKEN,
    TICK_HZ,
)
from openroboxing.studio import combination_record as cr

ANGLES = {name: 0.0 for name in G1.mujoco_joint_names}
TICKS_PER_TOKEN = SECONDS_PER_TOKEN * TICK_HZ


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


def test_the_horizon_shrinks_as_the_keyframe_is_approached():
    """The defect this fixes: the old code asked for the leg's full length every replan, so the
    keyframe was re-aimed 15 frames further out every 0.5 s and never arrived."""
    run, _ = runner([12])
    boundary = run.end_tick
    horizons = [
        run.intent_for(tick).horizon_tokens
        for tick in range(0, boundary)
        if run.intent_for(tick).replan
    ]
    assert horizons == sorted(horizons, reverse=True), horizons
    assert horizons[0] <= MAX_TOKENS


def test_the_implied_landing_tick_stays_put():
    """Pinning, stated directly: request R tokens at tick T and the plan ends at T + R * ticks-per-
    token. That sum must stay within 3 ticks of the leg's boundary for every replan."""
    run, _ = runner([12])
    boundary = run.end_tick
    for tick in range(0, boundary):
        intent = run.intent_for(tick)
        if not intent.replan or intent.pose is None:
            continue
        landing = tick + intent.horizon_tokens * TICKS_PER_TOKEN
        assert abs(landing - boundary) <= 3, (tick, intent.horizon_tokens, landing, boundary)


def test_no_replan_inside_the_final_min_tokens():
    run, _ = runner([12])
    boundary = run.end_tick
    floor_tick = boundary - MIN_TOKENS * TICKS_PER_TOKEN
    for tick in range(math.ceil(floor_tick) + 1, boundary):
        assert run.intent_for(tick).replan is False, tick


def test_a_long_leg_has_no_pose_target_until_the_keyframe_is_reachable():
    """55% of the rebuilt library's legs exceed MAX_TOKENS, so this is the majority path."""
    run, _ = runner([24])
    boundary = run.end_tick
    reachable_from = boundary - MAX_TOKENS * TICKS_PER_TOKEN
    assert run.intent_for(0).pose is None
    assert run.intent_for(0).horizon_tokens == MAX_TOKENS
    landed = run.intent_for(math.ceil(reachable_from) + 1)
    assert landed.pose is not None
    assert landed.horizon_tokens <= MAX_TOKENS


def test_the_hold_re_aims_at_the_final_pose_at_min_tokens():
    """Unchanged in behaviour from today: past the end the runner converges on the last pose."""
    run, _ = runner([6, 8])
    held = run.intent_for(run.end_tick + 500)
    assert held.replan is True
    assert held.pose is not None
    assert held.horizon_tokens == MIN_TOKENS


def test_every_horizon_is_a_length_the_model_can_be_asked_for():
    run, _ = runner([6, 24, 9])
    for tick in range(0, run.end_tick + 200):
        intent = run.intent_for(tick)
        if intent.replan:
            assert MIN_TOKENS <= intent.horizon_tokens <= MAX_TOKENS, (tick, intent.horizon_tokens)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv_mb/bin/python -m pytest tests/test_sequence_pinned.py -q --no-header 2>&1 | tail -5`
Expected: FAIL — `intent_for` currently returns `leg.horizon_tokens` constantly, so the shrinking,
the `pose is None` branch and the `replan` flag are all absent.

- [ ] **Step 3: Implement**

In `src/openroboxing/runtime/sequence.py`, add to the imports:

```python
from openroboxing.spec.constants import MAX_TOKENS, MIN_TOKENS, SECONDS_PER_TOKEN, TICK_HZ
```

(replacing the existing `from openroboxing.spec.constants import SECONDS_PER_TOKEN, TICK_HZ`).

Add this module-level helper immediately after `_tokens_to_ticks`:

```python
def _ticks_to_tokens(ticks: float) -> int:
    """Ticks to tokens, rounded **up**.

    Up, not to nearest: a plan that ends a frame short of its boundary leaves the generator's play
    cursor clamped on the plan's last frame, and ``get_context_mujoco_qpos`` then returns four copies
    of it (``full_agent.py:517-521``) — a zero-velocity context that tells the model the fighter is
    frozen. Rounding up overshoots by at most 3 frames instead, which the next leg's replan writes
    over.
    """
    return math.ceil(ticks / (SECONDS_PER_TOKEN * TICK_HZ))
```

and add `import math` to the top-level imports.

Replace the whole of `intent_for` with:

```python
    def intent_for(self, tick: int, facing_angle: float | None = None) -> GeneratorIntent:
        """The live leg's :class:`GeneratorIntent` at ``tick``, aimed at a **pinned** keyframe.

        Carries ``movement_angle`` and ``facing_angle`` through separately - `CLAUDE.md`'s named
        trap is leaving the former at its default, which silently means "straight ahead, always".

        **The keyframe does not move; the hole in front of it shrinks.** MotionBricks fills a hole
        between the 4 context frames and a target at the plan's last token. A leg's keyframe belongs
        at its recorded boundary tick, so the horizon asked for is the distance still to run —
        ``ceil(boundary - tick)`` in tokens — not the leg's full length. Asking for the full length
        every replan (what this did before `spec/intent.md` 3.2) re-aimed the keyframe
        ``REPLAN_DT * GENERATOR_HZ`` frames further out on every replan, so it receded and never
        landed: the "motions broken in pieces" defect.

        Three regimes, each meaning something different:

        - **hole > MAX_TOKENS** — the keyframe is not reachable inside one plan, so nothing is aimed
          at it: ``pose=None`` and a full-length plan of ambient ``walk_boxing`` shaped only by the
          leg's ``target_position``. 55 % of the rebuilt library's legs start here.
        - **MIN_TOKENS <= hole <= MAX_TOKENS** — the real in-between. This is where the recorded pose
          lands, within ±3 ticks of its boundary.
        - **hole < MIN_TOKENS** — no plan that short exists, so ``replan=False`` and the last plan is
          allowed to play out. This is what makes the landing exact rather than overshooting by up to
          ``MIN_TOKENS``.

        Past :attr:`end_tick` there is no future keyframe: the final leg's pose is re-aimed at
        ``MIN_TOKENS`` forever, which is the converge-and-hold behaviour the runtime already had.

        Args:
            facing_angle: the bearing to the opponent, world frame, measured this tick. It replaces
                the recorded heading in **both** places a heading reaches the generator - the target
                frame's ``target_heading`` and the ``facing_angle`` control signal - and a leg that
                does not travel (``Leg.is_still``) takes it as its ``movement_angle`` too, since a
                still leg has no direction of its own to travel in. ``None`` means *there is no
                opponent*: the Studio's rehearsal and the warp tools drive a lone fighter, and there
                the recording is the only heading there is.
        """
        index = self.leg_index(tick)
        leg = self._legs[index]
        facing = leg.facing_angle if facing_angle is None else facing_angle
        heading = leg.target_heading if facing_angle is None else facing_angle

        horizon, pose, replan = self._horizon_for(tick, index, leg)

        return GeneratorIntent(
            style=COMBINATION_CONTEXT,
            movement_angle=facing if leg.is_still else leg.movement_angle,
            facing_angle=facing,
            target_position=leg.target_position,
            target_heading=heading,
            pose=pose,
            horizon_tokens=horizon,
            replan=replan,
        )

    def _horizon_for(self, tick: int, index: int, leg: Leg) -> tuple[int, object | None, bool]:
        """``(horizon_tokens, pose, replan)`` for ``tick`` — the three regimes above."""
        if tick >= self.end_tick:
            # Holding: no future keyframe, so re-aim at the final pose with the shortest plan there
            # is. Anything longer would invent motion past a combination that has finished.
            return MIN_TOKENS, self._pose_for(leg, index), True

        remaining = _ticks_to_tokens(self._boundaries[index] - tick)
        if remaining > MAX_TOKENS:
            return MAX_TOKENS, None, True
        if remaining >= MIN_TOKENS:
            return remaining, self._pose_for(leg, index), True
        return MIN_TOKENS, self._pose_for(leg, index), False
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv_mb/bin/python -m pytest tests/test_sequence_pinned.py tests/test_sequence.py -q --no-header 2>&1 | tail -5`
Expected: all PASS. If a `test_sequence.py` test asserts `intent_for(...).horizon_tokens ==
leg.horizon_tokens`, that assertion encodes the defect being fixed — update it to assert the
pinned-horizon behaviour and say so in the commit message.

- [ ] **Step 5: Run the full suite**

Run: `.venv_mb/bin/python -m pytest tests/ -q --no-header 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/openroboxing/runtime/sequence.py tests/test_sequence_pinned.py tests/test_sequence.py
git commit -m "M7-T3: pin the keyframe; the horizon shrinks toward it instead of receding"
```

---

### Task 4: Context integrity — the 4 frames must never be one frame repeated

**Files:**
- Modify: `tests/test_reference_replan_flag.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reference_replan_flag.py`:

```python
class _CursorGenerator:
    """Models upstream's play cursor and its clamp, which is where frozen context comes from.

    `full_agent.get_next_frame` clamps the cursor to the plan's last index, and
    `get_context_mujoco_qpos` reads 4 frames from the cursor with the same clamp
    (`full_agent.py:503-521`). So a plan played to its end hands the next plan four copies of one
    frame — a zero-velocity context that tells the model the fighter is standing still. This stub
    reproduces exactly that behaviour so the runner's `ceil` rounding can be shown to avoid it.
    """

    def __init__(self) -> None:
        self._plan = self._make(6, 0.0)
        self._cursor = 0
        self._tick = 0.0

    @staticmethod
    def _make(tokens: int, start: float) -> np.ndarray:
        frames = np.zeros((tokens * 4, QPOS_DIM))
        frames[:, 3] = 1.0
        frames[:, 0] = start + np.arange(tokens * 4)  # strictly increasing: real motion
        return frames

    def next_frame(self) -> np.ndarray:
        frame = self._plan[self._cursor]
        self._cursor = max(0, min(self._cursor + 1, self._plan.shape[0] - 1))
        return frame

    def generate(self, intent, context_qpos, dt, *, force: bool = False) -> None:
        tokens = intent.horizon_tokens or 6
        self._tick += 1.0
        self._plan = self._make(tokens, self._tick * 1000.0)
        self._cursor = 0

    def context_qpos(self) -> np.ndarray:
        last = self._plan.shape[0] - 1
        rows = [self._plan[min(self._cursor + i, last)] for i in range(4)]
        return np.asarray(rows)


def test_the_context_is_never_one_frame_repeated() -> None:
    """A frozen context is silent: the fighter keeps moving, but the model plans as if it were
    standing still. Asserted across a whole combination including its hold."""
    from openroboxing.runtime import sequence, warp
    from openroboxing.runtime.conventions import G1
    from openroboxing.studio import combination_record as cr

    angles = {name: 0.0 for name in G1.mujoco_joint_names}
    keyframes = [cr.Keyframe(dict(angles), None, (0.0, 0.0), 0.0)]
    for i, token in enumerate([6, 24, 9], start=1):
        keyframes.append(cr.Keyframe(dict(angles), token, (0.1 * i, 0.0), 0.0))
    rec = cr.CombinationRecord(
        name="c", library_version="v0.2",
        source=cr.CombinationSource("t", 0, 100, False), keyframes=keyframes,
    )
    legs = warp.warp(rec, (0.0, 0.0), 0.0, rec.recorded_displacement)
    run = sequence.CombinationRunner(rec, legs, commit_at=0)

    generator = _CursorGenerator()
    stream = ReferenceStream(generator)
    frozen: list[int] = []

    original = generator.generate

    def watched(intent, context_qpos, dt, *, force: bool = False):
        if len(np.unique(np.asarray(context_qpos)[:, 0])) == 1:
            frozen.append(len(stream._frames))
        original(intent, context_qpos, dt, force=force)

    generator.generate = watched
    stream.ensure(run.intent_for, tick=run.end_tick + 100, ticks_ahead=10)

    assert not frozen, f"the context collapsed to a single repeated frame at frames {frozen[:10]}"
```

- [ ] **Step 2: Run to verify**

Run: `.venv_mb/bin/python -m pytest tests/test_reference_replan_flag.py -q --no-header 2>&1 | tail -5`
Expected: PASS. This is a regression test, not a bug hunt — Task 3's `ceil` rounding is what makes it
pass. To confirm it has teeth, temporarily change `_ticks_to_tokens` to use `math.floor`, re-run and
see it FAIL, then change it back.

- [ ] **Step 3: Commit**

```bash
git add tests/test_reference_replan_flag.py
git commit -m "M7-T4: regression test - the generator's context is never one frame repeated"
```

---

### Task 5: Re-measure `DRIFT_GAIN` under the new schedule

**Files:**
- Modify: `src/openroboxing/tools/measure_drift_gain.py:170-180`
- Modify: `src/openroboxing/spec/constants.py` (the `DRIFT_GAIN` value and docstring)
- Create: `docs/perf/2026-09-03-drift-gain-pinned.md`

This task is **required, not optional**: `warp.py` divides every residual by `DRIFT_GAIN`, and the
0.803 figure was measured with `force=True`, one clean plan per leg — a schedule this design no
longer uses. Measuring it after Half 2 as well would confound the schedule change with the library
change, which is why it sits here.

- [ ] **Step 1: Make the tool drive the real schedule**

In `src/openroboxing/tools/measure_drift_gain.py`, find:

```python
        # force=True: this measures the plan, not the replan schedule (spike_warp_tracking).
        generator.generate(intent, generator.context_qpos(), GENERATOR_DT, force=True)
```

Replace with:

```python
        # force=False and the ambient cadence: since `spec/intent.md` 3.2 the runtime never forces a
        # plan, so a gain measured on forced single plans is a gain for a schedule that no longer
        # exists. `intent.replan` is honoured here for the same reason `ReferenceStream` honours it.
        if intent.replan:
            generator.generate(intent, generator.context_qpos(), REPLAN_DT, force=False)
```

and add `REPLAN_DT` to the imports from `openroboxing.runtime.reference`.

- [ ] **Step 2: Run the measurement**

Run: `.venv_mb/bin/python -m openroboxing.tools.measure_drift_gain --help`
then the full measurement as that help text describes (it is a `slow` path and needs the GPU).
Record the median and range over the same 9 combinations × 4 distances the 2026-08-28 run used, so
the two are comparable.

- [ ] **Step 3: Write the measurement up**

Create `docs/perf/2026-09-03-drift-gain-pinned.md` recording: the command, the per-family medians,
the overall median and range, and an explicit comparison against the 0.803 measured on 2026-08-28.
State whether the change is inside the ±0.10 bar the constant is held to.

- [ ] **Step 4: Update the constant**

In `src/openroboxing/spec/constants.py`, set `DRIFT_GAIN` to the newly measured median and add to its
docstring, after the existing provenance:

```
**Re-measured 2026-09-03** under the pinned-keyframe schedule (`spec/intent.md` 3.2), which replaced
the forced single plan per leg the 2026-08-28 figure was measured on. A gain measured on a schedule
the runtime no longer runs is a gain for nothing; see docs/perf/2026-09-03-drift-gain-pinned.md.
```

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/tools/measure_drift_gain.py src/openroboxing/spec/constants.py docs/perf/2026-09-03-drift-gain-pinned.md
git commit -m "M7-T5: re-measure DRIFT_GAIN under the pinned-keyframe schedule"
```

---

## Phase 2 — Library: half the keyframes, double the legs

### Task 6: The new constants

**Files:**
- Modify: `src/openroboxing/spec/constants.py:47-116`

- [ ] **Step 1: Add and change the constants**

In `src/openroboxing/spec/constants.py`, immediately after `MIN_KEYFRAME_GAP_FRAMES`, add:

```python
MIN_TARGET_GAP_FRAMES: int = 2 * MIN_KEYFRAME_GAP_FRAMES
"""Closest two **targets** may sit, in corpus frames = 48 = 1.6 s.

Not the same quantity as :data:`MIN_KEYFRAME_GAP_FRAMES`, and the two must never be merged.
That one governs **detection** — how close two turning points may be found — and the measured
39/48 punch-capture rate depends on it. This one governs **selection**: which of those detected
poses become hard targets a plan is aimed at. Thinning runs after detection, so raising this does
not re-open the punch-capture measurement.

Derived, not chosen: doubling the detection floor is the smallest change that delivers the owner's
"longer than double" (2026-09-03). Measured over the shipped 130-combination library, it takes the
median leg from 9 tokens (1.20 s) to 17 tokens (2.27 s).
"""

MAX_TARGET_LEG_FRAMES: int = 96
"""Longest leg between two targets, in corpus frames = 24 tokens = 3.2 s.

**This replaces :data:`MAX_LEG_FRAMES` as the thing that caps a leg**, and the distinction is the
whole of `spec/intent.md` 3.2. A leg used to be exactly one plan, so the planner's 16-token maximum
capped it. Since 3.2 a long leg is an *untargeted phase plus a landing plan*, so the planner's
maximum caps a **plan** and no longer caps a **leg**.

The cap does not disappear, though: uncapped, measured legs reach 36 tokens (4.8 s) and a
combination runs past the duration the no-cancellation rule was sized for. 96 frames keeps a
2-leg combination at 6.4 s, inside the 7.6 s the shipped library already reaches.
"""

MAX_TARGET_LEG_TOKENS: int = MAX_TARGET_LEG_FRAMES // NUM_FRAMES_PER_TOKEN
"""24. What `leg_tokens` and the record validator bound a leg by — not :data:`MAX_TOKENS`."""
```

Replace the `MAX_LEG_FRAMES` docstring with:

```python
MAX_LEG_FRAMES: int = MAX_TOKENS * NUM_FRAMES_PER_TOKEN
"""The longest **plan** MotionBricks can produce, in corpus frames = 64 = 2.13 s.

**No longer the cap on a leg** — that is :data:`MAX_TARGET_LEG_FRAMES` since `spec/intent.md` 3.2.
Until 3.2 the two were the same number because a leg was exactly one plan; a leg is now an
untargeted phase plus a landing plan, so this bounds only the plan, and it is enforced at runtime in
``runtime/sequence.py`` rather than in the segmenter.
"""
```

Change the keyframe bounds:

```python
COMBINATION_MIN_KEYFRAMES: int = 2
COMBINATION_MAX_KEYFRAMES: int = 3
"""A combination is 2-3 keyframes, i.e. 1-2 legs. Was 3-6 before `spec/intent.md` 3.2.

Derived from duration, not taste. At up to :data:`MAX_TARGET_LEG_FRAMES` (3.2 s) per leg, 2 legs is
6.4 s and 3 would be 9.6 s - past the 7.6 s the shipped library reaches and past what the
no-cancellation rule was sized for (`docs/ASSUMPTIONS.md` §A23). Halving the keyframe count is what
makes each leg carry twice the motion; keeping the count at 6 would have doubled the combination
instead, which is a different and much riskier game-feel change.
"""
```

- [ ] **Step 2: Verify the arithmetic**

Run:
```bash
.venv_mb/bin/python -c "
from openroboxing.spec import constants as c
print(c.MIN_TARGET_GAP_FRAMES, c.MAX_TARGET_LEG_FRAMES, c.MAX_TARGET_LEG_TOKENS)
assert (c.MIN_TARGET_GAP_FRAMES, c.MAX_TARGET_LEG_FRAMES, c.MAX_TARGET_LEG_TOKENS) == (48, 96, 24)
print('ok')"
```
Expected: `48 96 24` then `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/openroboxing/spec/constants.py
git commit -m "M7-T6: target spacing and leg-length constants; a leg is no longer a plan"
```

---

### Task 7: `segment.thin_targets` and punch provenance

**Files:**
- Modify: `src/openroboxing/studio/segment.py:218-292`
- Create: `tests/test_segment_thinning.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_segment_thinning.py`:

```python
"""Thinning detected keyframes down to sparse targets, keeping the first and last punch."""

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


def test_keyframe_indices_reports_which_frames_were_punches():
    """Provenance is lost once a CombinationRecord is written, so it must come out of here."""
    qpos = np.zeros((400, 36))
    qpos[:, 3] = 1.0
    indices, punches = segment.keyframe_indices_with_provenance(qpos)
    assert isinstance(punches, set)
    assert punches.issubset(set(int(i) for i in indices))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv_mb/bin/python -m pytest tests/test_segment_thinning.py -q --no-header 2>&1 | tail -5`
Expected: FAIL with `AttributeError: module 'openroboxing.studio.segment' has no attribute
'thin_targets'`.

The last test needs a qpos that actually produces turning points; if a flat `np.zeros` array raises
`SegmentError` ("only 0 turning points"), replace it with a synthetic signal:

```python
    qpos = np.zeros((400, 36))
    qpos[:, 3] = 1.0
    t = np.arange(400)
    qpos[:, 7] = 0.6 * np.sin(2 * np.pi * t / 60.0)   # a shoulder swinging: real reversals
    qpos[:, 10] = 0.4 * np.sin(2 * np.pi * t / 90.0)
```

- [ ] **Step 3: Implement**

In `src/openroboxing/studio/segment.py`, add `MIN_TARGET_GAP_FRAMES` and `MAX_TARGET_LEG_FRAMES` to
the constants import, and add these two functions after `keyframe_indices`:

```python
def thin_targets(
    indices, punch_frames: set[int], *, min_gap: int = MIN_TARGET_GAP_FRAMES
) -> list[int]:
    """Thin detected keyframes to the sparse set a plan is actually aimed at.

    `spec/intent.md` 3.2: a leg carries twice the motion, so half the detected poses stop being hard
    targets and MotionBricks in-fills between the survivors instead. What survives, in priority
    order:

    1. the first and last keyframe — a combination must start and end where it was recorded;
    2. the first and last **punch** — the owner's decision, 2026-09-03: a combination's signature
       opening and closing strike stay recorded, interior ones become model-improvised;
    3. anything else still ``min_gap`` clear of everything already kept, punches considered first.

    Mandatory keyframes always win, even when they crowd each other: spacing is enforced on the
    optional ones only. Returns a sorted list, and is idempotent.
    """
    ordered = sorted(int(i) for i in indices)
    if len(ordered) <= 2:
        return ordered

    punches = [i for i in ordered if i in punch_frames]
    mandatory = {ordered[0], ordered[-1]}
    if punches:
        mandatory.add(punches[0])
        mandatory.add(punches[-1])

    kept = sorted(mandatory)
    # Punches first, so a punch and a fill competing for the same slot resolve to the punch.
    for frame in punches + [i for i in ordered if i not in punch_frames]:
        if frame in mandatory:
            continue
        if all(abs(frame - other) >= min_gap for other in kept):
            kept.append(frame)
            kept.sort()
    return kept


def keyframe_indices_with_provenance(
    qpos: np.ndarray,
    *,
    min_gap: int = MIN_KEYFRAME_GAP_FRAMES,
    max_gap: int = MAX_TARGET_LEG_FRAMES,
    reach_prominence: float = REACH_TURNING_PROMINENCE_M,
    fill_prominence: float = FILL_TURNING_PROMINENCE_M,
    model: mujoco.MjModel | None = None,
) -> tuple[np.ndarray, set[int]]:
    """:func:`keyframe_indices`, plus which of the frames came from ``reach`` — the punches.

    The provenance exists only here: a :class:`~openroboxing.studio.combination_record.Keyframe`
    stores joint angles and timing and nothing about how it was chosen, so thinning has to happen
    while this is still known.
    """
    reach, level, shift = body_signals(qpos, model)

    reach_points = turning_points(reach, reach_prominence)
    punches: list[int] = []
    for frame, _ in reach_points:
        if all(abs(frame - other) >= min_gap for other in punches):
            punches.append(frame)

    picked = list(punches)
    fill_points = turning_points(level, fill_prominence) + turning_points(shift, fill_prominence)
    fill_points.sort(key=lambda point: -point[1])
    for frame, _ in fill_points:
        if all(abs(frame - other) >= min_gap for other in picked):
            picked.append(frame)
    picked.sort()

    if len(picked) < COMBINATION_MIN_KEYFRAMES:
        raise SegmentError(
            f"only {len(picked)} turning points at prominence >= {fill_prominence} m "
            f"(reach >= {reach_prominence} m); a combination needs {COMBINATION_MIN_KEYFRAMES}"
        )

    all_points = turning_points(reach, 0.0) + turning_points(level, 0.0) + turning_points(shift, 0.0)
    all_points.sort(key=lambda point: -point[1])
    dense = densify(picked, all_points, min_gap=min_gap, max_gap=max_gap)
    return np.array(dense, dtype=int), set(punches)
```

Then change `keyframe_indices`'s default `max_gap` from `MAX_LEG_FRAMES` to `MAX_TARGET_LEG_FRAMES`,
and in `leg_tokens` replace both uses of the old cap:

```python
        if gap > MAX_TARGET_LEG_FRAMES:
            raise SegmentError(
                f"leg of {gap} frames is longer than the maximum leg {MAX_TARGET_LEG_FRAMES}; "
                "keyframe_indices densifies gaps, so reaching this means it was bypassed"
            )
        exact = gap / NUM_FRAMES_PER_TOKEN + residual
        chosen = max(MIN_TOKENS, min(MAX_TARGET_LEG_TOKENS, round(exact)))
```

**The `min(MAX_TOKENS, …)` → `min(MAX_TARGET_LEG_TOKENS, …)` change is the one that silently
defeats everything if missed**: left alone it truncates every merged leg back to 16 tokens and the
rebuilt library looks rebuilt while being unchanged. Import `MAX_TARGET_LEG_TOKENS` for it.

- [ ] **Step 4: Run to verify**

Run: `.venv_mb/bin/python -m pytest tests/test_segment_thinning.py tests/test_segment.py -q --no-header 2>&1 | tail -5`
Expected: the new tests PASS. `tests/test_segment.py:173` asserts `gaps <= MAX_LEG_FRAMES` — update it
to `MAX_TARGET_LEG_FRAMES`, and `:226` builds `leg_tokens([MAX_LEG_FRAMES + 1])` expecting a raise;
change it to `MAX_TARGET_LEG_FRAMES + 1`.

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/studio/segment.py tests/test_segment_thinning.py tests/test_segment.py
git commit -m "M7-T7: thin detected keyframes to sparse targets, keeping the first and last punch"
```

---

### Task 8: `build_from_take` thins, and the record validator follows

**Files:**
- Modify: `src/openroboxing/studio/combination_record.py:179-182` (validation) and `:268` (build)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_segment_thinning.py`:

```python
def test_a_merged_leg_is_a_valid_record():
    """A 24-token leg is legal since `spec/intent.md` 3.2: MAX_TOKENS caps a plan, not a leg."""
    from openroboxing.runtime.conventions import G1
    from openroboxing.spec.constants import MAX_TARGET_LEG_TOKENS
    from openroboxing.studio import combination_record as cr

    angles = {name: 0.0 for name in G1.mujoco_joint_names}
    rec = cr.CombinationRecord(
        name="c", library_version="v0.2",
        source=cr.CombinationSource("t", 0, 100, False),
        keyframes=[
            cr.Keyframe(dict(angles), None, (0.0, 0.0), 0.0),
            cr.Keyframe(dict(angles), MAX_TARGET_LEG_TOKENS, (0.2, 0.0), 0.0),
        ],
    )
    assert rec.keyframes[1].leg_tokens == MAX_TARGET_LEG_TOKENS
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv_mb/bin/python -m pytest tests/test_segment_thinning.py::test_a_merged_leg_is_a_valid_record -q --no-header 2>&1 | tail -5`
Expected: FAIL — `CombinationError: ... leg_tokens 24 outside [6, 16]`.

- [ ] **Step 3: Implement**

In `src/openroboxing/studio/combination_record.py`, change the import of `MAX_TOKENS` to
`MAX_TARGET_LEG_TOKENS` and replace the validation:

```python
            if not MIN_TOKENS <= keyframe.leg_tokens <= MAX_TARGET_LEG_TOKENS:
                raise CombinationError(
                    f"{record.name} keyframe {i}: leg_tokens {keyframe.leg_tokens} outside "
                    f"[{MIN_TOKENS}, {MAX_TARGET_LEG_TOKENS}]"
                )
```

In `build_from_take`, replace:

```python
    indices = segment.keyframe_indices(qpos, conventions=conventions)
```

with:

```python
    # Thinned to sparse targets before grouping: `spec/intent.md` 3.2 halves the number of poses a
    # plan is aimed at so each leg carries twice the motion. Provenance (which frames were punches)
    # only exists here — a Keyframe records timing and angles, not how it was chosen.
    detected, punches = segment.keyframe_indices_with_provenance(qpos)
    indices = np.array(segment.thin_targets(detected, punches), dtype=int)
```

Update the docstring paragraph that begins "Every leg is plannable by construction" to:

```
    Every leg is **reachable** by construction: :func:`segment.keyframe_indices_with_provenance`
    densifies any gap longer than ``MAX_TARGET_LEG_FRAMES``, and a leg longer than one plan is run as
    an untargeted phase followed by a landing in-between (``runtime/sequence.py``). A run whose legs
    cannot be tokenised raises rather than being dropped, because a silently skipped combination is a
    silently smaller library (`CLAUDE.md` invariant 5).
```

- [ ] **Step 4: Run to verify**

Run: `.venv_mb/bin/python -m pytest tests/ -q --no-header 2>&1 | tail -5`
Expected: PASS. Tests asserting the old 3–6 keyframe bounds must move to 2–3.

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/studio/combination_record.py tests/
git commit -m "M7-T8: build_from_take thins to sparse targets; a leg may exceed one plan"
```

---

### Task 9: Rebuild the library and re-derive the travel exclusion

**Files:**
- Modify: `src/openroboxing/poses/v0.2/combinations/*.json` (regenerated)
- Modify: `src/openroboxing/spec/constants.py` (`MAX_RECORDED_TRAVEL_M` if the gap moved)

- [ ] **Step 1: Find the rebuild entry point**

Run: `ls src/openroboxing/tools/ | grep -i "build\|harvest\|librar"`
then `.venv_mb/bin/python -m openroboxing.tools.<name> --help` for the one that builds the
combination library (`studio/harvest.py` is the harvester it wraps).

- [ ] **Step 2: Rebuild**

Run the build tool over `motions/`, writing to `src/openroboxing/poses/v0.2/combinations/`.

- [ ] **Step 3: Measure what came out**

Run:
```bash
.venv_mb/bin/python - <<'EOF'
import json, glob, collections
from statistics import median
legs=[]; kfs=collections.Counter(); dur=[]; travel=[]
import math
for p in sorted(glob.glob("src/openroboxing/poses/v0.2/combinations/*.json")):
    d=json.load(open(p)); ks=d["keyframes"]; kfs[len(ks)]+=1
    lt=[k["leg_tokens"] for k in ks[1:]]; legs+=lt
    dur.append(sum(lt)*4/30)
    dx,dy=ks[-1]["root_offset"]; travel.append(math.hypot(dx,dy))
legs.sort(); dur.sort(); travel.sort()
print("combinations:", sum(kfs.values()), "keyframe counts:", dict(sorted(kfs.items())))
print("legs: median %d (%.2f s) min %d max %d"%(median(legs), median(legs)*4/30, legs[0], legs[-1]))
print("duration s: min %.2f median %.2f max %.2f"%(dur[0], median(dur), dur[-1]))
print("recorded travel m, sorted:", [round(t,2) for t in travel])
EOF
```

Expected, from the pre-implementation simulation: keyframe counts within 2–3, median leg ≈ 17 tokens
(2.27 s), max leg ≤ 24, duration ≤ ~6.4 s. **If the median leg is still 9 tokens, the
`min(MAX_TOKENS, …)` clamp in `segment.leg_tokens` was not changed (Task 7, Step 3).**

- [ ] **Step 4: Re-derive `MAX_RECORDED_TRAVEL_M`**

Its 1.2 m value sits in "the widest gap in the distribution" (0.98 → 1.47) measured on the old
library. Leg boundaries have moved, so read the sorted travel list printed above and confirm a gap of
comparable width still brackets 1.2. If it does not, set the constant to the middle of the new widest
gap and rewrite the docstring's measurement paragraph with the new sorted values and date. Do not
leave the old numbers in place describing a library that no longer exists.

- [ ] **Step 5: Run the full suite**

Run: `.venv_mb/bin/python -m pytest tests/ -q --no-header 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/openroboxing/poses/v0.2/combinations src/openroboxing/spec/constants.py
git commit -m "M7-T9: rebuild the combination library on sparse targets"
```

---

### Task 10: Specs and CLAUDE.md

**Files:**
- Modify: `src/openroboxing/spec/intent.md`, `src/openroboxing/spec/combination.md`, `CLAUDE.md`
- Modify: `src/openroboxing/runtime/intents.py` (`SPEC_VERSION`)

- [ ] **Step 1: `spec/intent.md` → 3.2**

Replace the section "Forced plan lengths are back, and that has a cost" with a section
"The keyframe is pinned; the hole shrinks" stating the three regimes from Task 3, and add a changelog
entry:

```
- **3.2** (2026-09-03) — **a keyframe is pinned in absolute time.** `intent_for` requested the leg's
  full length on every replan, so the target was re-aimed `REPLAN_DT * GENERATOR_HZ` frames further
  out each time and never landed on its boundary — the "motions broken in pieces" defect. It now
  requests `ceil(boundary - tick)` in tokens, drops the pose target while the hole exceeds
  `MAX_TOKENS`, and stops replanning below `MIN_TOKENS`. **Withdraws 3.0's "consumed exactly"
  contract**, which was specified but never implemented; the same guarantee is now met by a
  mechanism that forces nothing. A leg is no longer one plan, so `MAX_TOKENS` stops bounding a leg
  (`MAX_TARGET_LEG_TOKENS` does).
```

- [ ] **Step 2: Bump the version constant**

In `src/openroboxing/runtime/intents.py`, set `SPEC_VERSION = "3.2"`.

- [ ] **Step 3: `spec/combination.md`**

Update the keyframe-count bounds (2–3), the leg-length cap (`MAX_TARGET_LEG_TOKENS`), and add the
thinning rule with its punch-preservation clause.

- [ ] **Step 4: `CLAUDE.md`**

In the canonical-rates table, replace the `Plan length`, `Leg` and `Combination` rows:

```
| Plan length | **the hole to the next keyframe**, 6–16 tokens | `sequence.py` asks for `ceil(boundary - tick)`, clamped |
| Leg | **0.8–3.2 s** | `MIN_TOKENS`–`MAX_TARGET_LEG_TOKENS` × 4 frames at 30 Hz |
| Combination | **2–3 keyframes, 1.6–6.4 s** | rebuilt on sparse targets, 2026-09-03 |
```

and update the `sequence.py` layout line to mention the pinned keyframe.

- [ ] **Step 5: Verify the spec/version pairing test**

Run: `.venv_mb/bin/python -m pytest tests/ -q --no-header 2>&1 | tail -3`
Expected: PASS — a test pairs `SPEC_VERSION` with the changelog and fails if one moves without the
other.

- [ ] **Step 6: Lint and commit**

```bash
bash lint.sh
git add -A
git commit -m "M7-T10: spec/intent.md 3.2 - the keyframe is pinned, and a leg is not a plan"
```

---

## Self-Review

**Spec coverage:** Half 1 → Tasks 1–4. `DRIFT_GAIN` → Task 5. Half 2 thinning → Tasks 6–8. Library
rebuild and `MAX_RECORDED_TRAVEL_M` → Task 9. Spec/doc changes → Task 10. Context integrity → Task 4.
The five tests the spec names map to Tasks 3 (pinning, replan floor, no-pose-above-cap, hold) and 4
(context integrity). Implementation order matches the spec's Half 1 → gain → Half 2 sequence.

**Types:** `thin_targets(indices, punch_frames, *, min_gap) -> list[int]` and
`keyframe_indices_with_provenance(qpos, ...) -> tuple[np.ndarray, set[int]]` are used consistently in
Tasks 7 and 8. `_horizon_for(tick, index, leg) -> tuple[int, object | None, bool]` is defined and
called once, in Task 3. `GeneratorIntent.replan` is added in Task 1 and read in Tasks 2, 3, 5.

**Known risk carried deliberately:** Task 5 needs a GPU and is the only task that cannot be verified
by the test suite alone. It is sequenced between the halves rather than at the end so its measurement
is attributable to the schedule change alone.
