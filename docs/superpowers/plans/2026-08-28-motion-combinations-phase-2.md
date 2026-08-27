# Motion Combinations — Phase 2: the match runtime

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A commit becomes a *combination* — the fighter runs a timed sequence of key poses that starts where it stands and ends on the ghost — and the walk approach is deleted.

**Architecture:** `runtime/warp.py` (phase 1, done) turns a record plus a placement into per-leg targets. This phase adds `runtime/sequence.py` to serve those legs by tick, reworks `intents.py` so a `Commit` carries a combination instead of a pose, and takes `spec/intent.md` to 3.0 by removing the approach, the dwell and the arrival test. `fight.py` follows.

**Tech Stack:** Python 3.10+, numpy, pytest, MuJoCo. Run everything with `.venv_mb/bin/python`.

---

## Scope: this is the match runtime only

Phase 1 measured the blast radius of removing the approach. `Loadout` alone appears in **23 files** — 9 tools, the server, the client and 8 test files — and there is a separate **sparring** subsystem (`server/sparring_app.py`, `client/sparring.js`, `spec/sparring_protocol.md`, `tools/serve_sparring.py`) with its own protocol that depends on `has_arrived` and `approach_timeout_ticks`. That is not one plan.

**This plan covers:** `runtime/sequence.py`, `runtime/intents.py`, `runtime/fight.py`, `runtime/reference.py`, `spec/intent.md` 3.0, and their tests. It produces working, testable software: two fighters running combinations under physics in a match.

**Plan 3 will cover** the peripheral surface — the 9 tools, the sparring subsystem, `server/host.py` + `protocol.py`, and the client's nine-per-page picker (design D6). Until then those keep compiling against a `Loadout` that still exists.

## Two measured facts this plan is built on

**The style must be `walk_boxing`.** Measured 2026-08-28: `walk` permits only 6–11 tokens
(`narrow_allowed_tokens` raises on 12), and combination legs run to 16. `walk_boxing` permits 6–16.
This also settles `CLAUDE.md`'s warning that `walk_boxing` leaves a fighter with no sideways gait —
it does not bite here, because travel now comes from `target_position` rather than from the gait
remap: a ghost 1 m to the side is reached as well as one 1 m ahead (0.79 m either way).

**The fighter reaches ~79 % of the drift it is asked for.** Measured over one combination at four
distances:

| drift asked | reached | fraction |
|---|---|---|
| 0.25 m | 0.22 m | 0.77 |
| 0.50 m | 0.41 m | 0.77 |
| 1.00 m | 0.81 m | 0.79 |
| 2.00 m | 1.65 m | 0.81 |

Near-constant, so it is a **gain**, not slop. `intents.py` already records the same effect for the
old approach ("one plan covers only ~54 % of the distance asked for at *any* length"), which the
open-ended walk-until-arrival hid. With the approach deleted there is nothing to hide it, so
uncorrected every move would land ~20 % short of the ghost. Task 1 measures the gain properly and
Task 2 applies it.

## Owner decision carried into this plan

A queued combination starts from where the fighter **actually is**, which is not exactly where the
previous ghost was. Asked 2026-08-28, the owner chose: **still reach the ghost.** Re-warp from the
real position and run whatever drift that needs, even above `APPROACH_SPEED_M_S`. Nothing is clamped
and nothing raises; a fighter knocked far off simply tracks badly and physics decides. The achieved
drift speed is recorded in the match record.

This means `warp()`'s `speed_ceiling` becomes an **issue-time validation** of the player's placement,
not an execution-time guard.

---

## File Structure

**Create:**

| Path | Responsibility |
|---|---|
| `src/openroboxing/runtime/sequence.py` | `CombinationRunner`: warped legs + tick → which leg is live |
| `src/openroboxing/tools/measure_drift_gain.py` | the Task 1 measurement |
| `tests/test_sequence.py`, `tests/test_intents_combinations.py` | |

**Modify:**

| Path | Change |
|---|---|
| `src/openroboxing/runtime/warp.py` | apply the measured drift gain; `speed_ceiling=None` disables the check |
| `src/openroboxing/spec/constants.py` | add `DRIFT_GAIN` |
| `src/openroboxing/runtime/intents.py` | `Commit` carries a combination; delete the approach; `SPEC_VERSION = "3.0"` |
| `src/openroboxing/spec/intent.md` | rewrite to 3.0 |
| `src/openroboxing/runtime/fight.py` | drive combinations; drop `has_arrived` / `has_settled` |
| `tests/test_intents.py`, `tests/test_fight.py` | follow |

---

### Task 1: Measure the drift gain

**Files:**
- Create: `src/openroboxing/tools/measure_drift_gain.py`
- Create: `docs/perf/2026-08-28-drift-gain.md`

The 0.79 above came from **one** combination. A constant that corrects every move must be measured
across a sample, and `CLAUDE.md` forbids inventing numbers.

- [ ] **Step 1: Write the tool**

Model it on `src/openroboxing/tools/spike_warp_tracking.py`, which already drives a combination leg
by leg — read that first and reuse its `leg_pose` helper shape. For each of **at least 8**
combinations spanning all three families (`shadow-boxing`, `ib-dodge`, `ib-combat-turn-jog`), and for
drifts of 0.25, 0.5, 1.0 and 2.0 m along the recorded direction:

- warp with the gain **disabled**, drive every leg with `style="walk_boxing"` and `force=True`,
  consuming each plan so the next leg's context follows on;
- record `reached / asked`, where `reached` is `hypot` of the final plan frame's root `(x, y)` and
  `asked` is `hypot` of the ghost.

Print a per-combination table and an overall median, mean, min and max of the fraction.

- [ ] **Step 2: Run it**

```bash
.venv_mb/bin/python -m openroboxing.tools.measure_drift_gain
```

- [ ] **Step 3: Decide, and write it up**

Write `docs/perf/2026-08-28-drift-gain.md` with the table, the command, and the decision.

**The bar:** if the fraction's spread across combinations is tight (say min and max within ±0.10 of
the median), a single `DRIFT_GAIN` constant is justified — record the median and its evidence. **If
it is not tight, STOP and report**: a per-combination gain is a schema change and a design decision,
not something to add quietly.

- [ ] **Step 4: Add the constant**

In `src/openroboxing/spec/constants.py`, after `APPROACH_SPEED_M_S`:

```python
DRIFT_GAIN: float = <MEDIAN FROM STEP 2>
"""Fraction of a commanded drift the generator actually covers, measured not assumed.

A warped combination aims each leg at a target position; MotionBricks converges toward it but
arrives short by a near-constant fraction, so `runtime/warp.py` divides the residual by this to
land on the ghost. Measured <DATE> over <N> combinations at 0.25-2.0 m:
median <X>, range <LO>-<HI>. See docs/perf/2026-08-28-drift-gain.md.

The old open-ended approach hid this by walking until it arrived; `spec/intent.md` 3.0 removes that,
so the correction has to be explicit. Re-measure after any submodule bump.
"""
```

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/tools/measure_drift_gain.py docs/perf/2026-08-28-drift-gain.md src/openroboxing/spec/constants.py
git commit -m "M6-T1: measure the drift gain - what fraction of a commanded drift is covered"
```

---

### Task 2: Apply the gain in the warp, and make the ceiling optional

**Files:**
- Modify: `src/openroboxing/runtime/warp.py`, `tests/test_warp.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_warp.py`:

```python
def test_the_residual_is_divided_by_the_drift_gain():
    """The generator covers DRIFT_GAIN of what it is asked for, so ask for more (M6-T1)."""
    from openroboxing.spec.constants import DRIFT_GAIN

    rec = record([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], [16, 16])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (1.0, 0.0))
    assert legs[-1].target_position[0] == pytest.approx(1.0 / DRIFT_GAIN)


def test_the_recorded_footwork_is_not_gained():
    """Only the residual is corrected; the recording is already the right size."""
    rec = record([(0.5, 0.0), (0.5, 0.0)], [0.0, 0.0], [16, 16])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (0.5, 0.0))
    assert legs[-1].target_position[0] == pytest.approx(0.5)


def test_a_none_ceiling_allows_any_drift():
    """Execution re-warps from where the fighter actually is and never refuses (owner, 2026-08-28)."""
    rec = record([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (50.0, 0.0), speed_ceiling=None)
    assert legs[-1].target_position[0] > 0.0


def test_the_ceiling_is_checked_before_the_gain():
    """The gain is a generator correction, not extra distance the player asked for."""
    rec = record([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], [16, 16])
    # 3.4 m over 4.27 s is 0.80 m/s - under the ceiling before the gain, over it after.
    warp.warp(rec, (0.0, 0.0), 0.0, (3.4, 0.0))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv_mb/bin/python -m pytest tests/test_warp.py -q --no-header 2>&1 | tail -4`
Expected: the four new tests fail.

- [ ] **Step 3: Implement**

In `warp()`:

- change the signature to `speed_ceiling: float | None = APPROACH_SPEED_M_S`;
- compute `drift_speed` from the **raw** residual and skip the check entirely when `speed_ceiling is
  None` — the check answers "did the player ask for something reachable", and the gain is a
  correction for the generator's shortfall, not extra distance the player requested;
- **after** the check, scale the residual by `1.0 / DRIFT_GAIN`;
- leave the rotated recorded offsets untouched — they are already the right size (D4).

Update the module docstring to state that the residual carries the gain and the recording does not.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv_mb/bin/python -m pytest tests/test_warp.py -q --no-header 2>&1 | tail -3`
Expected: `17 passed`

Note `test_every_library_combination_places_at_a_reachable_ghost` still passes unchanged: it places
each record at its own recorded displacement, where the residual is zero and the gain is a no-op.

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/runtime/warp.py tests/test_warp.py
git commit -m "M6-T2: correct the warp residual by the measured drift gain"
```

---

### Task 3: `CombinationRunner`

**Files:**
- Create: `src/openroboxing/runtime/sequence.py`
- Test: `tests/test_sequence.py`

The unit that answers "at this tick, which leg is the fighter on, and what is its intent". Pure —
no generator, no physics.

**The tick arithmetic, stated once:** legs are measured in **tokens**; the timeline runs in **ticks**.
One token is `SECONDS_PER_TOKEN * TICK_HZ` = 6.667 ticks. Leg `i` ends at
`commit_at + round(cumulative_tokens_i * SECONDS_PER_TOKEN * TICK_HZ)` — the same expression
`CombinationRecord.duration_ticks` already uses, so the last boundary equals `commit_at +
duration_ticks` by construction. Deriving boundaries from a per-call frame counter instead would
drift whenever a frame is skipped.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sequence.py`:

```python
"""CombinationRunner: which leg is live at a tick, and the intent it produces."""

from __future__ import annotations

import math

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


def test_intent_carries_the_legs_pose_target_and_forced_length():
    run, _ = runner([6, 8, 6])
    intent = run.intent_for(0)
    assert intent.style == sequence.COMBINATION_CONTEXT
    assert intent.horizon_tokens == 6
    assert intent.pose is not None
    assert intent.target_position is not None


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


def test_every_library_combination_runs_end_to_end():
    from openroboxing.paths import COMBINATION_DIR

    for path in sorted(COMBINATION_DIR.glob("*.json")):
        rec = cr.load(path)
        legs = warp.warp(rec, (0.0, 0.0), 0.0, rec.recorded_displacement)
        run = sequence.CombinationRunner(rec, legs, commit_at=0)
        seen = {run.leg_index(t) for t in range(run.end_tick)}
        assert seen == set(range(len(legs))), f"{path.name} skipped a leg: {sorted(seen)}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv_mb/bin/python -m pytest tests/test_sequence.py -q --no-header 2>&1 | tail -4`
Expected: no module named `openroboxing.runtime.sequence`.

- [ ] **Step 3: Implement**

Create `src/openroboxing/runtime/sequence.py` with a module docstring stating the conventions above
(ticks vs tokens, `walk_boxing`, and that a finished runner holds its last leg). Public surface:

```python
COMBINATION_CONTEXT = "walk_boxing"   # with the measurement below as its docstring

class SequenceError(RuntimeError): ...

class CombinationRunner:
    def __init__(self, record, legs, *, commit_at: int) -> None: ...
    @property
    def legs(self) -> tuple[Leg, ...]: ...
    @property
    def end_tick(self) -> int: ...
    def leg_index(self, tick: int) -> int: ...
    def is_finished(self, tick: int) -> bool: ...
    def intent_for(self, tick: int) -> GeneratorIntent: ...
```

`COMBINATION_CONTEXT` must carry the reason in its docstring: *measured 2026-08-28, `walk` permits
only 6–11 tokens and legs run to 16, so `walk_boxing` is the only clip that can express a forced leg
length; and because travel comes from `target_position` rather than the gait remap, its missing
lateral gait does not bite — a sideways ghost is reached as well as a forward one.*

`intent_for` builds a `GeneratorIntent` with `style=COMBINATION_CONTEXT`, the leg's
`movement_angle`, `facing_angle`, `target_position`, `target_heading`, `horizon_tokens`, and `pose`
— **a `PoseRecord`**, because `generator._install_pose_override` hands `intent.pose` to
`skeleton_fk.target_transforms`, which reads one. Build it from the leg's `joint_angles`.

`runtime` must not import `studio` at module level (see `generator.py`'s note). Import `PoseRecord`
inside the function, or under `TYPE_CHECKING` plus a local import — `warp.py` shows the pattern.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv_mb/bin/python -m pytest tests/test_sequence.py -q --no-header 2>&1 | tail -3`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/runtime/sequence.py tests/test_sequence.py
git commit -m "M6-T3: CombinationRunner - which leg is live at a tick, and its intent"
```

---

### Task 4: `spec/intent.md` 3.0

**Files:**
- Modify: `src/openroboxing/spec/intent.md`

Write the spec before the code that implements it (`CLAUDE.md` invariant 7). Read the current 2.2
first; this is a rewrite of the parts the approach owned, not a new document.

- [ ] **Step 1: Rewrite the affected sections**

3.0 must state:

1. **A commit carries a combination and a ghost.** Not a pose and a placement. The combination is
   3–6 recorded key poses with recorded timing (`spec/combination.md` 0.1).
2. **A commit starts in place.** There is no approach and no travel to a start. The fighter's
   position at the tick the commit begins is keyframe 0.
3. **A commit's span is known when it starts.** `end_tick = commit_at + record.duration_ticks`. This
   replaces arrival, the counted dwell and the timeout — 2.2's "the queue is not a schedule" becomes
   *the queue is a schedule from the moment each move starts*. Say so explicitly, because it reverses
   a documented decision.
4. **The ghost is where the final keyframe lands, and its heading is derived** — the fighter's
   heading plus the combination's recorded turn, never aimed at a target (design D5).
5. **Off-target queued moves still reach the ghost** (owner, 2026-08-28), running whatever drift that
   needs. The speed ceiling validates the player's placement at issue time only.
6. **The drift gain** and why it is needed once the approach is gone.

Removed and named as removed: `TRAVEL_CONTEXT`, the approach, `approach_timeout_ticks`,
`has_arrived`, `has_settled`, the counted dwell (`POSE_DWELL_TICKS`), `MAX_DWELL_TICKS`, and
`Placement.heading` as a player-set field. Keep a changelog entry saying what 3.0 removed and why —
2.2's own changelog is the model.

Unchanged and worth restating: **no cancellation**, `MAX_OUTSTANDING_COMMITS`, the
`COMMIT_HORIZON_TICKS` floor, and `OPENING_STANCE_CONTEXT` before the first commit of a round.

- [ ] **Step 2: Commit**

```bash
git add src/openroboxing/spec/intent.md
git commit -m "M6-T4: spec/intent.md 3.0 - a commit is a combination, and the approach is gone"
```

---

### Task 5: Rework `intents.py`

**Files:**
- Modify: `src/openroboxing/runtime/intents.py`
- Create: `tests/test_intents_combinations.py`
- Modify: `tests/test_intents.py`

The largest task. `intents.py` is 766 lines and roughly a third of it exists to run an approach that
no longer exists. **Read the whole module before changing it.**

- [ ] **Step 1: Write the failing tests**

Create `tests/test_intents_combinations.py` covering, at minimum:

- staging and committing a combination by name, and `IntentError` when the queue is full
  (`MAX_OUTSTANDING_COMMITS` unchanged);
- `commit_at` is stamped on the first `generator_intent` call at or after the horizon floor;
- `end_tick == commit_at + record.duration_ticks`, known as soon as the commit starts;
- the queue advances to the next commit exactly at `end_tick`, with no one-tick hole (2.2's
  hand-over rule survives);
- with the queue drained, the last combination's **final leg** intent is re-issued unchanged — the
  hold;
- before any commit, `OPENING_STANCE_CONTEXT`;
- `generator_intent` still raises when ticks go backwards;
- `generator_intent` no longer accepts `has_arrived` or `has_settled` (`TypeError`).

Then delete from `tests/test_intents.py` every test of the approach, the dwell and arrival, and every
`Loadout`-of-poses test that 3.0 makes meaningless. Deleting a test is a decision: list what you
deleted and why in the commit message.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv_mb/bin/python -m pytest tests/test_intents_combinations.py -q --no-header 2>&1 | tail -4`

- [ ] **Step 3: Implement**

- `Commit` carries `record: CombinationRecord`, `ghost: tuple[float, float]`, `issued_at`,
  `commit_at`, `ended_at`, and the `CombinationRunner` built when it starts. Delete `placement`,
  `strike_at`, `arrived`, `completed_by`, `is_approaching`.
- `Commit.end_tick` returns `commit_at + record.duration_ticks`, or `None` before it starts.
- `StagedIntent` holds a combination name and a ghost position; `Placement` loses its player-set
  heading (derive it with `warp.ghost_heading`).
- `IntentTimeline.generator_intent(tick, *, facing_angle=0.0, anchor=None)` — `anchor` is
  `() -> (position, heading)`, supplied by whoever knows where the fighter is, and is called **once,
  when a commit starts**, to build its runner via `warp(..., speed_ceiling=None)`. Delete
  `has_arrived`, `has_settled`, `_resolve_approach`, `_resolve_completion`,
  `approach_timeout_ticks`, `DEFAULT_APPROACH_TIMEOUT_TICKS` and `TRAVEL_CONTEXT`.
- `_hold_intent` keeps its two states, but "hold" now re-issues the last commit's **final leg**.
- `SPEC_VERSION = "3.0"`.
- Keep `Loadout` and `apply_adjustment` importable for now — Plan 3 removes them, and breaking 23
  files in this task would make it unreviewable. Leave a module comment saying so.

- [ ] **Step 4: Run to verify**

Run: `.venv_mb/bin/python -m pytest tests/test_intents_combinations.py tests/test_intents.py -q --no-header 2>&1 | tail -3`

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/runtime/intents.py tests/test_intents_combinations.py tests/test_intents.py
git commit -m "M6-T5: a commit is a combination - intents.py to spec/intent.md 3.0"
```

---

### Task 6: The three `reference.py` regressions

**Files:**
- Modify: `tests/test_sequence.py` (or a new `tests/test_reference_forced_length.py`)

`spec/intent.md` 2.0 moved to `horizon_tokens=None` deliberately, and `runtime/reference.py`'s
docstring records three defects that came from the forced-plan machinery it deleted. Phase 2
reintroduces forced lengths, so those three become explicit tests. **Read `reference.py`'s docstring
first** — it names all three precisely.

- [ ] **Step 1: Write the tests**

One test each:

1. **the final frame is not lost.** An 8-token leg is 32 generator frames; resampled 30→50 Hz it
   must not drop the frame that *is* the authored pose, at any tick alignment. Drive the alignment
   across all residues.
2. **no frame leaks into the next commit.** The last frame of one combination must not appear in the
   next one's first leg.
3. **the end-of-move replan does not bypass the ambient cadence.**

Task 1's spike showed plan length is honoured exactly on 30 of 30 legs, so 1 is expected to pass —
which is the point: it is a regression test, not a bug hunt. If any of the three fails, that is a real
finding; report it rather than adjusting the test.

- [ ] **Step 2: Run, then commit**

```bash
git add tests/
git commit -m "M6-T6: regression tests for the three defects forced plan lengths caused in 1.1"
```

---

### Task 7: `fight.py` drives combinations

**Files:**
- Modify: `src/openroboxing/runtime/fight.py`, `tests/test_fight.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_fight.py`, add: two fighters each run a combination to completion under physics; the
achieved end position is recorded; the queue advances at `end_tick`. Delete the approach and dwell
tests.

- [ ] **Step 2: Implement**

- `Fighter.timeline.generator_intent(...)` loses `has_arrived=` / `has_settled=` and gains
  `anchor=`, reading the fighter's live root `(x, y)` and yaw.
- Delete `Fight.has_arrived` and `Fight.has_settled`.
- Record the **achieved drift speed** per commit in the state trace, per the owner's decision — a
  move that could not reach its ghost must be visible in the record rather than silent.

- [ ] **Step 3: Run the full suite**

Run: `.venv_mb/bin/python -m pytest -q --no-header 2>&1 | tail -5`

Expect failures in `tests/test_server.py`, `tests/test_sparring_app.py`, `tests/test_replay.py`,
`tests/test_agent.py`, `tests/test_match.py` and `tests/test_scene.py` — those are Plan 3's surface.
**List them in your report.** Do not fix them here and do not delete them; a failing test that names
work still to do is more honest than a deleted one.

- [ ] **Step 4: Commit**

```bash
git add src/openroboxing/runtime/fight.py tests/test_fight.py
git commit -m "M6-T7: fight.py drives combinations; the approach is gone from the match loop"
```

---

## Definition of done

1. `tests/test_warp.py`, `tests/test_sequence.py`, `tests/test_intents_combinations.py` and
   `tests/test_fight.py` all green.
2. The rest of the suite fails **only** in the files Plan 3 owns, and those failures are listed.
3. `uvx ruff@0.16.2 check` clean on every file touched (the repo has ~95 pre-existing findings
   elsewhere; do not fix those here).
4. `spec/intent.md` is 3.0 and `intents.SPEC_VERSION` matches — a test pairs them.
5. `docs/perf/2026-08-28-drift-gain.md` exists and states the measured gain.
6. `CLAUDE.md` updated: the canonical-rates table still claims an approach leg, a sustained walk and
   a commit horizon that 3.0 removes.

## What this plan deliberately does not do

The 9 tools, the sparring subsystem, `server/host.py`, `server/protocol.py` and the client are
untouched, and `Loadout` still exists. Nobody can *play* a combination from a browser at the end of
this plan — a match harness can run one. Plan 3 is the surface, including D6's nine-per-page picker.
