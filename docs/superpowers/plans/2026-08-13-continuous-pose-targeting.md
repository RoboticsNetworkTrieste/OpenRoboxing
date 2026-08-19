# Continuous Pose Targeting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a commit one continuous intent — "be at this placement, in this pose" — that converges on both together and then holds the pose, replacing the two-phase walk-then-strike model.

**Architecture:** The pose is armed on every replan for the commit's whole life instead of only during a final forced plan, and the plan length is left to MotionBricks (`horizon_tokens=None`). The commit ends a measured dwell after arrival. When the queue drains, the last completed commit's intent stays armed, which *is* the hold. All forced-plan machinery in `runtime/reference.py` is deleted.

**Tech Stack:** Python 3.10, pytest, MuJoCo, ONNX Runtime, MotionBricks (upstream, unmodified except patch P0). Run everything with `.venv_mb/bin/python` from the repo root `/home/hpc-dev/GR00T-WholeBodyControl`.

**Design doc:** `docs/superpowers/specs/2026-08-13-continuous-pose-targeting-design.md`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/openroboxing/tools/measure_dwell.py` | CLI that measures the post-arrival dwell and per-pose generator error under continuous arming | **create** |
| `src/openroboxing/studio/rehearsal.py` | add `rehearse_approach()` — a replanned, pose-armed approach toward a placement | modify |
| `src/openroboxing/spec/constants.py` | `POSE_DWELL_TICKS`, `GENERATOR_POSE_TOLERANCE_RAD` | modify |
| `src/openroboxing/spec/rates.md` | document both numbers and their derivation | modify |
| `src/openroboxing/runtime/intents.py` | one continuous intent; hold on drained queue; `end_tick` from the dwell | modify |
| `src/openroboxing/runtime/reference.py` | delete forced-plan machinery; one replan cadence | modify |
| `src/openroboxing/runtime/generator.py` | drop `plan_key` from `GeneratorIntent` | modify |
| `src/openroboxing/server/host.py` | surface refused commits to the client (independent defect) | modify |
| `src/openroboxing/spec/intent.md` | → 2.0 | modify |
| `src/openroboxing/spec/pose_record.md` | admission tolerance + `horizon_tokens` meaning | modify |

Tasks 1–2 change no behaviour. Task 3 is the behaviour change and lands with its tests in one commit. Task 7 is independent of everything else and may be done first if preferred.

---

### Task 1: A replanned, pose-armed approach in the Studio bench

`studio/rehearsal.py` already has `rehearse()` (replans, no placement) and `rehearse_commit()` (one forced plan). Neither does what the new runtime will do. Add the third.

**Files:**
- Modify: `src/openroboxing/studio/rehearsal.py`
- Test: `tests/test_rehearsal.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rehearsal.py`:

```python
@pytest.mark.slow
def test_an_armed_approach_converges_on_both_the_placement_and_the_pose() -> None:
    """The claim continuous pose targeting rests on (design doc, "Why it is better, measured").

    Reproduce: .venv_mb/bin/python -m pytest tests/test_rehearsal.py -m slow -k armed_approach -v
    """
    import numpy as np

    from openroboxing.paths import POSE_DIR
    from openroboxing.studio import pose_record
    from openroboxing.studio.rehearsal import rehearse_approach

    pose = pose_record.load(POSE_DIR / "v0.1" / "hook-right.json")
    result = rehearse_approach(pose, travel_m=2.5, seconds=6.0)

    assert result.qpos.shape[1] == 36
    assert np.isfinite(result.qpos).all()

    # It gets there: the root ends far closer to the goal than it started.
    assert result.distance_to_goal[-1] < 0.40, result.distance_to_goal[-1]
    assert result.distance_to_goal[-1] < result.distance_to_goal[0] / 4

    # And it converges on the pose rather than wandering near it.
    assert result.pose_error_rad[-1] < np.radians(10.0), np.degrees(result.pose_error_rad[-1])
    assert result.pose_error_rad[-1] < result.pose_error_rad[0] / 2
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv_mb/bin/python -m pytest tests/test_rehearsal.py -m slow -k armed_approach -v
```

Expected: `ImportError: cannot import name 'rehearse_approach'`.

- [ ] **Step 3: Implement it**

Add to `src/openroboxing/studio/rehearsal.py`, after `rehearse_commit`:

```python
@dataclass(frozen=True)
class ApproachRehearsal:
    """A replanned approach with the pose armed throughout — what a commit becomes at intent 2.0.

    Conventions: ``qpos`` is ``(N, 36)`` MuJoCo qpos at ``GENERATOR_HZ``; ``pose_error_rad`` and
    ``distance_to_goal`` are per-frame and the same length as ``qpos``.
    """

    qpos: np.ndarray
    pose_error_rad: np.ndarray      # (N,) mean absolute joint error against the commanded pose
    distance_to_goal: np.ndarray    # (N,) metres from the root to the commanded placement
    pose_name: str


def rehearse_approach(
    pose: PoseRecord,
    *,
    travel_m: float,
    seconds: float,
    style: str = "walk_boxing",
    seed: int = 1234,
    prime_frames: int = 20,
    generator: MotionBricksGenerator | None = None,
) -> ApproachRehearsal:
    """Walk toward a placement with the pose armed on every replan, and keep what it produces.

    This is the measurement bench for `spec/intent.md` 2.0: unlike :func:`rehearse_commit` the plan
    is never consumed whole, and unlike :func:`rehearse` the pose is armed the entire time. The
    length is left to the model (``horizon_tokens=None``).
    """
    if travel_m <= 0.0:
        raise RehearsalError(f"travel_m must be positive, got {travel_m}")
    if seconds <= 0.0:
        raise RehearsalError(f"seconds must be positive, got {seconds}")

    owned = generator is None
    generator = generator or MotionBricksGenerator(GeneratorConfig(random_seed=seed))
    if style not in generator.clip_names:
        raise RehearsalError(
            f"unknown style {style!r}; available: {', '.join(sorted(generator.clip_names))}"
        )

    generator.reset(seed=seed)
    for _ in range(prime_frames):
        generator.next_frame()

    goal = np.asarray(generator.context_qpos()[-1][:2], dtype=float) + np.array([travel_m, 0.0])
    target_angles = pose.to_array()
    replan_every = int(round(REPLAN_DT * GENERATOR_HZ))

    frames, errors, distances = [], [], []
    for index in range(int(seconds * GENERATOR_HZ)):
        if index % replan_every == 0:
            intent = GeneratorIntent(
                style=style,
                target_position=(float(goal[0]), float(goal[1])),
                target_heading=0.0,
                facing_angle=0.0,
                pose=pose,
                horizon_tokens=None,
            )
            generator.generate(intent, generator.context_qpos(), dt=REPLAN_DT, force=True)

        frame = generator.next_frame()
        frames.append(frame)
        errors.append(float(np.abs(frame[7:] - target_angles).mean()))
        distances.append(float(np.linalg.norm(goal - np.asarray(frame[:2], dtype=float))))

    qpos = np.asarray(frames)
    if not np.isfinite(qpos).all():
        raise RehearsalError("the generator produced a non-finite frame")

    if owned:
        del generator

    return ApproachRehearsal(
        qpos=qpos,
        pose_error_rad=np.asarray(errors),
        distance_to_goal=np.asarray(distances),
        pose_name=pose.name,
    )
```

- [ ] **Step 4: Run it and watch it pass**

```bash
.venv_mb/bin/python -m pytest tests/test_rehearsal.py -m slow -k armed_approach -v
```

Expected: PASS. (Loads MotionBricks; allow ~30 s.)

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/studio/rehearsal.py tests/test_rehearsal.py
git commit -m "feat(studio): rehearse_approach - a replanned approach with the pose armed throughout

M4-T4: the measurement bench for spec/intent.md 2.0. Neither rehearse() (no pose) nor
rehearse_commit() (one forced plan) does what a continuous commit will do.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Measure the dwell and the admission tolerance

Two numbers the design cannot be implemented without. `CLAUDE.md` standing rule 3: never invent a number.

**Files:**
- Create: `src/openroboxing/tools/measure_dwell.py`
- Test: `tests/test_rehearsal.py`

- [ ] **Step 1: Write the failing test for the pure helper**

The tool's only non-I/O logic is "where does this error curve flatten out". Test that alone.

Append to `tests/test_rehearsal.py`:

```python
def test_settle_index_finds_where_a_curve_reaches_its_asymptote() -> None:
    """Reproduce: .venv_mb/bin/python -m pytest tests/test_rehearsal.py -k settle_index -v"""
    import numpy as np

    from openroboxing.tools.measure_dwell import settle_index

    # Falls for 10 samples, then flat. The asymptote is reached at index 10.
    curve = np.concatenate([np.linspace(1.0, 0.1, 11), np.full(20, 0.1)])
    assert settle_index(curve, from_index=0) == 10

    # Already flat: settles immediately.
    assert settle_index(np.full(15, 0.2), from_index=3) == 3


def test_settle_index_rejects_a_start_outside_the_curve() -> None:
    import numpy as np
    import pytest as _pytest

    from openroboxing.tools.measure_dwell import settle_index

    with _pytest.raises(ValueError):
        settle_index(np.zeros(5), from_index=5)
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv_mb/bin/python -m pytest tests/test_rehearsal.py -k settle_index -v
```

Expected: `ModuleNotFoundError: No module named 'openroboxing.tools.measure_dwell'`.

- [ ] **Step 3: Write the tool**

Create `src/openroboxing/tools/measure_dwell.py`:

```python
"""Measure the two numbers `spec/intent.md` 2.0 needs (M4-T4).

1. ``POSE_DWELL_TICKS`` - how long after arriving a fighter needs before the pose has settled,
   so the next queued commit does not cut the strike short.
2. The admission tolerance for ``generator_error_rad`` under continuous arming, which is looser
   than the forced-plan number every pose in ``poses/v0.1`` was admitted against.

Both are reported per pose and as a distribution. Nothing is written; the numbers go into
``spec/constants.py`` and ``spec/rates.md`` by hand, with this tool's output as their citation.

Usage
-----
    .venv_mb/bin/python -m openroboxing.tools.measure_dwell
    .venv_mb/bin/python -m openroboxing.tools.measure_dwell --travel 2.5 --seconds 8
"""

from __future__ import annotations

import argparse

import numpy as np

from openroboxing.paths import POSE_DIR
from openroboxing.spec.constants import ARRIVAL_RADIUS_M, GENERATOR_HZ, TICK_HZ
from openroboxing.studio import pose_record
from openroboxing.studio.rehearsal import rehearse_approach

#: How long a curve must stop improving before it counts as settled, in generator frames. One
#: replan interval: anything shorter cannot distinguish settling from the gap between two plans.
SETTLE_WINDOW_FRAMES = 15


def settle_index(curve: np.ndarray, *, from_index: int) -> int:
    """The first index at or after ``from_index`` where ``curve`` reaches its asymptote.

    The asymptote is the curve's minimum over the remainder; "reached" means within the noise band
    of that minimum, where the band is the spread of the final :data:`SETTLE_WINDOW_FRAMES` samples.
    Returns ``from_index`` when the curve is already flat.
    """
    curve = np.asarray(curve, dtype=float)
    if not 0 <= from_index < curve.shape[0]:
        raise ValueError(f"from_index {from_index} outside a curve of {curve.shape[0]} samples")

    tail = curve[from_index:]
    floor = float(tail.min())
    band = float(np.ptp(curve[-SETTLE_WINDOW_FRAMES:])) if curve.shape[0] >= SETTLE_WINDOW_FRAMES else 0.0
    reached = np.flatnonzero(tail <= floor + band)
    return int(from_index + (reached[0] if reached.size else 0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="measure_dwell",
        description="Measure POSE_DWELL_TICKS and the continuous-arming pose tolerance (M4-T4).",
    )
    parser.add_argument("--travel", type=float, default=2.5, help="metres to the placement")
    parser.add_argument("--seconds", type=float, default=8.0, help="how long to rehearse each pose")
    parser.add_argument("--library", default="v0.1", help="pose library under poses/")
    args = parser.parse_args(argv)

    records = pose_record.load_library(POSE_DIR / args.library)
    print(f"{len(records)} poses from {args.library}, placement {args.travel} m away\n")
    print(f"{'pose':<16} {'arrive_s':>9} {'settle_s':>9} {'dwell_ticks':>12} {'final_err':>10}")

    dwells, errors = [], []
    for name in sorted(records):
        result = rehearse_approach(records[name], travel_m=args.travel, seconds=args.seconds)

        inside = np.flatnonzero(result.distance_to_goal <= ARRIVAL_RADIUS_M)
        if inside.size == 0:
            print(f"{name:<16} {'never':>9} {'-':>9} {'-':>12} "
                  f"{np.degrees(result.pose_error_rad[-1]):>9.1f}d")
            continue

        arrived = int(inside[0])
        settled = settle_index(result.pose_error_rad, from_index=arrived)
        dwell_ticks = int(np.ceil((settled - arrived) / GENERATOR_HZ * TICK_HZ))
        final = float(result.pose_error_rad[settled])

        dwells.append(dwell_ticks)
        errors.append(final)
        print(f"{name:<16} {arrived / GENERATOR_HZ:>9.2f} {settled / GENERATOR_HZ:>9.2f} "
              f"{dwell_ticks:>12} {np.degrees(final):>9.1f}d")

    if not dwells:
        print("\nno pose arrived; nothing to report")
        return 1

    print(f"\nPOSE_DWELL_TICKS  = {max(dwells)}   "
          f"(upper end of {min(dwells)}..{max(dwells)}, so the slowest pose completes)")
    print(f"pose error        = {np.degrees(np.mean(errors)):.1f}d mean, "
          f"{np.degrees(max(errors)):.1f}d worst over {len(errors)} poses")
    print(f"GENERATOR_POSE_TOLERANCE_RAD >= {max(errors):.4f}  "
          f"({np.degrees(max(errors)):.1f}d, the worst pose in the library)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the unit test and watch it pass**

```bash
.venv_mb/bin/python -m pytest tests/test_rehearsal.py -k settle_index -v
```

Expected: 2 passed.

- [ ] **Step 5: Take the measurement**

```bash
.venv_mb/bin/python -m openroboxing.tools.measure_dwell 2>&1 | tail -20
```

Expected: a row per pose and the two summary numbers. **Record the printed output — it is the citation for the constants in Task 3.** If any pose prints `never`, stop and report: a pose that cannot be approached is a library problem, not a runtime one.

- [ ] **Step 6: Commit**

```bash
git add src/openroboxing/tools/measure_dwell.py tests/test_rehearsal.py
git commit -m "feat(tools): measure_dwell - the two numbers intent 2.0 needs

M4-T4: POSE_DWELL_TICKS and the continuous-arming pose tolerance, measured per pose
rather than chosen (CLAUDE.md standing rule 3).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: The constants

**Files:**
- Modify: `src/openroboxing/spec/constants.py`
- Modify: `src/openroboxing/spec/rates.md`
- Test: `tests/test_conventions.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_conventions.py`:

```python
def test_the_pose_dwell_is_a_sane_number_of_ticks() -> None:
    """Reproduce: .venv_mb/bin/python -m pytest tests/test_conventions.py -k pose_dwell -v"""
    from openroboxing.spec.constants import POSE_DWELL_TICKS, TICK_HZ

    assert isinstance(POSE_DWELL_TICKS, int)
    assert 0 < POSE_DWELL_TICKS <= 2 * TICK_HZ, "a dwell longer than 2 s is a different game"


def test_the_generator_pose_tolerance_is_in_radians() -> None:
    import math

    from openroboxing.spec.constants import GENERATOR_POSE_TOLERANCE_RAD

    assert 0.0 < GENERATOR_POSE_TOLERANCE_RAD < math.radians(30.0)
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv_mb/bin/python -m pytest tests/test_conventions.py -k "pose_dwell or pose_tolerance" -v
```

Expected: `ImportError: cannot import name 'POSE_DWELL_TICKS'`.

- [ ] **Step 3: Add the constants**

Append to `src/openroboxing/spec/constants.py`, after `MAX_OUTSTANDING_COMMITS`. **Replace `<N>` and
`<X.XXXX>` with the numbers Task 2 printed, and paste the tool's own output into the docstring.**

```python
POSE_DWELL_TICKS: int = <N>
"""How long a fighter stands in its committed pose after arriving, before the next commit starts.

Measured, not chosen (`tools/measure_dwell.py`, 2026-08-13): the interval from entering
ARRIVAL_RADIUS_M until the plan's joint error against the pose reaches its asymptote, taken at the
upper end of the library's distribution so the slowest pose still completes inside it.

Without it the next queued commit becomes current at the instant of arrival and the strike is cut
short - which is the "melted commits" the owner reported at 1.1. See spec/intent.md 2.0.
"""

GENERATOR_POSE_TOLERANCE_RAD: float = <X.XXXX>
"""How close the generator must get to an authored pose for it to be admitted.

Measured (`tools/measure_dwell.py`, 2026-08-13) as the worst pose in poses/v0.1 under continuous
arming. Looser than the forced-plan figure the library was originally admitted against, because a
continuously-armed pose is converged on rather than landed exactly - see spec/pose_record.md.
"""
```

- [ ] **Step 4: Document them in `spec/rates.md`**

Add to the table in `src/openroboxing/spec/rates.md`, with the same derivation text, and add a changelog line: `POSE_DWELL_TICKS and GENERATOR_POSE_TOLERANCE_RAD added for spec/intent.md 2.0, both measured by tools/measure_dwell.py.`

- [ ] **Step 5: Run the tests and watch them pass**

```bash
.venv_mb/bin/python -m pytest tests/test_conventions.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/openroboxing/spec/constants.py src/openroboxing/spec/rates.md tests/test_conventions.py
git commit -m "feat(spec): POSE_DWELL_TICKS and GENERATOR_POSE_TOLERANCE_RAD, both measured

M4-T4: cited to tools/measure_dwell.py rather than chosen.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: One continuous intent, and the hold

The behaviour change. `runtime/intents.py` only — the stream simplification is Task 5.

**Files:**
- Modify: `src/openroboxing/runtime/intents.py`
- Modify: `src/openroboxing/runtime/generator.py` (drop `plan_key`)
- Test: `tests/test_intents.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_intents.py`. These encode the three claims of the design.

```python
def test_a_walking_commit_already_carries_its_pose() -> None:
    """Intent 2.0: there is no poseless approach. Reproduce:
    .venv_mb/bin/python -m pytest tests/test_intents.py -k walking_commit_already -v
    """
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement(position=(3.0, 0.0), heading=0.0))
    timeline.commit(0)

    # Never arrives, so every frame below is mid-approach.
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


def test_a_commit_ends_a_dwell_after_it_arrives() -> None:
    from openroboxing.spec.constants import POSE_DWELL_TICKS

    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement(position=(0.5, 0.0), heading=0.0))
    commit = timeline.commit(0)

    for t in range(COMMIT_HORIZON_TICKS, COMMIT_HORIZON_TICKS + 5):
        timeline.generator_intent(t, has_arrived=lambda _c: True)

    assert commit.strike_at == COMMIT_HORIZON_TICKS
    assert commit.end_tick == commit.strike_at + POSE_DWELL_TICKS


def test_the_next_commit_starts_when_the_dwell_ends() -> None:
    timeline = _timeline()
    timeline.stage(pose_slot="1", placement=Placement(position=(0.5, 0.0), heading=0.0))
    first = timeline.commit(0)
    timeline.stage(pose_slot="2", placement=Placement(position=(1.0, 0.0), heading=0.0))
    second = timeline.commit(1)

    for t in range(COMMIT_HORIZON_TICKS, COMMIT_HORIZON_TICKS + 300):
        timeline.generator_intent(t, has_arrived=lambda _c: True)

    assert second.commit_at == first.end_tick, "queued moves still run back to back"
```

- [ ] **Step 2: Run them and watch them fail**

```bash
.venv_mb/bin/python -m pytest tests/test_intents.py -k "walking_commit_already or drained_queue_holds or dwell" -v
```

Expected: FAIL — `pose is None` mid-approach, and `after.style == "idle"`.

- [ ] **Step 3: Make `end_tick` come from the dwell**

In `src/openroboxing/runtime/intents.py`, in the `Commit` dataclass: delete the `duration_ticks: int`
field (line 231) and replace the `end_tick` property (lines 246-253) with:

```python
    @property
    def end_tick(self) -> int | None:
        """The tick this commit finishes, or ``None`` while it has not arrived yet.

        ``None`` means *later than any tick you can name*, not zero and not "already over".

        A commit ends :data:`POSE_DWELL_TICKS` after it arrives. The dwell is the move: the fighter
        stands in the pose it was commanded into. Before `spec/intent.md` 2.0 this was a pose-phase
        length derived from the pose's token count, which is what made a queued move cut the one in
        front of it short.
        """
        return None if self.strike_at is None else self.strike_at + POSE_DWELL_TICKS
```

Delete the `duration_ticks()` function (lines 274-282) and its `MIN_TOKENS`/`MAX_TOKENS`/
`SECONDS_PER_TOKEN` imports if they become unused. In `commit()` (line 530) delete the
`duration_ticks=duration_ticks(pose.horizon_tokens),` argument. Add `POSE_DWELL_TICKS` to the
`spec.constants` import at the top of the file.

- [ ] **Step 4: Arm the pose for the whole move, and hold on a drained queue**

Replace `generator_intent`'s body from line 577 to the end of the method with:

```python
        commit = self.current(tick)
        if commit is None:
            return self._hold_intent(facing_angle)

        if commit.commit_at is None:
            commit.commit_at = tick

        if commit.strike_at is None:
            self._resolve_approach(commit, tick, has_arrived)

        self._last_intent = self._intent_for(commit, facing_angle)
        return self._last_intent

    def _intent_for(self, commit: Commit, facing_angle: float) -> GeneratorIntent:
        """One continuous intent: *be at this placement, in this pose*.

        The pose is armed for the commit's whole life, not only once it has arrived, and the length
        is left to the model. `spec/intent.md` 2.0 - measured: an armed approach converges on the
        pose while it walks and arrives sooner than a poseless one.
        """
        placement = commit.placement
        return GeneratorIntent(
            style=commit.context,
            facing_angle=placement.heading if placement else facing_angle,
            target_position=placement.position if placement else None,
            target_heading=placement.heading if placement else None,
            pose=commit.pose,
            horizon_tokens=None,
        )

    def _hold_intent(self, facing_angle: float) -> GeneratorIntent:
        """What a fighter does with nothing to do: **hold the pose it was last commanded into.**

        Keeping the last commit's intent armed is the whole implementation. There is no idle clip
        and no freeze branch - the generator keeps in-betweening toward a target it has already
        reached, which is a fighter standing still in that pose. Because the intent carries its own
        ``target_heading`` a held fighter does not turn to track its opponent; re-orienting is paid
        for by the next commit. Both decided by the project owner, 2026-08-13.

        Before any commit has run in this round there is nothing to hold, so the ambient context
        stands in - that is a round's opening stance, not a fallback.
        """
        if self._last_intent is not None:
            return self._last_intent
        return GeneratorIntent(style=self.context, facing_angle=facing_angle)
```

In `__init__`, add `self._last_intent: GeneratorIntent | None = None`, and in whatever resets
per-round state set it back to `None`.

- [ ] **Step 5: Delete `plan_key`**

In `src/openroboxing/runtime/generator.py` delete the `plan_key: object | None = None` field (line 192)
and its docstring entry (lines 179-181). In `runtime/intents.py` delete `HOLD_CONTEXT` (lines
183-188) and `Commit.is_approaching`'s docstring reference to a pose phase — the method itself
stays, since `server/protocol.py:152` sends it and "executing but not yet arrived" is still exactly
what it means.

- [ ] **Step 6: Run the new tests and watch them pass**

```bash
.venv_mb/bin/python -m pytest tests/test_intents.py -k "walking_commit_already or drained_queue_holds or dwell" -v
```

Expected: 4 passed.

- [ ] **Step 7: Rewrite the tests that encoded the old model**

```bash
.venv_mb/bin/python -m pytest tests/test_intents.py -v 2>&1 | tail -40
```

Every failure here is a test asserting the two-phase rules. Rewrite each to assert the 2.0 rule it
corresponds to; **delete** the ones whose subject no longer exists (`HOLD_CONTEXT`,
`plan_key`, `duration_ticks`, "a walking commit carries no pose"). Do not weaken an assertion to
make it pass — if a test cannot be restated under 2.0, it is testing something the design deleted.

- [ ] **Step 8: Commit**

```bash
git add src/openroboxing/runtime/intents.py src/openroboxing/runtime/generator.py tests/test_intents.py
git commit -m "feat(intents): a commit is one continuous intent, and a drained queue holds its pose

M4-T4, spec/intent.md 2.0. The pose is armed for a commit's whole life instead of only
during a final forced plan, the length is left to MotionBricks, and end_tick comes from
the measured POSE_DWELL_TICKS. A drained queue keeps the last commit's intent armed
rather than switching to the idle clip.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Delete the forced-plan machinery

With no forced plans, `runtime/reference.py` has one path: consume a frame, replan at the cadence.

**Files:**
- Modify: `src/openroboxing/runtime/reference.py`
- Test: `tests/test_replay.py`, `tests/test_intents.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rehearsal.py` (it already has the generator fixtures):

```python
@pytest.mark.slow
def test_the_stream_holds_a_pose_indefinitely_without_raising() -> None:
    """Under 1.1 a plan outliving its move by more than one frame raised. Holding is now normal.

    Reproduce: .venv_mb/bin/python -m pytest tests/test_rehearsal.py -m slow -k holds_a_pose_indefinitely -v
    """
    from openroboxing.paths import POSE_DIR
    from openroboxing.runtime.generator import GeneratorConfig, GeneratorIntent, MotionBricksGenerator
    from openroboxing.runtime.reference import ReferenceStream
    from openroboxing.studio import pose_record

    pose = pose_record.load(POSE_DIR / "v0.1" / "guard.json")
    generator = MotionBricksGenerator(GeneratorConfig(random_seed=1234))
    stream = ReferenceStream(generator)

    held = GeneratorIntent(style="walk_boxing", pose=pose, horizon_tokens=None)
    stream.ensure(lambda _tick: held, tick=0)
    stream.ensure(lambda _tick: held, tick=500)   # 10 s of holding one pose

    assert stream.motion.shape[0] >= 500
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv_mb/bin/python -m pytest tests/test_rehearsal.py -m slow -k holds_a_pose_indefinitely -v
```

Expected: `ReferenceError: a committed plan ran out ... frames before its move ended`.

- [ ] **Step 3: Simplify `ensure`**

In `src/openroboxing/runtime/reference.py`, replace the loop body at lines 152-163 with:

```python
            intent = intent_at(self.tick_of_frame(len(self._frames)))
            self._frames.append(self.generator.next_frame())
            self._plan(intent, force=False)
```

Delete `_committed_frame` (lines 170-212), `_committed_plan_length` (lines 219-244),
`MAX_HELD_STRIKE_FRAMES` (lines 82-86), and the `_plan_key` / `_committed_plan_remaining` /
`_held_strike_frames` attributes wherever they are initialised or reset. Update the module docstring
table (around lines 8-40) so it describes one path rather than three.

- [ ] **Step 4: Run it and watch it pass**

```bash
.venv_mb/bin/python -m pytest tests/test_rehearsal.py -m slow -k holds_a_pose_indefinitely -v
```

Expected: PASS.

- [ ] **Step 5: Run the whole fast suite**

```bash
.venv_mb/bin/python -m pytest tests -q 2>&1 | tail -20
```

Expected: green, or failures only in tests that name the deleted machinery. Fix those the same way
as Task 4 Step 7.

- [ ] **Step 6: Commit**

```bash
git add src/openroboxing/runtime/reference.py tests/
git commit -m "refactor(reference): delete the forced-plan machinery

M4-T4, spec/intent.md 2.0. With the pose armed continuously there are no forced plans to
bind, consume exactly, or count leftover frames from. Removes _committed_frame,
_committed_plan_length and MAX_HELD_STRIKE_FRAMES, and with them the lost strike frame on
8-token poses, the frame leaked into the next commit, and the cadence bypass at
end-of-strike.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Prove it end to end

**Files:** none modified — this is the acceptance criterion.

- [ ] **Step 1: Run a match**

```bash
.venv_mb/bin/python -m openroboxing.tools.run_match --rounds 1 --round-seconds 20 --seed 7 --out /tmp/m.json
```

Expected: it completes, reports hits, and does not raise.

- [ ] **Step 2: Check every commit that threw has a `strike_at`**

```bash
.venv_mb/bin/python -c "
import json; r=json.load(open('/tmp/m.json'))['rounds'][0]
for c in r['commits']:
    print(c['fighter'], c['pose_name'], c['commit_at'], c['strike_at'], c['end_tick'], c['arrived'])
"
```

Expected: `end_tick - strike_at == POSE_DWELL_TICKS` for every completed commit, and `strike_at` is
non-null wherever the commit threw — `league/scoring.py:235` reads it to decide whether the fighters
were in range.

- [ ] **Step 3: Check scoring and replay still work**

```bash
.venv_mb/bin/python -m openroboxing.tools.score_match /tmp/m.json
.venv_mb/bin/python -m openroboxing.tools.replay_match /tmp/m.json --rescore
```

Expected: a scoreline, and knockdowns re-deriving from the trace.

- [ ] **Step 4: Watch it**

```bash
.venv_mb/bin/python -m openroboxing.tools.replay_match /tmp/m.json --video /tmp/m.mp4
```

Watch `/tmp/m.mp4`. The three reported symptoms are the acceptance test: the fighter should **stay**
in its pose after a move, consecutive moves should read as separate actions, and no commit should
vanish. **If any of the three is still visible, stop and report** rather than tuning — the design
is falsified and the fallback in the design doc is the next decision, not a parameter change.

- [ ] **Step 5: Commit nothing; record the result in the PR body.**

---

### Task 7: Surface refused commits (independent defect)

`QueuedPilot.act` catches every `IntentError` into `self.last_error`, which is written at
`server/host.py:112`, `:130`, `:161` and **read nowhere**. A player whose commit is refused sees
nothing at all. Independent of everything above.

**Files:**
- Modify: `src/openroboxing/server/host.py`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`:

```python
def test_a_refused_commit_tells_the_player_why() -> None:
    """A commit refused by the timeline must reach the client. Reproduce:
    .venv_mb/bin/python -m pytest tests/test_server.py -k refused_commit -v
    """
    from openroboxing.runtime.intents import IntentError
    from openroboxing.server.host import QueuedPilot

    class _Refuses:
        def commit(self, _tick):
            raise IntentError("the queue is full")

        def stage(self, **_kwargs):
            return None

    pilot = QueuedPilot()
    pilot.queue({"type": "commit"})
    errors = pilot.act(_Refuses(), tick=0)

    assert errors, "a refused commit produces something the host can send"
    assert "queue is full" in errors[0]
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv_mb/bin/python -m pytest tests/test_server.py -k refused_commit -v
```

Expected: FAIL — `act` returns `None`.

- [ ] **Step 3: Return the errors instead of swallowing them**

In `src/openroboxing/server/host.py`, change `QueuedPilot.act` to collect and return the messages rather
than storing the last one, replacing the `except IntentError as exc: self.last_error = str(exc)`
at line 161:

```python
            except IntentError as exc:
                refused.append(str(exc))
```

with `refused: list[str] = []` at the top of `act` and `return refused` at the end. Delete the
`last_error` attribute at lines 112 and 130 — nothing reads it, and `CLAUDE.md` prefers deleting to
disabling. At the call site in `MatchHost`, send each refusal to that seat with the existing
`protocol.error` builder, the same way a queue-full rejection is already reported at `:578-584`.

- [ ] **Step 4: Run it and watch it pass**

```bash
.venv_mb/bin/python -m pytest tests/test_server.py -k refused_commit -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/server/host.py tests/test_server.py
git commit -m "fix(host): a refused commit tells the player, instead of vanishing

QueuedPilot.act wrote every IntentError to self.last_error, which nothing read - so a
commit refused for a full queue, an unstaged pose or a bad slot produced no error, no log
and no UI. Reported by the project owner as 'some commits are skipped'.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Re-measure the library, and bump the specs

**Files:**
- Modify: `src/openroboxing/poses/v0.1/*.json`, `src/openroboxing/spec/intent.md`, `src/openroboxing/spec/pose_record.md`

- [ ] **Step 1: Re-measure admission**

```bash
.venv_mb/bin/python -m openroboxing.tools.build_library 2>&1 | tail -20
```

Expected: every pose re-measured against `GENERATOR_POSE_TOLERANCE_RAD`. **If a pose fails, stop and
report** — re-authoring it is a Studio job, not a runtime one.

- [ ] **Step 2: Re-measure the telegraph windows**

```bash
for p in guard jab-left jab-right hook-left hook-right uppercut-left uppercut-right slip-left slip-right cover; do
  .venv_mb/bin/python -m openroboxing.tools.measure_telegraph --pose src/openroboxing/poses/v0.1/$p.json
done
```

Expected: a window per pose. These are defined on generated motion, which this design changed, so
the recorded values are stale until this runs.

- [ ] **Step 3: Write `spec/intent.md` 2.0**

Replace §"A commit's span", §"What actually sets the floor" (keep the inert-horizon finding),
§"Arrival", and §"The pose target is reached — and must not be replanned over" with the 2.0 model:
one continuous intent, the pose armed throughout, model-chosen length, `end_tick` from
`POSE_DWELL_TICKS`, and the hold. Add a changelog entry citing the design doc and the measurements.

- [ ] **Step 4: Bump `spec/pose_record.md`**

Document that `horizon_tokens` is now the Studio's rehearsal parameter and the author's statement of
intended length, no longer forced on the generator; and that `generator_error_rad` is admitted
against `GENERATOR_POSE_TOLERANCE_RAD` measured under continuous arming. While there, fix the
pre-existing contradiction found on 2026-08-13: the spec says an admitted record with a null
`telegraph_ms` is invalid, but the validator requires only `generator_error_rad` and all ten shipped
poses have `telegraph_ms: null`.

- [ ] **Step 5: Run everything**

```bash
.venv_mb/bin/python -m pytest tests -q 2>&1 | tail -10
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/openroboxing/spec/ src/openroboxing/poses/
git commit -m "docs(spec): intent 2.0, and the library re-admitted under continuous arming

M4-T4. Also fixes the pose_record spec contradiction found 2026-08-13: the written rule
that a null telegraph_ms is invalid never matched the validator, and every shipped pose
violated it.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-review notes

**Spec coverage.** Every section of the design doc maps to a task: the model → Task 4; deletions →
Tasks 4 and 5; numbers to measure → Tasks 1–3; testing → the tests inside each task plus Task 6;
the independent defect → Task 7; spec bumps and re-admission → Task 8. The design's "not in scope"
list (input latency, queue depth, `TARGET_COMMIT_RATE`) has no tasks, correctly.

**Ordering.** Tasks 1–3 change no behaviour and are safe to land alone. Task 4 is the breaking
change and carries its test rewrites in the same commit. Task 7 is independent and may be done at
any point, including first.

**The one thing that can falsify this plan** is Task 6 Step 4: if the three reported symptoms are
still visible after the change, the design is wrong rather than mistuned, and the fallback
(a final forced plan, keeping the continuous arming) is the next decision — not a parameter.
