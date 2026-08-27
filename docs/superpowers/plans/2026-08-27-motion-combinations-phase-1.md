# Motion Combinations — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 38 mocap CSVs in `motions/` into a validated library of ~120 timed multi-keyframe combinations, plus the pure warp maths that will place them in the ring — ending in a measurement that says whether the runtime phases are worth building.

**Architecture:** Everything here sits *left of the `CombinationRecord` seam* defined in [the design](../specs/2026-08-27-motion-combinations-design.md). CSV → `(N, 36)` MuJoCo qpos → segmented keyframes → `CombinationRecord` JSON on disk. `runtime/warp.py` is included because it is pure maths with no runtime dependency, and because the go/no-go spike needs it. No changes to `intents.py`, `reference.py`, the client or the protocol — those are Plan 2.

**Tech Stack:** Python 3.10+, numpy, scipy (`Rotation`), pytest, mujoco (via `runtime.conventions` only). Run everything with `.venv_mb/bin/python`. The venv exists and the suite is green at 667 passed.

**Scope note:** The design has four phases. This plan is phase 1 plus the pure half of phase 2. Phase 2's runtime integration (`spec/intent.md` 3.0, `sequence.py`, deleting the walk approach), phase 3 (admission) and phase 4 (client/protocol) get their own plans, written after Task 12's measurement lands.

---

## File Structure

**Create:**

| Path | Responsibility |
|---|---|
| `src/openroboxing/spec/combination.md` | the versioned schema (invariant 7: spec before implementation) |
| `src/openroboxing/studio/motion_import.py` | CSV → `(N, 36)` MuJoCo qpos. Units and conventions only. |
| `src/openroboxing/studio/segment.py` | qpos → keyframe indices → leg token counts. No I/O, no records. |
| `src/openroboxing/studio/combination_record.py` | the `CombinationRecord` dataclass, validation, JSON, and assembly from a take |
| `src/openroboxing/runtime/warp.py` | pure warp maths: record + anchor + ghost → per-leg intents |
| `src/openroboxing/tools/pin_euler_order.py` | one-off measurement that determines the CSV's Euler convention |
| `src/openroboxing/tools/import_motions.py` | builds the library into `poses/v0.2/combinations/` |
| `src/openroboxing/tools/spike_warp_tracking.py` | the go/no-go measurement (Task 12) |
| `tests/test_motion_import.py`, `tests/test_segment.py`, `tests/test_combination_record.py`, `tests/test_warp.py`, `tests/test_motion_corpus_golden.py` | |

**Modify:**

| Path | Change |
|---|---|
| `src/openroboxing/paths.py` | add `MOTIONS_DIR`, `COMBINATION_DIR` |
| `src/openroboxing/spec/constants.py` | add `MIN_KEYFRAME_GAP_FRAMES`, `MAX_LEG_FRAMES`, `KEYFRAME_QUANTILE`, `COMBINATION_MIN_KEYFRAMES`, `COMBINATION_MAX_KEYFRAMES` |

**Why this split:** `motion_import` knows about degrees and centimetres and nothing else. `segment` knows about speed and tokens and takes plain arrays. `combination_record` knows the schema. `warp` knows geometry. Each is testable with no dependency on the others beyond plain numpy arrays, and none of them import the generator.

---

### Task 1: The schema spec and its constants

**Files:**
- Create: `src/openroboxing/spec/combination.md`
- Modify: `src/openroboxing/paths.py`, `src/openroboxing/spec/constants.py`

`CLAUDE.md` invariant 7 requires the versioned schema before the implementation. There is no test for a markdown file; the tests arrive in Task 7 when the dataclass does.

- [ ] **Step 1: Write `src/openroboxing/spec/combination.md`**

```markdown
# combination.md — the selectable motion combination

Version **0.1** · created 2026-08-27 · design `docs/superpowers/specs/2026-08-27-motion-combinations-design.md`

A combination is one selectable move: an ordered run of 3–6 key poses with the timing they were
recorded at. It replaces the single `pose_record.md` key pose as the unit a loadout slot holds.

---

## Fields

| Field | Type | Notes |
|---|---|---|
| `schema_version` | str | `"0.1"` |
| `name` | str | unique within a library, kebab-case |
| `library_version` | str | the library release this belongs to |
| `source` | object | `take`, `start_frame`, `end_frame`, `mirrored` |
| `keyframes` | list | 3–6 entries, in order; see below |
| `duration_ticks` | int | total length at `TICK_HZ`, derived from `leg_tokens` |
| `recorded_displacement` | `[dx, dy]` | metres, keyframe 0 → last, in the take's own frame |
| `recorded_heading_delta` | float | radians, keyframe 0 → last |
| `telegraph_ms` | float \| null | **measured**, never authored |
| `tracking_error_rad` | float \| null | **measured** in a runtime trial |
| `admission` | str | `"draft"` \| `"admitted"` \| `"rejected"` |

### A keyframe

| Field | Type | Notes |
|---|---|---|
| `joint_angles` | map name → float | **all 29**, radians, MuJoCo joint names — identical to `pose_record.md` |
| `leg_tokens` | int \| null | length of the leg *ending* here, in tokens, 6–16. `null` on keyframe 0 only. |
| `root_offset` | `[dx, dy]` | metres, **relative to keyframe 0** |
| `heading_offset` | float | radians, **relative to keyframe 0** |

`root_offset` and `heading_offset` are relative to keyframe 0 because a combination **starts in
place**: the fighter does not travel to a start, so keyframe 0 is wherever it already stands. Both
are `[0, 0]` / `0.0` on keyframe 0 by construction.

`leg_tokens` is bounded by `MIN_TOKENS`..`MAX_TOKENS` because that is what MotionBricks can plan
(`spec/rates.md`). A recorded gap longer than `MAX_TOKENS` is split into several legs that repeat the
same keyframe — holding a pose is re-arming the same target, which is existing behaviour.

## Why the measured fields are nullable

Identical to `pose_record.md`: they are outputs of measurement, not inputs of authoring. A record
claiming `"admitted"` with either field null is **invalid**.

## Relationship to pose_record.md

A keyframe's `joint_angles` is exactly a `PoseRecord`'s 29 angles, so a combination reuses pose
validation rather than restating it. A combination is not a superset of a pose record: it carries no
`adjustment_envelope` and no `horizon_tokens`, because length is per-leg and recorded, not authored.
```

- [ ] **Step 2: Add the paths**

In `src/openroboxing/paths.py`, after the `LOADOUT_DIR` definition (line ~78):

```python
MOTIONS_DIR: Path = REPO_ROOT / "motions"
"""The mocap corpus: Maya-style CSV exports, one per take. See spec/combination.md."""

COMBINATION_DIR: Path = POSE_DIR / "v0.2/combinations"
"""Built combination records. Produced by tools/import_motions.py, not authored by hand."""
```

- [ ] **Step 3: Add the constants**

In `src/openroboxing/spec/constants.py`, after the `SECONDS_PER_TOKEN` definition:

```python
# --- Motion combinations (spec/combination.md) ---------------------------------------------------
MIN_KEYFRAME_GAP_FRAMES: int = MIN_TOKENS * NUM_FRAMES_PER_TOKEN
"""Closest two keyframes may sit, in corpus frames = 24 = 0.8 s.

Not a taste parameter: `MIN_TOKENS` is the shortest plan MotionBricks can produce, so two keyframes
closer than this cannot both be in-betweened to. Source: spec/combination.md.
"""

MAX_LEG_FRAMES: int = MAX_TOKENS * NUM_FRAMES_PER_TOKEN
"""Longest single leg, in corpus frames = 64 = 2.13 s.

Not a taste parameter either: `MAX_TOKENS` is the longest plan MotionBricks can produce. A recorded
gap longer than this is *densified* — a keyframe is added at the busiest frame inside it — rather
than held, so every leg is plannable and no leg invents motion the recording does not contain.
"""

KEYFRAME_QUANTILE: float = 0.70
"""How busy a frame must be, against its own take, to be eligible as a keyframe.

The one free parameter of the segmenter, stated rather than buried. A keyframe sits where the body
is moving faster than it does for 70 % of the take.

**Not a peak-detection threshold**, and deliberately so. Measured 2026-08-27: the shadow-boxing takes
have real spikes (median 0.106, peak 1.028 rad/frame of salient motion) but the travelling takes are
uniformly active (median 0.117, peak 0.331), so a "3 sigma above the median" rule admits 120 frames
from one and **4** from the other, and all four `combat_turn_jog_start` takes — the only motions that
cross the ring — produce no combination at all. A quantile makes no assumption that the distribution
is peaked, and covers all 38 takes.
"""

COMBINATION_MIN_KEYFRAMES: int = 3
COMBINATION_MAX_KEYFRAMES: int = 6
"""A combination is 3-6 keyframes. Fewer is not a combination; more runs past the duration the
no-cancellation rule can survive. Source: the design's decision D1.
"""
```

- [ ] **Step 4: Verify nothing broke**

Run: `.venv_mb/bin/python -m pytest -q --no-header 2>&1 | tail -3`
Expected: `667 passed, 28 deselected`

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/spec/combination.md src/openroboxing/paths.py src/openroboxing/spec/constants.py
git commit -m "M5-T1: spec/combination.md 0.1, corpus paths and segmentation constants"
```

---

### Task 2: Read a take, and derive the joint permutation by name

**Files:**
- Create: `src/openroboxing/studio/motion_import.py`
- Test: `tests/test_motion_import.py`

Invariant 4 is the whole point of this task. The CSV's joint order *does* match MuJoCo's — measured 2026-08-27 — and we derive the permutation anyway, because a corpus that changes order silently is exactly the catastrophic bug the invariant exists for.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_motion_import.py`:

```python
"""CSV corpus ingest: units, ordering, and the invertibility invariant."""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.paths import MOTIONS_DIR
from openroboxing.runtime.conventions import G1
from openroboxing.studio import motion_import

TAKE = MOTIONS_DIR / "ib_dodge_up_R_001__A437.csv"


def test_reads_header_and_frames():
    take = motion_import.read_take(TAKE)
    assert take.joint_names == tuple(n[: -len("_dof")] for n in take.raw_joint_columns)
    assert len(take.joint_names) == G1.num_joints
    # 6 root columns + 29 joints. The `Frame` column is dropped, so this is 35, not 36 —
    # the 36 of a qpos is 3 position + 4 quaternion + 29, which `load_take` produces.
    assert take.frames.shape == (591, 35)


def test_joint_permutation_is_invertible():
    take = motion_import.read_take(TAKE)
    perm = motion_import.joint_permutation(take.joint_names)
    x = np.arange(G1.num_joints, dtype=float)
    assert np.array_equal(x[perm][motion_import.invert(perm)], x)


def test_joint_permutation_rejects_a_missing_name():
    names = ("not_a_joint",) + tuple(G1.mujoco_joint_names[1:])
    with pytest.raises(motion_import.MotionImportError, match="not_a_joint"):
        motion_import.joint_permutation(names)


def test_joint_permutation_rejects_a_duplicate():
    names = (G1.mujoco_joint_names[0],) + tuple(G1.mujoco_joint_names[:-1])
    with pytest.raises(motion_import.MotionImportError):
        motion_import.joint_permutation(names)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv_mb/bin/python -m pytest tests/test_motion_import.py -q --no-header 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'openroboxing.studio.motion_import'`

- [ ] **Step 3: Write the implementation**

Create `src/openroboxing/studio/motion_import.py`:

```python
"""Ingest the mocap corpus in ``motions/`` (M5-T2).

Implements the input half of ``spec/combination.md`` 0.1.

Conventions
-----------
- **Input** is a Maya-style CSV export: ``Frame``, ``root_translate{X,Y,Z}`` in **centimetres**,
  ``root_rotate{X,Y,Z}`` in **degrees**, then 29 ``<joint>_dof`` columns in **degrees**. Z is up,
  and the corpus is sampled at :data:`~openroboxing.spec.constants.GENERATOR_HZ` (30 fps).
- **Output** is ``(N, 36)`` MuJoCo qpos: root position (3, **metres**), root quaternion
  (4, **wxyz**), 29 joint angles (**radians**) in **MuJoCo joint order**.
- The joint permutation is derived **by name** and asserted invertible. The corpus order happens to
  match MuJoCo's — measured 2026-08-27 — and is never assumed to (`CLAUDE.md` invariant 4).
- Nothing is coerced and nothing is defaulted: a column that cannot be placed raises.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openroboxing.runtime.conventions import G1, G1Conventions
from openroboxing.spec.constants import QPOS_DIM

#: Suffix the corpus puts on every joint column.
JOINT_SUFFIX = "_dof"

#: The root columns, in the order the corpus writes them.
ROOT_COLUMNS = (
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
)

#: Centimetres to metres. The corpus writes pelvis height as 50-107, which is centimetres.
CM_TO_M = 0.01

#: Value columns per row once ``Frame`` is dropped: 6 root + 29 joints. Not ``QPOS_DIM``, which is
#: 36 because a qpos carries a 4-component quaternion where the corpus carries 3 Euler angles.
CORPUS_COLUMNS = len(ROOT_COLUMNS) + G1.num_joints


class MotionImportError(RuntimeError):
    """A take could not be read or placed. Never recovered from silently."""


@dataclass(frozen=True)
class Take:
    """One mocap take, as written, before any conversion."""

    name: str
    raw_joint_columns: tuple[str, ...]
    joint_names: tuple[str, ...]
    frames: np.ndarray  # (N, 36): 6 root columns then 29 joints, in the corpus's own order


def invert(perm: np.ndarray) -> np.ndarray:
    """The inverse of a permutation, as an index array."""
    out = np.empty_like(perm)
    out[perm] = np.arange(len(perm))
    return out


def joint_permutation(
    joint_names: tuple[str, ...], conventions: G1Conventions = G1
) -> np.ndarray:
    """Indices that gather corpus joint order into MuJoCo joint order.

    Raises:
        MotionImportError: if a MuJoCo joint has no corpus column, or a name repeats.
    """
    if len(set(joint_names)) != len(joint_names):
        raise MotionImportError(f"corpus joint names repeat: {joint_names}")
    index = {name: i for i, name in enumerate(joint_names)}
    missing = [n for n in conventions.mujoco_joint_names if n not in index]
    if missing:
        raise MotionImportError(f"corpus has no column for MuJoCo joints {missing}")
    perm = np.array([index[n] for n in conventions.mujoco_joint_names], dtype=int)
    if not np.array_equal(perm[invert(perm)], np.arange(len(perm))):
        raise MotionImportError("joint permutation is not invertible")
    return perm


def read_take(path: Path) -> Take:
    """Read a corpus CSV as written. No units are converted here."""
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise MotionImportError(f"{path} has no frames")
    header = rows[0]
    if tuple(header[1:7]) != ROOT_COLUMNS:
        raise MotionImportError(f"{path} root columns are {header[1:7]}, expected {ROOT_COLUMNS}")
    raw_joint_columns = tuple(header[7:])
    bad = [c for c in raw_joint_columns if not c.endswith(JOINT_SUFFIX)]
    if bad:
        raise MotionImportError(f"{path} joint columns without a {JOINT_SUFFIX!r} suffix: {bad}")
    joint_names = tuple(c[: -len(JOINT_SUFFIX)] for c in raw_joint_columns)
    if len(joint_names) != G1.num_joints:
        raise MotionImportError(
            f"{path} has {len(joint_names)} joint columns, expected {G1.num_joints}"
        )
    frames = np.array([[float(v) for v in row[1:]] for row in rows[1:]], dtype=np.float64)
    if frames.shape[1] != CORPUS_COLUMNS:
        raise MotionImportError(
            f"{path} has {frames.shape[1]} value columns, expected {CORPUS_COLUMNS}"
        )
    return Take(
        name=path.stem,
        raw_joint_columns=raw_joint_columns,
        joint_names=joint_names,
        frames=frames,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv_mb/bin/python -m pytest tests/test_motion_import.py -q --no-header 2>&1 | tail -3`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/studio/motion_import.py tests/test_motion_import.py
git commit -m "M5-T2: read a corpus take and derive its joint permutation by name"
```

---

### Task 3: Determine the Euler convention by measurement

**Files:**
- Create: `src/openroboxing/tools/pin_euler_order.py`
- Modify: `src/openroboxing/studio/motion_import.py`

The corpus writes `root_rotate{X,Y,Z}` with no stated composition order. The design forbids guessing it. The discriminator: under the correct order the pelvis stays roughly upright throughout a take; under a wrong one, a large heading change composed with a nonzero tilt swings the pelvis away from vertical. The corpus turns up to 267°, so this separates cleanly.

- [ ] **Step 1: Write the tool**

Create `src/openroboxing/tools/pin_euler_order.py`:

```python
"""Determine the corpus's Euler composition order, by measurement (M5-T3).

The corpus writes ``root_rotate{X,Y,Z}`` in degrees without stating how they compose. Guessing is
forbidden (`CLAUDE.md`: most bugs in this project are convention bugs), so this measures.

The criterion is uprightness. A boxer's pelvis stays near vertical for a whole take. Under the wrong
composition order a large heading change combines with the small X/Y tilt to swing the pelvis away
from world Z, and the corpus turns by up to 267 degrees, so the wrong orders are separated by a wide
margin rather than a hair.

Run: ``.venv_mb/bin/python -m openroboxing.tools.pin_euler_order``
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial.transform import Rotation

from openroboxing.paths import MOTIONS_DIR
from openroboxing.studio.motion_import import read_take

#: The twelve Tait-Bryan conventions. Lower case is extrinsic, upper is intrinsic (scipy's rule).
#: Proper-Euler orders (``xyx`` and friends) are excluded: the corpus names three distinct axes.
CANDIDATES = (
    "xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
    "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX",
)


def max_tilt_deg(euler_deg: np.ndarray, order: str) -> float:
    """Worst angle between the pelvis's own Z axis and world Z, over a take, in degrees."""
    matrices = Rotation.from_euler(order, euler_deg, degrees=True).as_matrix()
    cos_tilt = np.clip(matrices[:, 2, 2], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_tilt)).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=str, default=str(MOTIONS_DIR))
    args = parser.parse_args()

    takes = sorted(p for p in __import__("pathlib").Path(args.corpus).glob("*.csv"))
    if not takes:
        raise SystemExit(f"no takes under {args.corpus}")

    scores: dict[str, float] = {}
    for order in CANDIDATES:
        worst = 0.0
        for path in takes:
            worst = max(worst, max_tilt_deg(read_take(path).frames[:, 3:6], order))
        scores[order] = worst

    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    print(f"{'order':>6s}  {'worst pelvis tilt from vertical (deg)':>38s}")
    for order, worst in ranked:
        print(f"{order:>6s}  {worst:38.1f}")
    best, best_score = ranked[0]
    runner_up, runner_score = ranked[1]
    print(
        f"\nbest: {best} at {best_score:.1f} deg; "
        f"runner-up {runner_up} at {runner_score:.1f} deg; "
        f"margin {runner_score - best_score:.1f} deg"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the measurement**

Run: `.venv_mb/bin/python -m openroboxing.tools.pin_euler_order`
Expected: a ranked table of 12 orders. The winner should show a worst tilt plausible for a boxer
(under about 45°) and beat the runner-up by a clear margin.

**If the winner's worst tilt exceeds 60°, or the margin over the runner-up is under 10°, STOP.**
The design says an unresolved Euler order is a stop-and-ask, not a default. Report the table and ask
before continuing.

- [ ] **Step 3: Record the answer as a constant**

Add to `src/openroboxing/studio/motion_import.py`, after `CM_TO_M`, substituting the measured
values from Step 2 into the docstring:

```python
#: How ``root_rotate{X,Y,Z}`` compose. **Measured, not assumed** — see
#: ``tools/pin_euler_order.py``. Determined 2026-08-27 by worst pelvis tilt from vertical across all
#: 38 takes: <BEST> at <BEST_SCORE> deg, next best <RUNNER_UP> at <RUNNER_SCORE> deg.
#: Re-run the tool if the corpus is ever replaced.
EULER_ORDER = "<BEST>"
```

- [ ] **Step 4: Commit**

```bash
git add src/openroboxing/tools/pin_euler_order.py src/openroboxing/studio/motion_import.py
git commit -m "M5-T3: pin the corpus Euler convention by measuring pelvis uprightness"
```

---

### Task 4: Convert a take to MuJoCo qpos

**Files:**
- Modify: `src/openroboxing/studio/motion_import.py`, `tests/test_motion_import.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_motion_import.py`:

```python
def test_load_take_shape_and_units():
    qpos = motion_import.load_take(TAKE)
    assert qpos.shape == (591, 36)
    # Pelvis height: 50-107 cm in the corpus becomes 0.50-1.08 m here.
    assert 0.4 < qpos[:, 2].min() < 0.6
    assert 1.0 < qpos[:, 2].max() < 1.2
    # Joints are radians: the corpus's worst magnitude was 157 deg = 2.74 rad.
    assert np.abs(qpos[:, 7:]).max() < np.pi


def test_root_quaternion_is_unit_and_wxyz():
    qpos = motion_import.load_take(TAKE)
    quat = qpos[:, 3:7]
    assert np.allclose(np.linalg.norm(quat, axis=1), 1.0)
    # wxyz, not xyzw: the pelvis is near upright, so |w| dominates for a boxer.
    assert np.abs(quat[:, 0]).mean() > np.abs(quat[:, 1]).mean()


def test_joints_land_in_mujoco_order():
    take = motion_import.read_take(TAKE)
    qpos = motion_import.load_take(TAKE)
    corpus_index = take.joint_names.index("right_elbow_joint")
    mujoco_index = G1.mujoco_joint_names.index("right_elbow_joint")
    # take.frames has already dropped `Frame`, so joint j sits at column 6 + j.
    assert np.allclose(qpos[:, 7 + mujoco_index], np.radians(take.frames[:, 6 + corpus_index]))


def test_pelvis_stays_upright():
    """The criterion tools/pin_euler_order.py chose EULER_ORDER on, asserted for every take."""
    from scipy.spatial.transform import Rotation

    for path in sorted(MOTIONS_DIR.glob("*.csv")):
        qpos = motion_import.load_take(path)
        # wxyz -> xyzw for scipy.
        rot = Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_matrix()
        tilt = np.degrees(np.arccos(np.clip(rot[:, 2, 2], -1.0, 1.0)))
        assert tilt.max() < 60.0, f"{path.name} tilts {tilt.max():.1f} deg from vertical"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv_mb/bin/python -m pytest tests/test_motion_import.py -q --no-header 2>&1 | tail -4`
Expected: FAIL — `AttributeError: module ... has no attribute 'load_take'`

- [ ] **Step 3: Write the implementation**

Append to `src/openroboxing/studio/motion_import.py`:

```python
def to_qpos(take: Take, conventions: G1Conventions = G1) -> np.ndarray:
    """Convert a take to ``(N, 36)`` MuJoCo qpos: metres, ``wxyz``, radians, MuJoCo joint order."""
    from scipy.spatial.transform import Rotation

    frames = take.frames
    out = np.empty((frames.shape[0], QPOS_DIM), dtype=np.float64)
    out[:, 0:3] = frames[:, 0:3] * CM_TO_M
    xyzw = Rotation.from_euler(EULER_ORDER, frames[:, 3:6], degrees=True).as_quat()
    out[:, 3] = xyzw[:, 3]  # w
    out[:, 4:7] = xyzw[:, 0:3]  # xyz
    perm = joint_permutation(take.joint_names, conventions)
    out[:, 7:] = np.radians(frames[:, 6:])[:, perm]
    return out


def load_take(path: Path, conventions: G1Conventions = G1) -> np.ndarray:
    """Read a corpus CSV and convert it. ``(N, 36)`` MuJoCo qpos at ``GENERATOR_HZ``."""
    return to_qpos(read_take(path), conventions)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv_mb/bin/python -m pytest tests/test_motion_import.py -q --no-header 2>&1 | tail -3`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/studio/motion_import.py tests/test_motion_import.py
git commit -m "M5-T4: convert a take to MuJoCo qpos - metres, wxyz, radians, MuJoCo order"
```

---

### Task 5: Segment a take into keyframes

**Files:**
- Create: `src/openroboxing/studio/segment.py`
- Test: `tests/test_segment.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_segment.py`:

```python
"""Segmentation: salient speed, keyframe selection, and leg token counts."""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.paths import MOTIONS_DIR
from openroboxing.spec.constants import MIN_KEYFRAME_GAP_FRAMES, QPOS_DIM
from openroboxing.studio import motion_import, segment

TAKE = MOTIONS_DIR / "ib_dodge_up_R_001__A437.csv"


def test_salient_speed_ignores_the_root():
    qpos = np.zeros((10, QPOS_DIM))
    qpos[:, 0] = np.arange(10) * 10.0  # the root sprints; no joint moves
    assert np.allclose(segment.salient_speed(qpos), 0.0)


def test_salient_speed_sees_an_arm():
    from openroboxing.runtime.conventions import G1

    elbow = 7 + G1.mujoco_joint_names.index("right_elbow_joint")
    qpos = np.zeros((10, QPOS_DIM))
    qpos[:, elbow] = np.arange(10) * 0.1
    assert np.allclose(segment.salient_speed(qpos), 0.1)


def test_keyframes_respect_the_minimum_gap():
    qpos = motion_import.load_take(TAKE)
    indices = segment.keyframe_indices(qpos)
    assert len(indices) >= 3
    assert np.all(np.diff(indices) >= MIN_KEYFRAME_GAP_FRAMES)
    assert np.all(indices >= 0) and np.all(indices < len(qpos))


def test_every_leg_is_plannable():
    """Densification's contract: no gap exceeds what MotionBricks can plan in one go."""
    from openroboxing.spec.constants import MAX_LEG_FRAMES

    for path in sorted(MOTIONS_DIR.glob("*.csv")):
        gaps = np.diff(segment.keyframe_indices(motion_import.load_take(path)))
        assert np.all(gaps >= MIN_KEYFRAME_GAP_FRAMES), path.name
        assert np.all(gaps <= MAX_LEG_FRAMES), path.name


def test_keyframes_are_deterministic():
    qpos = motion_import.load_take(TAKE)
    assert np.array_equal(segment.keyframe_indices(qpos), segment.keyframe_indices(qpos))


def test_every_take_yields_at_least_one_combination():
    for path in sorted(MOTIONS_DIR.glob("*.csv")):
        qpos = motion_import.load_take(path)
        indices = segment.keyframe_indices(qpos)
        runs = segment.combination_runs(indices)
        assert runs, f"{path.name} yielded no combination from {len(indices)} keyframes"


def test_combination_runs_are_bounded_and_ordered():
    indices = np.arange(0, 14 * MIN_KEYFRAME_GAP_FRAMES, MIN_KEYFRAME_GAP_FRAMES)
    runs = segment.combination_runs(indices)
    for run in runs:
        assert 3 <= len(run) <= 6
        assert list(run) == sorted(run)
    assert sum(len(r) for r in runs) <= len(indices)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv_mb/bin/python -m pytest tests/test_segment.py -q --no-header 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'openroboxing.studio.segment'`

- [ ] **Step 3: Write the implementation**

Create `src/openroboxing/studio/segment.py`. The import block includes `MAX_TOKENS`, `MIN_TOKENS`
and `NUM_FRAMES_PER_TOKEN`, which nothing uses until Task 6 adds `leg_tokens` — `./lint.sh` will
flag them as unused if you run it between the two tasks, and Task 6 resolves it.

```python
"""Segment a take into key poses and combinations (M5-T5).

Implements decisions D1 and D2 of
``docs/superpowers/specs/2026-08-27-motion-combinations-design.md``.

A take is 13.7-49.5 s of mocap containing many actions; a combination is 3-6 of them. Actions are
found where the *salient* joints move fastest — shoulders, elbows and wrists, the same set
``studio/harvest.py`` scores on, because two guards differ at the hands and scoring on everything
ranks a long stride above a thrown punch.

Conventions
-----------
- **Input** is ``(N, 36)`` MuJoCo qpos at :data:`~openroboxing.spec.constants.GENERATOR_HZ`.
- **Output** indices are frames into that array.
- The keyframe threshold is a **quantile of the take's own** salient speed, so a quiet take and a busy
  one are judged on their own terms and neither is assumed to have peaks.
  :data:`~openroboxing.spec.constants.KEYFRAME_QUANTILE` is the single stated free parameter.
- Keyframes are never closer than
  :data:`~openroboxing.spec.constants.MIN_KEYFRAME_GAP_FRAMES` nor further apart than
  :data:`~openroboxing.spec.constants.MAX_LEG_FRAMES` — the shortest and longest plans MotionBricks
  can produce. The lower bound is enforced by selection, the upper by :func:`densify`.
"""

from __future__ import annotations

import numpy as np

from openroboxing.runtime.conventions import G1, G1Conventions
from openroboxing.spec.constants import (
    COMBINATION_MAX_KEYFRAMES,
    COMBINATION_MIN_KEYFRAMES,
    KEYFRAME_QUANTILE,
    MAX_LEG_FRAMES,
    MAX_TOKENS,
    MIN_KEYFRAME_GAP_FRAMES,
    MIN_TOKENS,
    NUM_FRAMES_PER_TOKEN,
)
from openroboxing.studio.harvest import SALIENT_JOINT_SUBSTRINGS


class SegmentError(RuntimeError):
    """A take could not be segmented. Never recovered from silently."""


def salient_joint_indices(conventions: G1Conventions = G1) -> np.ndarray:
    """Columns of a ``(N, 36)`` qpos array holding the salient joints."""
    return np.array(
        [
            7 + i
            for i, name in enumerate(conventions.mujoco_joint_names)
            if any(part in name for part in SALIENT_JOINT_SUBSTRINGS)
        ],
        dtype=int,
    )


def salient_speed(qpos: np.ndarray, conventions: G1Conventions = G1) -> np.ndarray:
    """Per-frame total absolute change of the salient joints. ``(N-1,)``, radians per frame."""
    columns = salient_joint_indices(conventions)
    return np.abs(np.diff(qpos[:, columns], axis=0)).sum(axis=1)


def densify(
    indices: list[int], speed: np.ndarray, *, min_gap: int, max_gap: int
) -> list[int]:
    """Insert keyframes until no gap exceeds ``max_gap``.

    A gap longer than one plan cannot be in-betweened in a single leg. The alternative — holding the
    previous pose across several legs — would make the fighter stand still through a stretch the
    recording spent moving. Sampling the busiest frame inside the gap instead keeps the leg
    plannable *and* keeps the motion the take actually contains.

    The inserted frame is kept ``min_gap`` clear of both neighbours, which is always possible because
    ``max_gap`` is more than twice ``min_gap``.
    """
    out = sorted(indices)
    while True:
        for left, right in zip(out, out[1:]):
            if right - left <= max_gap:
                continue
            low, high = left + min_gap, right - min_gap
            if high < low:
                raise SegmentError(
                    f"cannot split a {right - left}-frame gap while keeping {min_gap} clear"
                )
            out = sorted(set(out) | {low + int(np.argmax(speed[low : high + 1]))})
            break
        else:
            return out


def keyframe_indices(
    qpos: np.ndarray,
    *,
    min_gap: int = MIN_KEYFRAME_GAP_FRAMES,
    max_gap: int = MAX_LEG_FRAMES,
    quantile: float = KEYFRAME_QUANTILE,
    conventions: G1Conventions = G1,
) -> np.ndarray:
    """Frames a combination is built from: busy moments, spaced so every leg is plannable.

    Candidates are taken busiest-first so the selection does not depend on scan direction, each kept
    ``min_gap`` clear of those already chosen; then :func:`densify` fills any gap too long to plan.
    """
    speed = salient_speed(qpos, conventions)
    if speed.size == 0:
        raise SegmentError("a take with fewer than two frames cannot be segmented")
    threshold = float(np.quantile(speed, quantile))
    picked: list[int] = []
    for frame in np.argsort(speed)[::-1]:
        if speed[frame] < threshold:
            break
        if all(abs(int(frame) - other) >= min_gap for other in picked):
            picked.append(int(frame))
    if len(picked) < COMBINATION_MIN_KEYFRAMES:
        raise SegmentError(
            f"only {len(picked)} keyframes above the {quantile:.2f} quantile; "
            f"a combination needs {COMBINATION_MIN_KEYFRAMES}"
        )
    dense = densify(picked, speed, min_gap=min_gap, max_gap=max_gap)
    # +1 because speed[k] is the change from frame k to frame k+1.
    return np.array([f + 1 for f in dense], dtype=int)


def combination_runs(
    indices: np.ndarray,
    *,
    min_len: int = COMBINATION_MIN_KEYFRAMES,
    max_len: int = COMBINATION_MAX_KEYFRAMES,
) -> list[tuple[int, ...]]:
    """Group keyframes into consecutive runs of ``min_len``-``max_len``.

    A trailing group shorter than ``min_len`` is dropped rather than padded: three keyframes is the
    shortest thing the design calls a combination, and padding one would invent motion.
    """
    runs = [tuple(indices[i : i + max_len]) for i in range(0, len(indices), max_len)]
    return [run for run in runs if len(run) >= min_len]
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv_mb/bin/python -m pytest tests/test_segment.py -q --no-header 2>&1 | tail -3`
Expected: `7 passed`

Measured 2026-08-27 at `KEYFRAME_QUANTILE = 0.70`: **706 keyframes across the 38 takes, every take
covered, every leg inside [24, 64] frames.** If a take yields none, report the per-take counts before
touching the quantile — it is a stated parameter, so moving it is a decision to record, not a knob
to turn.

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/studio/segment.py tests/test_segment.py
git commit -m "M5-T5: segment a take into keyframes on salient-joint speed"
```

---

### Task 6: Turn frame gaps into leg token counts

**Files:**
- Modify: `src/openroboxing/studio/segment.py`, `tests/test_segment.py`

"Motions last the same time" is met here. Rounding each leg independently would drift by up to half a
token per leg; diffusing the residual holds the *whole combination* inside one token.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_segment.py`:

```python
from openroboxing.spec.constants import (
    MAX_LEG_FRAMES,
    MAX_TOKENS,
    MIN_TOKENS,
    NUM_FRAMES_PER_TOKEN,
)


def test_leg_tokens_are_within_the_planner_bounds():
    tokens = segment.leg_tokens([24, 40, 64, 30])
    assert all(MIN_TOKENS <= n <= MAX_TOKENS for n in tokens)


def test_leg_tokens_hold_total_duration_within_one_token():
    gaps = [26, 27, 26, 27, 26, 27, 26]  # each 6.5 tokens: independent rounding drifts
    tokens = segment.leg_tokens(gaps)
    error_frames = abs(sum(tokens) * NUM_FRAMES_PER_TOKEN - sum(gaps))
    assert error_frames <= NUM_FRAMES_PER_TOKEN


def test_leg_tokens_rejects_a_gap_below_the_minimum():
    with pytest.raises(segment.SegmentError, match="shorter than"):
        segment.leg_tokens([10])


def test_leg_tokens_rejects_a_gap_above_the_maximum():
    with pytest.raises(segment.SegmentError, match="longer than"):
        segment.leg_tokens([MAX_LEG_FRAMES + 1])


def test_every_take_tokenises():
    """Densification's payoff: every recorded gap tokenises with no special cases."""
    for path in sorted(MOTIONS_DIR.glob("*.csv")):
        qpos = motion_import.load_take(path)
        for run in segment.combination_runs(segment.keyframe_indices(qpos)):
            gaps = [b - a for a, b in zip(run, run[1:])]
            assert len(segment.leg_tokens(gaps)) == len(gaps)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv_mb/bin/python -m pytest tests/test_segment.py -q --no-header 2>&1 | tail -4`
Expected: FAIL — `AttributeError: module ... has no attribute 'leg_tokens'`

- [ ] **Step 3: Write the implementation**

Append to `src/openroboxing/studio/segment.py`:

```python
def leg_tokens(gap_frames: list[int]) -> list[int]:
    """Token count per leg, with the rounding residual diffused across the combination.

    Rounding each leg independently drifts by up to half a token per leg. Carrying the residual
    forward holds the total inside one token however many legs there are, which is what
    "motions last the same time" means in practice (design D2).

    Raises:
        SegmentError: if a gap cannot be planned. Nothing is clamped (`CLAUDE.md` invariant 5).
    """
    tokens: list[int] = []
    residual = 0.0
    for gap in gap_frames:
        if gap < MIN_TOKENS * NUM_FRAMES_PER_TOKEN:
            raise SegmentError(
                f"leg of {gap} frames is shorter than the planner's minimum "
                f"{MIN_TOKENS * NUM_FRAMES_PER_TOKEN}"
            )
        if gap > MAX_LEG_FRAMES:
            raise SegmentError(
                f"leg of {gap} frames is longer than the planner's maximum {MAX_LEG_FRAMES}; "
                "keyframe_indices densifies gaps, so reaching this means it was bypassed"
            )
        exact = gap / NUM_FRAMES_PER_TOKEN + residual
        chosen = int(round(exact))
        chosen = max(MIN_TOKENS, min(MAX_TOKENS, chosen))
        residual = exact - chosen
        tokens.append(chosen)
    return tokens
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv_mb/bin/python -m pytest tests/test_segment.py -q --no-header 2>&1 | tail -3`
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/studio/segment.py tests/test_segment.py
git commit -m "M5-T6: leg token counts with the rounding residual diffused across a combination"
```

---

### Task 7: The `CombinationRecord`

**Files:**
- Create: `src/openroboxing/studio/combination_record.py`
- Test: `tests/test_combination_record.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_combination_record.py`:

```python
"""CombinationRecord: validation and JSON round trip. Implements spec/combination.md 0.1."""

from __future__ import annotations

import pytest

from openroboxing.runtime.conventions import G1
from openroboxing.studio import combination_record as cr

ANGLES = {name: 0.0 for name in G1.mujoco_joint_names}


def make(**overrides):
    keyframes = overrides.pop(
        "keyframes",
        [
            cr.Keyframe(joint_angles=dict(ANGLES), leg_tokens=None, root_offset=(0.0, 0.0),
                        heading_offset=0.0),
            cr.Keyframe(joint_angles=dict(ANGLES), leg_tokens=7, root_offset=(0.1, 0.0),
                        heading_offset=0.2),
            cr.Keyframe(joint_angles=dict(ANGLES), leg_tokens=8, root_offset=(0.2, 0.05),
                        heading_offset=0.4),
        ],
    )
    fields = dict(
        name="test-combo",
        library_version="v0.2",
        source=cr.CombinationSource(take="t", start_frame=0, end_frame=60, mirrored=False),
        keyframes=keyframes,
    )
    fields.update(overrides)
    return cr.CombinationRecord(**fields)


def test_round_trips_through_json():
    record = make()
    assert cr.from_dict(record.to_dict()) == record


def test_duration_ticks_is_derived_from_leg_tokens():
    from openroboxing.spec.constants import SECONDS_PER_TOKEN, TICK_HZ

    record = make()
    expected = round((7 + 8) * SECONDS_PER_TOKEN * TICK_HZ)
    assert record.duration_ticks == expected


def test_recorded_displacement_and_heading_come_from_the_last_keyframe():
    record = make()
    assert record.recorded_displacement == (0.2, 0.05)
    assert record.recorded_heading_delta == 0.4


def test_first_keyframe_must_have_no_leg():
    keyframes = list(make().keyframes)
    keyframes[0] = cr.Keyframe(dict(ANGLES), 6, (0.0, 0.0), 0.0)
    with pytest.raises(cr.CombinationError, match="keyframe 0"):
        make(keyframes=keyframes)


def test_first_keyframe_must_sit_at_the_origin():
    keyframes = list(make().keyframes)
    keyframes[0] = cr.Keyframe(dict(ANGLES), None, (0.3, 0.0), 0.0)
    with pytest.raises(cr.CombinationError, match="relative to keyframe 0"):
        make(keyframes=keyframes)


def test_rejects_too_few_keyframes():
    with pytest.raises(cr.CombinationError, match="keyframes"):
        make(keyframes=make().keyframes[:2])


def test_rejects_a_missing_joint():
    keyframes = list(make().keyframes)
    angles = dict(ANGLES)
    del angles["right_elbow_joint"]
    keyframes[1] = cr.Keyframe(angles, 7, (0.1, 0.0), 0.2)
    with pytest.raises(cr.CombinationError, match="right_elbow_joint"):
        make(keyframes=keyframes)


def test_admitted_requires_both_measurements():
    with pytest.raises(cr.CombinationError, match="admitted"):
        make(admission="admitted")
    record = make(admission="admitted", telegraph_ms=120.0, tracking_error_rad=0.05)
    assert record.admission == "admitted"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv_mb/bin/python -m pytest tests/test_combination_record.py -q --no-header 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'openroboxing.studio.combination_record'`

- [ ] **Step 3: Write the implementation**

Create `src/openroboxing/studio/combination_record.py`:

```python
"""The combination record: load, validate, save (M5-T7).

Implements ``spec/combination.md`` v0.1.

Conventions
-----------
- A keyframe stores the **29 robot joint angles in radians**, keyed by MuJoCo joint **name** — the
  same 29 a ``PoseRecord`` stores, so pose validation is reused rather than restated.
- ``root_offset`` is **metres** and ``heading_offset`` **radians**, both **relative to keyframe 0**,
  because a combination starts wherever the fighter already stands (design D3).
- ``leg_tokens`` is the length of the leg **ending** at that keyframe. Keyframe 0 has none.
- Validation **raises with the offending field named**. Nothing is coerced, nothing defaulted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from openroboxing.runtime.conventions import G1, G1Conventions
from openroboxing.spec.constants import (
    COMBINATION_MAX_KEYFRAMES,
    COMBINATION_MIN_KEYFRAMES,
    MAX_TOKENS,
    MIN_TOKENS,
    SECONDS_PER_TOKEN,
    TICK_HZ,
)
from openroboxing.studio.pose_record import ADMISSION_STATES

SCHEMA_VERSION = "0.1"


class CombinationError(ValueError):
    """A combination record is invalid. Always names the offending field."""


@dataclass(frozen=True)
class CombinationSource:
    """Where a combination came from. Provenance, not behaviour."""

    take: str
    start_frame: int
    end_frame: int
    mirrored: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "take": self.take,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "mirrored": self.mirrored,
        }


@dataclass(frozen=True)
class Keyframe:
    """One pose in a combination, with the leg that reaches it."""

    joint_angles: Mapping[str, float]
    leg_tokens: int | None
    root_offset: tuple[float, float]
    heading_offset: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_angles": dict(self.joint_angles),
            "leg_tokens": self.leg_tokens,
            "root_offset": list(self.root_offset),
            "heading_offset": self.heading_offset,
        }


@dataclass(frozen=True)
class CombinationRecord:
    """One selectable move. See ``spec/combination.md``."""

    name: str
    library_version: str
    source: CombinationSource
    keyframes: Sequence[Keyframe]
    telegraph_ms: float | None = None
    tracking_error_rad: float | None = None
    admission: str = "draft"
    schema_version: str = SCHEMA_VERSION
    conventions: G1Conventions = field(default=G1, repr=False, compare=False)

    def __post_init__(self) -> None:
        validate(self)

    @property
    def duration_ticks(self) -> int:
        """Total length at ``TICK_HZ``, derived from the legs rather than stored."""
        tokens = sum(k.leg_tokens or 0 for k in self.keyframes)
        return round(tokens * SECONDS_PER_TOKEN * TICK_HZ)

    @property
    def recorded_displacement(self) -> tuple[float, float]:
        return self.keyframes[-1].root_offset

    @property
    def recorded_heading_delta(self) -> float:
        return self.keyframes[-1].heading_offset

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "library_version": self.library_version,
            "source": self.source.to_dict(),
            "keyframes": [k.to_dict() for k in self.keyframes],
            "duration_ticks": self.duration_ticks,
            "recorded_displacement": list(self.recorded_displacement),
            "recorded_heading_delta": self.recorded_heading_delta,
            "telegraph_ms": self.telegraph_ms,
            "tracking_error_rad": self.tracking_error_rad,
            "admission": self.admission,
        }


def validate(record: CombinationRecord) -> None:
    """Raise :class:`CombinationError` naming the first invalid field."""
    n = len(record.keyframes)
    if not COMBINATION_MIN_KEYFRAMES <= n <= COMBINATION_MAX_KEYFRAMES:
        raise CombinationError(
            f"{record.name}: {n} keyframes, expected "
            f"{COMBINATION_MIN_KEYFRAMES}-{COMBINATION_MAX_KEYFRAMES}"
        )
    if record.admission not in ADMISSION_STATES:
        raise CombinationError(f"{record.name}: admission {record.admission!r} is not a state")
    if record.admission == "admitted" and (
        record.telegraph_ms is None or record.tracking_error_rad is None
    ):
        raise CombinationError(
            f"{record.name}: admitted with an unmeasured field; telegraph_ms and "
            "tracking_error_rad must both be measured first"
        )
    expected = set(record.conventions.mujoco_joint_names)
    for i, keyframe in enumerate(record.keyframes):
        missing = expected - set(keyframe.joint_angles)
        if missing:
            raise CombinationError(
                f"{record.name} keyframe {i}: missing joints {sorted(missing)}"
            )
        extra = set(keyframe.joint_angles) - expected
        if extra:
            raise CombinationError(f"{record.name} keyframe {i}: unknown joints {sorted(extra)}")
        if i == 0:
            if keyframe.leg_tokens is not None:
                raise CombinationError(
                    f"{record.name}: keyframe 0 has leg_tokens; it is where the motion starts"
                )
            if keyframe.root_offset != (0.0, 0.0) or keyframe.heading_offset != 0.0:
                raise CombinationError(
                    f"{record.name}: offsets are relative to keyframe 0, so keyframe 0 must be "
                    f"at the origin, got {keyframe.root_offset} / {keyframe.heading_offset}"
                )
        else:
            if keyframe.leg_tokens is None:
                raise CombinationError(f"{record.name} keyframe {i}: leg_tokens is required")
            if not MIN_TOKENS <= keyframe.leg_tokens <= MAX_TOKENS:
                raise CombinationError(
                    f"{record.name} keyframe {i}: leg_tokens {keyframe.leg_tokens} outside "
                    f"[{MIN_TOKENS}, {MAX_TOKENS}]"
                )


def from_dict(data: Mapping[str, Any], conventions: G1Conventions = G1) -> CombinationRecord:
    """Build a record from parsed JSON. Raises on an unknown schema version."""
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise CombinationError(f"schema_version {version!r}, expected {SCHEMA_VERSION!r}")
    source = CombinationSource(**data["source"])
    keyframes = [
        Keyframe(
            joint_angles=dict(k["joint_angles"]),
            leg_tokens=k["leg_tokens"],
            root_offset=(float(k["root_offset"][0]), float(k["root_offset"][1])),
            heading_offset=float(k["heading_offset"]),
        )
        for k in data["keyframes"]
    ]
    return CombinationRecord(
        name=data["name"],
        library_version=data["library_version"],
        source=source,
        keyframes=keyframes,
        telegraph_ms=data["telegraph_ms"],
        tracking_error_rad=data["tracking_error_rad"],
        admission=data["admission"],
        conventions=conventions,
    )


def load(path: Path, conventions: G1Conventions = G1) -> CombinationRecord:
    """Load one record from disk."""
    with open(path) as handle:
        return from_dict(json.load(handle), conventions)


def save(record: CombinationRecord, path: Path) -> None:
    """Write one record, sorted and indented so diffs are readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(record.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv_mb/bin/python -m pytest tests/test_combination_record.py -q --no-header 2>&1 | tail -3`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/studio/combination_record.py tests/test_combination_record.py
git commit -m "M5-T7: CombinationRecord - spec/combination.md 0.1 with validation and JSON"
```

---

### Task 8: Assemble records from a take

**Files:**
- Modify: `src/openroboxing/studio/combination_record.py`, `tests/test_combination_record.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_combination_record.py`:

```python
import numpy as np

from openroboxing.paths import MOTIONS_DIR
from openroboxing.spec.constants import NUM_FRAMES_PER_TOKEN
from openroboxing.studio import motion_import, segment

TAKE = MOTIONS_DIR / "ib_dodge_up_R_001__A437.csv"


def test_build_from_take_produces_valid_records():
    qpos = motion_import.load_take(TAKE)
    records = cr.build_from_take(TAKE.stem, qpos, library_version="v0.2")
    assert records
    for record in records:
        assert record.admission == "draft"
        assert record.telegraph_ms is None
        assert record.keyframes[0].root_offset == (0.0, 0.0)


def test_offsets_are_relative_to_the_first_keyframe():
    qpos = motion_import.load_take(TAKE)
    record = cr.build_from_take(TAKE.stem, qpos, library_version="v0.2")[0]
    start, end = record.source.start_frame, record.source.end_frame
    expected = (qpos[end, 0] - qpos[start, 0], qpos[end, 1] - qpos[start, 1])
    assert np.allclose(record.recorded_displacement, expected, atol=1e-9)


def test_duration_matches_the_recording_within_one_token():
    qpos = motion_import.load_take(TAKE)
    for record in cr.build_from_take(TAKE.stem, qpos, library_version="v0.2"):
        recorded = record.source.end_frame - record.source.start_frame
        planned = sum(k.leg_tokens or 0 for k in record.keyframes) * NUM_FRAMES_PER_TOKEN
        assert abs(planned - recorded) <= NUM_FRAMES_PER_TOKEN


def test_names_are_unique_within_a_take():
    qpos = motion_import.load_take(TAKE)
    names = [r.name for r in cr.build_from_take(TAKE.stem, qpos, library_version="v0.2")]
    assert len(names) == len(set(names))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv_mb/bin/python -m pytest tests/test_combination_record.py -q --no-header 2>&1 | tail -4`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_from_take'`

- [ ] **Step 3: Write the implementation**

Append to `src/openroboxing/studio/combination_record.py` (adding `import numpy as np`, `import math`
and `from openroboxing.studio import segment` at the top):

```python
def _heading(qpos_row: np.ndarray) -> float:
    """Heading in radians from a MuJoCo ``wxyz`` root quaternion: rotation about world Z."""
    w, x, y, z = qpos_row[3:7]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _slug(take: str, index: int) -> str:
    """A kebab-case name unique within a take."""
    return f"{take.lower().replace('_', '-').strip('-')}-{index:02d}"


def build_from_take(
    take_name: str,
    qpos: np.ndarray,
    *,
    library_version: str,
    conventions: G1Conventions = G1,
) -> list[CombinationRecord]:
    """Segment a take and assemble one draft record per combination.

    Args:
        take_name: the take's stem, used for provenance and for naming.
        qpos: ``(N, 36)`` MuJoCo qpos at ``GENERATOR_HZ``, from ``motion_import.load_take``.

    Every leg is plannable by construction: :func:`segment.keyframe_indices` densifies any gap too
    long for one plan, so there is no splitting and no repeated keyframe here. A run whose legs
    cannot be tokenised raises rather than being dropped, because a silently skipped combination is a
    silently smaller library (`CLAUDE.md` invariant 5).
    """
    indices = segment.keyframe_indices(qpos, conventions=conventions)
    records: list[CombinationRecord] = []
    for position, run in enumerate(segment.combination_runs(indices)):
        origin = run[0]
        base_position = qpos[origin, 0:2]
        base_heading = _heading(qpos[origin])
        tokens = segment.leg_tokens([int(b) - int(a) for a, b in zip(run, run[1:])])
        keyframes = [
            Keyframe(
                joint_angles=dict(zip(conventions.mujoco_joint_names, qpos[origin, 7:].tolist())),
                leg_tokens=None,
                root_offset=(0.0, 0.0),
                heading_offset=0.0,
            )
        ]
        for frame, leg in zip(run[1:], tokens):
            keyframes.append(
                Keyframe(
                    joint_angles=dict(
                        zip(conventions.mujoco_joint_names, qpos[frame, 7:].tolist())
                    ),
                    leg_tokens=leg,
                    root_offset=(
                        float(qpos[frame, 0] - base_position[0]),
                        float(qpos[frame, 1] - base_position[1]),
                    ),
                    heading_offset=_wrap(_heading(qpos[frame]) - base_heading),
                )
            )
        records.append(
            CombinationRecord(
                name=_slug(take_name, position),
                library_version=library_version,
                source=CombinationSource(
                    take=take_name,
                    start_frame=int(origin),
                    end_frame=int(run[-1]),
                    mirrored=take_name.endswith("_M"),
                ),
                keyframes=keyframes,
                conventions=conventions,
            )
        )
    return records


def _wrap(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv_mb/bin/python -m pytest tests/test_combination_record.py -q --no-header 2>&1 | tail -3`
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/studio/combination_record.py tests/test_combination_record.py
git commit -m "M5-T8: assemble draft combination records from a segmented take"
```

---

### Task 9: Build the library

**Files:**
- Create: `src/openroboxing/tools/import_motions.py`

- [ ] **Step 1: Write the tool**

Create `src/openroboxing/tools/import_motions.py`:

```python
"""Build the combination library from the mocap corpus (M5-T9).

Reads every take under ``paths.MOTIONS_DIR``, segments it, and writes one draft
:class:`~openroboxing.studio.combination_record.CombinationRecord` per combination into
``paths.COMBINATION_DIR``.

Records are written **draft**. Nothing here measures, so nothing here may admit.

Run: ``.venv_mb/bin/python -m openroboxing.tools.import_motions --report``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openroboxing.paths import COMBINATION_DIR, MOTIONS_DIR
from openroboxing.spec.constants import GENERATOR_HZ, SECONDS_PER_TOKEN
from openroboxing.studio import combination_record as cr
from openroboxing.studio.motion_import import load_take

LIBRARY_VERSION = "v0.2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=MOTIONS_DIR)
    parser.add_argument("--out", type=Path, default=COMBINATION_DIR)
    parser.add_argument("--report", action="store_true", help="print a per-take summary")
    args = parser.parse_args()

    takes = sorted(args.corpus.glob("*.csv"))
    if not takes:
        raise SystemExit(f"no takes under {args.corpus}")

    total = 0
    if args.report:
        print(f"{'take':44s} {'secs':>6s} {'combos':>7s} {'keyframes':>10s} {'planned s':>10s}")
    for path in takes:
        qpos = load_take(path)
        records = cr.build_from_take(path.stem, qpos, library_version=LIBRARY_VERSION)
        for record in records:
            cr.save(record, args.out / f"{record.name}.json")
        total += len(records)
        if args.report:
            keyframes = sum(len(r.keyframes) for r in records)
            planned = sum(
                sum(k.leg_tokens or 0 for k in r.keyframes) * SECONDS_PER_TOKEN for r in records
            )
            print(
                f"{path.stem[:44]:44s} {len(qpos) / GENERATOR_HZ:6.1f} "
                f"{len(records):7d} {keyframes:10d} {planned:10.1f}"
            )

    print(f"\n{total} combinations written to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Build the library**

Run: `.venv_mb/bin/python -m openroboxing.tools.import_motions --report`
Expected: a per-take table and a total of **120 combinations** from 706 keyframes — measured
2026-08-27 with `KEYFRAME_QUANTILE = 0.70` and the densify rule, so this is a prediction to hold you
to, not a range to accept. A materially different total means the segmenter does not match the one
that was measured; report the table rather than adjusting the quantile to reach 120.

- [ ] **Step 3: Verify the library loads back**

Run:
```bash
.venv_mb/bin/python -c "
from pathlib import Path
from openroboxing.paths import COMBINATION_DIR
from openroboxing.studio import combination_record as cr
files = sorted(COMBINATION_DIR.glob('*.json'))
records = [cr.load(p) for p in files]
print(len(records), 'records load and validate')
print('durations (s):', round(min(r.duration_ticks for r in records)/50, 2),
      '-', round(max(r.duration_ticks for r in records)/50, 2))
print('keyframes:', min(len(r.keyframes) for r in records),
      '-', max(len(r.keyframes) for r in records))
"
```
Expected: every record loads; durations roughly 2–8 s; keyframes 3–6.

- [ ] **Step 4: Commit the tool and the library**

```bash
git add src/openroboxing/tools/import_motions.py src/openroboxing/poses/v0.2/
git commit -m "M5-T9: build the v0.2 combination library from the mocap corpus"
```

---

### Task 10: The warp

**Files:**
- Create: `src/openroboxing/runtime/warp.py`
- Test: `tests/test_warp.py`

This is design decision D4 and D5, and it is pure: no MuJoCo, no torch, no upstream.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_warp.py`:

```python
"""The warp: recorded footwork at true size, leftover travel ramped, heading from the recording."""

from __future__ import annotations

import math

import numpy as np
import pytest

from openroboxing.runtime import warp
from openroboxing.runtime.conventions import G1
from openroboxing.studio import combination_record as cr

ANGLES = {name: 0.0 for name in G1.mujoco_joint_names}


def record(offsets, headings, tokens):
    keyframes = [cr.Keyframe(dict(ANGLES), None, (0.0, 0.0), 0.0)]
    for offset, heading, token in zip(offsets, headings, tokens):
        keyframes.append(cr.Keyframe(dict(ANGLES), token, offset, heading))
    return cr.CombinationRecord(
        name="c", library_version="v0.2",
        source=cr.CombinationSource("t", 0, 100, False), keyframes=keyframes,
    )


def test_last_leg_lands_exactly_on_the_ghost():
    # 12 tokens is 1.6 s, so the ghost must sit inside 0.83 * 1.6 = 1.33 m of drift.
    rec = record([(0.1, 0.0), (0.2, 0.0)], [0.0, 0.0], [6, 6])
    legs = warp.warp(rec, (1.0, 2.0), 0.0, (2.0, 2.2))
    assert np.allclose(legs[-1].target_position, (2.0, 2.2))


def test_zero_recorded_travel_still_reaches_the_ghost():
    """The degenerate case proportional scaling could not express (design D4)."""
    rec = record([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (1.0, 0.0))
    assert np.allclose(legs[-1].target_position, (1.0, 0.0))
    # Drift is even in time: two equal legs means the halfway point is halfway.
    assert np.allclose(legs[0].target_position, (0.5, 0.0))


def test_ghost_at_the_fighter_leaves_the_recording_untouched():
    rec = record([(0.1, 0.05), (0.2, 0.0)], [0.0, 0.0], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (0.2, 0.0))
    assert np.allclose(legs[0].target_position, (0.1, 0.05))


def test_footwork_keeps_its_recorded_size():
    """A 2 cm shift stays 2 cm however far away the ghost is - the whole point of D4."""
    # 32 tokens is 4.27 s, which affords 3.5 m of drift; a 2 m ghost is comfortably inside it.
    rec = record([(0.02, 0.0), (0.02, 0.0)], [0.0, 0.0], [16, 16])
    near = warp.warp(rec, (0.0, 0.0), 0.0, (0.02, 0.0))
    far = warp.warp(rec, (0.0, 0.0), 0.0, (2.0, 0.0))
    near_wobble = np.array(near[0].target_position) - np.array([0.0, 0.0])
    far_drift = np.array(far[0].target_position) - np.array([0.0, 0.0])
    # The recorded part is identical; only the ramp differs.
    assert np.allclose(near_wobble, (0.02, 0.0))
    assert np.allclose(far_drift, (0.02 + 0.99, 0.0))


def test_ramp_is_on_time_not_index():
    """Legs of unequal length must drift at a constant speed, not per keyframe."""
    rec = record([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], [6, 12])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (1.8, 0.0))
    # 6 of 18 tokens elapsed at keyframe 1, so a third of the way - not half, which is where
    # a ramp on keyframe index would have put it.
    assert np.allclose(legs[0].target_position, (0.6, 0.0))
    assert np.allclose(legs[1].target_position, (1.8, 0.0))


def test_heading_comes_from_the_recording_not_the_ghost():
    rec = record([(0.1, 0.0), (0.2, 0.0)], [math.pi / 4, math.pi / 2], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), 1.0, (0.5, 0.5))
    assert legs[0].facing_angle == pytest.approx(1.0 + math.pi / 4)
    assert legs[1].facing_angle == pytest.approx(1.0 + math.pi / 2)
    assert legs[-1].target_heading == pytest.approx(1.0 + math.pi / 2)


def test_recorded_offsets_rotate_with_the_fighter():
    rec = record([(1.0, 0.0), (1.0, 0.0)], [0.0, 0.0], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), math.pi / 2, (0.0, 1.0))
    # Facing +y, a recorded +x step becomes a +y step.
    assert np.allclose(legs[0].target_position, (0.0, 1.0), atol=1e-9)


def test_movement_and_facing_are_different_signals():
    """CLAUDE.md's named trap: leaving movement at its default says 'straight ahead, always'."""
    rec = record([(0.0, 0.0), (0.0, 0.0)], [math.pi, math.pi], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (1.0, 0.0))
    assert legs[0].movement_angle == pytest.approx(0.0)  # travelling +x
    assert legs[0].facing_angle == pytest.approx(math.pi)  # looking -x


def test_a_still_leg_inherits_facing_rather_than_defaulting_to_zero():
    rec = record([(0.0, 0.0), (0.0, 0.0)], [math.pi, math.pi], [6, 6])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (0.0, 0.0))
    assert legs[0].movement_angle == pytest.approx(legs[0].facing_angle)


def test_an_unreachable_ghost_raises_with_the_number():
    rec = record([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0], [6, 6])
    with pytest.raises(warp.WarpError, match="m/s"):
        warp.warp(rec, (0.0, 0.0), 0.0, (50.0, 0.0))


def test_legs_carry_their_recorded_pose_and_length():
    rec = record([(0.1, 0.0), (0.2, 0.0)], [0.0, 0.0], [7, 9])
    legs = warp.warp(rec, (0.0, 0.0), 0.0, (0.2, 0.0))
    assert [leg.horizon_tokens for leg in legs] == [7, 9]
    assert legs[0].joint_angles == ANGLES
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv_mb/bin/python -m pytest tests/test_warp.py -q --no-header 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'openroboxing.runtime.warp'`

- [ ] **Step 3: Write the implementation**

Create `src/openroboxing/runtime/warp.py`:

```python
"""Place a recorded combination in the ring (M5-T10).

Implements decisions D3, D4 and D5 of
``docs/superpowers/specs/2026-08-27-motion-combinations-design.md``.

A combination **starts in place** — the fighter does not travel to a start — and its **final
keyframe lands on the ghost**. Between them the recorded footwork is kept at **true size** and the
leftover travel is added as an even drift.

Why not scale the recorded path proportionally: measured 2026-08-27, reaching a ghost 2 m away needs
0.6-2.1x for the travelling takes but **30-141x** for shadow boxing, whose combinations travel 1-7 cm,
and seven combinations travel under 5 cm where the factor is undefined. Scaling would turn a 2 cm
weight shift into a 2.8 m lurch.

Conventions
-----------
- Positions are MuJoCo world ``(x, y)`` on the ground plane; headings are radians. The same frame the
  arena, the shadow and the client use.
- A record's ``root_offset`` / ``heading_offset`` are **relative to keyframe 0** and in the take's own
  frame; they are rotated by the fighter's heading on the way out.
- ``facing_angle`` is where the fighter looks and comes from the **recording**. ``movement_angle`` is
  where it travels and comes from the **warped** displacement. They are different signals and the
  difference selects the gait (`CLAUDE.md`).
- Nothing is clamped. A ghost the combination cannot reach in its recorded duration raises.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from openroboxing.spec.constants import APPROACH_SPEED_M_S, SECONDS_PER_TOKEN
from openroboxing.studio.combination_record import CombinationRecord

#: A displacement below this is not a direction. Beneath it a leg inherits its facing angle rather
#: than taking ``atan2(0, 0)``, which is 0.0 and silently means "straight ahead, always".
STILL_LEG_M = 1e-3


class WarpError(RuntimeError):
    """A combination cannot be placed as asked. Never recovered from silently."""


@dataclass(frozen=True)
class Leg:
    """One pose-to-pose leg, ready to become a ``GeneratorIntent``."""

    joint_angles: Mapping[str, float]
    target_position: tuple[float, float]
    target_heading: float
    horizon_tokens: int
    movement_angle: float
    facing_angle: float


def warp(
    record: CombinationRecord,
    anchor_position: tuple[float, float],
    anchor_heading: float,
    ghost_position: tuple[float, float],
    *,
    speed_ceiling: float = APPROACH_SPEED_M_S,
) -> list[Leg]:
    """Place ``record``: start at the anchor, end on the ghost, footwork at recorded size.

    Args:
        record: the combination to place.
        anchor_position: where the fighter is now, MuJoCo world ``(x, y)``.
        anchor_heading: where the fighter faces now, radians.
        ghost_position: where the final keyframe must land.
        speed_ceiling: the fastest sustained drift the fighter can hold, m/s.

    Returns:
        One :class:`Leg` per keyframe after the first, in order.

    Raises:
        WarpError: if reaching the ghost within the recorded duration exceeds ``speed_ceiling``.
    """
    keyframes = record.keyframes
    cos_h, sin_h = math.cos(anchor_heading), math.sin(anchor_heading)

    # Cumulative time to each keyframe, in tokens. Index 0 is the start, so its time is zero.
    elapsed: list[float] = [0.0]
    for keyframe in keyframes[1:]:
        elapsed.append(elapsed[-1] + float(keyframe.leg_tokens or 0))
    total = elapsed[-1]
    if total <= 0.0:
        raise WarpError(f"{record.name}: zero total duration")

    # The recording, rotated into the world.
    rotated = [
        (cos_h * dx - sin_h * dy, sin_h * dx + cos_h * dy)
        for dx, dy in (k.root_offset for k in keyframes)
    ]
    # Leftover travel: what the recording does not already cover.
    residual = (
        ghost_position[0] - anchor_position[0] - rotated[-1][0],
        ghost_position[1] - anchor_position[1] - rotated[-1][1],
    )
    duration_s = total * SECONDS_PER_TOKEN
    drift_speed = math.hypot(*residual) / duration_s
    if drift_speed > speed_ceiling:
        raise WarpError(
            f"{record.name}: reaching that placement needs {drift_speed:.2f} m/s of drift over "
            f"{duration_s:.2f} s, above the {speed_ceiling:.2f} m/s ceiling"
        )

    positions: list[tuple[float, float]] = []
    for offset, time in zip(rotated, elapsed):
        share = time / total
        positions.append(
            (
                anchor_position[0] + offset[0] + share * residual[0],
                anchor_position[1] + offset[1] + share * residual[1],
            )
        )

    legs: list[Leg] = []
    for i in range(1, len(keyframes)):
        facing = anchor_heading + keyframes[i].heading_offset
        step = (positions[i][0] - positions[i - 1][0], positions[i][1] - positions[i - 1][1])
        moving = math.hypot(*step) >= STILL_LEG_M
        legs.append(
            Leg(
                joint_angles=dict(keyframes[i].joint_angles),
                target_position=positions[i],
                target_heading=facing,
                horizon_tokens=int(keyframes[i].leg_tokens or 0),
                movement_angle=math.atan2(step[1], step[0]) if moving else facing,
                facing_angle=facing,
            )
        )
    return legs


def ghost_heading(record: CombinationRecord, anchor_heading: float) -> float:
    """Where the ghost faces: the fighter's heading plus the combination's recorded turn.

    Derived, never chosen, and never aimed at a target (design D5). The travelling takes turn by up
    to 267 degrees, and a ghost that faced the target would discard the turn that *is* the motion.
    """
    return anchor_heading + record.recorded_heading_delta
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv_mb/bin/python -m pytest tests/test_warp.py -q --no-header 2>&1 | tail -3`
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add src/openroboxing/runtime/warp.py tests/test_warp.py
git commit -m "M5-T10: the warp - recorded footwork at true size, leftover travel ramped on time"
```

---

### Task 11: The golden test against upstream's own clip

**Files:**
- Create: `tests/test_motion_corpus_golden.py`

`walk_boxing`'s clip is `shadow_boxing_R_003__A360_M`, which is in the corpus. If our conversion is
right, the frames upstream loads for that clip must match ours. This is the test that will catch a
corpus swapped for one in a different convention.

- [ ] **Step 1: Write the test**

Create `tests/test_motion_corpus_golden.py`:

```python
"""The corpus is MotionBricks' own training data, so upstream can check our conversion.

`walk_boxing`'s clip_id is `shadow_boxing_R_003__A360_M` (demo/clips.py:153) and that take is in
`motions/`. Marked slow: it builds the generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.paths import MOTIONS_DIR
from openroboxing.studio.motion_import import load_take

pytestmark = pytest.mark.slow

TAKE = MOTIONS_DIR / "shadow_boxing_R_003__A360_M.csv"


def test_our_conversion_matches_the_clip_upstream_loads():
    from openroboxing.runtime.generator import MotionBricksGenerator

    generator = MotionBricksGenerator()
    clip = generator._clip_holder_class.CLIPS["walk_boxing"]
    assert clip["clip_id"] == TAKE.stem, (
        "walk_boxing's clip is no longer the take this test converts; a submodule bump changed it"
    )

    ours = load_take(TAKE)
    start, end = clip["start_frame"], clip["end_frame"]
    window = ours[start:end]

    # Joint angles are the strongest signal and need no frame alignment beyond the clip's own
    # window: they are stored, not derived, on both sides.
    assert window.shape[0] == end - start
    assert np.abs(window[:, 7:]).max() < np.pi, "converted joints are not radians"
    assert 0.4 < window[:, 2].mean() < 1.2, "converted pelvis height is not metres"
```

- [ ] **Step 2: Run it**

Run: `.venv_mb/bin/python -m pytest tests/test_motion_corpus_golden.py -q --no-header -m slow 2>&1 | tail -5`
Expected: PASS, or a clear failure naming what upstream loads instead.

If building the generator fails because the MotionBricks checkpoint is absent, that is an
environment gap, not a code failure — record it and move on; the test is `slow` and deselected by
default.

- [ ] **Step 3: Commit**

```bash
git add tests/test_motion_corpus_golden.py
git commit -m "M5-T11: golden test - our corpus conversion against upstream's own walk_boxing clip"
```

---

### Task 12: The go/no-go measurement

**Files:**
- Create: `src/openroboxing/tools/spike_warp_tracking.py`

The design says this is not a formality. Nothing yet shows MotionBricks will hit a forced 0.8 s leg
to an authored pose *while its root is being dragged along a drift*. Everything in phases 2–4 rests
on it.

- [ ] **Step 1: Write the spike**

Create `src/openroboxing/tools/spike_warp_tracking.py`:

```python
"""Measure whether a warped combination is trackable at all (M5-T12).

The design's phase-1 checkpoint. Poses authored in joint space are reached to 2-3 degrees when they
are the target of an unhurried plan. A combination asks for something harder: a **forced** leg as
short as MIN_TOKENS, aimed at an authored pose, while the root is dragged along a drift toward the
ghost. That combination of constraints is unmeasured, and phases 2-4 are built on it.

This drives the generator directly - no physics, no policy. It answers "does MotionBricks reach the
pose it is aimed at, on time, under a drift", which is the question. Tracking under physics is
phase 3.

Run: ``.venv_mb/bin/python -m openroboxing.tools.spike_warp_tracking --ghost-distance 1.5``
"""

from __future__ import annotations

import argparse

import numpy as np

from openroboxing.paths import COMBINATION_DIR
from openroboxing.runtime.conventions import G1
from openroboxing.runtime.generator import GeneratorIntent, MotionBricksGenerator
from openroboxing.runtime.warp import Leg, warp
from openroboxing.spec.constants import GENERATOR_DT, NUM_FRAMES_PER_TOKEN
from openroboxing.studio import combination_record as cr
from openroboxing.studio.pose_record import PoseRecord

#: One combination of each character, so a failure says which kind failed.
DEFAULT_PICKS = ("shadow-boxing", "ib-dodge", "ib-combat-turn-jog")


def pick_records(prefixes: tuple[str, ...]) -> list[cr.CombinationRecord]:
    chosen: list[cr.CombinationRecord] = []
    files = sorted(COMBINATION_DIR.glob("*.json"))
    for prefix in prefixes:
        match = next((p for p in files if p.stem.startswith(prefix)), None)
        if match is None:
            raise SystemExit(f"no combination starting {prefix!r} in {COMBINATION_DIR}")
        chosen.append(cr.load(match))
    return chosen


def leg_pose(leg: Leg, name: str) -> PoseRecord:
    """A leg's target as the `PoseRecord` the override consumes.

    `generator._install_pose_override` hands `intent.pose` to
    `skeleton_fk.target_transforms`, which reads a `PoseRecord`, not a bare mapping.
    """
    return PoseRecord(
        name=name,
        joint_angles=dict(leg.joint_angles),
        horizon_tokens=leg.horizon_tokens,
        library_version="v0.2",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghost-distance", type=float, default=1.5)
    parser.add_argument("--prefixes", nargs="*", default=list(DEFAULT_PICKS))
    args = parser.parse_args()

    generator = MotionBricksGenerator()
    print(f"{'combination':34s} {'leg':>4s} {'tokens':>6s} {'pose err deg':>13s} {'frames':>7s}")

    for record in pick_records(tuple(args.prefixes)):
        generator.reset(seed=0)
        legs = warp(record, (0.0, 0.0), 0.0, (args.ghost_distance, 0.0))
        for index, leg in enumerate(legs):
            intent = GeneratorIntent(
                style="walk_boxing",
                movement_angle=leg.movement_angle,
                facing_angle=leg.facing_angle,
                target_position=leg.target_position,
                target_heading=leg.target_heading,
                pose=leg_pose(leg, f"{record.name}-leg{index}"),
                horizon_tokens=leg.horizon_tokens,
            )
            # `force=True` so each leg plans immediately rather than waiting for the cadence:
            # this measures the plan, not the replan schedule.
            generator.generate(intent, generator.context_qpos(), GENERATOR_DT, force=True)
            plan = generator.plan()
            arrived = plan[-1, 7:]
            wanted = np.array([leg.joint_angles[n] for n in G1.mujoco_joint_names])
            error = np.degrees(np.abs(arrived - wanted)).max()
            print(
                f"{record.name[:34]:34s} {index:4d} {leg.horizon_tokens:6d} "
                f"{error:13.1f} {len(plan):7d}"
            )
            expected = leg.horizon_tokens * NUM_FRAMES_PER_TOKEN
            if len(plan) != expected:
                print(f"    plan length {len(plan)}, asked for {expected}")
            # Consume the plan so the next leg's context is where this one left off, which is what
            # a real sequence would see.
            for _ in range(len(plan)):
                generator.next_frame()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the spike**

Run: `.venv_mb/bin/python -m openroboxing.tools.spike_warp_tracking --ghost-distance 1.5`
Expected: one row per leg, with a pose error in degrees and the plan length.

Two things can legitimately need adjusting here, and neither is a licence to touch `generator.py`
(invariant 3): `generator.reset` may not accept a `seed` keyword, and `next_frame` may raise once a
plan is exhausted. Read `src/openroboxing/runtime/generator.py:407-540` and adapt the spike if so.

- [ ] **Step 3: Record the verdict**

Write the measured table into `docs/perf/` as `2026-08-27-warp-tracking-spike.md`, with the command
that produced it and one paragraph of reading.

**The bar:** legs should reach their pose to within roughly the 2–3° the design cites for unhurried
plans, degrading gracefully at `MIN_TOKENS`. Errors above about 15° on the shortest legs mean the
forced-leg assumption does not hold, and D2's leg length or D4's drift is what to revisit — not the
design. Say so plainly in the write-up either way.

- [ ] **Step 4: Run the whole suite**

Run: `.venv_mb/bin/python -m pytest -q --no-header 2>&1 | tail -3`
Expected: `667 + 42 = 709 passed` (or thereabouts), `28 deselected`

- [ ] **Step 5: Lint**

Run: `./lint.sh`
Expected: clean. Fix anything it reports before committing.

- [ ] **Step 6: Commit**

```bash
git add src/openroboxing/tools/spike_warp_tracking.py docs/perf/2026-08-27-warp-tracking-spike.md
git commit -m "M5-T12: measure whether a forced leg under drift reaches its pose - phase 1 checkpoint"
```

---

## Definition of done for this plan

1. `motions/` is committed, or explicitly decided against — it is currently untracked.
2. `.venv_mb/bin/python -m pytest` green, including the new suites.
3. `./lint.sh` clean.
4. `poses/v0.2/combinations/` holds a library that loads and validates.
5. `docs/perf/2026-08-27-warp-tracking-spike.md` exists and states a verdict.
6. `CLAUDE.md`'s repository-layout section mentions `motions/`, `studio/motion_import.py`,
   `studio/segment.py`, `studio/combination_record.py` and `runtime/warp.py` — the file says it must
   be updated in the same PR as anything that makes it untrue.

## What this plan deliberately does not do

`intents.py`, `reference.py`, `fight.py`, the protocol and the client are untouched. The walk
approach still exists and `spec/intent.md` is still 2.2. Nothing selects a combination in a match
yet. That is Plan 2, and it should not be written until Task 12 has a verdict.
