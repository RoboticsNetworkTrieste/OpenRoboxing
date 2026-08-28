"""Segment a take into key poses and combinations (M5-T5).

Implements decisions D1 and D2 of
``docs/superpowers/specs/2026-08-27-motion-combinations-design.md``, corrected 2026-08-28: D1
originally called for segmenting on "salient-joint speed peaks" (joint-space speed, quantile
threshold). That rule is the defect this module now fixes — see "Why turning points, not speed
peaks" below and the correction recorded in the design doc itself.

The bug this replaces
----------------------
A punch's peak *speed* is halfway through the extension. At full extension the hand is momentarily
**stationary** — that is where the motion reverses, and a jab that never reverses is not a jab. The
old rule (a quantile of joint-space speed) therefore sampled the moments *between* poses and never
the poses themselves: MotionBricks in-betweens between two mid-swing configurations, which reads as
"statuary positioning" rather than a punch landing. Reported by the project owner 2026-08-27.

Why turning points, not speed peaks
------------------------------------
A key pose is where the body's own motion **turns around** — a local extremum, not a local rate
maximum. In one dimension a punch's extension distance rises, peaks (zero velocity, the reversal)
and falls; the reversal is the pose, and it is a minimum of *speed*, not a maximum. Framed this way
segmenting is peak-detection on position, not on its derivative.

Why Cartesian body space, not joint space
------------------------------------------
This is the mirror image of the argument ``telegraph.py`` already makes: *"Distance is measured in
Cartesian body space, not joint space, via forward kinematics. Two poses can differ a lot in joint
angles while the fists barely move, and it is the fists the opponent watches."* Here the direction of
the argument reverses but the reason is the same signal: during a punch several joints keep rotating
through the reversal (the shoulder can still be rotating while the wrist has stopped, or elbow and
shoulder can trade off extension) while the **fist** — the thing that actually turns around — sits
still for an instant. A joint-space extremum finder measured this and recovered only 5-8 of 13
punches in a shadow-boxing take; the Cartesian one below recovers 13 of 13 in the same take.

Three signals are computed per frame via forward kinematics on the G1 model
(:data:`~openroboxing.paths.G1_29DOF_SIM_XML`, the simulation-ready revision — see
``CLAUDE.md``'s "traps found the hard way"):

- **reach** — the further wrist's distance from the pelvis. Punches.
- **level** — the pelvis's height above the mean ankle height. Ducking, rising.
- **shift** — the pelvis's horizontal distance from the foot midpoint. Slips, leans, weight
  transfer.

Why prioritised, not unioned
-----------------------------
**Measured, not a taste call.** Unioning all three signals' turning points and picking the strongest
first drops punch capture from 39/48 to 14/48 across a 7-take sample spanning all three motion
families (shadow boxing, dodges, jog-turns): with only :data:`MIN_KEYFRAME_GAP_FRAMES` of spacing
budget, a slip's prominence routinely outranks and evicts a nearby punch, because ``shift`` and
``level`` vary almost everywhere a fighter moves while ``reach`` is quiet except during a strike.

So **reach is taken first**, greedily strongest-first, each keyframe kept
:data:`MIN_KEYFRAME_GAP_FRAMES` clear of those already chosen — this alone must capture the punches,
because nothing downstream may unseat one. **Then level and shift fill remaining space only**: they
may occupy frames no reach keyframe claimed, and may never displace one. Verified result of this
design over the same 7-take sample: **39 of 48 punches captured, 119 keyframes**. The 9 misses are
punches sitting closer than :data:`MIN_KEYFRAME_GAP_FRAMES` to a stronger one — the irreducible cost
of the generator's shortest plannable leg, not a defect in the rule.

Conventions
-----------
- **Input** is ``(N, 36)`` MuJoCo qpos at :data:`~openroboxing.spec.constants.GENERATOR_HZ`.
- **Output** indices are frames into that array, already sorted and deduplicated.
- Reach turning points are eligible at prominence >= :data:`REACH_TURNING_PROMINENCE_M`; level and
  shift fill turning points at >= :data:`FILL_TURNING_PROMINENCE_M`. Both are metres, both measured
  — see the constants' own docstrings for what pinned each value.
- Keyframes are never closer than
  :data:`~openroboxing.spec.constants.MIN_KEYFRAME_GAP_FRAMES` nor further apart than
  :data:`~openroboxing.spec.constants.MAX_LEG_FRAMES` — the shortest and longest plans MotionBricks
  can produce. The lower bound is enforced by selection, the upper by :func:`densify`, which now
  inserts at the strongest turning point inside the gap (any of the three signals) rather than at
  the gap's busiest frame, for the same reason the top-level selection changed: the busiest frame in
  a gap is a mid-swing frame, not a pose.
- Once a combination's keyframes are fixed, :func:`leg_tokens` converts each gap in frames to a leg
  length in tokens. Rounding each leg independently would drift by up to half a token per leg, so the
  rounding residual is carried forward and diffused across the combination, holding the *total*
  duration within one token of the recording however many legs it has (design D2).
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from openroboxing.paths import G1_29DOF_SIM_XML
from openroboxing.runtime.conventions import G1, G1Conventions
from openroboxing.spec.constants import (
    COMBINATION_MAX_KEYFRAMES,
    COMBINATION_MIN_KEYFRAMES,
    FILL_TURNING_PROMINENCE_M,
    MAX_LEG_FRAMES,
    MAX_TOKENS,
    MIN_KEYFRAME_GAP_FRAMES,
    MIN_TOKENS,
    NUM_FRAMES_PER_TOKEN,
    REACH_TURNING_PROMINENCE_M,
)

if TYPE_CHECKING:
    import mujoco

#: Both wrists: a punch may be thrown with either hand and reach must not miss the off-hand.
REACH_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")

#: The feet, which ``level`` and ``shift`` are measured against — "how low" and "how far
#: off-centre" are both foot-relative questions.
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")

PELVIS_BODY = "pelvis"


class SegmentError(RuntimeError):
    """A take could not be segmented. Never recovered from silently."""


def _body_id(model: mujoco.MjModel, name: str) -> int:
    """A body id resolved by name, raising rather than silently returning -1 (invariant 4)."""
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise SegmentError(f"body {name!r} not in {G1_29DOF_SIM_XML}")
    return body_id


def body_signals(
    qpos: np.ndarray, model: mujoco.MjModel | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three Cartesian signals key poses turn around on. Each ``(N,)``, metres.

    Computed via ``mujoco.mj_kinematics`` on :data:`~openroboxing.paths.G1_29DOF_SIM_XML` — forward
    kinematics only, no dynamics, so this is cheap: ~30k calls over the whole 38-take corpus takes
    seconds. ``model`` may be supplied to reuse one across many takes instead of loading it per call.
    """
    import mujoco

    arr = np.asarray(qpos, dtype=np.float64)
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(G1_29DOF_SIM_XML))
    data = mujoco.MjData(model)

    pelvis_id = _body_id(model, PELVIS_BODY)
    reach_ids = [_body_id(model, name) for name in REACH_BODIES]
    foot_ids = [_body_id(model, name) for name in FOOT_BODIES]

    n = arr.shape[0]
    reach = np.empty(n)
    level = np.empty(n)
    shift = np.empty(n)
    for i in range(n):
        data.qpos[:] = arr[i]
        mujoco.mj_kinematics(model, data)
        pelvis = data.xpos[pelvis_id]
        wrists = data.xpos[reach_ids]
        feet = data.xpos[foot_ids]
        reach[i] = float(np.linalg.norm(wrists - pelvis, axis=-1).max())
        level[i] = float(pelvis[2] - feet[:, 2].mean())
        shift[i] = float(np.linalg.norm(pelvis[:2] - feet[:, :2].mean(axis=0)))
    return reach, level, shift


def turning_points(signal: np.ndarray, min_prominence: float = 0.0) -> list[tuple[int, float]]:
    """Local maxima and minima of ``signal`` — the frames where it reverses — strongest first.

    Both directions are turning points: a punch's extension is a ``reach`` maximum, its retraction
    to guard a ``reach`` minimum, and both are places the body actually pauses. ``min_prominence``
    (same units as ``signal``) is the peak-detection floor; 0 returns every turning point, which is
    what :func:`densify` searches so its fallback covers as much of the gap as the signals contain.
    """
    from scipy.signal import find_peaks

    maxima, max_props = find_peaks(signal, prominence=min_prominence)
    minima, min_props = find_peaks(-signal, prominence=min_prominence)
    points = list(zip(maxima.tolist(), max_props["prominences"].tolist(), strict=True))
    points += list(zip(minima.tolist(), min_props["prominences"].tolist(), strict=True))
    points.sort(key=lambda point: -point[1])
    return points


def densify(
    indices: list[int], turning: list[tuple[int, float]], *, min_gap: int, max_gap: int
) -> list[int]:
    """Insert keyframes until no gap exceeds ``max_gap``.

    A gap longer than one plan cannot be in-betweened in a single leg. The alternative — holding the
    previous pose across several legs — would make the fighter stand still through a stretch the
    recording spent moving. The inserted frame is the strongest turning point inside the gap (any of
    reach, level or shift — ``turning`` is the union, sorted strongest-first by :func:`turning_points`
    at zero prominence) rather than the gap's busiest frame: the busiest frame is mid-swing, same bug
    the top-level selection fixes. If the gap contains no turning point at all, the fallback is its
    midpoint.

    The inserted frame is kept ``min_gap`` clear of both neighbours, which is always possible because
    ``max_gap`` is more than twice ``min_gap``.
    """
    out = sorted(indices)
    while True:
        for left, right in pairwise(out):
            if right - left <= max_gap:
                continue
            low, high = left + min_gap, right - min_gap
            if high < low:
                raise SegmentError(
                    f"cannot split a {right - left}-frame gap while keeping {min_gap} clear"
                )
            chosen = next(
                (frame for frame, _ in turning if low <= frame <= high and frame not in out),
                (low + high) // 2,
            )
            out = sorted(set(out) | {chosen})
            break
        else:
            return out


def keyframe_indices(
    qpos: np.ndarray,
    *,
    min_gap: int = MIN_KEYFRAME_GAP_FRAMES,
    max_gap: int = MAX_LEG_FRAMES,
    reach_prominence: float = REACH_TURNING_PROMINENCE_M,
    fill_prominence: float = FILL_TURNING_PROMINENCE_M,
    conventions: G1Conventions = G1,
    model: mujoco.MjModel | None = None,
) -> np.ndarray:
    """Frames a combination is built from: where the body turns around, spaced so every leg is
    plannable.

    ``reach`` turning points are taken first, greedily strongest-first, each kept ``min_gap`` clear
    of those already chosen — nothing may unseat one of these once picked. ``level`` and ``shift``
    turning points are added next as fill only, into whatever space remains. Finally
    :func:`densify` closes any gap still longer than ``max_gap``. See the module docstring for why
    this order — and not a union of all three — is the one that captures punches.

    ``conventions`` is accepted for compatibility with the caller
    (``combination_record.build_from_take``, which segments and assembles keyframes against the same
    :class:`G1Conventions` in one pass) but is not used here: the forward kinematics this function
    needs always run against :data:`~openroboxing.paths.G1_29DOF_SIM_XML`, a physical-model question
    unrelated to joint-order conventions.
    """
    del conventions
    if qpos.shape[0] < 2:
        raise SegmentError("a take with fewer than two frames cannot be segmented")

    reach, level, shift = body_signals(qpos, model)

    reach_points = turning_points(reach, reach_prominence)
    picked: list[int] = []
    for frame, _ in reach_points:
        if all(abs(frame - other) >= min_gap for other in picked):
            picked.append(frame)

    # Fill only after reach has had first pick: level and shift may occupy space no reach keyframe
    # claimed, but nothing here may displace one already chosen.
    fill_points = turning_points(level, fill_prominence) + turning_points(shift, fill_prominence)
    fill_points.sort(key=lambda point: -point[1])
    for frame, _ in fill_points:
        if all(abs(frame - other) >= min_gap for other in picked):
            picked.append(frame)
    picked.sort()

    # Gated on reach-plus-fill, not reach alone: a quiet take (one punch, or none) still has ducks
    # and weight shifts to seed from, and `densify` below only bridges gaps between existing seeds —
    # it cannot bootstrap a combination out of fewer than `COMBINATION_MIN_KEYFRAMES` of them.
    if len(picked) < COMBINATION_MIN_KEYFRAMES:
        raise SegmentError(
            f"only {len(picked)} turning points at prominence >= {fill_prominence} m "
            f"(reach >= {reach_prominence} m); a combination needs {COMBINATION_MIN_KEYFRAMES}"
        )

    all_points = turning_points(reach, 0.0) + turning_points(level, 0.0) + turning_points(shift, 0.0)
    all_points.sort(key=lambda point: -point[1])
    dense = densify(picked, all_points, min_gap=min_gap, max_gap=max_gap)
    return np.array(dense, dtype=int)


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
        chosen = max(MIN_TOKENS, min(MAX_TOKENS, round(exact)))
        residual = exact - chosen
        tokens.append(chosen)
    return tokens
