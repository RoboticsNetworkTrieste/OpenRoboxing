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
- Once a combination's keyframes are fixed, :func:`leg_tokens` converts each gap in frames to a leg
  length in tokens. Rounding each leg independently would drift by up to half a token per leg, so the
  rounding residual is carried forward and diffused across the combination, holding the *total*
  duration within one token of the recording however many legs it has (design D2).
"""

from __future__ import annotations

from itertools import pairwise

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
        for left, right in pairwise(out):
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
