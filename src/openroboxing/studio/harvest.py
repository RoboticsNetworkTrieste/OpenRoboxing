"""Harvest candidate key poses out of generated motion (M2-T5).

Harvesting is one route, not the only one
----------------------------------------
This module was written believing authored poses were ignored by the generator. That was a
measurement error (`spec/upstream_notes.md`): poses authored directly in joint space are reached to
2–3°, so a library does **not** have to be harvested.

Harvesting remains useful for what it is actually good at — finding configurations that are natural
for a given style, and seeding an author who would rather start from something plausible than from a
blank set of joint angles. It is a source of proposals, not a workaround.

This is a *proposal* step, not an authoring step. A candidate is only a pose because a person says it
is a jab; the code can find distinctive configurations, and cannot know what a jab is.

Conventions
-----------
- **Input** is a ``(N, 36)`` qpos stream in MuJoCo order, at any rate.
- **Output** records are ``draft`` and carry no measurements. Admission is a separate, later step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openroboxing.runtime.conventions import G1, G1Conventions
from openroboxing.spec.constants import MAX_TOKENS, MIN_TOKENS, QPOS_DIM
from openroboxing.studio.pose_record import PoseRecord, PoseSource

#: Joints whose motion makes a boxing pose distinctive. Legs and waist carry stance, but two guards
#: differ at the hands; scoring on everything would rank a long stride above a thrown punch.
SALIENT_JOINT_SUBSTRINGS = ("shoulder", "elbow", "wrist")


class HarvestError(RuntimeError):
    """Candidates could not be harvested. Never recovered from silently."""


@dataclass(frozen=True)
class Candidate:
    """One harvested frame, with enough provenance to find it again."""

    frame: int
    angles: np.ndarray  # (29,) MuJoCo order
    distinctiveness: float
    style: str
    seed: int

    def to_record(
        self,
        name: str,
        *,
        horizon_tokens: int = 8,
        library_version: str = "v0.1-draft",
        conventions: G1Conventions = G1,
    ) -> PoseRecord:
        """A draft record for this candidate. Named by a human, because naming is the judgment."""
        if not MIN_TOKENS <= horizon_tokens <= MAX_TOKENS:
            raise HarvestError(
                f"horizon_tokens {horizon_tokens} outside [{MIN_TOKENS}, {MAX_TOKENS}]"
            )
        return PoseRecord(
            name=name,
            joint_angles=dict(zip(conventions.mujoco_joint_names, self.angles.tolist())),
            horizon_tokens=horizon_tokens,
            library_version=library_version,
            source=PoseSource(clip=self.style, start_frame=self.frame, end_frame=self.frame),
        )


def salient_joints(conventions: G1Conventions = G1) -> np.ndarray:
    """Indices of the joints candidate scoring looks at."""
    indices = [
        i
        for i, name in enumerate(conventions.mujoco_joint_names)
        if any(part in name for part in SALIENT_JOINT_SUBSTRINGS)
    ]
    if not indices:
        raise HarvestError(
            f"no joint name contains any of {SALIENT_JOINT_SUBSTRINGS}; the naming has changed"
        )
    return np.array(indices)


def distinctiveness(qpos: np.ndarray, conventions: G1Conventions = G1) -> np.ndarray:
    """Per-frame distance from the run's own mean pose, over the salient joints.

    Measured against the run's mean rather than a fixed reference so it adapts to whatever the style
    is doing: the frames that stand out of *this* motion are the ones worth looking at.
    """
    arr = np.asarray(qpos, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != QPOS_DIM:
        raise HarvestError(f"expected (N, {QPOS_DIM}) qpos, got shape {arr.shape}")

    joints = arr[:, 7:][:, salient_joints(conventions)]
    return np.linalg.norm(joints - joints.mean(axis=0), axis=1)


def harvest(
    qpos: np.ndarray,
    *,
    count: int = 8,
    min_separation: int = 10,
    skip_frames: int = 0,
    style: str = "walk_boxing",
    seed: int = 1234,
    conventions: G1Conventions = G1,
) -> list[Candidate]:
    """Pick the most distinctive, well-separated frames of a motion.

    Args:
        count: how many candidates to return.
        min_separation: minimum frames between two candidates. Without it the top scores all land on
            consecutive frames of one punch and every candidate is the same pose.
        skip_frames: ignore this many frames at the start. The generator begins from a neutral
            standing pose and takes a beat to settle into the style, and those transient frames
            otherwise sweep the ranking — they are far from the run's mean without being anything a
            player would throw. Measured on `walk_boxing`: the top four candidates were all warmup,
            arms at the sides.

    Returns:
        Candidates in descending distinctiveness. Fewer than ``count`` if the motion is too short to
        hold that many separated frames — which is reported, not padded.
    """
    if count < 1:
        raise HarvestError(f"count must be at least 1, got {count}")
    if min_separation < 1:
        raise HarvestError(f"min_separation must be at least 1, got {min_separation}")
    if skip_frames < 0:
        raise HarvestError(f"skip_frames must not be negative, got {skip_frames}")

    arr = np.asarray(qpos, dtype=np.float64)
    if skip_frames >= arr.shape[0]:
        raise HarvestError(
            f"skip_frames {skip_frames} leaves nothing of a {arr.shape[0]}-frame motion"
        )

    # Scored after the skip, so the discarded warmup does not shift the mean the rest is judged
    # against.
    scores = np.full(arr.shape[0], -np.inf)
    scores[skip_frames:] = distinctiveness(arr[skip_frames:], conventions)

    chosen: list[int] = []
    for frame in np.argsort(-scores):
        if len(chosen) >= count or not np.isfinite(scores[frame]):
            break
        if all(abs(int(frame) - taken) >= min_separation for taken in chosen):
            chosen.append(int(frame))

    return [
        Candidate(
            frame=frame,
            angles=arr[frame, 7:].copy(),
            distinctiveness=float(scores[frame]),
            style=style,
            seed=seed,
        )
        for frame in chosen
    ]
