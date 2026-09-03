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

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from openroboxing.runtime.conventions import G1, G1Conventions, quat_wxyz_to_yaw
from openroboxing.spec.constants import (
    COMBINATION_MAX_KEYFRAMES,
    COMBINATION_MIN_KEYFRAMES,
    MAX_TARGET_LEG_TOKENS,
    MIN_TOKENS,
    SECONDS_PER_TOKEN,
    TICK_HZ,
)
from openroboxing.studio import segment
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

    def __post_init__(self) -> None:
        # Normalise so equality (and JSON round trips) never trip on list-vs-tuple or an unfrozen
        # mapping type. `Mapping`/`Sequence` are accepted at the boundary; the canonical stored
        # form is always a plain dict and a 2-tuple.
        object.__setattr__(self, "joint_angles", dict(self.joint_angles))
        object.__setattr__(self, "root_offset", tuple(self.root_offset))

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
        # Normalise to a tuple so a `list` and a `tuple` of identical keyframes compare equal —
        # dataclass equality compares the sequence itself, not just its elements.
        object.__setattr__(self, "keyframes", tuple(self.keyframes))
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
            # MAX_TARGET_LEG_TOKENS, not MAX_TOKENS: since `spec/intent.md` 3.2 a leg is no longer
            # one plan, so the planner's per-plan maximum does not bound it. A long leg runs an
            # untargeted phase and then a landing in-between (`runtime/sequence.py`).
            if not MIN_TOKENS <= keyframe.leg_tokens <= MAX_TARGET_LEG_TOKENS:
                raise CombinationError(
                    f"{record.name} keyframe {i}: leg_tokens {keyframe.leg_tokens} outside "
                    f"[{MIN_TOKENS}, {MAX_TARGET_LEG_TOKENS}]"
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


def _heading(qpos_row: np.ndarray) -> float:
    """Heading in radians from a MuJoCo ``wxyz`` root quaternion: rotation about world Z.

    Delegates to the shared formula (`runtime.conventions.quat_wxyz_to_yaw`) so a recorded take's
    heading and a live fighter's heading (`runtime.fight`) are never two independent derivations of
    the same convention.
    """
    return quat_wxyz_to_yaw(qpos_row[3:7])


def _wrap(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return -((-angle + math.pi) % (2.0 * math.pi) - math.pi)


def _slug(take: str, index: int) -> str:
    """A kebab-case name unique within a take."""
    stem = take.lower().replace("_", "-")
    while "--" in stem:
        stem = stem.replace("--", "-")
    return f"{stem.strip('-')}-{index:02d}"


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

    Every leg is **reachable** by construction: :func:`segment.keyframe_indices_with_provenance`
    densifies any gap longer than ``MAX_TARGET_LEG_FRAMES``, and a leg longer than one plan is run as
    an untargeted phase followed by a landing in-between (``runtime/sequence.py``), so there is no
    splitting and no repeated keyframe here. A run whose legs cannot be tokenised raises rather than
    being dropped, because a silently skipped combination is a silently smaller library
    (`CLAUDE.md` invariant 5).
    """
    # Thinned to sparse targets before grouping: `spec/intent.md` 3.2 halves the number of poses a
    # plan is aimed at so each leg carries twice the motion. Provenance — which frames were punches —
    # only exists at this point, because a Keyframe records angles and timing, not how it was chosen.
    detected, punches = segment.keyframe_indices_with_provenance(qpos, conventions=conventions)
    indices = np.array(segment.thin_targets(detected, punches), dtype=int)
    records: list[CombinationRecord] = []
    for position, run in enumerate(segment.combination_runs(indices)):
        origin = int(run[0])
        base_position = qpos[origin, 0:2]
        base_heading = _heading(qpos[origin])
        tokens = segment.leg_tokens([int(b) - int(a) for a, b in pairwise(run)])
        keyframes = [
            Keyframe(
                joint_angles=dict(zip(conventions.mujoco_joint_names, qpos[origin, 7:].tolist())),
                leg_tokens=None,
                root_offset=(0.0, 0.0),
                heading_offset=0.0,
            )
        ]
        for frame, leg in zip(run[1:], tokens, strict=True):
            keyframes.append(
                Keyframe(
                    joint_angles=dict(
                        zip(conventions.mujoco_joint_names, qpos[int(frame), 7:].tolist())
                    ),
                    leg_tokens=leg,
                    root_offset=(
                        float(qpos[int(frame), 0] - base_position[0]),
                        float(qpos[int(frame), 1] - base_position[1]),
                    ),
                    heading_offset=_wrap(_heading(qpos[int(frame)]) - base_heading),
                )
            )
        records.append(
            CombinationRecord(
                name=_slug(take_name, position),
                library_version=library_version,
                source=CombinationSource(
                    take=take_name,
                    start_frame=origin,
                    end_frame=int(run[-1]),
                    mirrored=take_name.endswith("_M"),
                ),
                keyframes=keyframes,
                conventions=conventions,
            )
        )
    return records
