"""Pose record: load, validate, and expand to the generator's skeleton (M2-T2).

Implements ``spec/pose_record.md`` v0.1.

Conventions
-----------
- A record stores the **29 robot joint angles in radians**, keyed by MuJoCo joint **name**. Never by
  index — `CLAUDE.md` invariant 4.
- The generator wants the 34-joint ``g1skel34`` skeleton. That form is *derived*
  (:func:`to_skeleton_angles`), never stored, so there is one source of truth.
- Validation **raises with the offending field named**. Nothing is coerced and no missing angle is
  defaulted: a pose that is wrong should fail at load, not drive a robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from openroboxing.runtime.conventions import G1, G1Conventions
from openroboxing.spec.constants import MAX_TOKENS, MIN_TOKENS

SCHEMA_VERSION = "0.2"

ADMISSION_STATES = ("draft", "admitted", "rejected")

#: Skeleton joints with no robot counterpart. Values are derived, never authored — see
#: spec/pose_record.md. `pelvis` comes from placement; the rest are unactuated on the 29-DOF G1.
DERIVED_SKELETON_JOINTS = (
    "pelvis_joint",  # from `pelvis_skel`; the root, owned by placement rather than by the pose
    "left_toe_base",
    "right_toe_base",
    "left_hand_roll_joint",
    "right_hand_roll_joint",
)


class PoseRecordError(ValueError):
    """A pose record is invalid. Always names the offending field."""


@dataclass(frozen=True)
class PoseSource:
    """Where an authored pose came from. Provenance, not behaviour."""

    clip: str
    start_frame: int
    end_frame: int

    def to_dict(self) -> dict[str, Any]:
        return {"clip": self.clip, "start_frame": self.start_frame, "end_frame": self.end_frame}


@dataclass(frozen=True)
class PoseRecord:
    """One authored key pose. See ``spec/pose_record.md``."""

    name: str
    joint_angles: dict[str, float]
    horizon_tokens: int
    library_version: str
    source: PoseSource | None = None
    adjustment_envelope: dict[str, float] = field(default_factory=dict)
    #: Worst-joint error, radians, between the commanded pose and the generator's plan endpoint.
    #: "Will MotionBricks produce this pose at all?" — the first gate, and the one admission turns on.
    generator_error_rad: float | None = None
    #: Worst-joint error, radians, between the reference motion and what the robot actually did under
    #: physics. A different question from the above, and it needs a runtime trial to answer.
    tracking_error_rad: float | None = None
    telegraph_ms: float | None = None
    admission: str = "draft"
    schema_version: str = SCHEMA_VERSION

    # -- conversions ------------------------------------------------------------------------------
    def to_array(self, conventions: G1Conventions = G1) -> np.ndarray:
        """The 29 angles as an array in MuJoCo order."""
        return np.array(
            [self.joint_angles[n] for n in conventions.mujoco_joint_names], dtype=np.float64
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "joint_angles": dict(self.joint_angles),
            "horizon_tokens": self.horizon_tokens,
            "source": self.source.to_dict() if self.source else None,
            "adjustment_envelope": dict(self.adjustment_envelope),
            "telegraph_ms": self.telegraph_ms,
            "generator_error_rad": self.generator_error_rad,
            "tracking_error_rad": self.tracking_error_rad,
            "admission": self.admission,
            "library_version": self.library_version,
        }

    def is_admitted(self) -> bool:
        return self.admission == "admitted"


def to_skeleton_angles(
    record: PoseRecord, skeleton_joint_names: list[str], conventions: G1Conventions = G1
) -> dict[str, float]:
    """Expand a 29-joint record onto the generator's skeleton.

    The 5 skeleton joints with no robot counterpart are filled with 0.0: ``pelvis`` is owned by
    placement rather than by the pose, and the toe and hand-roll joints are unactuated on this robot.

    Raises:
        PoseRecordError: if a skeleton joint is neither in the record nor a known derived joint —
            which would mean the skeleton changed and this mapping is stale.
    """
    out: dict[str, float] = {}
    for skel_name in skeleton_joint_names:
        robot_name = skel_name[:-5] + "_joint" if skel_name.endswith("_skel") else skel_name
        if robot_name in record.joint_angles:
            out[skel_name] = record.joint_angles[robot_name]
        elif robot_name in DERIVED_SKELETON_JOINTS:
            out[skel_name] = 0.0
        else:
            raise PoseRecordError(
                f"skeleton joint {skel_name!r} maps to {robot_name!r}, which is neither in the "
                "record nor a known derived joint; the skeleton or the mapping has changed"
            )
    return out


# -- validation ------------------------------------------------------------------------------------
def _joint_limits(conventions: G1Conventions = G1) -> dict[str, tuple[float, float]]:
    """Per-joint limits read from the MJCF, never a hard-coded table."""
    import mujoco

    from openroboxing.paths import G1_29DOF_XML

    model = mujoco.MjModel.from_xml_path(str(G1_29DOF_XML))
    limits: dict[str, tuple[float, float]] = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name in conventions.mujoco_joint_names and model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            limits[name] = (float(lo), float(hi))
    return limits


def validate(
    record: PoseRecord, conventions: G1Conventions = G1, check_limits: bool = True
) -> None:
    """Raise :class:`PoseRecordError` naming the first offending field."""
    if record.schema_version != SCHEMA_VERSION:
        raise PoseRecordError(
            f"schema_version: expected {SCHEMA_VERSION!r}, got {record.schema_version!r}"
        )
    if not record.name or " " in record.name:
        raise PoseRecordError(f"name: must be non-empty and space-free, got {record.name!r}")

    expected = set(conventions.mujoco_joint_names)
    actual = set(record.joint_angles)
    missing, extra = expected - actual, actual - expected
    if missing:
        raise PoseRecordError(f"joint_angles: missing {len(missing)} joints: {sorted(missing)}")
    if extra:
        raise PoseRecordError(f"joint_angles: unknown joints: {sorted(extra)}")
    for joint, angle in record.joint_angles.items():
        if not np.isfinite(angle):
            raise PoseRecordError(f"joint_angles[{joint!r}]: not finite ({angle})")

    if not MIN_TOKENS <= record.horizon_tokens <= MAX_TOKENS:
        raise PoseRecordError(
            f"horizon_tokens: {record.horizon_tokens} outside [{MIN_TOKENS}, {MAX_TOKENS}]"
        )

    if record.admission not in ADMISSION_STATES:
        raise PoseRecordError(f"admission: {record.admission!r} not one of {ADMISSION_STATES}")
    if record.admission == "admitted" and record.generator_error_rad is None:
        # The rule that stops an unmeasured pose reaching a match. `telegraph_ms` is deliberately
        # *not* required: a telegraph is what a player reads, not a quantity an offline proxy can
        # settle, and gating on a number nobody trusts would only launder it. It is recorded when
        # available and tuned in playtest (WORKPLAN M4-T4). See spec/pose_record.md.
        raise PoseRecordError(
            "generator_error_rad: required when admission is 'admitted' (M2-T5)"
        )

    for joint, bound in record.adjustment_envelope.items():
        if joint not in expected:
            raise PoseRecordError(f"adjustment_envelope: unknown joint {joint!r}")
        if not np.isfinite(bound) or bound <= 0:
            raise PoseRecordError(
                f"adjustment_envelope[{joint!r}]: bound must be positive, got {bound}"
            )

    if check_limits:
        for joint, (lo, hi) in _joint_limits(conventions).items():
            angle = record.joint_angles[joint]
            if not lo <= angle <= hi:
                raise PoseRecordError(
                    f"joint_angles[{joint!r}]: {angle:.4f} outside the model's limits "
                    f"[{lo:.4f}, {hi:.4f}]"
                )
            bound = record.adjustment_envelope.get(joint)
            if bound is not None and not (lo <= angle - bound and angle + bound <= hi):
                raise PoseRecordError(
                    f"adjustment_envelope[{joint!r}]: +-{bound} around {angle:.4f} leaves the "
                    f"model's limits [{lo:.4f}, {hi:.4f}]"
                )


def from_dict(data: dict[str, Any], validate_record: bool = True, **kwargs) -> PoseRecord:
    """Build a record from a plain dict, raising on anything unexpected."""
    required = ("name", "joint_angles", "horizon_tokens", "library_version")
    for key in required:
        if key not in data:
            raise PoseRecordError(f"{key}: required field is missing")

    source_data = data.get("source")
    source = None
    if source_data is not None:
        try:
            source = PoseSource(
                clip=source_data["clip"],
                start_frame=int(source_data["start_frame"]),
                end_frame=int(source_data["end_frame"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PoseRecordError(f"source: malformed ({exc})") from exc

    record = PoseRecord(
        name=data["name"],
        joint_angles={str(k): float(v) for k, v in data["joint_angles"].items()},
        horizon_tokens=int(data["horizon_tokens"]),
        library_version=str(data["library_version"]),
        source=source,
        adjustment_envelope={
            str(k): float(v) for k, v in (data.get("adjustment_envelope") or {}).items()
        },
        telegraph_ms=data.get("telegraph_ms"),
        generator_error_rad=data.get("generator_error_rad"),
        tracking_error_rad=data.get("tracking_error_rad"),
        admission=data.get("admission", "draft"),
        schema_version=data.get("schema_version", SCHEMA_VERSION),
    )
    if validate_record:
        validate(record, **kwargs)
    return record


def load(path: Path, validate_record: bool = True, **kwargs) -> PoseRecord:
    """Load and validate one pose record from JSON."""
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise PoseRecordError(f"{path}: not valid JSON ({exc})") from exc
    try:
        return from_dict(data, validate_record=validate_record, **kwargs)
    except PoseRecordError as exc:
        raise PoseRecordError(f"{path}: {exc}") from exc


def save(record: PoseRecord, path: Path, validate_record: bool = True, **kwargs) -> None:
    """Validate then write a pose record as JSON."""
    if validate_record:
        validate(record, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n")


def load_library(directory: Path, **kwargs) -> dict[str, PoseRecord]:
    """Load every ``*.json`` in a directory, keyed by pose name. Raises on a duplicate name."""
    records: dict[str, PoseRecord] = {}
    for path in sorted(Path(directory).glob("*.json")):
        record = load(path, **kwargs)
        if record.name in records:
            raise PoseRecordError(f"{path}: duplicate pose name {record.name!r}")
        records[record.name] = record
    return records
