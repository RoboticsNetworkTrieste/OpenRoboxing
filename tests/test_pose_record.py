"""M2-T2 acceptance: pose record schema validation.

Acceptance criterion from WORKPLAN.md M2-T2:
  schema validation round-trips; an invalid record fails with a specific message naming the
  offending field.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_pose_record.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.runtime.conventions import G1
from openroboxing.spec.constants import MAX_TOKENS, MIN_TOKENS, NUM_JOINTS
from openroboxing.studio.pose_record import (
    DERIVED_SKELETON_JOINTS,
    PoseRecord,
    PoseRecordError,
    PoseSource,
    from_dict,
    load,
    load_library,
    save,
    to_skeleton_angles,
    validate,
)


def _record(**overrides) -> PoseRecord:
    """A valid draft record: the default standing pose, which is always within limits."""
    from openroboxing.runtime.obs import default_angles

    angles = dict(zip(G1.mujoco_joint_names, default_angles(G1, "mujoco")))
    base = {
        "name": "guard-high",
        "joint_angles": angles,
        "horizon_tokens": 8,
        "library_version": "v0.1",
        "source": PoseSource(clip="walk_boxing", start_frame=25, end_frame=35),
    }
    base.update(overrides)
    return PoseRecord(**base)


# --- the acceptance criterion ---------------------------------------------------------------------
def test_round_trip_through_json(tmp_path) -> None:
    record = _record()
    path = tmp_path / "guard-high.json"
    save(record, path)
    loaded = load(path)
    assert loaded.name == record.name
    assert loaded.horizon_tokens == record.horizon_tokens
    assert loaded.source.clip == "walk_boxing"
    assert np.allclose(loaded.to_array(), record.to_array())


def test_round_trip_through_dict() -> None:
    record = _record()
    assert np.allclose(from_dict(record.to_dict()).to_array(), record.to_array())


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda d: d.pop("name"), "name"),
        (lambda d: d.pop("horizon_tokens"), "horizon_tokens"),
        (lambda d: d.pop("library_version"), "library_version"),
        (lambda d: d["joint_angles"].pop("left_knee_joint"), "joint_angles"),
        (lambda d: d["joint_angles"].update({"not_a_joint": 0.0}), "joint_angles"),
        (lambda d: d.update({"horizon_tokens": MAX_TOKENS + 1}), "horizon_tokens"),
        (lambda d: d.update({"horizon_tokens": MIN_TOKENS - 1}), "horizon_tokens"),
        (lambda d: d.update({"admission": "probably-fine"}), "admission"),
        (lambda d: d.update({"schema_version": "0.0"}), "schema_version"),
        (lambda d: d.update({"adjustment_envelope": {"nope_joint": 0.1}}), "adjustment_envelope"),
        (
            lambda d: d.update({"adjustment_envelope": {"left_elbow_joint": -0.1}}),
            "adjustment_envelope",
        ),
        (lambda d: d["joint_angles"].update({"left_knee_joint": float("nan")}), "joint_angles"),
    ],
)
def test_invalid_record_names_the_offending_field(mutate, expected) -> None:
    data = _record().to_dict()
    mutate(data)
    with pytest.raises(PoseRecordError, match=expected):
        from_dict(data)


# --- the rule that keeps unmeasured poses out of matches --------------------------------------------
def test_admitted_does_not_require_a_telegraph() -> None:
    """A telegraph is what a player reads, not something an offline proxy settles.

    The M2-T3 proxy gave mirrored poses windows differing by 433 ms, so gating admission on it would
    only launder a number nobody trusts. Recorded when available, tuned in playtest (M4-T4).
    """
    validate(_record(admission="admitted", generator_error_rad=0.1))


def test_admitted_requires_a_measured_generator_error() -> None:
    """The gate that stops a pose the generator will not produce reaching a match."""
    with pytest.raises(PoseRecordError, match="generator_error_rad"):
        validate(_record(admission="admitted", telegraph_ms=120.0))


def test_admitted_with_the_measurements_is_valid() -> None:
    validate(
        _record(
            admission="admitted",
            telegraph_ms=120.0,
            generator_error_rad=0.1,
            tracking_error_rad=0.2,
        )
    )


def test_draft_needs_no_measurements() -> None:
    validate(_record())
    assert not _record().is_admitted()


# --- joint limits come from the model, not a table --------------------------------------------------
def test_angle_outside_the_model_limits_is_rejected() -> None:
    data = _record().to_dict()
    data["joint_angles"]["left_knee_joint"] = 99.0
    with pytest.raises(PoseRecordError, match="outside the model's limits"):
        from_dict(data)


def test_envelope_that_leaves_the_limits_is_rejected() -> None:
    """The envelope's corners must be reachable, since admission only covers the corners."""
    from openroboxing.runtime.obs import default_angles

    angles = dict(zip(G1.mujoco_joint_names, default_angles(G1, "mujoco")))
    record = _record(joint_angles=angles, adjustment_envelope={"left_knee_joint": 50.0})
    with pytest.raises(PoseRecordError, match="leaves the"):
        validate(record)


# --- the 29 -> 34 expansion -------------------------------------------------------------------------
def test_expansion_to_the_generator_skeleton() -> None:
    """Every skeleton joint is filled: 29 from the record, 5 derived."""
    import sys

    from openroboxing.paths import MOTIONBRICKS_ROOT

    if str(MOTIONBRICKS_ROOT) not in sys.path:
        sys.path.insert(0, str(MOTIONBRICKS_ROOT))
    skeletons = pytest.importorskip("motionbricks.motionlib.core.skeletons.g1")

    names = [n for n, _ in skeletons.G1Skeleton34.bone_order_names_with_parents]
    expanded = to_skeleton_angles(_record(), names)
    assert len(expanded) == len(names) == 34
    assert set(expanded) == set(names)


def test_all_robot_joints_exist_in_the_skeleton() -> None:
    """The premise of storing 29: the skeleton is a strict superset of the robot."""
    import sys

    from openroboxing.paths import MOTIONBRICKS_ROOT

    if str(MOTIONBRICKS_ROOT) not in sys.path:
        sys.path.insert(0, str(MOTIONBRICKS_ROOT))
    skeletons = pytest.importorskip("motionbricks.motionlib.core.skeletons.g1")

    names = [n for n, _ in skeletons.G1Skeleton34.bone_order_names_with_parents]
    stripped = {n[:-5] + "_joint" if n.endswith("_skel") else n for n in names}
    assert set(G1.mujoco_joint_names) <= stripped, "a robot joint is missing from the skeleton"
    assert stripped - set(G1.mujoco_joint_names) == set(DERIVED_SKELETON_JOINTS)


def test_unknown_skeleton_joint_raises() -> None:
    with pytest.raises(PoseRecordError, match="neither in the record nor"):
        to_skeleton_angles(_record(), ["some_new_joint"])


# --- arrays and libraries ---------------------------------------------------------------------------
def test_to_array_is_in_mujoco_order() -> None:
    from openroboxing.runtime.obs import default_angles

    assert np.allclose(_record().to_array(), default_angles(G1, "mujoco"))
    assert _record().to_array().shape == (NUM_JOINTS,)


def test_library_load_and_duplicate_names(tmp_path) -> None:
    save(_record(name="jab-left"), tmp_path / "a.json")
    save(_record(name="hook-right"), tmp_path / "b.json")
    library = load_library(tmp_path)
    assert set(library) == {"jab-left", "hook-right"}

    save(_record(name="jab-left"), tmp_path / "c.json")
    with pytest.raises(PoseRecordError, match="duplicate pose name"):
        load_library(tmp_path)


def test_malformed_json_names_the_file(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(PoseRecordError, match="not valid JSON"):
        load(path)
