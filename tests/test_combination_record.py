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
    fields = {
        "name": "test-combo",
        "library_version": "v0.2",
        "source": cr.CombinationSource(take="t", start_frame=0, end_frame=60, mirrored=False),
        "keyframes": keyframes,
    }
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


def test_rejects_a_leg_outside_the_planner_bounds():
    keyframes = list(make().keyframes)
    keyframes[1] = cr.Keyframe(dict(ANGLES), 99, (0.1, 0.0), 0.2)
    with pytest.raises(cr.CombinationError, match="leg_tokens"):
        make(keyframes=keyframes)


def test_admitted_requires_both_measurements():
    with pytest.raises(cr.CombinationError, match="admitted"):
        make(admission="admitted")
    record = make(admission="admitted", telegraph_ms=120.0, tracking_error_rad=0.05)
    assert record.admission == "admitted"


def test_save_and_load_round_trip(tmp_path):
    record = make()
    path = tmp_path / "c.json"
    cr.save(record, path)
    assert cr.load(path) == record


def test_load_rejects_an_unknown_schema_version(tmp_path):
    import json

    data = make().to_dict()
    data["schema_version"] = "9.9"
    path = tmp_path / "c.json"
    path.write_text(json.dumps(data))
    with pytest.raises(cr.CombinationError, match="schema_version"):
        cr.load(path)
