"""The built library is a committed artefact; these tests are what keep it honest."""

from __future__ import annotations

import pytest

from openroboxing.paths import COMBINATION_DIR
from openroboxing.spec.constants import (
    COMBINATION_MAX_KEYFRAMES,
    COMBINATION_MIN_KEYFRAMES,
    NUM_FRAMES_PER_TOKEN,
)
from openroboxing.studio import combination_record as cr


@pytest.fixture(scope="module")
def library():
    paths = sorted(COMBINATION_DIR.glob("*.json"))
    assert paths, f"no combinations in {COMBINATION_DIR}; run tools.import_motions"
    return [cr.load(p) for p in paths]


def test_every_record_loads_and_validates(library):
    assert len(library) == 120


def test_names_are_unique(library):
    names = [r.name for r in library]
    assert len(names) == len(set(names))


def test_every_record_is_draft_and_unmeasured(library):
    for record in library:
        assert record.admission == "draft"
        assert record.telegraph_ms is None
        assert record.tracking_error_rad is None


def test_keyframe_counts_are_in_range(library):
    for record in library:
        assert COMBINATION_MIN_KEYFRAMES <= len(record.keyframes) <= COMBINATION_MAX_KEYFRAMES


def test_durations_match_their_recordings_within_one_token(library):
    for record in library:
        recorded = record.source.end_frame - record.source.start_frame
        planned = sum(k.leg_tokens or 0 for k in record.keyframes) * NUM_FRAMES_PER_TOKEN
        assert abs(planned - recorded) <= NUM_FRAMES_PER_TOKEN, record.name


def test_both_mirrors_are_present(library):
    mirrored = sum(1 for r in library if r.source.mirrored)
    assert mirrored == len(library) - mirrored
