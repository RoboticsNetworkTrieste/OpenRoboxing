"""Harvesting candidate poses, and rendering them (M2-T5 infrastructure).

Harvest tests are analytic — a synthesised motion with known extremes — so they check the *selection*
rule rather than the generator. The render tests need a GPU and are marked slow.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_harvest.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.runtime.conventions import G1
from openroboxing.spec.constants import MAX_TOKENS, NUM_JOINTS, QPOS_DIM
from openroboxing.studio.harvest import (
    SALIENT_JOINT_SUBSTRINGS,
    Candidate,
    HarvestError,
    distinctiveness,
    harvest,
    salient_joints,
)


def _motion(frames: int = 200, spikes: tuple[int, ...] = (50, 120, 180)) -> np.ndarray:
    """A quiet motion with a few frames where an arm swings out."""
    qpos = np.zeros((frames, QPOS_DIM))
    qpos[:, 2] = 0.793
    qpos[:, 3] = 1.0
    elbow = 7 + G1.mujoco_joint_names.index("left_elbow_joint")
    for spike in spikes:
        qpos[spike, elbow] = 1.5
    return qpos


# --- selection -------------------------------------------------------------------------------------
def test_finds_the_distinctive_frames() -> None:
    spikes = (50, 120, 180)
    candidates = harvest(_motion(spikes=spikes), count=3, min_separation=5)

    assert len(candidates) == 3
    assert {c.frame for c in candidates} == set(spikes)


def test_candidates_come_back_in_descending_distinctiveness() -> None:
    qpos = _motion(spikes=(50, 120, 180))
    elbow = 7 + G1.mujoco_joint_names.index("left_elbow_joint")
    qpos[120, elbow] = 3.0  # make one clearly the biggest

    candidates = harvest(qpos, count=3, min_separation=5)
    assert candidates[0].frame == 120
    scores = [c.distinctiveness for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_separation_stops_every_candidate_being_the_same_pose() -> None:
    """Without it the top scores are consecutive frames of one punch."""
    qpos = _motion(frames=200, spikes=())
    elbow = 7 + G1.mujoco_joint_names.index("left_elbow_joint")
    qpos[100:110, elbow] = 1.5  # one long punch

    close = harvest(qpos, count=4, min_separation=1)
    apart = harvest(qpos, count=4, min_separation=30)

    assert max(c.frame for c in close) - min(c.frame for c in close) < 30
    frames = sorted(c.frame for c in apart)
    assert all(b - a >= 30 for a, b in zip(frames, frames[1:]))


def test_the_warmup_can_be_skipped() -> None:
    """The generator starts from a neutral pose; those frames otherwise sweep the ranking."""
    qpos = _motion(frames=200, spikes=(150,))
    elbow = 7 + G1.mujoco_joint_names.index("left_elbow_joint")
    qpos[0:20, elbow] = 3.0  # a big transient at the start

    assert harvest(qpos, count=1, min_separation=5)[0].frame < 20
    assert harvest(qpos, count=1, min_separation=5, skip_frames=30)[0].frame == 150


def test_fewer_candidates_than_asked_for_is_reported_not_padded() -> None:
    candidates = harvest(_motion(frames=40, spikes=(10, 25)), count=10, min_separation=15)
    assert 0 < len(candidates) < 10
    assert len({c.frame for c in candidates}) == len(candidates)


def test_scoring_looks_at_the_salient_joints_only() -> None:
    """A stride must not outrank a punch."""
    indices = salient_joints()
    names = [G1.mujoco_joint_names[i] for i in indices]
    assert names, "no salient joints resolved"
    assert all(any(part in name for part in SALIENT_JOINT_SUBSTRINGS) for name in names)
    assert not any("knee" in name for name in names)

    qpos = _motion(frames=100, spikes=())
    knee = 7 + G1.mujoco_joint_names.index("left_knee_joint")
    qpos[40, knee] = 2.0  # a big leg movement, no arm movement
    assert distinctiveness(qpos).max() == pytest.approx(0.0, abs=1e-9)


# --- failing loudly ----------------------------------------------------------------------------------
def test_bad_qpos_shape_raises() -> None:
    with pytest.raises(HarvestError, match="expected"):
        harvest(np.zeros((10, 20)))


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"count": 0}, "count must be at least 1"),
        ({"min_separation": 0}, "min_separation must be at least 1"),
        ({"skip_frames": -1}, "skip_frames must not be negative"),
        ({"skip_frames": 500}, "leaves nothing"),
    ],
)
def test_bad_arguments_raise(kwargs, expected) -> None:
    with pytest.raises(HarvestError, match=expected):
        harvest(_motion(), **kwargs)


# --- records ------------------------------------------------------------------------------------------
def test_a_candidate_becomes_a_valid_draft_record() -> None:
    from openroboxing.studio.pose_record import validate

    candidate = harvest(_motion(), count=1, min_separation=5)[0]
    record = candidate.to_record(name="guard-high")

    validate(record)
    assert record.admission == "draft", "harvesting proposes; it does not admit"
    assert record.telegraph_ms is None and record.generator_error_rad is None
    assert record.source.clip == "walk_boxing"
    assert record.source.start_frame == candidate.frame
    assert np.allclose(record.to_array(), candidate.angles)
    assert len(record.joint_angles) == NUM_JOINTS


def test_an_out_of_range_horizon_raises() -> None:
    candidate = harvest(_motion(), count=1, min_separation=5)[0]
    with pytest.raises(HarvestError, match="outside"):
        candidate.to_record(name="x", horizon_tokens=MAX_TOKENS + 1)


def test_candidates_are_frozen() -> None:
    candidate = harvest(_motion(), count=1, min_separation=5)[0]
    assert isinstance(candidate, Candidate)
    with pytest.raises(Exception):
        candidate.frame = 3  # type: ignore[misc]


# --- rendering -------------------------------------------------------------------------------------------
def test_contact_sheet_tiles_and_checks_its_inputs() -> None:
    from openroboxing.studio.render import RenderError, contact_sheet

    tiles = [np.full((20, 10, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
    sheet = contact_sheet(tiles, ["a", "b", "c"], columns=2)
    assert sheet.ndim == 3 and sheet.shape[2] == 3
    assert sheet.shape[0] > 20 and sheet.shape[1] > 20

    with pytest.raises(RenderError, match="3 images but 2 labels"):
        contact_sheet(tiles, ["a", "b"])
    with pytest.raises(RenderError, match="nothing to tile"):
        contact_sheet([], [])
    with pytest.raises(RenderError, match="tiles must match"):
        contact_sheet([tiles[0], np.zeros((5, 5, 3), np.uint8)], ["a", "b"])


def test_png_round_trips_through_a_reader(tmp_path) -> None:
    """Written by hand to avoid an image dependency, so it is worth checking it is really a PNG."""
    from openroboxing.studio.render import save_png

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(17, 23, 3), dtype=np.uint8)
    path = save_png(image, tmp_path / "x.png")

    import zlib

    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    start = raw.index(b"IDAT") + 4
    end = raw.index(b"IEND") - 8
    rows = zlib.decompress(raw[start:end])
    stride = 23 * 3 + 1
    decoded = np.array(
        [list(rows[r * stride + 1 : (r + 1) * stride]) for r in range(17)], dtype=np.uint8
    ).reshape(17, 23, 3)
    assert np.array_equal(decoded, image)


@pytest.mark.slow
def test_renders_a_pose() -> None:
    from openroboxing.runtime.obs import default_angles
    from openroboxing.studio.pose_record import PoseRecord
    from openroboxing.studio.render import render_pose

    record = PoseRecord(
        name="standing",
        joint_angles=dict(zip(G1.mujoco_joint_names, default_angles(G1, "mujoco"))),
        horizon_tokens=8,
        library_version="dev",
    )
    image = render_pose(record, width=160, height=200)
    assert image.shape == (200, 160, 3)
    assert image.std() > 5, "the render is a flat colour; nothing was drawn"
