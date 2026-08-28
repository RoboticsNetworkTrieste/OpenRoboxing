"""The sparring bench's DebugTap: state derivation, the viz transform, the recorder.

Ported for `spec/intent.md` 3.0 (B3): a commit is a combination, not a placement and a pose, so the
machine state loses the approach/dwell split (`APPROACH`/`DWELL` collapse into `RUNNING`) and the
fixtures below build `Commit`s from `CombinationRecord`s rather than `PoseRecord` + `Placement`.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_sparring_tap.py -v
"""

from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest

from openroboxing.runtime.conventions import G1
from openroboxing.runtime.intents import Commit
from openroboxing.server.sparring_tap import (
    HOLD,
    MACHINE_STATES,
    OPENING,
    RUNNING,
    WAITING,
    DebugTap,
    TapError,
    derive_machine_state,
    viz_ghost,
    viz_world_path,
    yaw_of_quat_wxyz,
)
from openroboxing.spec.constants import QPOS_DIM
from openroboxing.studio.combination_record import CombinationRecord, CombinationSource, Keyframe

ANGLES = {name: 0.0 for name in G1.mujoco_joint_names}


def _combination(name: str = "jab-cross", *, tokens=(6, 6)) -> CombinationRecord:
    """A small, admitted combination — cheap enough to build `Commit`s from directly.

    Mirrors the builder in `test_intents_combinations.py`, `test_fight.py`, `test_warp.py`: each
    test file in this repo keeps its own copy rather than sharing a fixtures module.
    """
    keyframes = [Keyframe(dict(ANGLES), None, (0.0, 0.0), 0.0)]
    for i, token in enumerate(tokens, start=1):
        keyframes.append(Keyframe(dict(ANGLES), token, (0.1 * i, 0.0), 0.0))
    return CombinationRecord(
        name=name,
        library_version="v0.2",
        source=CombinationSource("t", 0, 100, False),
        keyframes=keyframes,
        telegraph_ms=180.0,
        tracking_error_rad=0.1,
        admission="admitted",
    )


def _commit(issued_at: int, *, commit_at=None, ended_at=None) -> Commit:
    """A `Commit` with its span set directly — `spec/intent.md` 3.0's fields are plain, settable
    once a commit has started (`commit_at`, `ended_at`), unlike 2.2's watched-for `strike_at`."""
    commit = Commit(record=_combination(), ghost=(1.0, 0.0), issued_at=issued_at)
    commit.commit_at = commit_at
    commit.ended_at = ended_at
    return commit


class TestMachineState:
    def test_no_commits_is_the_opening_stance(self) -> None:
        assert derive_machine_state([], tick=0) == OPENING

    def test_issued_but_not_current_is_waiting(self) -> None:
        # Inside its horizon window: issued, not yet executing.
        assert derive_machine_state([_commit(10)], tick=15) == WAITING

    def test_executing_is_running(self) -> None:
        """3.0 has one running state, not two: a combination is one continuous piece of recorded
        motion, so there is no approach-then-dwell split left for `RUNNING` to distinguish."""
        commit = _commit(10, commit_at=40, ended_at=140)
        assert derive_machine_state([commit], tick=60) == RUNNING

    def test_a_finished_commit_with_nothing_behind_is_held(self) -> None:
        commit = _commit(10, commit_at=40, ended_at=140)
        assert derive_machine_state([commit], tick=140) == HOLD

    def test_a_commit_not_yet_finished_is_still_running_right_up_to_its_end(self) -> None:
        """`end_tick` is exact arithmetic at 3.0 — no `None`-means-still-running rule survives it."""
        commit = _commit(10, commit_at=40, ended_at=140)
        assert commit.end_tick == 140
        assert derive_machine_state([commit], tick=139) == RUNNING

    def test_a_queued_commit_behind_a_finished_one_is_waiting(self) -> None:
        done = _commit(10, commit_at=40, ended_at=140)
        queued = _commit(120)  # issued while `done` was still running; not started itself
        assert derive_machine_state([done, queued], tick=141) == WAITING

    def test_a_scrub_before_any_commit_started_is_opening_even_with_a_non_empty_log(self) -> None:
        """A commit issued *later* in the session must not make a scrub to an earlier tick read
        `HOLD` just because the commit log, as a whole, is non-empty — `Commit.end_tick` is `None`
        until a commit starts, and scrubbing to a tick before that must still say `OPENING`."""
        later = _commit(500, commit_at=520, ended_at=600)
        assert derive_machine_state([later], tick=0) == OPENING

    def test_the_names_match_the_indices(self) -> None:
        assert MACHINE_STATES[OPENING] == "OPENING"
        assert MACHINE_STATES[WAITING] == "WAITING"
        assert MACHINE_STATES[RUNNING] == "RUNNING"
        assert MACHINE_STATES[HOLD] == "HOLD"
        assert len(MACHINE_STATES) == 4, "APPROACH/DWELL collapsed into RUNNING at 3.0"


# -- the visualisation transform --------------------------------------------------------------------
def _ref_motion(rows: int = 10) -> np.ndarray:
    """A generator-frame motion whose root walks +x at 0.1 m/frame, identity orientation."""
    motion = np.zeros((rows, QPOS_DIM))
    motion[:, 0] = np.arange(rows) * 0.1
    motion[:, 2] = 0.79
    motion[:, 3] = 1.0  # identity wxyz
    return motion


class TestVizTransform:
    def test_yaw_of_identity_is_zero(self) -> None:
        assert yaw_of_quat_wxyz([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_yaw_of_a_quarter_turn(self) -> None:
        half = math.pi / 4
        assert yaw_of_quat_wxyz([math.cos(half), 0.0, 0.0, math.sin(half)]) == pytest.approx(
            math.pi / 2
        )

    def test_row_zero_lands_on_the_robot(self) -> None:
        path = viz_world_path(_ref_motion(), tick=3, robot_xy=(2.0, -1.0), apply_yaw=1.3)
        assert path[0] == pytest.approx([2.0, -1.0])

    def test_the_displacement_is_rotated_by_apply_yaw(self) -> None:
        # The reference advances +x; with a quarter-turn alignment the world sees +y.
        path = viz_world_path(_ref_motion(), tick=0, robot_xy=(0.0, 0.0), apply_yaw=math.pi / 2)
        assert path[5] == pytest.approx([0.0, 0.5], abs=1e-12)

    def test_coherence_with_to_generator_frame(self) -> None:
        """The viz transform inverts the rotation `fight.to_generator_frame` applies.

        The core maps world -> generator as ``gen = context + R(-yaw)(world - robot)``. Build a
        generator-frame point that way, put it in the motion, and the viz path must return the
        original world point.
        """
        yaw, robot, world_target = 0.7, np.array([0.4, -0.2]), np.array([2.0, 1.5])
        context = np.array([5.0, 3.0])  # wherever the generator believes it is
        c, s = math.cos(-yaw), math.sin(-yaw)
        gen_point = context + np.array(
            [
                c * (world_target - robot)[0] - s * (world_target - robot)[1],
                s * (world_target - robot)[0] + c * (world_target - robot)[1],
            ]
        )
        motion = _ref_motion(3)
        motion[0, 0:2] = context
        motion[2, 0:2] = gen_point
        path = viz_world_path(motion, tick=0, robot_xy=robot, apply_yaw=yaw)
        assert path[2] == pytest.approx(world_target, abs=1e-12)

    def test_the_ghost_clamps_at_the_motions_end(self) -> None:
        ghost = viz_ghost(
            _ref_motion(10),
            tick=5,
            lookahead=45,
            robot_xy=(0.0, 0.0),
            apply_yaw=0.0,
            joint_names=G1.mujoco_joint_names,
        )
        # Clamped to the last row: displacement 0.9 - 0.5 = 0.4 in x.
        assert ghost["x"] == pytest.approx(0.4)
        assert ghost["z"] == pytest.approx(0.79)
        assert len(ghost["angles"]) == 29
        assert set(ghost["angles"]) == set(G1.mujoco_joint_names)


# -- the recorder ------------------------------------------------------------------------------------
def _row(tick: int, nq: int = 79) -> dict:
    return {
        "qpos": np.full(nq, float(tick), dtype=np.float32),
        "ref_red": np.zeros(QPOS_DIM),
        "ref_blue": np.zeros(QPOS_DIM),
        "err_red": np.full(29, 0.01 * tick),
        "action_red": np.zeros(29),
        "root_h_red": 0.79,
        "root_h_blue": 0.78,
        "separation": 1.0,
        "dist_target": float("nan"),
        "dist_plan": float("nan"),
        "step_ms": 7.0,
        "machine": OPENING,
        "commit_ordinal": -1,
    }


class TestDebugTap:
    def test_appends_and_reads_back(self) -> None:
        tap = DebugTap()
        for tick in range(100):
            tap.append(tick, **_row(tick))
        assert tap.window() == (0, 99)
        row = tap.at(42)
        assert row["tick"] == 42
        assert row["qpos"][0] == pytest.approx(42.0)
        assert row["machine"] == OPENING

    def test_the_ring_drops_the_oldest(self) -> None:
        tap = DebugTap(max_ticks=50)
        for tick in range(100):
            tap.append(tick, **_row(tick))
        assert tap.window() == (50, 99)
        with pytest.raises(TapError, match="outside the recording window"):
            tap.at(10)

    def test_a_gap_in_the_ticks_is_refused(self) -> None:
        tap = DebugTap()
        tap.append(0, **_row(0))
        with pytest.raises(TapError, match="does not follow"):
            tap.append(2, **_row(2))

    def test_a_partial_row_is_refused(self) -> None:
        tap = DebugTap()
        row = _row(0)
        row.pop("step_ms")
        with pytest.raises(TapError, match="missing columns"):
            tap.append(0, **row)

    def test_series_downsamples_and_derives_the_error(self) -> None:
        tap = DebugTap()
        for tick in range(100):
            tap.append(tick, **_row(tick))
        tap.replans.append((30, False, 44))
        tap.replans.append((90, True, 40))
        tap.keyframe_events.append((25, 0, 1, 0.03, 0.09))
        tap.keyframe_events.append((95, 0, 2, 0.02, 0.05))

        series = tap.series(0, 99, stride=10)
        assert series["tick"] == list(range(0, 100, 10))
        assert len(series["err_mean"]) == 10
        assert series["err_mean"][5] == pytest.approx(0.5, rel=1e-5)  # |0.01 * 50|
        assert series["replans"] == [[30, False, 44], [90, True, 40]]
        assert series["keyframe_events"] == [[25, 0, 1, 0.03, 0.09], [95, 0, 2, 0.02, 0.05]]

    def test_series_carries_no_bare_nan(self) -> None:
        """A gap is ``null``, never ``NaN``.

        Python's ``json`` emits bare ``NaN`` happily and JavaScript's ``JSON.parse`` refuses it, so
        one un-executing tick used to poison the whole payload: the client's fetch threw, the
        charts never received a point, and the bench looked like it recorded nothing (2026-08-17 —
        970 of 1052 samples were NaN and the whole strip stayed blank).
        """
        tap = DebugTap()
        for tick in range(10):
            row = _row(tick)  # dist_target is NaN in every row: no commit is executing
            if tick == 3:
                row["step_ms"] = float("inf")
            tap.append(tick, **row)

        series = tap.series(0, 9)
        assert series["dist"] == [None] * 10
        assert series["step_ms"][3] is None
        json.dumps(series, allow_nan=False)  # raises on any non-finite that slipped through

    def test_events_survive_trimming(self) -> None:
        tap = DebugTap(max_ticks=50)
        for tick in range(100):
            tap.append(tick, **_row(tick))
        tap.replans.append((10, False, 44))  # before the window
        tap.replans.append((80, False, 44))
        tap.keyframe_events.append((10, 0, 0, 0.01, 0.02))  # before the window
        tap.keyframe_events.append((80, 0, 1, 0.03, 0.04))
        assert tap.series(50, 99)["replans"] == [[80, False, 44]]
        assert tap.series(50, 99)["keyframe_events"] == [[80, 0, 1, 0.03, 0.04]]

    def test_npz_round_trips(self) -> None:
        tap = DebugTap()
        for tick in range(20):
            tap.append(tick, **_row(tick))
        tap.replans.append((5, True, 40))
        tap.keyframe_events.append((7, 0, 1, 0.04, 0.11))

        archive = np.load(io.BytesIO(tap.to_npz_bytes()))
        assert archive["tick"].shape == (20,)
        assert archive["qpos"].shape == (20, 79)
        assert archive["err_red"].shape == (20, 29)
        assert archive["replans"].tolist() == [[5, 1, 40]]
        assert archive["keyframe_events"].tolist() == [[7.0, 0.0, 1.0, 0.04, 0.11]]
        assert int(archive["window_start"]) == 0

    def test_clear_forgets_keyframe_events_too(self) -> None:
        tap = DebugTap()
        tap.append(0, **_row(0))
        tap.keyframe_events.append((0, 0, 0, 0.0, 0.0))
        tap.clear()
        assert tap.keyframe_events == []
        assert len(tap) == 0
