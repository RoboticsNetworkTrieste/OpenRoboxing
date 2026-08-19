"""The sparring bench's DebugTap: state derivation, the viz transform, the recorder.

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
from openroboxing.runtime.intents import Commit, Placement
from openroboxing.server.sparring_tap import (
    APPROACH,
    DWELL,
    HOLD,
    MACHINE_STATES,
    OPENING,
    WAITING,
    DebugTap,
    TapError,
    derive_machine_state,
    viz_ghost,
    viz_world_path,
    yaw_of_quat_wxyz,
)
from openroboxing.spec.constants import POSE_DWELL_TICKS, QPOS_DIM
from openroboxing.studio.pose_record import PoseRecord


def _pose(name: str = "jab-left") -> PoseRecord:
    from openroboxing.runtime.obs import default_angles

    return PoseRecord(
        name=name,
        joint_angles=dict(zip(G1.mujoco_joint_names, default_angles(G1, "mujoco"))),
        horizon_tokens=8,
        library_version="v0.1",
        admission="admitted",
        telegraph_ms=180.0,
        generator_error_rad=0.1,
    )


def _commit(
    issued_at: int, *, commit_at=None, strike_at=None, arrived=None, ended_at=None
) -> Commit:
    commit = Commit(
        pose=_pose(),
        context="walk_boxing",
        placement=Placement(position=(1.0, 0.0), heading=0.0),
        issued_at=issued_at,
        slot="1",
    )
    commit.commit_at = commit_at
    commit.strike_at = strike_at
    commit.arrived = arrived
    # Since `spec/intent.md` 2.2 a span is stamped as the move runs, so a fixture that wants a
    # *finished* commit has to say when it finished; there is no rule that derives it.
    commit.ended_at = ended_at
    return commit


class TestMachineState:
    def test_no_commits_is_the_opening_stance(self) -> None:
        assert derive_machine_state([], tick=0) == OPENING

    def test_issued_but_not_current_is_waiting(self) -> None:
        # Inside its horizon window: issued, not yet executing.
        assert derive_machine_state([_commit(10)], tick=15) == WAITING

    def test_executing_without_arrival_is_the_approach(self) -> None:
        commit = _commit(10, commit_at=40)
        assert derive_machine_state([commit], tick=60) == APPROACH

    def test_arrived_and_holding_the_strike_is_the_dwell(self) -> None:
        commit = _commit(10, commit_at=40, strike_at=100, arrived=True)
        assert derive_machine_state([commit], tick=110) == DWELL

    def test_a_finished_commit_with_nothing_behind_is_held(self) -> None:
        ended = 100 + POSE_DWELL_TICKS
        commit = _commit(10, commit_at=40, strike_at=100, arrived=True, ended_at=ended)
        assert commit.end_tick == ended
        assert derive_machine_state([commit], tick=ended) == HOLD

    def test_a_commit_that_has_struck_but_not_ended_is_still_the_dwell(self) -> None:
        """`end_tick` is None until the move is over, and None means *later than any tick*."""
        commit = _commit(10, commit_at=40, strike_at=100, arrived=True)
        assert commit.end_tick is None
        assert derive_machine_state([commit], tick=100 + 10 * POSE_DWELL_TICKS) == DWELL

    def test_a_queued_commit_behind_a_finished_one_is_waiting(self) -> None:
        done = _commit(10, commit_at=40, strike_at=100, arrived=True, ended_at=174)
        queued = _commit(150)
        assert derive_machine_state([done, queued], tick=done.end_tick + 1) == WAITING

    def test_the_names_match_the_indices(self) -> None:
        assert MACHINE_STATES[OPENING] == "OPENING"
        assert MACHINE_STATES[WAITING] == "WAITING"
        assert MACHINE_STATES[APPROACH] == "APPROACH"
        assert MACHINE_STATES[DWELL] == "DWELL"
        assert MACHINE_STATES[HOLD] == "HOLD"


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

        series = tap.series(0, 99, stride=10)
        assert series["tick"] == list(range(0, 100, 10))
        assert len(series["err_mean"]) == 10
        assert series["err_mean"][5] == pytest.approx(0.5, rel=1e-5)  # |0.01 * 50|
        assert series["replans"] == [[30, False, 44], [90, True, 40]]

    def test_series_carries_no_bare_nan(self) -> None:
        """A gap is ``null``, never ``NaN``.

        Python's ``json`` emits bare ``NaN`` happily and JavaScript's ``JSON.parse`` refuses it, so
        one un-approached tick used to poison the whole payload: the client's fetch threw, the
        charts never received a point, and the bench looked like it recorded nothing (2026-08-17 —
        970 of 1052 samples were NaN and the whole strip stayed blank).
        """
        tap = DebugTap()
        for tick in range(10):
            row = _row(tick)  # dist_target is NaN in every row: no placement is being approached
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
        assert tap.series(50, 99)["replans"] == [[80, False, 44]]

    def test_npz_round_trips(self) -> None:
        tap = DebugTap()
        for tick in range(20):
            tap.append(tick, **_row(tick))
        tap.replans.append((5, True, 40))

        archive = np.load(io.BytesIO(tap.to_npz_bytes()))
        assert archive["tick"].shape == (20,)
        assert archive["qpos"].shape == (20, 79)
        assert archive["err_red"].shape == (20, 29)
        assert archive["replans"].tolist() == [[5, 1, 40]]
        assert int(archive["window_start"]) == 0
