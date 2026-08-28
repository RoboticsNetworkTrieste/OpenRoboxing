"""Rehearsal: driving the generator from an authored pose.

The fast tests use a stub generator, so the contract of :func:`rehearse` is checked without loading a
checkpoint. The slow test drives the real thing and is the one that proves an armed pose actually
reaches the model.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_rehearsal.py -v
    .venv_mb/bin/python -m pytest tests/test_rehearsal.py -v -m slow   # needs a GPU
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.runtime.conventions import G1
from openroboxing.runtime.generator import GeneratorIntent
from openroboxing.spec.constants import GENERATOR_HZ, QPOS_DIM
from openroboxing.studio.pose_record import PoseRecord
from openroboxing.studio.rehearsal import (
    REPLAN_DT,
    Rehearsal,
    RehearsalError,
    rehearse,
    rehearse_approach,
)


def _record(name: str = "dev-jab") -> PoseRecord:
    from openroboxing.runtime.obs import default_angles

    angles = dict(zip(G1.mujoco_joint_names, default_angles(G1, "mujoco")))
    angles["left_shoulder_pitch_joint"] = -0.6
    return PoseRecord(
        name=name, joint_angles=angles, horizon_tokens=6, library_version="dev"
    )


class _StubGenerator:
    """Enough of :class:`MotionBricksGenerator` to exercise the rehearsal loop."""

    clip_names = ("walk_boxing", "walk", "idle")

    def __init__(self) -> None:
        self.calls: list[GeneratorIntent] = []
        self.frame = 0
        self.reset_seeds: list[int] = []

    def reset(self, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)
        self.frame = 0

    def next_frame(self) -> np.ndarray:
        qpos = np.zeros(QPOS_DIM)
        qpos[2] = 0.793
        qpos[3] = 1.0
        qpos[7] = 0.001 * self.frame  # something that changes, so streams are distinguishable
        self.frame += 1
        return qpos

    def generate(self, intent, context_qpos, dt) -> None:
        self.calls.append(intent)

    def context_qpos(self) -> np.ndarray:
        return np.zeros((4, QPOS_DIM))


# --- the contract ----------------------------------------------------------------------------------
def test_produces_the_requested_length_at_the_generator_rate() -> None:
    stub = _StubGenerator()
    result = rehearse(_record(), seconds=2.0, generator=stub)

    assert isinstance(result, Rehearsal)
    assert result.qpos.shape == (int(2.0 * GENERATOR_HZ), QPOS_DIM)
    assert result.rate_hz == GENERATOR_HZ
    assert result.seconds == pytest.approx(2.0)


def test_the_armed_pose_reaches_the_generator() -> None:
    stub = _StubGenerator()
    record = _record()
    rehearse(record, seconds=0.5, generator=stub)

    assert stub.calls, "the generator was never asked to plan"
    assert all(call.pose is record for call in stub.calls), "the pose was not armed on every replan"


def test_no_pose_means_upstream_behaviour() -> None:
    stub = _StubGenerator()
    result = rehearse(None, seconds=0.5, generator=stub)

    assert result.pose_name is None
    assert all(call.pose is None for call in stub.calls)


def test_the_style_argument_wins_over_the_intent() -> None:
    """A caller must not be able to arm a different style by passing a stale intent."""
    stub = _StubGenerator()
    rehearse(
        _record(),
        style="walk",
        seconds=0.5,
        generator=stub,
        intent=GeneratorIntent(style="idle", facing_angle=0.75),
    )
    assert all(call.style == "walk" for call in stub.calls)
    assert all(call.facing_angle == 0.75 for call in stub.calls), "the rest of the intent was lost"


def test_the_pose_argument_wins_over_the_intent() -> None:
    stub = _StubGenerator()
    wanted = _record("wanted")
    rehearse(
        wanted,
        seconds=0.5,
        generator=stub,
        intent=GeneratorIntent(pose=_record("stale")),
    )
    assert all(call.pose is wanted for call in stub.calls)


def test_the_seed_is_applied_before_generating() -> None:
    stub = _StubGenerator()
    rehearse(_record(), seconds=0.5, seed=4321, generator=stub)
    assert stub.reset_seeds == [4321]


def test_is_reproducible_from_its_recorded_fields() -> None:
    a = rehearse(_record(), seconds=0.5, seed=7, generator=_StubGenerator())
    b = rehearse(_record(), seconds=0.5, seed=7, generator=_StubGenerator())
    assert np.array_equal(a.qpos, b.qpos)
    assert (a.style, a.seed, a.pose_name) == (b.style, b.seed, b.pose_name)


# --- failing loudly ----------------------------------------------------------------------------------
def test_unknown_style_raises_and_lists_what_exists() -> None:
    with pytest.raises(RehearsalError, match="unknown style"):
        rehearse(_record(), style="uppercut", generator=_StubGenerator())


def test_non_positive_seconds_raises() -> None:
    with pytest.raises(RehearsalError, match="seconds must be positive"):
        rehearse(_record(), seconds=0.0, generator=_StubGenerator())


def test_a_non_finite_frame_raises() -> None:
    class _Broken(_StubGenerator):
        def next_frame(self) -> np.ndarray:
            qpos = super().next_frame()
            qpos[9] = np.nan
            return qpos

    with pytest.raises(RehearsalError, match="non-finite"):
        rehearse(_record(), seconds=0.5, generator=_Broken())


# --- the real thing ------------------------------------------------------------------------------------
@pytest.mark.slow
def test_arming_a_pose_changes_what_the_generator_produces() -> None:
    """An armed pose measurably steers a *replanning* run.

    Only that the motion moves — a replanning run never consumes a plan's tail, so it is the wrong
    place to ask whether the pose is reached. That is
    :func:`test_an_authored_boxing_pose_is_actually_reached`.
    """
    from openroboxing.runtime.generator import GeneratorConfig, MotionBricksGenerator

    generator = MotionBricksGenerator(GeneratorConfig(random_seed=1234))
    common = dict(style="walk_boxing", seconds=2.0, seed=1234, generator=generator)

    with_pose = rehearse(_record(), **common).qpos
    without_pose = rehearse(None, **common).qpos

    difference = np.abs(with_pose[:, 7:] - without_pose[:, 7:]).max()
    assert difference > 0.05, (
        f"arming a pose changed generation by only {difference:.4f} rad; the override is inert"
    )


# --- the correction: a commit is one plan, read to its end -----------------------------------
def test_a_commit_forces_one_plan_and_reads_it_whole() -> None:
    """The bug this guards: replanning discards each plan's tail, which is where the target lands."""
    from openroboxing.studio.rehearsal import rehearse_commit

    class _Planning(_StubGenerator):
        def __init__(self) -> None:
            super().__init__()
            self.forced: list[bool] = []

        def generate(self, intent, context_qpos, dt, *, force=False):
            self.calls.append(intent)
            self.forced.append(force)

        def plan(self) -> np.ndarray:
            qpos = np.zeros((40, QPOS_DIM))
            qpos[:, 2] = 0.793
            qpos[:, 3] = 1.0
            return qpos

    stub = _Planning()
    result = rehearse_commit(_record(), generator=stub, prime_frames=5)

    assert stub.forced == [True], "the plan must be forced, or the replan guard silently skips it"
    assert len(stub.calls) == 1, "a commit is exactly one plan"
    assert result.qpos.shape == (40, QPOS_DIM), "the whole plan must be returned, not a prefix"
    assert result.pose_name == _record().name


def test_a_commit_passes_the_poses_horizon() -> None:
    from openroboxing.studio.rehearsal import rehearse_commit

    class _Planning(_StubGenerator):
        def generate(self, intent, context_qpos, dt, *, force=False):
            self.calls.append(intent)

        def plan(self) -> np.ndarray:
            qpos = np.zeros((30, QPOS_DIM))
            qpos[:, 3] = 1.0
            return qpos

    stub = _Planning()
    rehearse_commit(_record(), generator=stub)
    assert stub.calls[0].horizon_tokens == _record().horizon_tokens


def test_reachability_is_measured_at_the_plan_endpoint() -> None:
    """Not at the closest frame of a replanning run — that is the measurement that was wrong."""
    from openroboxing.studio.rehearsal import measure_reachability

    record = _record()
    target = record.to_array()

    class _Arriving(_StubGenerator):
        def generate(self, intent, context_qpos, dt, *, force=False):
            self.calls.append(intent)

        def plan(self) -> np.ndarray:
            qpos = np.zeros((20, QPOS_DIM))
            qpos[:, 3] = 1.0
            qpos[:, 7:] = 0.0          # every frame far from the target...
            qpos[-1, 7:] = target      # ...except the last, which arrives
            return qpos

    reach = measure_reachability(record, generator=_Arriving())
    assert reach.max_error_rad == pytest.approx(0.0, abs=1e-9)
    assert reach.best_frame == 19, "the endpoint is frame N-1"
    assert reach.frames == 20


@pytest.mark.slow
def test_an_authored_boxing_pose_is_actually_reached() -> None:
    """The claim the pose library rests on, against the real generator."""
    from openroboxing.runtime.generator import GeneratorConfig, MotionBricksGenerator
    from openroboxing.runtime.obs import default_angles
    from openroboxing.studio.rehearsal import measure_reachability

    angles = dict(zip(G1.mujoco_joint_names, default_angles(G1, "mujoco")))
    angles["left_shoulder_pitch_joint"] = -1.2   # a straight left, well outside walk_boxing
    angles["left_elbow_joint"] = 0.15
    jab = PoseRecord(
        name="jab-left", joint_angles=angles, horizon_tokens=8, library_version="dev"
    )

    generator = MotionBricksGenerator(GeneratorConfig(random_seed=1234))
    reach = measure_reachability(jab, generator=generator)

    assert reach.max_error_deg < 20.0, (
        f"the generator missed the commanded pose by {reach.max_error_deg:.1f} deg "
        f"at {reach.worst_joint}"
    )


# --- a continuous commit: replanned, with the pose armed the whole way -------------------------
#: How far the approach stub creeps along +x per frame, in metres. Arbitrary, but small enough that
#: a 2 s run does not overshoot a 1 m goal.
_STUB_STEP_M = 0.01


class _Approaching(_StubGenerator):
    """A stub that records the cadence, and walks so the distance signal actually moves.

    Unlike :class:`_StubGenerator` its ``context_qpos`` tracks the frames it has emitted, as the real
    generator's does — otherwise priming cannot be observed to move the goal.
    """

    def __init__(self) -> None:
        super().__init__()
        self.forced: list[bool] = []
        self.generated_at: list[int] = []
        self.last = np.zeros(QPOS_DIM)

    def generate(self, intent, context_qpos, dt, *, force=False):
        self.calls.append(intent)
        self.forced.append(force)
        self.generated_at.append(self.frame)

    def next_frame(self) -> np.ndarray:
        qpos = super().next_frame()
        qpos[0] = _STUB_STEP_M * self.frame  # creeps along +x, so distance_to_goal falls
        self.last = qpos
        return qpos

    def context_qpos(self) -> np.ndarray:
        context = np.zeros((4, QPOS_DIM))
        context[-1] = self.last
        return context


def test_the_pose_is_armed_on_every_replan_of_an_approach() -> None:
    """The one behaviour the function exists for: unlike a commit, the pose never stops being armed.

    Guards the two-phase regression the design replaces — a plan armed once and then approached
    without a pose is what never converged.
    """
    stub = _Approaching()
    record = _record()
    seconds, prime = 2.0, 20
    result = rehearse_approach(
        record, travel_m=1.0, seconds=seconds, prime_frames=prime, generator=stub
    )

    cycle = int(round(REPLAN_DT * GENERATOR_HZ))
    frames = int(seconds * GENERATOR_HZ)
    assert len(stub.calls) == frames // cycle, "one plan per replan cycle, no more and no less"
    # `generated_at` counts from the stub's first frame, so the priming frames offset it.
    assert stub.generated_at == [prime + i for i in range(0, frames, cycle)], (
        "the replans are not evenly spaced one REPLAN_DT apart"
    )
    assert all(call.pose is record for call in stub.calls), "the pose was not armed on every replan"
    assert all(call.horizon_tokens is None for call in stub.calls), (
        "the model must choose the plan length; forcing the pose's horizon is what plateaus short"
    )
    assert all(stub.forced), "an unforced replan is skipped by the upstream cadence gate"
    assert result.qpos.shape == (frames, QPOS_DIM)


def test_an_approach_reports_a_per_frame_error_against_both_targets() -> None:
    stub = _Approaching()
    result = rehearse_approach(_record(), travel_m=1.0, seconds=1.0, generator=stub)

    frames = int(1.0 * GENERATOR_HZ)
    assert result.pose_error_rad.shape == (frames,)
    assert result.distance_to_goal.shape == (frames,)
    assert result.pose_name == _record().name
    assert result.distance_to_goal[-1] < result.distance_to_goal[0], "the stub walks toward the goal"


@pytest.mark.parametrize("prime_frames", [0, 20, 50])
def test_the_goal_is_relative_to_where_priming_left_the_root(prime_frames: int) -> None:
    """``travel_m`` is a distance to cover, not a coordinate — so priming must move the goal too.

    Were the goal absolute, a longer prime would start the run closer to it and the same
    ``travel_m`` would mean a different approach.
    """
    result = rehearse_approach(
        _record(),
        travel_m=1.0,
        seconds=1.0,
        prime_frames=prime_frames,
        generator=_Approaching(),
    )
    # One frame of the stub's creep separates the goal from the first measured frame, whatever the
    # prime: the goal was set from the context priming left behind.
    assert result.distance_to_goal[0] == pytest.approx(1.0 - _STUB_STEP_M, abs=1e-9)


def test_the_approach_carries_the_placement_it_was_asked_for() -> None:
    stub = _Approaching()
    rehearse_approach(_record(), travel_m=2.0, seconds=1.0, generator=stub)

    assert all(call.target_position is not None for call in stub.calls), "no placement was sent"
    assert all(call.target_heading is not None for call in stub.calls), (
        "target_position without target_heading makes upstream drop the placement silently"
    )


# --- failing loudly ----------------------------------------------------------------------------
@pytest.mark.parametrize("travel_m", [0.0, -1.0])
def test_non_positive_travel_raises(travel_m: float) -> None:
    with pytest.raises(RehearsalError, match="travel_m must be positive"):
        rehearse_approach(_record(), travel_m=travel_m, seconds=1.0, generator=_Approaching())


def test_non_positive_seconds_raises_for_an_approach() -> None:
    with pytest.raises(RehearsalError, match="seconds must be positive"):
        rehearse_approach(_record(), travel_m=1.0, seconds=0.0, generator=_Approaching())


def test_an_approach_with_an_unknown_style_raises_and_lists_what_exists() -> None:
    with pytest.raises(RehearsalError, match="unknown style"):
        rehearse_approach(
            _record(), travel_m=1.0, seconds=1.0, style="uppercut", generator=_Approaching()
        )


def test_a_non_finite_approach_frame_raises() -> None:
    class _Broken(_Approaching):
        def next_frame(self) -> np.ndarray:
            qpos = super().next_frame()
            qpos[9] = np.nan
            return qpos

    with pytest.raises(RehearsalError, match="non-finite"):
        rehearse_approach(_record(), travel_m=1.0, seconds=1.0, generator=_Broken())


@pytest.mark.slow
def test_an_armed_approach_converges_on_both_the_placement_and_the_pose() -> None:
    """The claim continuous pose targeting rests on (design doc, "Why it is better, measured").

    Reproduce: .venv_mb/bin/python -m pytest tests/test_rehearsal.py -m slow -k armed_approach -v
    """
    from openroboxing.paths import POSE_DIR
    from openroboxing.spec.constants import ARRIVAL_RADIUS_M
    from openroboxing.studio import pose_record
    from openroboxing.studio.rehearsal import REPLAN_DT, rehearse_approach

    pose = pose_record.load(POSE_DIR / "v0.1" / "hook-right.json")
    result = rehearse_approach(pose, travel_m=2.5, seconds=6.0)

    assert result.qpos.shape[1] == QPOS_DIM
    assert np.isfinite(result.qpos).all()

    # It gets there: inside the radius the runtime itself calls arrived (`fight.has_arrived`), so
    # this bench's pass criterion cannot drift from the rule it exists to justify.
    assert result.distance_to_goal[-1] < ARRIVAL_RADIUS_M, result.distance_to_goal[-1]
    assert result.distance_to_goal[-1] < result.distance_to_goal[0] / 4

    # And it converges on the pose rather than wandering near it. Read over the whole final replan
    # cycle: the error oscillates within each cycle, and because `seconds` is a multiple of
    # REPLAN_DT the last index systematically lands on the most favourable phase (5.99 deg against
    # 6.84 mean / 7.41 max over the cycle). Measuring at [-1] would flatter it.
    cycle = int(round(REPLAN_DT * GENERATOR_HZ))
    settled = result.pose_error_rad[-cycle:]
    assert settled.max() < np.radians(10.0), np.degrees(settled.max())
    assert settled.mean() < result.pose_error_rad[0] / 2, np.degrees(settled.mean())


@pytest.mark.slow
def test_the_stream_holds_a_pose_indefinitely_without_raising() -> None:
    """Under 1.1 a plan outliving its move by more than one frame raised. Holding is now normal.

    Reproduce: .venv_mb/bin/python -m pytest tests/test_rehearsal.py -m slow -k holds_a_pose -v
    """
    from openroboxing.paths import POSE_DIR
    from openroboxing.runtime.generator import (
        GeneratorConfig,
        GeneratorIntent,
        MotionBricksGenerator,
    )
    from openroboxing.runtime.reference import ReferenceStream
    from openroboxing.studio import pose_record

    pose = pose_record.load(POSE_DIR / "v0.1" / "guard.json")
    generator = MotionBricksGenerator(GeneratorConfig(random_seed=1234))
    stream = ReferenceStream(generator)

    held = GeneratorIntent(style="walk_boxing", pose=pose, horizon_tokens=None)
    stream.ensure(lambda _tick: held, tick=0)
    stream.ensure(lambda _tick: held, tick=500)   # 10 s of holding one pose

    assert stream.motion.shape[0] >= 500
