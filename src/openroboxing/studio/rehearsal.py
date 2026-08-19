"""Rehearse a pose: drive the generator from an authored key pose and keep what it produces.

This is the studio's bench. A rehearsal is *kinematic only* — the generator's own output, no physics,
no policy — which is what the telegraph window is defined on (WORKPLAN M2-T3: "for a generated
motion"). Whether the robot can actually execute the motion is a separate measurement taken under
physics, and it is the other half of admission.

Conventions
-----------
- **Output** is ``(N, 36)`` MuJoCo ``qpos`` at :data:`~openroboxing.spec.constants.GENERATOR_HZ`,
  not the 50 Hz control rate. Pass ``rate_hz=GENERATOR_HZ`` to anything that measures it.
- A rehearsal is reproducible from ``(pose, style, seed, seconds)`` alone, so a recorded measurement
  can always be re-derived.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from openroboxing.runtime.generator import (
    GeneratorConfig,
    GeneratorIntent,
    MotionBricksGenerator,
)
from openroboxing.spec.constants import GENERATOR_HZ, QPOS_DIM

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openroboxing.studio.pose_record import PoseRecord

#: How often the generator is asked to replan, in seconds. Matches `SingleFighterWorld`, so a
#: rehearsal and a physics trial see the same replan cadence.
REPLAN_DT = 0.5


class RehearsalError(RuntimeError):
    """A rehearsal could not be produced. Never recovered from silently."""


@dataclass(frozen=True)
class Rehearsal:
    """One generated motion, and everything needed to reproduce it."""

    qpos: np.ndarray  # (N, 36) at GENERATOR_HZ
    style: str
    seed: int
    pose_name: str | None

    @property
    def rate_hz(self) -> float:
        return float(GENERATOR_HZ)

    @property
    def seconds(self) -> float:
        return self.qpos.shape[0] / GENERATOR_HZ


def rehearse(
    pose: PoseRecord | None,
    *,
    style: str = "walk_boxing",
    seconds: float = 2.0,
    seed: int = 1234,
    generator: MotionBricksGenerator | None = None,
    intent: GeneratorIntent | None = None,
) -> Rehearsal:
    """Generate the motion an authored pose produces.

    Args:
        pose: the key pose to reach, or ``None`` for the unmodified clip-sampled behaviour — which is
            what a guard baseline is.
        generator: reuse an existing generator. Building one loads a checkpoint onto the GPU, so
            measuring a library of poses should build one and pass it in.
        intent: the surrounding control signals. ``style`` and ``pose`` are always taken from this
            function's arguments, so callers cannot arm a different pose by accident.

    Returns:
        A :class:`Rehearsal` holding ``(N, 36)`` qpos at ``GENERATOR_HZ``.
    """
    if seconds <= 0:
        raise RehearsalError(f"seconds must be positive, got {seconds}")

    owned = generator is None
    generator = generator or MotionBricksGenerator(GeneratorConfig(random_seed=seed))
    if style not in generator.clip_names:
        raise RehearsalError(
            f"unknown style {style!r}; available: {', '.join(sorted(generator.clip_names))}"
        )

    base = intent or GeneratorIntent()
    armed = GeneratorIntent(
        style=style,
        movement_angle=base.movement_angle,
        facing_angle=base.facing_angle,
        target_position=base.target_position,
        target_heading=base.target_heading,
        pose=pose,
    )

    generator.reset(seed=seed)
    frames: list[np.ndarray] = []
    wanted = int(round(seconds * GENERATOR_HZ))
    guard = 0
    while len(frames) < wanted:
        if guard > 10 * wanted + 1000:
            raise RehearsalError(
                f"generator produced {len(frames)} of {wanted} frames before stalling"
            )
        guard += 1
        frames.append(generator.next_frame())
        generator.generate(armed, generator.context_qpos(), dt=REPLAN_DT)

    qpos = np.asarray(frames[:wanted], dtype=np.float64)
    if qpos.shape != (wanted, QPOS_DIM):
        raise RehearsalError(f"expected ({wanted}, {QPOS_DIM}) qpos, got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise RehearsalError("the generator produced a non-finite frame")

    if owned:
        del generator  # the caller did not ask us to hold a checkpoint open

    return Rehearsal(
        qpos=qpos, style=style, seed=seed, pose_name=pose.name if pose is not None else None
    )


def rehearse_commit(
    pose: PoseRecord,
    *,
    style: str = "walk_boxing",
    seed: int = 1234,
    prime_frames: int = 20,
    generator: MotionBricksGenerator | None = None,
    intent: GeneratorIntent | None = None,
) -> Rehearsal:
    """Generate **one** plan for a pose and return it whole. This is what a commit is.

    MotionBricks is an in-betweening model: a plan runs from the current context to the target pose,
    and the target is the plan's **last** frame. :func:`rehearse` replans every
    :data:`REPLAN_DT` seconds, which throws away each plan's tail — so a move whose plan is longer
    than the replan interval never arrives, and measuring it that way makes a perfectly reachable
    pose look unreachable. Measured: the same jab is 65.3° off at frame 15 and 7.8° off at frame 60.

    That is also the runtime rule it implies. A commit owns the timeline
    (``runtime/intents.py``); while it is executing the fighter must **not** replan, or the move is
    truncated before it lands.

    Args:
        prime_frames: frames consumed before planning, so the context is the fighter's recent motion
            rather than whatever the reset buffer holds.

    Returns:
        A :class:`Rehearsal` holding the plan, ``(N, 36)`` at ``GENERATOR_HZ``. Its last frame is the
        commanded pose.
    """
    owned = generator is None
    generator = generator or MotionBricksGenerator(GeneratorConfig(random_seed=seed))
    if style not in generator.clip_names:
        raise RehearsalError(
            f"unknown style {style!r}; available: {', '.join(sorted(generator.clip_names))}"
        )
    if prime_frames < 0:
        raise RehearsalError(f"prime_frames must not be negative, got {prime_frames}")

    base = intent or GeneratorIntent()
    armed = GeneratorIntent(
        style=style,
        movement_angle=base.movement_angle,
        facing_angle=base.facing_angle,
        target_position=base.target_position,
        target_heading=base.target_heading,
        pose=pose,
        horizon_tokens=pose.horizon_tokens,
    )

    generator.reset(seed=seed)
    for _ in range(prime_frames):
        generator.next_frame()
    generator.generate(armed, generator.context_qpos(), dt=REPLAN_DT, force=True)

    qpos = generator.plan()
    if qpos.shape[0] < 2:
        raise RehearsalError(f"the generator returned a {qpos.shape[0]}-frame plan")
    if not np.isfinite(qpos).all():
        raise RehearsalError("the generator produced a non-finite frame")

    if owned:
        del generator

    return Rehearsal(qpos=qpos, style=style, seed=seed, pose_name=pose.name)


@dataclass(frozen=True)
class ApproachRehearsal:
    """A replanned approach with the pose armed throughout — what a commit becomes at intent 2.0.

    Conventions: ``qpos`` is ``(N, 36)`` MuJoCo qpos at ``GENERATOR_HZ``; ``pose_error_rad`` and
    ``distance_to_goal`` are per-frame and the same length as ``qpos``. Both errors are measured in
    the **generator's own frame** — see :func:`rehearse_approach` on why that is not the world frame.

    Both signals oscillate within each replan cycle, because each new plan starts from the context
    and converges over its own length. Read them over a whole cycle (the final
    ``REPLAN_DT * GENERATOR_HZ`` frames), not at a single index: sampling at ``[-1]`` when
    ``seconds`` is a multiple of :data:`REPLAN_DT` systematically lands on the most favourable phase.
    Measured below: 5.99° at ``[-1]`` against 6.84° mean / 7.41° max over the final cycle.
    """

    qpos: np.ndarray
    pose_error_rad: np.ndarray  # (N,) mean absolute joint error against the commanded pose
    distance_to_goal: np.ndarray  # (N,) metres from the root to the commanded placement
    pose_name: str


def rehearse_approach(
    pose: PoseRecord,
    *,
    travel_m: float,
    seconds: float,
    style: str = "walk_boxing",
    seed: int = 1234,
    prime_frames: int = 20,
    generator: MotionBricksGenerator | None = None,
) -> ApproachRehearsal:
    """Walk toward a placement with the pose armed on every replan, and keep what it produces.

    This is the measurement bench for `spec/intent.md` 2.0: unlike :func:`rehearse_commit` the plan
    is never consumed whole, and unlike :func:`rehearse` the pose is armed the entire time. The
    length is left to the model (``horizon_tokens=None``).

    Measured (``hook-right``, ``travel_m=2.5``, ``seconds=6.0``, seed 1234): the root closes
    **2.500 m → 0.028 m** and the pose error falls **17.01° → 5.99°** mean (6.84° mean / 7.41° max
    over the final replan cycle). The same approach with the pose **not** armed closes the distance
    just as well — 2.500 m → 0.003 m — but its pose error *rises*, 17.01° → 18.53° (19.21° mean over
    the final cycle). Travel is the placement's doing; the convergence on the pose is the override's,
    and without it there is none.

    Args:
        travel_m: how far the placement sits from where priming left the root, along ``+x`` of the
            **generator's own frame** — which is *not* the MuJoCo world frame. The generator plans in
            a frame that differs from the world by a yaw (and an origin) once a fighter is under
            physics; ``runtime/fight.py`` converts between the two at exactly one boundary
            (``_placement_for``). ``goal``, the intent's ``target_position`` and the ``frame[:2]``
            this measures against are all on the generator's side of it, so this bench never needs
            the conversion and its metres are not comparable to a world-frame distance.
            Relative, so the rehearsal does not depend on where a reset happens to put the fighter.
        prime_frames: frames consumed before the first plan, so the context is the fighter's recent
            motion rather than whatever the reset buffer holds. As :func:`rehearse_commit`.

    Returns:
        An :class:`ApproachRehearsal` holding ``int(seconds * GENERATOR_HZ)`` frames, and the two
        per-frame errors the design's claim is stated in.
    """
    if travel_m <= 0.0:
        raise RehearsalError(f"travel_m must be positive, got {travel_m}")
    if seconds <= 0.0:
        raise RehearsalError(f"seconds must be positive, got {seconds}")
    if prime_frames < 0:
        raise RehearsalError(f"prime_frames must not be negative, got {prime_frames}")

    owned = generator is None
    generator = generator or MotionBricksGenerator(GeneratorConfig(random_seed=seed))
    if style not in generator.clip_names:
        raise RehearsalError(
            f"unknown style {style!r}; available: {', '.join(sorted(generator.clip_names))}"
        )

    generator.reset(seed=seed)
    for _ in range(prime_frames):
        generator.next_frame()

    goal = np.asarray(generator.context_qpos()[-1][:2], dtype=float) + np.array([travel_m, 0.0])
    target_angles = pose.to_array()
    replan_every = int(round(REPLAN_DT * GENERATOR_HZ))

    frames: list[np.ndarray] = []
    errors: list[float] = []
    distances: list[float] = []
    for index in range(int(seconds * GENERATOR_HZ)):
        if index % replan_every == 0:
            intent = GeneratorIntent(
                style=style,
                target_position=(float(goal[0]), float(goal[1])),
                target_heading=0.0,
                facing_angle=0.0,
                pose=pose,
                horizon_tokens=None,
            )
            generator.generate(intent, generator.context_qpos(), dt=REPLAN_DT, force=True)

        frame = generator.next_frame()
        frames.append(frame)
        errors.append(float(np.abs(frame[7:] - target_angles).mean()))
        distances.append(float(np.linalg.norm(goal - np.asarray(frame[:2], dtype=float))))

    qpos = np.asarray(frames, dtype=np.float64)
    if qpos.shape[1:] != (QPOS_DIM,):
        raise RehearsalError(f"expected (N, {QPOS_DIM}) qpos, got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise RehearsalError("the generator produced a non-finite frame")

    if owned:
        del generator  # the caller did not ask us to hold a checkpoint open

    return ApproachRehearsal(
        qpos=qpos,
        pose_error_rad=np.asarray(errors, dtype=np.float64),
        distance_to_goal=np.asarray(distances, dtype=np.float64),
        pose_name=pose.name,
    )


@dataclass(frozen=True)
class Reachability:
    """How close the generator gets to a commanded pose. The first admission gate.

    Distinct from the *policy* tracking error, which asks whether the robot can execute a motion the
    generator did produce. This asks the earlier question: will the generator produce it at all?
    """

    mean_error_rad: float
    max_error_rad: float
    worst_joint: str
    best_frame: int
    frames: int

    @property
    def mean_error_deg(self) -> float:
        return float(np.degrees(self.mean_error_rad))

    @property
    def max_error_deg(self) -> float:
        return float(np.degrees(self.max_error_rad))

    def passes(self, tolerance_rad: float) -> bool:
        return self.max_error_rad <= tolerance_rad


def measure_reachability(
    pose: PoseRecord,
    *,
    style: str = "walk_boxing",
    seed: int = 1234,
    generator: MotionBricksGenerator | None = None,
) -> Reachability:
    """Rehearse a pose as a commit and report how close the plan's endpoint came to it.

    Measured at the plan's **last frame**, because that is where an in-between puts its target. An
    earlier version of this took the closest frame of a *replanning* rehearsal, which never consumes
    a plan's tail and therefore reported a reachable pose as unreachable by an order of magnitude.
    See :func:`rehearse_commit`.
    """
    from openroboxing.runtime.conventions import G1

    result = rehearse_commit(pose, style=style, seed=seed, generator=generator)
    target = pose.to_array()
    endpoint = np.abs(result.qpos[-1, 7:] - target)

    return Reachability(
        mean_error_rad=float(endpoint.mean()),
        max_error_rad=float(endpoint.max()),
        worst_joint=G1.mujoco_joint_names[int(endpoint.argmax())],
        best_frame=int(result.qpos.shape[0] - 1),
        frames=int(result.qpos.shape[0]),
    )
