"""Single-fighter runtime: MuJoCo physics driven by MotionBricks through GEAR-SONIC.

One G1, physics on, policy at 50 Hz, reference motion produced by the generator and converted by the
bridge. This is where the M1 pieces first run as a live loop rather than a replay.

Conventions
-----------
- **Control** at ``TICK_HZ`` (50 Hz); physics substeps at the model's own timestep, as many as fit in
  one control tick.
- **qpos** is ``[root pos (3), root quat wxyz (4), joints (29)]`` — MuJoCo order throughout.
- **qvel** is ``[linear (3, world frame), angular (3, body frame), joints (29)]``. MuJoCo reports a
  free joint's angular velocity in the body frame, which is the convention the IMU term expects.
- **Torques**: ``tau = kp * (q_target - q) - kd * dq``, matching the deploy's PD form
  (``base_sim.py::compute_body_torques`` with ``tau_ff = 0`` and ``dq_target = 0``), clamped to the
  motors' effort limits.
- **Generator lookahead**: the encoder needs the reference 45 ticks ahead, so the generator is always
  pulled far enough in front of the consumed tick. Falling behind raises rather than padding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openroboxing.paths import G1_29DOF_SCENE_XML
from openroboxing.runtime.bridge import compute_apply_delta_heading, encoder_input
from openroboxing.runtime.conventions import G1
from openroboxing.runtime.generator import GeneratorIntent, MotionBricksGenerator
from openroboxing.runtime.obs import ObservationBuilder, RobotState, default_angles
from openroboxing.runtime.policy import (
    GearSonicPolicy,
    action_to_joint_target,
    effort_limits,
    pd_kd,
    pd_kp,
)
from openroboxing.runtime.reference import (
    LOOKAHEAD_TICKS,
    REPLAN_DT,
    ReferenceStream,
)
from openroboxing.spec.constants import HISTORY_LEN, NUM_JOINTS, TICK_DT, TICK_HZ

__all__ = ["LOOKAHEAD_TICKS", "REPLAN_DT", "RunLog", "SingleFighterWorld", "WorldError"]


class WorldError(RuntimeError):
    """The world could not be built or stepped. Never recovered from silently."""


@dataclass
class RunLog:
    """Per-tick record of a run. This log *is* the run's result."""

    tick: list[int] = field(default_factory=list)
    root_height: list[float] = field(default_factory=list)
    joint_tracking_error: list[np.ndarray] = field(default_factory=list)
    root_position: list[np.ndarray] = field(default_factory=list)
    reference_root_position: list[np.ndarray] = field(default_factory=list)
    fell: bool = False
    fell_at_tick: int | None = None

    def summary(self) -> dict[str, float]:
        err = (
            np.array(self.joint_tracking_error)
            if self.joint_tracking_error
            else np.zeros((1, NUM_JOINTS))
        )
        return {
            "ticks": float(len(self.tick)),
            "seconds": len(self.tick) / TICK_HZ,
            "mean_joint_error_rad": float(np.abs(err).mean()),
            "max_joint_error_rad": float(np.abs(err).max()),
            "min_root_height_m": float(np.min(self.root_height)) if self.root_height else 0.0,
            "final_root_height_m": float(self.root_height[-1]) if self.root_height else 0.0,
            "fell": float(self.fell),
        }

    def per_joint_error(self) -> np.ndarray:
        """Mean absolute tracking error per joint, MuJoCo order."""
        if not self.joint_tracking_error:
            return np.zeros(NUM_JOINTS)
        return np.abs(np.array(self.joint_tracking_error)).mean(axis=0)


class SingleFighterWorld:
    """One G1 in MuJoCo, tracking a generated reference motion.

    Args:
        style: clip name driving the generator, e.g. ``"walk_boxing"``.
        seed: seeds both numpy and torch inside the generator.
        fall_height: root height below which the run is declared a fall.
    """

    def __init__(
        self,
        style: str = "walk_boxing",
        seed: int = 1234,
        fall_height: float = 0.4,
        scene_xml=G1_29DOF_SCENE_XML,
        policy: GearSonicPolicy | None = None,
        generator: MotionBricksGenerator | None = None,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.style = style
        self.fall_height = fall_height

        if not scene_xml.exists():
            raise WorldError(f"scene not found: {scene_xml}")
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)

        expected_qpos = 7 + NUM_JOINTS
        if self.model.nq != expected_qpos:
            raise WorldError(
                f"{scene_xml.name} has nq={self.model.nq}, expected {expected_qpos} "
                "(free joint + 29 hinges); is this the 29-DOF scene?"
            )

        self.substeps = max(1, int(round(TICK_DT / self.model.opt.timestep)))

        self.policy = policy or GearSonicPolicy()
        self.generator = generator or MotionBricksGenerator()
        self.builder = ObservationBuilder()

        self._kp = pd_kp(G1, "mujoco")
        self._kd = pd_kd(G1, "mujoco")
        self._defaults = default_angles(G1, "mujoco")
        self._actuator_for_joint = self._map_actuators_to_joints()
        self._joint_ids = self._joint_ids_in_mujoco_order()

        # Clamp to the MODEL's own limits, not the table in policy_parameters.hpp. They disagree:
        # the MJCF gives left_hip_pitch +-88 Nm (a 7520_14) where the header lists it as a 7520_22 at
        # 139 Nm, with a comment noting the motor was upgraded. Which field carries the limit differs
        # between the two shipped G1 files, so `effort_limits` reads whichever the model declares.
        self._effort_limit = effort_limits(self.model, self._actuator_for_joint, self._joint_ids)

        self.reference = ReferenceStream(self.generator)
        self._apply_delta_heading = None
        self._last_action = np.zeros(NUM_JOINTS)
        self._tick = 0

    def _map_actuators_to_joints(self) -> np.ndarray:
        """Actuator index for each MuJoCo joint, derived by name (`CLAUDE.md` invariant 4).

        Returns an array indexed by joint position holding the actuator that drives it. Raises if any
        joint is unactuated or any actuator drives an unknown joint.
        """
        mujoco = self._mujoco
        joint_to_actuator: dict[str, int] = {}
        for a in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[a, 0])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is None:
                raise WorldError(f"actuator {a} drives an unnamed joint")
            joint_to_actuator[name] = a

        missing = [n for n in G1.mujoco_joint_names if n not in joint_to_actuator]
        if missing:
            raise WorldError(f"joints with no actuator: {missing}")
        return np.array([joint_to_actuator[n] for n in G1.mujoco_joint_names], dtype=int)

    def _joint_ids_in_mujoco_order(self) -> np.ndarray:
        """Model joint id per joint, by name — the other half of reading a limit off the joint."""
        mujoco = self._mujoco
        ids = []
        for name in G1.mujoco_joint_names:
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise WorldError(f"joint {name!r} is not in the model")
            ids.append(joint)
        return np.array(ids, dtype=int)

    # -- setup ------------------------------------------------------------------------------------
    def reset(self, seed: int = 1234) -> None:
        """Reset physics to the default standing pose and reseed the generator."""
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, 0.793]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:] = self._defaults
        self.data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

        self.generator.reset(seed=seed)
        self.builder.reset()
        self.reference.reset()
        self._last_action = np.zeros(NUM_JOINTS)
        self._tick = 0

        ambient = GeneratorIntent(style=self.style)
        self.reference.ensure(lambda _tick: ambient, tick=0)
        self._apply_delta_heading = compute_apply_delta_heading(
            init_base_quat_wxyz=self.data.qpos[3:7].copy(),
            init_ref_root_quat_wxyz=self.reference.motion[0, 3:7],
        )

        # Prime the observation history so the first real tick has a full window.
        for _ in range(HISTORY_LEN):
            self.builder.push(self.robot_state())

    # -- state ------------------------------------------------------------------------------------
    def robot_state(self) -> RobotState:
        """Read the simulator into the conventions the observation builder expects."""
        return RobotState(
            joint_pos=self.data.qpos[7:].copy(),
            joint_vel=self.data.qvel[6:].copy(),
            base_quat=self.data.qpos[3:7].copy(),
            base_ang_vel=self.data.qvel[3:6].copy(),
            last_action=self._last_action,
        )

    # -- stepping ---------------------------------------------------------------------------------
    def step(self, intent_at) -> dict:
        """Advance one 50 Hz control tick. Returns the tick's record.

        Args:
            intent_at: ``tick -> GeneratorIntent``, asked per generated frame for the tick that
                frame will be played at. It is a callable rather than one intent because frames run
                ahead of the tick that consumes them — see ``runtime/reference.py``.
        """
        self.reference.ensure(intent_at, self._tick)
        self.reference.require(self._tick)

        self.builder.push(self.robot_state())

        enc = encoder_input(
            tick=self._tick,
            motion_50hz=self.reference.motion,
            base_quat_wxyz=self.data.qpos[3:7].copy(),
            motion_joint_vel_50hz=self.reference.velocities,
            apply_delta_heading=self._apply_delta_heading,
        )
        action, _ = self.policy.step(enc, self.builder)
        self._last_action = action

        target = action_to_joint_target(action)
        self._apply_torques_and_step(target)

        reference = self.reference.motion[self._tick]
        record = {
            "tick": self._tick,
            "root_height": float(self.data.qpos[2]),
            "joint_tracking_error": self.data.qpos[7:].copy() - reference[7:],
            "root_position": self.data.qpos[0:3].copy(),
            "reference_root_position": reference[0:3].copy(),
        }
        self._tick += 1
        return record

    def _apply_torques_and_step(self, joint_target: np.ndarray) -> None:
        """PD to the target, clamp to effort limits, then run the physics substeps."""
        for _ in range(self.substeps):
            q = self.data.qpos[7:]
            dq = self.data.qvel[6:]
            tau = self._kp * (joint_target - q) - self._kd * dq
            np.clip(tau, -self._effort_limit, self._effort_limit, out=tau)
            self.data.ctrl[self._actuator_for_joint] = tau
            self._mujoco.mj_step(self.model, self.data)

    def run(
        self,
        seconds: float,
        intent: GeneratorIntent | None = None,
        intent_for_tick=None,
    ) -> RunLog:
        """Run for ``seconds`` of simulated time, stopping early if the fighter falls.

        Args:
            intent: a fixed intent for the whole run.
            intent_for_tick: ``tick -> GeneratorIntent``, for a run driven by a commit queue. Takes
                precedence over ``intent``; this is how :class:`~openroboxing.runtime.intents.
                IntentTimeline` drives a fighter.
        """
        if intent_for_tick is None:
            fixed = intent or GeneratorIntent(style=self.style)

            def intent_for_tick(_tick: int) -> GeneratorIntent:
                return fixed

        log = RunLog()
        for _ in range(int(seconds * TICK_HZ)):
            record = self.step(intent_for_tick)
            log.tick.append(record["tick"])
            log.root_height.append(record["root_height"])
            log.joint_tracking_error.append(record["joint_tracking_error"])
            log.root_position.append(record["root_position"])
            log.reference_root_position.append(record["reference_root_position"])
            if record["root_height"] < self.fall_height:
                log.fell = True
                log.fell_at_tick = record["tick"]
                break
        return log
