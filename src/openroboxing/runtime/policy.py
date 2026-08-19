"""GEAR-SONIC policy execution via ONNX Runtime.

Two networks run per fighter per control tick:

1. the **encoder** (tokenizer) turns a 1762-dim view of the reference motion into 64 tokens;
2. the **decoder** (policy) turns 994 dims — those tokens plus 10 frames of robot-state history —
   into 29 actions.

The reference runs both inline at 50 Hz: its ``GatherTokenState`` calls ``Encode()`` as a side effect
of assembling the first policy term (``g1_deploy_onnx_ref.cpp:1657-1666``). We do the same, so a
fighter costs two inferences per tick — the number M1-T7's budget has to carry.

Conventions
-----------
- **Rate**: 50 Hz, one :meth:`GearSonicPolicy.step` per control tick.
- **Actions** come out in **IsaacLab order**, raw, exactly as the reference stores them in
  ``last_action`` (``:3121``). Feed them straight back into ``obs.RobotState.last_action``.
- **Joint targets** are a separate step: :func:`action_to_joint_target` applies
  ``target = action * action_scale + default_angle`` and returns **MuJoCo order**, ready for the
  simulator's PD controller.
- **dtype**: the ONNX graphs take float32. Observations are assembled in float64 and narrowed at the
  boundary; the narrowing is the only precision loss in the chain.

Numerics
--------
Against the golden capture, the CPU execution provider reproduces the reference's tokens **exactly**
(the encoder's outputs are quantised to multiples of 1/8) and its actions to ~8e-06, well inside the
1e-3 M1-T4 tolerance. CPU is the default here because it is deterministic and this is a parity path;
`providers` is exposed for when M1-T7 needs throughput.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openroboxing.paths import POLICY_DECODER_ONNX, POLICY_ENCODER_ONNX
from openroboxing.runtime.conventions import G1, G1Conventions
from openroboxing.runtime.obs import default_angles
from openroboxing.spec.constants import (
    ACTION_DIM,
    DAMPING_RATIO,
    ENCODER_INPUT_DIM,
    MOTOR_ARMATURE,
    MOTOR_EFFORT_LIMIT,
    MOTOR_TYPE_BY_JOINT,
    NATURAL_FREQ,
    PD_GAIN_MULTIPLIER_BY_JOINT,
    POLICY_INPUT_DIM,
    TOKEN_DIM,
)


class PolicyError(RuntimeError):
    """The policy could not be loaded or run. Never recovered from silently."""


def _motor_property(joint: str, table: dict[str, float]) -> float:
    motor = MOTOR_TYPE_BY_JOINT.get(joint)
    if motor is None:
        raise PolicyError(f"no motor type recorded for joint {joint!r}")
    if motor not in table:
        raise PolicyError(f"motor type {motor!r} (joint {joint!r}) missing from table")
    return table[motor]


def stiffness(conventions: G1Conventions = G1, order: str = "mujoco") -> np.ndarray:
    """Per-joint PD stiffness, ``armature * omega^2``, ordered by joint name."""
    names = _names(conventions, order)
    return np.array(
        [_motor_property(n, MOTOR_ARMATURE) * NATURAL_FREQ**2 for n in names], dtype=np.float64
    )


def damping(conventions: G1Conventions = G1, order: str = "mujoco") -> np.ndarray:
    """Per-joint PD damping, ``2 * zeta * armature * omega``, ordered by joint name."""
    names = _names(conventions, order)
    return np.array(
        [2.0 * DAMPING_RATIO * _motor_property(n, MOTOR_ARMATURE) * NATURAL_FREQ for n in names],
        dtype=np.float64,
    )


def pd_kp(conventions: G1Conventions = G1, order: str = "mujoco") -> np.ndarray:
    """PD position gains, i.e. stiffness with the per-joint multiplier applied.

    Six joints (both ankles, waist roll and pitch) carry a 2x multiplier that ``g1_action_scale``
    does **not** — see ``PD_GAIN_MULTIPLIER_BY_JOINT``. Use this for control, never
    :func:`stiffness`, which is the unmultiplied value the action scale divides by.
    """
    names = _names(conventions, order)
    mult = np.array([PD_GAIN_MULTIPLIER_BY_JOINT.get(n, 1.0) for n in names], dtype=np.float64)
    return stiffness(conventions, order) * mult


def pd_kd(conventions: G1Conventions = G1, order: str = "mujoco") -> np.ndarray:
    """PD damping gains, with the same per-joint multiplier as :func:`pd_kp`."""
    names = _names(conventions, order)
    mult = np.array([PD_GAIN_MULTIPLIER_BY_JOINT.get(n, 1.0) for n in names], dtype=np.float64)
    return damping(conventions, order) * mult


def action_scale(conventions: G1Conventions = G1, order: str = "mujoco") -> np.ndarray:
    """Per-joint action scale, ``0.25 * effort_limit / stiffness``, ordered by joint name."""
    names = _names(conventions, order)
    stiff = stiffness(conventions, order)
    limits = np.array([_motor_property(n, MOTOR_EFFORT_LIMIT) for n in names], dtype=np.float64)
    return 0.25 * limits / stiff


def _names(conventions: G1Conventions, order: str) -> tuple[str, ...]:
    if order == "mujoco":
        return conventions.mujoco_joint_names
    if order == "isaaclab":
        return conventions.isaaclab_joint_names
    raise PolicyError(f"unknown joint order {order!r}")


def action_to_joint_target(action: np.ndarray, conventions: G1Conventions = G1) -> np.ndarray:
    """Convert a raw policy action into a joint position target.

    Implements ``target = action * action_scale + default_angle``
    (``policy_parameters.hpp:29``, applied at ``g1_deploy_onnx_ref.cpp:3120-3122``).

    Args:
        action: ``(29,)`` raw policy output, **IsaacLab order**.

    Returns:
        ``(29,)`` joint position targets in **MuJoCo order**, with default angles included — what a
        simulator's PD controller expects.
    """
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (ACTION_DIM,):
        raise PolicyError(f"action: expected ({ACTION_DIM},), got {a.shape}")
    return conventions.to_mujoco(a) * action_scale(conventions, "mujoco") + default_angles(
        conventions, "mujoco"
    )


def effort_limits(model, actuator_ids, joint_ids) -> np.ndarray:
    """Torque each actuator may apply, read from the compiled model rather than a table.

    Read from the model because the model is what MuJoCo enforces: it clamps to whichever limit is
    declared regardless of what a header says, so honouring it keeps applied and reported torque
    equal. The two shipped G1 files disagree about *where* to declare it and agree about the numbers:

    - ``g1_29dof_old.xml`` (behind ``scene_29dof.xml``) puts ``ctrlrange`` on each ``<motor>``;
    - ``g1_29dof.xml`` (what the arena composes) puts ``actuatorfrcrange`` on each ``<joint>`` and
      leaves the motors bare.

    Both compile to 88 / 139 / 50 / 25 / 5 N·m. Reading only one of them returns an array of **zeros**
    — which does not look like a failure, it looks like a robot with no muscles, and the fighter
    simply melts. Hence: every source is tried, and finding none raises (`CLAUDE.md` invariant 5).

    Args:
        model: a compiled ``mujoco.MjModel``.
        actuator_ids: actuator index per joint, in the order the caller drives them.
        joint_ids: the joint each of those actuators drives, same order.
    """
    actuator_ids = np.asarray(actuator_ids, dtype=int)
    joint_ids = np.asarray(joint_ids, dtype=int)
    if actuator_ids.shape != joint_ids.shape:
        raise PolicyError(
            f"actuator_ids {actuator_ids.shape} and joint_ids {joint_ids.shape} must match"
        )

    limits = np.zeros(actuator_ids.shape, dtype=np.float64)
    for source, limited, ranges, ids in (
        ("actuator forcerange", model.actuator_forcelimited, model.actuator_forcerange, actuator_ids),
        ("actuator ctrlrange", model.actuator_ctrllimited, model.actuator_ctrlrange, actuator_ids),
        ("joint actuatorfrcrange", model.jnt_actfrclimited, model.jnt_actfrcrange, joint_ids),
    ):
        unset = limits <= 0.0
        if not unset.any():
            break
        declared = np.asarray(limited)[ids].astype(bool) & unset
        limits[declared] = np.abs(np.asarray(ranges)[ids][declared]).max(axis=1)

    if (limits <= 0.0).any():
        missing = np.flatnonzero(limits <= 0.0).tolist()
        raise PolicyError(
            f"no torque limit declared for actuators {missing}; checked forcerange, ctrlrange and "
            "the joint's actuatorfrcrange. Clipping to zero would leave the robot unable to move "
            "and would look like a physics problem"
        )
    return limits


class GearSonicPolicy:
    """The shipped GEAR-SONIC encoder + decoder, run through ONNX Runtime."""

    def __init__(
        self,
        encoder_path: Path = POLICY_ENCODER_ONNX,
        decoder_path: Path = POLICY_DECODER_ONNX,
        providers: list[str] | None = None,
        warmup: int = 2,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment problem
            raise PolicyError(
                "onnxruntime is not installed; install it into the venv "
                "(VIRTUAL_ENV=.venv_mb uv pip install onnxruntime)"
            ) from exc

        for path in (encoder_path, decoder_path):
            if not path.exists():
                raise PolicyError(f"model not found: {path}")

        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        chosen = providers or ["CPUExecutionProvider"]

        self._encoder = ort.InferenceSession(str(encoder_path), options, providers=chosen)
        self._decoder = ort.InferenceSession(str(decoder_path), options, providers=chosen)

        self._enc_in = self._encoder.get_inputs()[0].name
        self._enc_out = self._encoder.get_outputs()[0].name
        self._dec_in = self._decoder.get_inputs()[0].name
        self._dec_out = self._decoder.get_outputs()[0].name

        self._check_dim(self._encoder, ENCODER_INPUT_DIM, "encoder")
        self._check_dim(self._decoder, POLICY_INPUT_DIM, "decoder")

        for _ in range(warmup):
            self.encode(np.zeros(ENCODER_INPUT_DIM))
            self.act(np.zeros(POLICY_INPUT_DIM))

    @staticmethod
    def _check_dim(session, expected: int, label: str) -> None:
        shape = session.get_inputs()[0].shape
        actual = shape[-1]
        if isinstance(actual, int) and actual != expected:
            raise PolicyError(
                f"{label} expects input width {actual}, spec says {expected}; "
                "the shipped weights and spec/constants.py disagree"
            )

    def encode(self, encoder_input: np.ndarray) -> np.ndarray:
        """Run the tokenizer. ``(1762,)`` in, ``(64,)`` out."""
        x = np.asarray(encoder_input, dtype=np.float32)
        if x.shape != (ENCODER_INPUT_DIM,):
            raise PolicyError(f"encoder_input: expected ({ENCODER_INPUT_DIM},), got {x.shape}")
        out = self._encoder.run([self._enc_out], {self._enc_in: x[None, :]})[0]
        return np.asarray(out, dtype=np.float64).reshape(TOKEN_DIM)

    def act(self, policy_input: np.ndarray) -> np.ndarray:
        """Run the policy. ``(994,)`` in, ``(29,)`` out in IsaacLab order, raw."""
        x = np.asarray(policy_input, dtype=np.float32)
        if x.shape != (POLICY_INPUT_DIM,):
            raise PolicyError(f"policy_input: expected ({POLICY_INPUT_DIM},), got {x.shape}")
        out = self._decoder.run([self._dec_out], {self._dec_in: x[None, :]})[0]
        return np.asarray(out, dtype=np.float64).reshape(ACTION_DIM)

    def step(self, encoder_input: np.ndarray, builder) -> tuple[np.ndarray, np.ndarray]:
        """One control tick: tokenize, assemble, act.

        Mirrors the reference's ordering, where the encoder runs as a side effect of gathering
        ``token_state``.

        Args:
            encoder_input: ``(1762,)`` tokenizer input for this tick.
            builder: a ready :class:`~openroboxing.runtime.obs.ObservationBuilder`.

        Returns:
            ``(action, token_state)``.
        """
        token_state = self.encode(encoder_input)
        action = self.act(builder.policy_input(token_state))
        return action, token_state
