"""Observation assembly for the GEAR-SONIC policy.

Reproduces, in Python, the observation vector the C++ reference builds. This is the riskiest module
in the project: a wrong term does not crash, it produces confident nonsense. Correctness is defined
by `tests/test_obs_parity.py` replaying the golden capture, not by this file looking reasonable.

What is actually active
-----------------------
The shipped `policy/release/observation_config.yaml` enables **six** policy terms and, in `g1`
encoder mode, only **three** informative encoder terms. The registry in the C++ offers ~70; the rest
are inert. Confirmed against the golden capture — see `spec/upstream_notes.md` §Q2. In particular the
720-dim SMPL block and every VR-3point term are identically zero and are deliberately not
implemented here.

Conventions
-----------
- **Rate**: one observation per 50 Hz control tick.
- **Input joint order**: :class:`RobotState` takes ``joint_pos`` / ``joint_vel`` in **MuJoCo
  (hardware/motor) order with the default standing angles included** — i.e. exactly what the robot
  and the reference's ``q.csv`` / ``dq.csv`` report.
- **Internal joint order**: everything stored in the history is **IsaacLab order**, and joint
  positions have the default angles **subtracted** (``g1_deploy_onnx_ref.cpp:2847``).
- **``last_action``** is the raw policy output, already in IsaacLab order, stored unmodified
  (``:3121``). Despite its header comment, the reference's ``action.csv`` is written with no
  transform at all (``state_logger.cpp:302``), so it is directly comparable.
- **Quaternions** are ``wxyz`` (MuJoCo / Unitree IMU convention).
- **History layout**: each term is laid out frame-major, ``[frame, value]``, **oldest first** —
  index 0 is the oldest frame and index ``HISTORY_LEN-1`` is the current tick. Measured from the
  capture with an exact zero residual, and matching the C++ ``newest_first = false`` default.

Failure behaviour
-----------------
`CLAUDE.md` invariant 5: no silent fallbacks. Building an observation before the history has been
filled raises rather than zero-padding, because a zero-padded observation is a plausible-looking
vector that silently misdrives the robot. The reference has a mode-fallback loop
(``g1_deploy_onnx_ref.cpp:2036``); it is deliberately not copied.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from openroboxing.runtime.conventions import G1, G1Conventions
from openroboxing.spec.constants import (
    DEFAULT_ANGLES_BY_JOINT,
    ENCODER_INPUT_DIM,
    HISTORY_LEN,
    NUM_JOINTS,
    POLICY_INPUT_DIM,
    TOKEN_DIM,
)

# Policy (decoder) input, in the order policy/release/observation_config.yaml lists them.
POLICY_TERMS: tuple[tuple[str, int], ...] = (
    ("token_state", TOKEN_DIM),
    ("his_base_angular_velocity_10frame_step1", 3 * HISTORY_LEN),
    ("his_body_joint_positions_10frame_step1", NUM_JOINTS * HISTORY_LEN),
    ("his_body_joint_velocities_10frame_step1", NUM_JOINTS * HISTORY_LEN),
    ("his_last_actions_10frame_step1", NUM_JOINTS * HISTORY_LEN),
    ("his_gravity_dir_10frame_step1", 3 * HISTORY_LEN),
)

# Encoder (tokenizer) input, in order. `required` marks the terms the g1 mode computes; the rest are
# left at zero by the per-mode fill (g1_deploy_onnx_ref.cpp:2041-2089).
ENCODER_TERMS: tuple[tuple[str, int, bool], ...] = (
    ("encoder_mode_4", 4, True),
    ("motion_joint_positions_10frame_step5", 290, True),
    ("motion_joint_velocities_10frame_step5", 290, True),
    ("motion_root_z_position_10frame_step5", 10, False),
    ("motion_root_z_position", 1, False),
    ("motion_anchor_orientation", 6, False),
    ("motion_anchor_orientation_10frame_step5", 60, True),
    ("motion_joint_positions_lowerbody_10frame_step5", 120, False),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120, False),
    ("vr_3point_local_target", 9, False),
    ("vr_3point_local_orn_target", 12, False),
    ("smpl_joints_10frame_step1", 720, False),
    ("smpl_anchor_orientation_10frame_step1", 60, False),
    ("motion_joint_positions_wrists_10frame_step1", 60, False),
)

#: Gravity in the world frame. Rotated into the body frame to form `his_gravity_dir`.
_WORLD_GRAVITY_DIR = np.array([0.0, 0.0, -1.0])


class ObservationError(RuntimeError):
    """An observation could not be built. Never recovered from silently."""


def term_offsets(terms) -> dict[str, tuple[int, int]]:
    """Map term name -> (start, stop) column indices, in declaration order."""
    out: dict[str, tuple[int, int]] = {}
    offset = 0
    for term in terms:
        name, dim = term[0], term[1]
        out[name] = (offset, offset + dim)
        offset += dim
    return out


POLICY_OFFSETS = term_offsets(POLICY_TERMS)
ENCODER_OFFSETS = term_offsets(ENCODER_TERMS)


def default_angles(conventions: G1Conventions = G1, order: str = "mujoco") -> np.ndarray:
    """Default standing angles as an array, ordered by joint *name*.

    Args:
        order: ``"mujoco"`` or ``"isaaclab"``.

    Raises:
        ObservationError: if the model's joint names and the constant's keys disagree.
    """
    if order == "mujoco":
        names = conventions.mujoco_joint_names
    elif order == "isaaclab":
        names = conventions.isaaclab_joint_names
    else:
        raise ObservationError(f"unknown joint order {order!r}")

    missing = [n for n in names if n not in DEFAULT_ANGLES_BY_JOINT]
    if missing:
        raise ObservationError(f"no default angle recorded for joints: {missing}")
    return np.array([DEFAULT_ANGLES_BY_JOINT[n] for n in names], dtype=np.float64)


def projected_gravity(base_quat_wxyz: np.ndarray) -> np.ndarray:
    """Gravity direction expressed in the base frame.

    Mirrors ``GatherHisGravityDir`` (``g1_deploy_onnx_ref.cpp:1624-1625``): rotate the world-frame
    down-vector by the *conjugate* of the base orientation.

    Args:
        base_quat_wxyz: base orientation, ``wxyz``, shape ``(..., 4)``.

    Returns:
        Unit-ish vector of shape ``(..., 3)``.
    """
    q = np.asarray(base_quat_wxyz, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ObservationError(f"expected a wxyz quaternion, got shape {q.shape}")
    # conjugate: negate the vector part
    w = q[..., 0]
    vec = -q[..., 1:]
    v = np.broadcast_to(_WORLD_GRAVITY_DIR, vec.shape)
    # quat_rotate_d: v*(2w^2-1) + 2w*cross(qvec, v) + 2*qvec*dot(qvec, v)
    a = v * (2.0 * w[..., None] ** 2 - 1.0)
    b = np.cross(vec, v) * (2.0 * w[..., None])
    c = vec * (2.0 * np.sum(vec * v, axis=-1)[..., None])
    return a + b + c


@dataclass(frozen=True)
class RobotState:
    """One tick of measured robot state, in the conventions the robot itself reports.

    Attributes:
        joint_pos: ``(29,)`` MuJoCo order, **default angles included**.
        joint_vel: ``(29,)`` MuJoCo order.
        base_quat: ``(4,)`` base IMU orientation, ``wxyz``.
        base_ang_vel: ``(3,)`` base IMU angular velocity.
        last_action: ``(29,)`` previous raw policy output, **IsaacLab order**.
    """

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    base_quat: np.ndarray
    base_ang_vel: np.ndarray
    last_action: np.ndarray

    def __post_init__(self) -> None:
        for name, expected in (
            ("joint_pos", NUM_JOINTS),
            ("joint_vel", NUM_JOINTS),
            ("base_quat", 4),
            ("base_ang_vel", 3),
            ("last_action", NUM_JOINTS),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (expected,):
                raise ObservationError(
                    f"RobotState.{name}: expected shape ({expected},), got {value.shape}"
                )
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class _HistoryFrame:
    """One tick, converted into the conventions the observation history stores."""

    body_q: np.ndarray  # IsaacLab order, defaults subtracted
    body_dq: np.ndarray  # IsaacLab order
    last_action: np.ndarray  # IsaacLab order, raw policy output
    base_ang_vel: np.ndarray
    gravity_dir: np.ndarray


class ObservationBuilder:
    """Rolling history of robot state, assembled into the 994-dim policy input.

    Push one :class:`RobotState` per 50 Hz tick, then call :meth:`policy_input`. The builder holds
    ``HISTORY_LEN`` frames; until that many have been pushed, :meth:`policy_input` raises.
    """

    def __init__(self, conventions: G1Conventions = G1) -> None:
        self._conv = conventions
        self._defaults_mujoco = default_angles(conventions, "mujoco")
        self._frames: deque[_HistoryFrame] = deque(maxlen=HISTORY_LEN)

    @property
    def ready(self) -> bool:
        """True once a full history window has been pushed."""
        return len(self._frames) == HISTORY_LEN

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()

    def push(self, state: RobotState) -> None:
        """Convert one tick of measured state into history conventions and append it."""
        self._frames.append(
            _HistoryFrame(
                # reorder to IsaacLab and subtract the defaults (cpp:2847)
                body_q=self._conv.to_isaaclab(state.joint_pos - self._defaults_mujoco),
                body_dq=self._conv.to_isaaclab(state.joint_vel),
                last_action=state.last_action,  # already IsaacLab order (cpp:3121)
                base_ang_vel=state.base_ang_vel,
                gravity_dir=projected_gravity(state.base_quat),
            )
        )

    # -- individual terms -------------------------------------------------------------------------
    def _stack(self, attr: str) -> np.ndarray:
        """Frame-major, oldest-first stack of one history field, flattened."""
        return np.concatenate([getattr(f, attr) for f in self._frames])

    def his_body_joint_positions(self) -> np.ndarray:
        """290 dims: 10 frames x 29 joints, IsaacLab order, defaults subtracted."""
        return self._stack("body_q")

    def his_body_joint_velocities(self) -> np.ndarray:
        """290 dims: 10 frames x 29 joints, IsaacLab order."""
        return self._stack("body_dq")

    def his_last_actions(self) -> np.ndarray:
        """290 dims: 10 frames x 29 previous raw policy outputs, IsaacLab order."""
        return self._stack("last_action")

    def his_base_angular_velocity(self) -> np.ndarray:
        """30 dims: 10 frames x 3, base IMU angular velocity."""
        return self._stack("base_ang_vel")

    def his_gravity_dir(self) -> np.ndarray:
        """30 dims: 10 frames x 3, gravity projected into the base frame."""
        return self._stack("gravity_dir")

    # -- assembly ---------------------------------------------------------------------------------
    def policy_input(self, token_state: np.ndarray) -> np.ndarray:
        """Assemble the 994-dim decoder input.

        Args:
            token_state: ``(64,)`` encoder output for this tick. Produced by the tokenizer
                (``runtime/policy.py``), not by this module — in the reference the encoder runs
                inline as a side effect of gathering this very term.

        Raises:
            ObservationError: if the history is not yet full, or a term has the wrong width.
        """
        if not self.ready:
            raise ObservationError(
                f"history holds {len(self._frames)} of {HISTORY_LEN} frames; "
                "push more ticks before building an observation"
            )
        token = np.asarray(token_state, dtype=np.float64)
        if token.shape != (TOKEN_DIM,):
            raise ObservationError(f"token_state: expected ({TOKEN_DIM},), got {token.shape}")

        parts = {
            "token_state": token,
            "his_base_angular_velocity_10frame_step1": self.his_base_angular_velocity(),
            "his_body_joint_positions_10frame_step1": self.his_body_joint_positions(),
            "his_body_joint_velocities_10frame_step1": self.his_body_joint_velocities(),
            "his_last_actions_10frame_step1": self.his_last_actions(),
            "his_gravity_dir_10frame_step1": self.his_gravity_dir(),
        }

        out = np.empty(POLICY_INPUT_DIM, dtype=np.float64)
        for name, dim in POLICY_TERMS:
            lo, hi = POLICY_OFFSETS[name]
            value = parts[name]
            if value.shape != (dim,):
                raise ObservationError(f"term {name}: expected ({dim},), got {value.shape}")
            out[lo:hi] = value
        return out


def split_policy_input(vector: np.ndarray) -> dict[str, np.ndarray]:
    """Split a 994-dim policy input into its named terms, for per-term error reporting."""
    v = np.asarray(vector)
    if v.shape[-1] != POLICY_INPUT_DIM:
        raise ObservationError(f"expected {POLICY_INPUT_DIM} columns, got shape {v.shape}")
    return {name: v[..., lo:hi] for name, (lo, hi) in POLICY_OFFSETS.items()}


def split_encoder_input(vector: np.ndarray) -> dict[str, np.ndarray]:
    """Split a 1762-dim encoder input into its named terms."""
    v = np.asarray(vector)
    if v.shape[-1] != ENCODER_INPUT_DIM:
        raise ObservationError(f"expected {ENCODER_INPUT_DIM} columns, got shape {v.shape}")
    return {name: v[..., lo:hi] for name, (lo, hi) in ENCODER_OFFSETS.items()}
