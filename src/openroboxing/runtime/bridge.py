"""Bridge: MotionBricks output → GEAR-SONIC reference inputs.

The generator emits 36-dim MuJoCo ``qpos`` at 30 Hz; the policy consumes reference motion at 50 Hz in
IsaacLab order. This module is the conversion, and it is a convention minefield — every function here
states what it takes and what it returns.

Conventions
-----------
- **Input**: ``(N, 36)`` qpos frames at ``GENERATOR_HZ`` — 3 root position, 4 root quaternion
  ``wxyz``, 29 joints in **MuJoCo order**.
- **Output**: the same layout at ``TICK_HZ``, plus velocities.
- **Resampling**: positions and joints interpolate **linearly**; the root quaternion uses **slerp**,
  because componentwise interpolation of a quaternion is not a rotation.
- **Velocities are finite-differenced *after* resampling**, never before (`CLAUDE.md`). Differencing
  first and then interpolating smears each sample across neighbours and understates peaks.
- **Encoder terms are lookahead, not history.** Measured from the golden capture: the reference
  motion advances exactly one frame per 50 Hz tick, and the encoder is fed frames
  ``t, t+5, ..., t+45`` — 0.9 s ahead. Beyond the end of the motion the reference clamps to the last
  frame (``g1_deploy_onnx_ref.cpp:652-654``); we do the same.

What the encoder actually needs
-------------------------------
In ``g1`` mode only three terms carry information (``spec/upstream_notes.md`` §Q2):
``motion_joint_positions_10frame_step5``, ``motion_joint_velocities_10frame_step5`` and
``motion_anchor_orientation_10frame_step5``. Everything else is structurally zero and is not built.
"""

from __future__ import annotations

import numpy as np

from openroboxing.runtime.conventions import G1, G1Conventions
from openroboxing.runtime.obs import ENCODER_OFFSETS
from openroboxing.spec.constants import (
    ENCODER_INPUT_DIM,
    GENERATOR_HZ,
    HISTORY_LEN,
    NUM_JOINTS,
    QPOS_DIM,
    TICK_DT,
    TICK_HZ,
)

#: Stride between the reference frames the encoder samples, in 50 Hz ticks.
ENCODER_FRAME_STRIDE = 5

_ROOT_POS = slice(0, 3)
_ROOT_QUAT = slice(3, 7)
_JOINTS = slice(7, 36)


class BridgeError(RuntimeError):
    """A conversion could not be performed. Never recovered from silently."""


def _check_qpos(frames: np.ndarray, what: str = "frames") -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != QPOS_DIM:
        raise BridgeError(f"{what}: expected (N, {QPOS_DIM}), got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise BridgeError(f"{what}: need at least 2 frames to resample, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise BridgeError(f"{what}: contains NaN or inf")
    return arr


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between two arrays of ``wxyz`` quaternions.

    Args:
        q0, q1: ``(N, 4)`` unit quaternions.
        t: ``(N,)`` interpolation parameter in ``[0, 1]``.

    Returns:
        ``(N, 4)`` interpolated quaternions.

    Antipodal inputs are handled by flipping ``q1`` — ``q`` and ``-q`` are the same rotation, and
    without the flip the interpolation takes the long way round.
    """
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64).copy()
    if a.shape != b.shape or a.shape[-1] != 4:
        raise BridgeError(
            f"slerp: expected matching (N, 4) quaternions, got {a.shape} and {b.shape}"
        )

    dot = np.sum(a * b, axis=-1)
    flip = dot < 0.0
    b[flip] = -b[flip]
    dot = np.abs(dot)

    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    out = np.empty_like(a)

    # Near-parallel: slerp is numerically unstable, and lerp+normalise is indistinguishable.
    close = dot > 1.0 - 1e-9
    if close.any():
        lerped = a[close] + t[close] * (b[close] - a[close])
        norms = np.linalg.norm(lerped, axis=-1, keepdims=True)
        out[close] = lerped / norms

    far = ~close
    if far.any():
        theta = np.arccos(np.clip(dot[far], -1.0, 1.0)).reshape(-1, 1)
        sin_theta = np.sin(theta)
        tf = t[far]
        out[far] = (np.sin((1.0 - tf) * theta) * a[far] + np.sin(tf * theta) * b[far]) / sin_theta
    return out


def resample_qpos(
    frames: np.ndarray,
    source_hz: float = GENERATOR_HZ,
    target_hz: float = TICK_HZ,
) -> np.ndarray:
    """Resample a qpos stream, linearly for positions and by slerp for the root quaternion.

    The output spans the same wall-clock duration as the input. Sample ``k`` of the output is at time
    ``k / target_hz``, and the last input frame is included, so a 30 Hz clip of ``N`` frames becomes
    ``floor((N-1) * target_hz / source_hz) + 1`` frames at 50 Hz.

    Args:
        frames: ``(N, 36)`` qpos at ``source_hz``.

    Returns:
        ``(M, 36)`` qpos at ``target_hz``.
    """
    src = _check_qpos(frames)
    n = src.shape[0]
    duration = (n - 1) / source_hz
    m = int(np.floor(duration * target_hz)) + 1

    # position of each output sample in *source frame index* units
    src_index = np.arange(m) * (source_hz / target_hz)
    lo = np.floor(src_index).astype(int)
    lo = np.clip(lo, 0, n - 2)
    hi = lo + 1
    frac = src_index - lo

    out = np.empty((m, QPOS_DIM), dtype=np.float64)
    weight = frac[:, None]
    out[:, _ROOT_POS] = src[lo][:, _ROOT_POS] * (1 - weight) + src[hi][:, _ROOT_POS] * weight
    out[:, _JOINTS] = src[lo][:, _JOINTS] * (1 - weight) + src[hi][:, _JOINTS] * weight
    out[:, _ROOT_QUAT] = slerp(src[lo][:, _ROOT_QUAT], src[hi][:, _ROOT_QUAT], frac)
    return out


def finite_difference_velocities(values: np.ndarray, dt: float = TICK_DT) -> np.ndarray:
    """Per-sample velocities of an already-resampled series.

    Uses a central difference in the interior and one-sided differences at the ends, so the result
    has the same length as the input and no sample is invented.

    Args:
        values: ``(N, K)`` positions, already at the target rate.
        dt: seconds between samples.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise BridgeError(f"expected a 2-D (N, K) array, got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise BridgeError("need at least 2 samples to difference")
    if dt <= 0:
        raise BridgeError(f"dt must be positive, got {dt}")

    out = np.empty_like(arr)
    out[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
    out[0] = (arr[1] - arr[0]) / dt
    out[-1] = (arr[-1] - arr[-2]) / dt
    return out


def joint_velocities(frames_50hz: np.ndarray, dt: float = TICK_DT) -> np.ndarray:
    """Joint velocities of a resampled qpos stream, in the stream's own (MuJoCo) order."""
    return finite_difference_velocities(_check_qpos(frames_50hz)[:, _JOINTS], dt)


def rotation_6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """The 6-D rotation representation the encoder uses.

    First two columns of the rotation matrix, flattened **row-wise**
    (``g1_deploy_onnx_ref.cpp:681-685``).

    Args:
        quat_wxyz: ``(..., 4)``.

    Returns:
        ``(..., 6)``.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    if q.shape[-1] != 4:
        raise BridgeError(f"expected a wxyz quaternion, got shape {q.shape}")
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # columns 0 and 1 of the rotation matrix
    c00 = 1 - 2 * (y * y + z * z)
    c10 = 2 * (x * y + w * z)
    c20 = 2 * (x * z - w * y)
    c01 = 2 * (x * y - w * z)
    c11 = 1 - 2 * (x * x + z * z)
    c21 = 2 * (y * z + w * x)
    return np.stack([c00, c01, c10, c11, c20, c21], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of a ``wxyz`` quaternion."""
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``wxyz`` quaternion arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def heading_quat(q: np.ndarray) -> np.ndarray:
    """The yaw-only part of a ``wxyz`` quaternion (``calc_heading_quat_d``)."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise BridgeError(f"expected a wxyz quaternion, got shape {q.shape}")
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw / 2.0
    return np.stack([np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=-1)


def compute_apply_delta_heading(
    init_base_quat_wxyz: np.ndarray,
    init_ref_root_quat_wxyz: np.ndarray,
    delta_heading: float = 0.0,
) -> np.ndarray:
    """Yaw alignment between the reference motion and the robot at the moment the motion started.

    Mirrors ``ComputeApplyDeltaHeading`` (``g1_deploy_onnx_ref.cpp:591-604``)::

        apply = heading(init_base) * conj(heading(init_ref_root))
        apply = yaw(delta_heading) * apply          # if the operator nudged the heading

    This is **runtime state, not motion data**: it depends on where the robot was facing when the
    clip began. It is constant for the life of a motion.
    """
    apply = quat_multiply(
        heading_quat(init_base_quat_wxyz), quat_conjugate(heading_quat(init_ref_root_quat_wxyz))
    )
    if delta_heading != 0.0:
        half = delta_heading / 2.0
        yaw_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        apply = quat_multiply(yaw_quat, apply)
    return apply


def anchor_orientation(
    base_quat_wxyz: np.ndarray,
    ref_root_quat_wxyz: np.ndarray,
    apply_delta_heading: np.ndarray = IDENTITY_QUAT,
) -> np.ndarray:
    """The reference root orientation relative to the robot base, as 6-D.

    Mode 0 of ``GatherMotionAnchorOrientationMutiFrame``
    (``g1_deploy_onnx_ref.cpp:662, 674-685``)::

        conj(base) * (apply_delta_heading * ref_root)   ->  6-D

    ``apply_delta_heading`` is easy to forget and its omission is not obviously wrong — it produces a
    plausible O(1) error. See :func:`compute_apply_delta_heading`.
    """
    aligned = quat_multiply(apply_delta_heading, ref_root_quat_wxyz)
    return rotation_6d(quat_multiply(quat_conjugate(base_quat_wxyz), aligned))


def lookahead_indices(tick: int, num_frames: int, stride: int, motion_length: int) -> np.ndarray:
    """Reference-frame indices the encoder samples at ``tick``, clamped at the motion's end.

    Mirrors ``g1_deploy_onnx_ref.cpp:648-655``: advance by ``frame_idx * stride`` and clamp to the
    final frame rather than wrapping or extrapolating.
    """
    idx = tick + np.arange(num_frames) * stride
    return np.clip(idx, 0, motion_length - 1)


def encoder_input(
    tick: int,
    motion_50hz: np.ndarray,
    base_quat_wxyz: np.ndarray,
    motion_joint_vel_50hz: np.ndarray | None = None,
    apply_delta_heading: np.ndarray = IDENTITY_QUAT,
    conventions: G1Conventions = G1,
) -> np.ndarray:
    """Assemble the 1762-dim tokenizer input for one tick, in ``g1`` mode.

    Only the three informative terms are written; everything else stays zero, exactly as the
    reference's per-mode fill leaves it.

    Args:
        tick: index into ``motion_50hz`` for the current control tick.
        motion_50hz: ``(N, 36)`` reference motion already resampled to 50 Hz, MuJoCo order.
        base_quat_wxyz: ``(4,)`` the robot's current base orientation.
        motion_joint_vel_50hz: ``(N, 29)`` joint velocities; finite-differenced from
            ``motion_50hz`` if not supplied.

    Returns:
        ``(1762,)``.
    """
    motion = _check_qpos(motion_50hz, "motion_50hz")
    if motion_joint_vel_50hz is None:
        motion_joint_vel_50hz = joint_velocities(motion)
    vel = np.asarray(motion_joint_vel_50hz, dtype=np.float64)
    if vel.shape != (motion.shape[0], NUM_JOINTS):
        raise BridgeError(
            f"motion_joint_vel_50hz: expected {(motion.shape[0], NUM_JOINTS)}, got {vel.shape}"
        )

    idx = lookahead_indices(tick, HISTORY_LEN, ENCODER_FRAME_STRIDE, motion.shape[0])

    out = np.zeros(ENCODER_INPUT_DIM, dtype=np.float64)

    lo, hi = ENCODER_OFFSETS["motion_joint_positions_10frame_step5"]
    out[lo:hi] = conventions.to_isaaclab(motion[idx][:, _JOINTS]).reshape(-1)

    lo, hi = ENCODER_OFFSETS["motion_joint_velocities_10frame_step5"]
    out[lo:hi] = conventions.to_isaaclab(vel[idx]).reshape(-1)

    lo, hi = ENCODER_OFFSETS["motion_anchor_orientation_10frame_step5"]
    base = np.broadcast_to(np.asarray(base_quat_wxyz, dtype=np.float64), (HISTORY_LEN, 4))
    delta = np.broadcast_to(np.asarray(apply_delta_heading, dtype=np.float64), (HISTORY_LEN, 4))
    out[lo:hi] = anchor_orientation(base, motion[idx][:, _ROOT_QUAT], delta).reshape(-1)

    return out
