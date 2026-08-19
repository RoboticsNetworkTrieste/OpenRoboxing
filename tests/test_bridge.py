"""M1-T5 acceptance: the generator → policy bridge.

Acceptance criterion from WORKPLAN.md M1-T5:
  feeding a known analytic qpos ramp produces the expected resampled series; a round-trip test on a
  real generated clip shows no NaNs, no discontinuities at segment boundaries, and velocities
  consistent with finite differences of the positions.

The "real generated clip" here is the reference motion from the golden capture, which is a genuine
motion stream at the control rate. Driving MotionBricks itself belongs to M1-T6.

Reproduce with:
    .venv_mb/bin/python -m pytest tests/test_bridge.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.paths import GOLDEN_POLICY_IO_DIR
from openroboxing.runtime.bridge import (
    ENCODER_FRAME_STRIDE,
    BridgeError,
    anchor_orientation,
    encoder_input,
    finite_difference_velocities,
    joint_velocities,
    lookahead_indices,
    quat_multiply,
    resample_qpos,
    rotation_6d,
    slerp,
)
from openroboxing.runtime.obs import ENCODER_OFFSETS
from openroboxing.spec.constants import (
    GENERATOR_HZ,
    HISTORY_LEN,
    NUM_JOINTS,
    QPOS_DIM,
    TICK_DT,
    TICK_HZ,
)

FIXTURE = GOLDEN_POLICY_IO_DIR / "golden.npz"


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    if not FIXTURE.exists():
        pytest.skip(f"golden fixture not present at {FIXTURE}")
    with np.load(FIXTURE, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _ramp(n: int, slope: float = 1.0) -> np.ndarray:
    """A qpos stream where every position channel ramps linearly and the root stays upright."""
    frames = np.zeros((n, QPOS_DIM))
    t = np.arange(n) / GENERATOR_HZ
    frames[:, 0:3] = (slope * t)[:, None] * np.array([1.0, 2.0, 3.0])
    frames[:, 3] = 1.0  # identity quaternion, wxyz
    frames[:, 7:] = (slope * t)[:, None] * np.linspace(0.1, 1.0, NUM_JOINTS)
    return frames


# --- the analytic ramp ----------------------------------------------------------------------------
def test_ramp_resamples_to_the_expected_series() -> None:
    """A linear ramp must stay exactly linear through resampling, at the new rate."""
    n = 31  # 1.0 s at 30 Hz
    out = resample_qpos(_ramp(n))

    expected_len = int(np.floor((n - 1) / GENERATOR_HZ * TICK_HZ)) + 1
    assert out.shape == (expected_len, QPOS_DIM)

    t_out = np.arange(expected_len) / TICK_HZ
    assert np.allclose(out[:, 0:3], t_out[:, None] * np.array([1.0, 2.0, 3.0]))
    assert np.allclose(out[:, 7:], t_out[:, None] * np.linspace(0.1, 1.0, NUM_JOINTS))


def test_ramp_velocities_equal_the_analytic_slope() -> None:
    """d/dt of a ramp is its slope, everywhere including the endpoints."""
    out = resample_qpos(_ramp(31))
    vel = joint_velocities(out)
    expected = np.linspace(0.1, 1.0, NUM_JOINTS)
    assert np.allclose(vel, expected[None, :])


def test_resampling_preserves_duration_and_endpoints() -> None:
    frames = _ramp(31)
    out = resample_qpos(frames)
    assert np.allclose(out[0], frames[0])
    # the final 50 Hz sample lands on the final 30 Hz sample when the duration divides evenly
    assert np.allclose(out[-1, 0:3], frames[-1, 0:3], atol=1e-9)


def test_five_to_three_ratio() -> None:
    """30 -> 50 Hz is 5:3 — every 3 generator frames become 5 ticks."""
    out = resample_qpos(_ramp(4))  # 3 intervals
    assert out.shape[0] == 6  # ticks 0..5 inclusive


# --- velocities are differenced AFTER resampling --------------------------------------------------
def test_velocity_is_a_central_difference_in_the_interior() -> None:
    values = np.cumsum(np.random.default_rng(0).standard_normal((50, 3)), axis=0)
    vel = finite_difference_velocities(values, TICK_DT)
    assert np.allclose(vel[1:-1], (values[2:] - values[:-2]) / (2 * TICK_DT))
    assert np.allclose(vel[0], (values[1] - values[0]) / TICK_DT)
    assert np.allclose(vel[-1], (values[-1] - values[-2]) / TICK_DT)


def test_velocity_length_matches_input() -> None:
    values = np.zeros((17, 4))
    assert finite_difference_velocities(values).shape == values.shape


def test_differencing_before_resampling_would_differ() -> None:
    """Guard the ordering rule: the two orders genuinely disagree on a curved signal.

    If this ever passes trivially the test has stopped protecting anything.
    """
    t30 = np.arange(31) / GENERATOR_HZ
    frames = np.zeros((31, QPOS_DIM))
    frames[:, 3] = 1.0
    frames[:, 7] = np.sin(2 * np.pi * 3.0 * t30)  # 3 Hz, fast relative to 30 Hz sampling

    after = joint_velocities(resample_qpos(frames))[:, 0]
    before_30 = finite_difference_velocities(frames[:, 7:8], 1.0 / GENERATOR_HZ)
    padded = np.repeat(before_30, 1, axis=0)
    before = np.interp(
        np.arange(after.size) / TICK_HZ, np.arange(padded.shape[0]) / GENERATOR_HZ, padded[:, 0]
    )
    assert np.abs(after - before).max() > 1e-3


# --- quaternions ----------------------------------------------------------------------------------
def test_slerp_endpoints_and_midpoint() -> None:
    q0 = np.array([[1.0, 0.0, 0.0, 0.0]])
    half = np.pi / 2
    q1 = np.array([[np.cos(half / 2), 0.0, 0.0, np.sin(half / 2)]])  # 90 deg about z
    assert np.allclose(slerp(q0, q1, np.array([0.0])), q0)
    assert np.allclose(slerp(q0, q1, np.array([1.0])), q1)
    mid = slerp(q0, q1, np.array([0.5]))[0]
    quarter = np.array([np.cos(half / 4), 0.0, 0.0, np.sin(half / 4)])  # 45 deg about z
    assert np.allclose(mid, quarter)


def test_slerp_stays_on_the_unit_sphere() -> None:
    rng = np.random.default_rng(7)
    q0 = rng.standard_normal((200, 4))
    q0 /= np.linalg.norm(q0, axis=1, keepdims=True)
    q1 = rng.standard_normal((200, 4))
    q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
    out = slerp(q0, q1, rng.random(200))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_slerp_takes_the_short_way_round() -> None:
    """q and -q are the same rotation; without the antipodal flip slerp goes the long way."""
    q0 = np.array([[1.0, 0.0, 0.0, 0.0]])
    q1 = np.array([[-1.0, 0.0, 0.0, 0.0]])  # same rotation, opposite sign
    mid = slerp(q0, q1, np.array([0.5]))[0]
    assert np.abs(np.abs(mid[0]) - 1.0) < 1e-9


def test_resampled_quaternions_stay_unit() -> None:
    rng = np.random.default_rng(11)
    frames = np.zeros((40, QPOS_DIM))
    q = rng.standard_normal((40, 4))
    frames[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    out = resample_qpos(frames)
    assert np.allclose(np.linalg.norm(out[:, 3:7], axis=1), 1.0)


def test_rotation_6d_of_identity() -> None:
    assert np.allclose(rotation_6d(np.array([1.0, 0.0, 0.0, 0.0])), [1, 0, 0, 1, 0, 0])


def test_quat_multiply_matches_identity() -> None:
    rng = np.random.default_rng(3)
    q = rng.standard_normal((10, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    identity = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
    assert np.allclose(quat_multiply(identity, q), q)


# --- against the real motion stream ---------------------------------------------------------------
def test_real_clip_round_trip_has_no_nans_or_discontinuities(golden) -> None:
    """The captured reference motion, resampled down to 30 Hz and back, must stay smooth."""
    motion = golden["target_motion"]
    down = motion[::5][:60]  # 50 Hz -> 10 Hz, a harsher round trip than 30 Hz
    up = resample_qpos(down, source_hz=TICK_HZ / 5, target_hz=TICK_HZ)

    assert np.isfinite(up).all()
    assert np.allclose(np.linalg.norm(up[:, 3:7], axis=1), 1.0)

    # no step changes: the largest per-tick jump must stay in family with the median
    steps = np.abs(np.diff(up[:, 7:], axis=0)).max(axis=1)
    assert steps.max() < 20.0 * np.median(steps[steps > 0]) + 1e-6


def test_velocities_are_consistent_with_position_differences(golden) -> None:
    motion = golden["target_motion"][:200]
    vel = joint_velocities(motion)
    recomputed = finite_difference_velocities(motion[:, 7:], TICK_DT)
    assert np.allclose(vel, recomputed)


def _solve_apply_delta_heading(golden) -> np.ndarray:
    """Recover `apply_delta_heading` from the capture.

    It is runtime state (where the robot faced when the clip started) and the capture does not record
    it, but it is recoverable: invert the 6-D back to a rotation and solve
    ``delta = base * base_to_ref * conj(ref_raw)``.
    """
    from openroboxing.runtime.bridge import quat_conjugate

    lo, hi = ENCODER_OFFSETS["motion_anchor_orientation_10frame_step5"]
    sixd = golden["encoder_input"][:, lo:hi].reshape(-1, HISTORY_LEN, 6)[:, 0, :]

    # 6-D holds columns 0 and 1 flattened row-wise; rebuild column 2 as their cross product.
    c0 = sixd[:, [0, 2, 4]]
    c1 = sixd[:, [1, 3, 5]]
    c0 = c0 / np.linalg.norm(c0, axis=1, keepdims=True)
    c1 = c1 - np.sum(c0 * c1, axis=1, keepdims=True) * c0
    c1 = c1 / np.linalg.norm(c1, axis=1, keepdims=True)
    rot = np.stack([c0, c1, np.cross(c0, c1)], axis=-1)

    w = np.sqrt(np.clip(1 + rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2], 0, None)) / 2
    base_to_ref = np.stack(
        [
            w,
            (rot[:, 2, 1] - rot[:, 1, 2]) / (4 * w),
            (rot[:, 0, 2] - rot[:, 2, 0]) / (4 * w),
            (rot[:, 1, 0] - rot[:, 0, 1]) / (4 * w),
        ],
        axis=-1,
    )
    delta = quat_multiply(
        quat_multiply(golden["state_base_quat"], base_to_ref),
        quat_conjugate(golden["target_motion"][:, 3:7]),
    )
    return delta * np.sign(delta[:, :1])  # quaternion double cover


def test_recovered_delta_heading_is_constant_and_yaw_only(golden) -> None:
    """`apply_delta_heading` must behave as ComputeApplyDeltaHeading says it does.

    It aligns the reference's initial heading with the robot's, so it is a pure yaw rotation and is
    fixed for the life of a motion. If this fails, the anchor-orientation formula is misunderstood.
    """
    delta = _solve_apply_delta_heading(golden)
    # Tolerance reflects the recovery path, not real drift: it inverts a 6-D representation whose
    # inputs come from target_motion.csv at ~6 significant digits (~4e-6 on O(1) values). Measured
    # spread is ~2e-7, i.e. constant to within the fixture's own precision.
    assert np.abs(delta.std(axis=0)).max() < 1e-6, "delta_heading drifts; it should be constant"
    assert np.abs(delta[:, 1:3]).max() < 1e-6, "delta_heading has roll/pitch; it should be yaw-only"


def test_anchor_orientation_matches_the_reference(golden) -> None:
    """Reproduce the encoder's `motion_anchor_orientation_10frame_step5` from raw quaternions.

    `apply_delta_heading` is supplied (recovered above) because it is runtime state the capture does
    not carry — in the live runtime it comes from `compute_apply_delta_heading` at motion start.

    Tolerance is 1e-5, not the parity gate's 1e-4, because target_motion.csv is still written at the
    stock ~6 significant digits — patch P1 widened only the policy and encoder dumps.
    """
    lo, hi = ENCODER_OFFSETS["motion_anchor_orientation_10frame_step5"]
    theirs = golden["encoder_input"][:, lo:hi].reshape(-1, HISTORY_LEN, 6)
    delta = _solve_apply_delta_heading(golden)[0]

    n = golden["target_motion"].shape[0]
    last_t = n - ENCODER_FRAME_STRIDE * (HISTORY_LEN - 1) - 1
    assert last_t > 50

    for frame in range(HISTORY_LEN):
        offset = frame * ENCODER_FRAME_STRIDE
        ours = anchor_orientation(
            golden["state_base_quat"][:last_t],
            golden["target_motion"][offset : offset + last_t, 3:7],
            np.broadcast_to(delta, (last_t, 4)),
        )
        err = float(np.abs(ours - theirs[:last_t, frame]).max())
        assert err < 1e-5, f"frame {frame}: max abs err {err:.3e}"


def test_omitting_delta_heading_is_badly_wrong(golden) -> None:
    """Guard the trap: leaving apply_delta_heading out gives a plausible but O(1)-wrong answer."""
    lo, hi = ENCODER_OFFSETS["motion_anchor_orientation_10frame_step5"]
    theirs = golden["encoder_input"][:, lo:hi].reshape(-1, HISTORY_LEN, 6)[:, 0, :]
    naive = anchor_orientation(golden["state_base_quat"], golden["target_motion"][:, 3:7])
    assert np.abs(naive - theirs).max() > 0.5


def test_heading_quat_of_a_yaw_only_quaternion_is_itself(golden) -> None:
    from openroboxing.runtime.bridge import heading_quat

    delta = _solve_apply_delta_heading(golden)[0]
    assert np.allclose(heading_quat(delta), delta, atol=1e-9)


def test_encoder_input_reproduces_the_reference_motion_terms(golden) -> None:
    """Assemble the tokenizer input from the motion stream and match the captured one.

    Joint velocities are excluded: the reference reads them from the motion's own joint_vel.csv,
    which the capture does not carry, so they cannot be reconstructed from target_motion alone.
    """
    n = golden["target_motion"].shape[0]
    last_t = n - ENCODER_FRAME_STRIDE * (HISTORY_LEN - 1) - 1
    tick = 40
    assert tick < last_t

    ours = encoder_input(
        tick=tick,
        motion_50hz=golden["target_motion"],
        base_quat_wxyz=golden["state_base_quat"][tick],
        apply_delta_heading=_solve_apply_delta_heading(golden)[0],
    )
    theirs = golden["encoder_input"][tick]

    for term in ("motion_joint_positions_10frame_step5", "motion_anchor_orientation_10frame_step5"):
        lo, hi = ENCODER_OFFSETS[term]
        err = float(np.abs(ours[lo:hi] - theirs[lo:hi]).max())
        assert err < 1e-5, f"{term}: max abs err {err:.3e}"


def test_encoder_input_leaves_inactive_terms_zero(golden) -> None:
    ours = encoder_input(
        tick=10,
        motion_50hz=golden["target_motion"],
        base_quat_wxyz=golden["state_base_quat"][10],
    )
    for term in ("smpl_joints_10frame_step1", "vr_3point_local_target", "encoder_mode_4"):
        lo, hi = ENCODER_OFFSETS[term]
        assert np.all(ours[lo:hi] == 0.0), f"{term} should be zero in g1 mode"


def test_lookahead_clamps_at_the_end_of_the_motion() -> None:
    idx = lookahead_indices(
        tick=95, num_frames=HISTORY_LEN, stride=ENCODER_FRAME_STRIDE, motion_length=100
    )
    assert idx[0] == 95
    assert (idx <= 99).all()
    assert idx[-1] == 99  # clamped, not wrapped


def test_lookahead_is_forward_in_time() -> None:
    idx = lookahead_indices(
        tick=0, num_frames=HISTORY_LEN, stride=ENCODER_FRAME_STRIDE, motion_length=1000
    )
    assert list(idx) == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]


# --- failure behaviour ----------------------------------------------------------------------------
def test_wrong_qpos_width_raises() -> None:
    with pytest.raises(BridgeError, match="expected"):
        resample_qpos(np.zeros((10, 30)))


def test_single_frame_raises() -> None:
    with pytest.raises(BridgeError, match="at least 2 frames"):
        resample_qpos(np.zeros((1, QPOS_DIM)))


def test_nan_input_raises() -> None:
    frames = _ramp(10)
    frames[3, 8] = np.nan
    with pytest.raises(BridgeError, match="NaN"):
        resample_qpos(frames)


def test_bad_dt_raises() -> None:
    with pytest.raises(BridgeError, match="dt must be positive"):
        finite_difference_velocities(np.zeros((5, 2)), dt=0.0)
