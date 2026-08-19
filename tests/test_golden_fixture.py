"""T2 acceptance: the golden fixture loads and matches the M0-T2 observation table.

Acceptance criterion from WORKPLAN.md M1-T2:
  tests/fixtures/golden_policy_io/ contains aligned arrays and the reproduction command; a smoke
  test loads them and checks shapes against the observation table from M0-T2.

These are structural assertions about the *reference*, not about our code. If one fails after a
re-capture, the reference or its config changed and spec/upstream_notes.md is stale.

Reproduce with:
    .venv_mb/bin/python -m pytest tests/test_golden_fixture.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.paths import GOLDEN_POLICY_IO_DIR
from openroboxing.spec.constants import (
    ENCODER_INPUT_DIM,
    HISTORY_LEN,
    NUM_JOINTS,
    POLICY_INPUT_DIM,
    QPOS_DIM,
)

FIXTURE = GOLDEN_POLICY_IO_DIR / "golden.npz"

# Policy input, in observation_config.yaml order. See spec/upstream_notes.md Q2.
POLICY_TERMS: tuple[tuple[str, int], ...] = (
    ("token_state", 64),
    ("his_base_angular_velocity_10frame_step1", 30),
    ("his_body_joint_positions_10frame_step1", 290),
    ("his_body_joint_velocities_10frame_step1", 290),
    ("his_last_actions_10frame_step1", 290),
    ("his_gravity_dir_10frame_step1", 30),
)

# Encoder input, in order, with whether the term is required in `g1` mode (mode_id 0).
ENCODER_TERMS: tuple[tuple[str, int, bool], ...] = (
    ("encoder_mode_4", 4, True),  # required, but writes mode id 0 + 3 zeros -> all zero
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

# encoder_mode_4 is required by the mode but is identically zero for g1 (mode id 0).
ZERO_EVEN_THOUGH_REQUIRED = {"encoder_mode_4"}


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    if not FIXTURE.exists():
        pytest.skip(f"golden fixture not present at {FIXTURE}; see its README to re-capture")
    with np.load(FIXTURE, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _offsets(terms) -> dict[str, tuple[int, int]]:
    out, off = {}, 0
    for term in terms:
        name, dim = term[0], term[1]
        out[name] = (off, off + dim)
        off += dim
    return out


def test_shapes_match_the_shipped_onnx(golden) -> None:
    assert golden["policy_input"].shape[1] == POLICY_INPUT_DIM
    assert golden["encoder_input"].shape[1] == ENCODER_INPUT_DIM
    assert golden["target_motion"].shape[1] == QPOS_DIM


def test_arrays_are_row_aligned(golden) -> None:
    n = int(golden["num_ticks"])
    assert golden["policy_input"].shape[0] == n
    assert golden["encoder_input"].shape[0] == n
    assert golden["target_motion"].shape[0] == n


def test_term_dimensions_sum_to_the_totals() -> None:
    assert sum(d for _, d in POLICY_TERMS) == POLICY_INPUT_DIM
    assert sum(d for _, d, _ in ENCODER_TERMS) == ENCODER_INPUT_DIM


def test_no_nans_or_infs(golden) -> None:
    for key in ("policy_input", "encoder_input", "target_motion"):
        assert np.isfinite(golden[key]).all(), f"{key} contains NaN or inf"


def test_encoder_zero_fill_holds_for_g1_mode(golden) -> None:
    """The claim that shrinks M1-T3: unrequired encoder terms are identically zero.

    If this fails, the capture was not in `g1` mode and the fixture is not what obs.py must match.
    """
    enc = golden["encoder_input"]
    for name, (lo, hi) in _offsets(ENCODER_TERMS).items():
        required = next(r for n, _, r in ENCODER_TERMS if n == name)
        peak = float(np.abs(enc[:, lo:hi]).max())
        if required and name not in ZERO_EVEN_THOUGH_REQUIRED:
            assert peak > 0.0, f"{name} is required in g1 mode but is all zero"
        else:
            assert peak == 0.0, f"{name} should be zero in g1 mode but peaks at {peak}"


def test_only_640_encoder_dims_are_non_zero(golden) -> None:
    enc = golden["encoder_input"]
    non_zero_cols = int((np.abs(enc).max(axis=0) > 0).sum())
    assert non_zero_cols == 640, (
        f"expected 640 informative encoder dims in g1 mode, got {non_zero_cols}; "
        "spec/upstream_notes.md Q2 is stale"
    )


def test_encoder_mode_is_g1(golden) -> None:
    """encoder_mode_4 writes the mode id as a scalar, then three zeros. g1 is mode 0."""
    lo, hi = _offsets(ENCODER_TERMS)["encoder_mode_4"]
    assert np.all(golden["encoder_input"][:, lo:hi] == 0.0)


def test_gravity_is_a_unit_direction(golden) -> None:
    """his_gravity_dir is a projected gravity direction: 10 frames of a 3-vector, each ~unit norm."""
    lo, hi = _offsets(POLICY_TERMS)["his_gravity_dir_10frame_step1"]
    g = golden["policy_input"][:, lo:hi].reshape(-1, HISTORY_LEN, 3)
    norms = np.linalg.norm(g, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"norms range {norms.min()}..{norms.max()}"


@pytest.mark.parametrize(
    "term", ["his_body_joint_positions_10frame_step1", "his_body_joint_velocities_10frame_step1"]
)
def test_history_layout_is_frame_major_oldest_first(golden, term: str) -> None:
    """Frame k at tick t+1 equals frame k+1 at tick t: the buffer shifts, newest enters last.

    This pins the memory layout obs.py must reproduce. Established from the capture with an exact
    zero residual, so any tolerance here would be too generous.
    """
    lo, hi = _offsets(POLICY_TERMS)[term]
    frames = golden["policy_input"][:, lo:hi].reshape(-1, HISTORY_LEN, NUM_JOINTS)
    residual = np.abs(frames[1:, :-1] - frames[:-1, 1:]).max()
    assert residual == 0.0, f"{term}: expected oldest-first frame-major, residual {residual}"


def test_raw_input_signals_are_present(golden) -> None:
    """Without the raw inputs, a parity test only proves self-consistency, not parity."""
    expected = {
        "state_q": NUM_JOINTS,
        "state_dq": NUM_JOINTS,
        "state_action": NUM_JOINTS,
        "state_base_ang_vel": 3,
        "state_base_quat": 4,
        "state_token_state": 64,
        "state_encoder_mode": 1,
    }
    for name, width in expected.items():
        assert name in golden, f"{name} missing; re-capture with --enable-csv-logs"
        assert golden[name].shape == (int(golden["num_ticks"]), width)


def test_state_logs_align_with_the_policy_dump(golden) -> None:
    """The dump flags and StateLogger are separate code paths; prove they share a row index.

    token_state appears in both. It must match exactly at zero shift and not at any other shift,
    otherwise the raw inputs are off by a tick and every parity result would be subtly wrong.
    """
    from_policy = golden["policy_input"][:, 0:64]
    from_logger = golden["state_token_state"]
    assert np.abs(from_policy - from_logger).max() == 0.0, "state logs are not tick-aligned"

    for shift in (-2, -1, 1, 2):
        rolled = np.roll(from_logger, shift, axis=0)[2:-2]
        err = np.abs(from_policy[2:-2] - rolled).max()
        assert err > 0.0, f"shift {shift:+d} also matches — alignment check is not discriminating"


def test_capture_is_in_g1_encoder_mode(golden) -> None:
    assert np.all(golden["state_encoder_mode"] == 0.0)


def test_base_quaternion_is_unit_norm(golden) -> None:
    norms = np.linalg.norm(golden["state_base_quat"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), f"norms {norms.min()}..{norms.max()}"


def test_fixture_window_contains_real_motion(golden) -> None:
    """A window where the reference is static would pass a parity test trivially."""
    joints = golden["target_motion"][:, 7:]
    per_tick = np.abs(np.diff(joints, axis=0)).sum(axis=1)
    assert per_tick.mean() > 0.1, f"reference barely moves (mean |dq|/tick = {per_tick.mean()})"


def _round_to_significant_digits(v: np.ndarray, digits: int) -> np.ndarray:
    """Round to `digits` significant digits, elementwise. Zeros pass through."""
    out = np.zeros_like(v)
    nz = v != 0
    magnitude = np.floor(np.log10(np.abs(v[nz])))
    scale = 10.0 ** (digits - 1 - magnitude)
    out[nz] = np.round(v[nz] * scale) / scale
    return out


def test_dump_precision_is_wide_enough_for_the_gate(golden) -> None:
    """Patch P1: joint velocities reach O(30), where 6 significant digits quantises at ~1e-4.

    The stock dump used default ostream precision (6 significant digits), which would put every
    value exactly on a 6-significant-digit grid. With P1 the dump carries 9, so most values must
    fall *off* that grid.
    """
    lo, hi = _offsets(POLICY_TERMS)["his_body_joint_velocities_10frame_step1"]
    block = golden["policy_input"][:, lo:hi]
    peak = float(np.abs(block).max())
    assert peak > 10.0, "expected O(10-100) joint velocities; is this the right capture?"

    values = block.ravel()
    values = values[values != 0]
    off_grid = np.mean(values != _round_to_significant_digits(values, 6))
    assert off_grid > 0.5, (
        f"only {off_grid:.1%} of values carry more than 6 significant digits — "
        "patch P1 (dump precision) is probably missing from the binary that produced this fixture"
    )
