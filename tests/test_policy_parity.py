"""M1-T4 acceptance: the Python policy reproduces the C++ reference's actions.

Acceptance criterion from WORKPLAN.md M1-T4:
  replaying the golden observations through the Python policy reproduces the C++ actions to
  < 1e-3 max absolute error.

Reproduce with:
    .venv_mb/bin/python -m pytest tests/test_policy_parity.py -v -s
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.paths import GOLDEN_POLICY_IO_DIR
from openroboxing.runtime.conventions import G1
from openroboxing.runtime.obs import default_angles
from openroboxing.runtime.policy import (
    GearSonicPolicy,
    PolicyError,
    action_scale,
    action_to_joint_target,
    damping,
    stiffness,
)
from openroboxing.spec.constants import ACTION_PARITY_TOL, NUM_JOINTS

FIXTURE = GOLDEN_POLICY_IO_DIR / "golden.npz"

# How many ticks to replay. Every tick costs two ONNX inferences on CPU, so the suite stays quick.
N_TICKS = 120


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    if not FIXTURE.exists():
        pytest.skip(f"golden fixture not present at {FIXTURE}")
    with np.load(FIXTURE, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="module")
def policy() -> GearSonicPolicy:
    pytest.importorskip("onnxruntime")
    return GearSonicPolicy()


# --- the acceptance criterion --------------------------------------------------------------------
def test_action_parity(golden, policy) -> None:
    """decoder(policy_input[t]) must reproduce the action the reference emitted at tick t.

    The reference logs `last_action`, so the action computed at tick t appears in the log at t+1.
    """
    ours = np.vstack([policy.act(golden["policy_input"][t]) for t in range(N_TICKS)])
    theirs = golden["state_action"][1 : N_TICKS + 1]

    err = np.abs(ours - theirs)
    max_err = float(err.max())
    print(f"\n  replayed {N_TICKS} ticks")
    print(f"  action max abs err  = {max_err:.3e}   (tolerance {ACTION_PARITY_TOL:.0e})")
    print(f"  action mean abs err = {err.mean():.3e}")

    worst = np.unravel_index(np.argmax(err), err.shape)
    assert max_err < ACTION_PARITY_TOL, (
        f"action parity FAILED: {max_err:.3e} at tick {worst[0]}, joint {worst[1]} "
        f"(ours={ours[worst]!r} theirs={theirs[worst]!r})"
    )


def test_action_log_offset_is_one_tick(golden, policy) -> None:
    """Pin the offset: only t+1 matches, so a future re-capture cannot silently shift it."""
    n = 40
    ours = np.vstack([policy.act(golden["policy_input"][t]) for t in range(n)])
    errs = {
        shift: float(np.abs(ours[: n - 2] - golden["state_action"][shift : shift + n - 2]).max())
        for shift in (0, 1, 2)
    }
    assert errs[1] < ACTION_PARITY_TOL
    assert errs[0] > 0.1 and errs[2] > 0.1, f"offset is not discriminating: {errs}"


def test_encoder_reproduces_reference_tokens(golden, policy) -> None:
    """The tokenizer is exact: its outputs are quantised, so they round-trip bit-for-bit."""
    ours = np.vstack([policy.encode(golden["encoder_input"][t]) for t in range(N_TICKS)])
    theirs = golden["state_token_state"][:N_TICKS]
    max_err = float(np.abs(ours - theirs).max())
    print(f"\n  token max abs err = {max_err:.3e}")
    assert max_err < ACTION_PARITY_TOL


def test_step_matches_running_the_two_stages_separately(golden, policy) -> None:
    """`step` is the reference's ordering: encode, then assemble, then act."""
    from openroboxing.runtime.obs import HISTORY_LEN, ObservationBuilder, RobotState

    builder = ObservationBuilder()
    for t in range(HISTORY_LEN):
        builder.push(
            RobotState(
                joint_pos=golden["state_q"][t],
                joint_vel=golden["state_dq"][t],
                base_quat=golden["state_base_quat"][t],
                base_ang_vel=golden["state_base_ang_vel"][t],
                last_action=golden["state_action"][t],
            )
        )
    t = HISTORY_LEN - 1
    action, token = policy.step(golden["encoder_input"][t], builder)
    assert np.abs(token - golden["state_token_state"][t]).max() < ACTION_PARITY_TOL

    # `step` acts on OUR assembled observation; `act` here is given the reference's. Those differ by
    # the fixture's ~5e-08 decimal rounding, so the outputs cannot be bit-identical - the network
    # amplifies that input error by roughly 40x. Still ~500x inside the tolerance.
    drift = float(np.abs(action - policy.act(golden["policy_input"][t])).max())
    assert drift < ACTION_PARITY_TOL, f"assembled-vs-reference action drift {drift:.3e}"


# --- action -> joint target ----------------------------------------------------------------------
def test_action_scale_matches_the_reference_values(golden) -> None:
    """Spot-check the derived scales against the C++ formula for two different motor types.

    action_scale = 0.25 * effort_limit / (armature * omega^2)
    """
    omega = 10.0 * 2.0 * 3.1415926535
    scales = action_scale(G1, "mujoco")
    names = G1.mujoco_joint_names

    # left_knee_joint is a 7520_22; left_wrist_pitch_joint is a 4010.
    knee = scales[names.index("left_knee_joint")]
    assert knee == pytest.approx(0.25 * 139.0 / (0.025101925 * omega**2))
    wrist = scales[names.index("left_wrist_pitch_joint")]
    assert wrist == pytest.approx(0.25 * 5.0 / (0.00425 * omega**2))


def test_zero_action_gives_the_default_pose() -> None:
    target = action_to_joint_target(np.zeros(NUM_JOINTS))
    assert np.allclose(target, default_angles(G1, "mujoco"))


def test_joint_target_is_in_mujoco_order() -> None:
    """A one-hot action on an IsaacLab index must move the matching *named* joint in MuJoCo order."""
    isaac_idx = 5
    action = np.zeros(NUM_JOINTS)
    action[isaac_idx] = 1.0
    target = action_to_joint_target(action)
    moved = np.flatnonzero(~np.isclose(target, default_angles(G1, "mujoco")))
    assert moved.size == 1
    assert G1.mujoco_joint_names[moved[0]] == G1.isaaclab_joint_names[isaac_idx]


def test_gains_are_positive_and_named() -> None:
    for fn in (stiffness, damping, action_scale):
        values = fn(G1, "mujoco")
        assert values.shape == (NUM_JOINTS,)
        assert (values > 0).all(), f"{fn.__name__} produced a non-positive gain"


# --- failure behaviour ----------------------------------------------------------------------------
def test_wrong_input_width_raises(policy) -> None:
    with pytest.raises(PolicyError, match="policy_input"):
        policy.act(np.zeros(100))
    with pytest.raises(PolicyError, match="encoder_input"):
        policy.encode(np.zeros(100))


def test_missing_model_raises() -> None:
    from pathlib import Path

    with pytest.raises(PolicyError, match="model not found"):
        GearSonicPolicy(encoder_path=Path("/nonexistent/encoder.onnx"))
