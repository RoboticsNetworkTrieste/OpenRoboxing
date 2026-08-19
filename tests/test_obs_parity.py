"""M1-T3 — THE M1 GATE: observation parity against the C++ reference.

Acceptance criterion from WORKPLAN.md M1-T3:
  pytest tests/test_obs_parity.py replays the golden fixture and matches the C++ `policy_input`
  vector to max absolute error < 1e-4 on every tick. Report per-term error so a failure localises
  immediately.

This is a genuine parity test, not a self-consistency check: the raw signals (`state_q`, `state_dq`,
`state_action`, `state_base_quat`, `state_base_ang_vel`) are fed to `obs.py`, and the *assembled*
result is compared with what the reference actually fed its network. `token_state` is supplied from
the capture because the tokenizer is `runtime/policy.py`'s job (M1-T4), not this module's.

Reproduce with:
    .venv_mb/bin/python -m pytest tests/test_obs_parity.py -v -s
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.paths import GOLDEN_POLICY_IO_DIR
from openroboxing.runtime.obs import (
    POLICY_TERMS,
    ObservationBuilder,
    ObservationError,
    RobotState,
    projected_gravity,
    split_policy_input,
)
from openroboxing.spec.constants import HISTORY_LEN, OBS_PARITY_TOL

FIXTURE = GOLDEN_POLICY_IO_DIR / "golden.npz"


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    if not FIXTURE.exists():
        pytest.skip(f"golden fixture not present at {FIXTURE}")
    with np.load(FIXTURE, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="module")
def replay(golden) -> dict[str, np.ndarray]:
    """Replay the raw capture through ObservationBuilder.

    Returns the assembled vectors and the reference vectors for the ticks where a full history
    window is available (the first HISTORY_LEN-1 ticks cannot be reconstructed: their history
    extends before the start of the fixture window).
    """
    builder = ObservationBuilder()
    n = int(golden["num_ticks"])

    ours, theirs, ticks = [], [], []
    for t in range(n):
        builder.push(
            RobotState(
                joint_pos=golden["state_q"][t],
                joint_vel=golden["state_dq"][t],
                base_quat=golden["state_base_quat"][t],
                base_ang_vel=golden["state_base_ang_vel"][t],
                last_action=golden["state_action"][t],
            )
        )
        if not builder.ready:
            continue
        ours.append(builder.policy_input(token_state=golden["state_token_state"][t]))
        theirs.append(golden["policy_input"][t])
        ticks.append(t)

    return {
        "ours": np.array(ours),
        "theirs": np.array(theirs),
        "ticks": np.array(ticks),
    }


# --- THE GATE -----------------------------------------------------------------------------------
def test_observation_parity_gate(replay) -> None:
    """Max absolute error over every tick and every dimension must be < 1e-4."""
    ours, theirs = replay["ours"], replay["theirs"]
    assert ours.shape == theirs.shape

    err = np.abs(ours - theirs)
    max_err = float(err.max())

    # Per-term report, printed regardless of outcome so a failure localises immediately.
    print(f"\n  replayed {ours.shape[0]} ticks of {ours.shape[1]} dims")
    print(f"  {'term':44s} {'max abs err':>12s}  {'mean abs err':>12s}")
    mine = split_policy_input(ours)
    ref = split_policy_input(theirs)
    for name, _ in POLICY_TERMS:
        d = np.abs(mine[name] - ref[name])
        print(f"  {name:44s} {d.max():12.3e}  {d.mean():12.3e}")
    print(f"  {'TOTAL':44s} {max_err:12.3e}")

    worst = np.unravel_index(np.argmax(err), err.shape)
    assert max_err < OBS_PARITY_TOL, (
        f"observation parity FAILED: max abs err {max_err:.3e} >= {OBS_PARITY_TOL:.0e} "
        f"at tick {replay['ticks'][worst[0]]}, dim {worst[1]} "
        f"(ours={ours[worst]!r} theirs={theirs[worst]!r})"
    )


@pytest.mark.parametrize("term", [name for name, _ in POLICY_TERMS])
def test_each_term_individually(replay, term: str) -> None:
    """Same bound per term, so a failure names the term rather than a dimension index."""
    mine = split_policy_input(replay["ours"])[term]
    ref = split_policy_input(replay["theirs"])[term]
    max_err = float(np.abs(mine - ref).max())
    assert max_err < OBS_PARITY_TOL, f"{term}: max abs err {max_err:.3e}"


def test_replay_covers_almost_every_tick(replay, golden) -> None:
    """Only the first HISTORY_LEN-1 ticks are unreconstructable."""
    expected = int(golden["num_ticks"]) - (HISTORY_LEN - 1)
    assert replay["ours"].shape[0] == expected


# --- the pieces the gate depends on -------------------------------------------------------------
def test_projected_gravity_matches_the_reference_term(golden) -> None:
    """Computed straight from base_quat, independent of any history bookkeeping."""
    from openroboxing.runtime.obs import POLICY_OFFSETS

    lo, hi = POLICY_OFFSETS["his_gravity_dir_10frame_step1"]
    # the current tick is the LAST frame of the oldest-first history
    newest = golden["policy_input"][:, lo:hi].reshape(-1, HISTORY_LEN, 3)[:, -1, :]
    ours = projected_gravity(golden["state_base_quat"])
    assert np.abs(ours - newest).max() < OBS_PARITY_TOL


def test_gravity_is_unit_norm(golden) -> None:
    g = projected_gravity(golden["state_base_quat"])
    assert np.allclose(np.linalg.norm(g, axis=-1), 1.0, atol=1e-9)


def test_identity_quaternion_gives_straight_down() -> None:
    assert np.allclose(projected_gravity(np.array([1.0, 0.0, 0.0, 0.0])), [0.0, 0.0, -1.0])


def test_newest_history_frame_is_the_current_tick(golden) -> None:
    """Pins oldest-first: the last frame of the joint-position history is this tick's measurement."""
    from openroboxing.runtime.conventions import G1
    from openroboxing.runtime.obs import POLICY_OFFSETS, default_angles

    lo, hi = POLICY_OFFSETS["his_body_joint_positions_10frame_step1"]
    newest = golden["policy_input"][:, lo:hi].reshape(-1, HISTORY_LEN, 29)[:, -1, :]
    ours = G1.to_isaaclab(golden["state_q"] - default_angles(G1, "mujoco"))
    assert np.abs(ours - newest).max() < OBS_PARITY_TOL


# --- how the encoder's motion terms are built (a finding M1-T5 depends on) -----------------------
def test_encoder_motion_frames_are_future_reference_frames_at_stride_5(golden) -> None:
    """`motion_joint_positions_10frame_step5[f]` is the reference motion at tick `t + 5f`.

    Two things follow, and M1-T5 (the bridge) depends on both:
      * the reference motion advances exactly one frame per 50 Hz control tick;
      * the encoder is fed 0.9 s of *lookahead* (10 frames x stride 5 = 45 ticks), not history.

    Tolerance is 1e-5 rather than the gate's 1e-4 because `target_motion.csv` is still written at the
    stock ~6 significant digits - patch P1 widened only the policy and encoder dumps.
    """
    from openroboxing.runtime.conventions import G1
    from openroboxing.runtime.obs import ENCODER_OFFSETS

    lo, hi = ENCODER_OFFSETS["motion_joint_positions_10frame_step5"]
    block = golden["encoder_input"][:, lo:hi].reshape(-1, HISTORY_LEN, 29)
    motion_isaac = G1.to_isaaclab(golden["target_motion"][:, 7:36])

    n = block.shape[0]
    last_t = n - 5 * (HISTORY_LEN - 1) - 1  # need t + 45 in range
    assert last_t > 50, "fixture window too short to check the lookahead"

    for frame in range(HISTORY_LEN):
        ours = motion_isaac[frame * 5 : frame * 5 + last_t]
        theirs = block[:last_t, frame]
        err = float(np.abs(ours - theirs).max())
        assert err < 1e-5, f"frame {frame} (tick t+{5 * frame}): max abs err {err:.3e}"


def test_encoder_motion_uses_isaaclab_order(golden) -> None:
    """target_motion.csv is written in MuJoCo order; the encoder term is IsaacLab order.

    Without the remap the error is O(1), so this pins the direction of the conversion.
    """
    from openroboxing.runtime.obs import ENCODER_OFFSETS

    lo, hi = ENCODER_OFFSETS["motion_joint_positions_10frame_step5"]
    frame0 = golden["encoder_input"][:, lo:hi].reshape(-1, HISTORY_LEN, 29)[:, 0, :]
    unmapped = golden["target_motion"][:, 7:36]
    assert np.abs(frame0 - unmapped).max() > 0.1, "expected MuJoCo-order motion to NOT match"


# --- failure behaviour --------------------------------------------------------------------------
def test_building_before_history_is_full_raises() -> None:
    builder = ObservationBuilder()
    zeros = np.zeros(29)
    builder.push(
        RobotState(
            joint_pos=zeros,
            joint_vel=zeros,
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_ang_vel=np.zeros(3),
            last_action=zeros,
        )
    )
    assert not builder.ready
    with pytest.raises(ObservationError, match="push more ticks"):
        builder.policy_input(token_state=np.zeros(64))


def test_wrong_shaped_state_raises() -> None:
    with pytest.raises(ObservationError, match="joint_pos"):
        RobotState(
            joint_pos=np.zeros(23),
            joint_vel=np.zeros(29),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_ang_vel=np.zeros(3),
            last_action=np.zeros(29),
        )


def test_wrong_shaped_token_state_raises(golden) -> None:
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
    with pytest.raises(ObservationError, match="token_state"):
        builder.policy_input(token_state=np.zeros(32))
