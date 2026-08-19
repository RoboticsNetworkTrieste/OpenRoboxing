"""T1 acceptance: name-derived MuJoCo ↔ IsaacLab mappings.

Acceptance criterion from WORKPLAN.md M1-T1:
  property test over 1000 random vectors, ``to_mujoco(to_isaaclab(x)) == x`` exactly;
  constructing the mapping with a deliberately renamed joint raises.

Reproduce with:
    .venv_mb/bin/python -m pytest tests/test_conventions.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.runtime import conventions as C
from openroboxing.spec.constants import NUM_JOINTS

RNG_SEED = 20260807
N_TRIALS = 1000


# --- the acceptance criterion -------------------------------------------------------------------
def test_joint_roundtrip_is_exact_over_1000_random_vectors() -> None:
    rng = np.random.default_rng(RNG_SEED)
    for _ in range(N_TRIALS):
        x = rng.standard_normal(NUM_JOINTS)
        assert np.array_equal(C.to_mujoco(C.to_isaaclab(x)), x)
        assert np.array_equal(C.to_isaaclab(C.to_mujoco(x)), x)


def test_renamed_joint_raises() -> None:
    """A deliberately renamed joint must fail loudly, not silently drop or reorder."""
    bad = list(C.G1.isaaclab_joint_names)
    bad[7] = "left_hip_yaw_joint_RENAMED"
    with pytest.raises(C.ConventionError, match="name sets differ"):
        C.build_conventions(isaaclab_joint_names=bad)


def test_duplicated_joint_raises() -> None:
    bad = list(C.G1.isaaclab_joint_names)
    bad[7] = bad[6]
    with pytest.raises(C.ConventionError, match="duplicate IsaacLab joint names"):
        C.build_conventions(isaaclab_joint_names=bad)


def test_dropped_joint_raises() -> None:
    with pytest.raises(C.ConventionError, match="name sets differ"):
        C.build_conventions(isaaclab_joint_names=list(C.G1.isaaclab_joint_names)[:-1])


# --- the mapping agrees with the deployed C++ ---------------------------------------------------
def test_derived_mapping_matches_cpp_reference() -> None:
    """The whole point of deriving by name: it must reproduce what the robot actually runs."""
    assert C.G1.mujoco_to_isaaclab == C._CPP_MUJOCO_TO_ISAACLAB
    assert C.G1.isaaclab_to_mujoco == C._CPP_ISAACLAB_TO_MUJOCO


def test_gather_convention_matches_cpp_usage() -> None:
    """Pin the gather semantics, the trap documented in the module docstring.

    Mirrors g1_deploy_onnx_ref.cpp:3120 — an IsaacLab-ordered action vector is consumed in MuJoCo
    (motor) order as ``mujoco[i] = isaac[isaaclab_to_mujoco[i]]``.
    """
    isaac = np.arange(NUM_JOINTS, dtype=float)
    mj = C.to_mujoco(isaac)
    for i in range(NUM_JOINTS):
        assert mj[i] == isaac[C.G1.isaaclab_to_mujoco[i]]

    mj2 = np.arange(NUM_JOINTS, dtype=float) * 3.0
    isaac2 = C.to_isaaclab(mj2)
    for i in range(NUM_JOINTS):
        assert isaac2[i] == mj2[C.G1.mujoco_to_isaaclab[i]]


def test_mapping_moves_names_not_just_numbers() -> None:
    """Position k in IsaacLab order must hold the joint of that name from MuJoCo order."""
    for k, name in enumerate(C.G1.isaaclab_joint_names):
        assert C.G1.mujoco_joint_names[C.G1.mujoco_to_isaaclab[k]] == name


# --- structural invariants ----------------------------------------------------------------------
def test_mappings_are_permutations_and_mutually_inverse() -> None:
    n = NUM_JOINTS
    assert sorted(C.G1.mujoco_to_isaaclab) == list(range(n))
    assert sorted(C.G1.isaaclab_to_mujoco) == list(range(n))
    assert all(C.G1.mujoco_to_isaaclab[C.G1.isaaclab_to_mujoco[k]] == k for k in range(n))
    assert all(C.G1.isaaclab_to_mujoco[C.G1.mujoco_to_isaaclab[k]] == k for k in range(n))


def test_joint_counts() -> None:
    assert C.G1.num_joints == NUM_JOINTS
    assert len(C.G1.mujoco_joint_names) == NUM_JOINTS
    assert "floating_base_joint" not in C.G1.mujoco_joint_names


def test_batched_joint_roundtrip() -> None:
    """Mappings act on the last axis, so history buffers round-trip too."""
    rng = np.random.default_rng(RNG_SEED + 1)
    x = rng.standard_normal((10, 4, NUM_JOINTS))
    assert np.array_equal(C.to_mujoco(C.to_isaaclab(x)), x)


def test_wrong_joint_count_raises() -> None:
    with pytest.raises(C.ConventionError, match="expected 29 joints"):
        C.to_isaaclab(np.zeros(23))


# --- bodies -------------------------------------------------------------------------------------
def test_body_roundtrip_and_permutation() -> None:
    n = C.G1.num_bodies
    assert sorted(C.G1.mujoco_to_isaaclab_body) == list(range(n))
    rng = np.random.default_rng(RNG_SEED + 2)
    x = rng.standard_normal(n)
    assert np.array_equal(C.G1.bodies_to_mujoco(C.G1.bodies_to_isaaclab(x)), x)


def test_body_roundtrip_with_trailing_vector_axis() -> None:
    """Per-body positions (n,3) and quaternions (n,4) reorder on axis -2."""
    n = C.G1.num_bodies
    rng = np.random.default_rng(RNG_SEED + 3)
    for k in (3, 4):
        x = rng.standard_normal((n, k))
        assert np.array_equal(C.G1.bodies_to_mujoco(C.G1.bodies_to_isaaclab(x)), x)


def test_body_names_move_with_the_mapping() -> None:
    for k, name in enumerate(C.G1.isaaclab_body_names):
        assert C.G1.mujoco_body_names[C.G1.mujoco_to_isaaclab_body[k]] == name


def test_world_body_excluded() -> None:
    assert "world" not in C.G1.mujoco_body_names
    assert C.G1.mujoco_body_names[0] == "pelvis"


# --- quaternions --------------------------------------------------------------------------------
def test_quaternion_roundtrip_and_ordering() -> None:
    rng = np.random.default_rng(RNG_SEED + 4)
    q = rng.standard_normal((N_TRIALS, 4))
    assert np.array_equal(C.quat_xyzw_to_wxyz(C.quat_wxyz_to_xyzw(q)), q)

    w, x, y, z = 0.1, 0.2, 0.3, 0.4
    assert np.array_equal(C.quat_wxyz_to_xyzw(np.array([w, x, y, z])), np.array([x, y, z, w]))
    assert np.array_equal(C.quat_xyzw_to_wxyz(np.array([x, y, z, w])), np.array([w, x, y, z]))


def test_bad_quaternion_shape_raises() -> None:
    with pytest.raises(C.ConventionError, match="quaternion"):
        C.quat_wxyz_to_xyzw(np.zeros(3))


def test_the_pose_dwell_is_a_sane_number_of_ticks() -> None:
    """Reproduce: .venv_mb/bin/python -m pytest tests/test_conventions.py -k pose_dwell -v"""
    from openroboxing.spec.constants import POSE_DWELL_TICKS, TICK_HZ

    assert isinstance(POSE_DWELL_TICKS, int)
    assert 0 < POSE_DWELL_TICKS <= 2 * TICK_HZ, "a dwell longer than 2 s is a different game"


def test_the_generator_pose_tolerance_is_in_radians() -> None:
    import math

    from openroboxing.spec.constants import GENERATOR_POSE_TOLERANCE_RAD

    assert 0.0 < GENERATOR_POSE_TOLERANCE_RAD < math.radians(30.0)
