"""CSV corpus ingest: units, ordering, and the invertibility invariant."""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.paths import MOTIONS_DIR
from openroboxing.runtime.conventions import G1
from openroboxing.studio import motion_import

TAKE = MOTIONS_DIR / "ib_dodge_up_R_001__A437.csv"


def test_reads_header_and_frames():
    take = motion_import.read_take(TAKE)
    assert take.joint_names == tuple(n[: -len("_dof")] for n in take.raw_joint_columns)
    assert len(take.joint_names) == G1.num_joints
    # 6 root columns + 29 joints. The `Frame` column is dropped, so this is 35, not 36 —
    # the 36 of a qpos is 3 position + 4 quaternion + 29, which `load_take` produces.
    assert take.frames.shape == (591, 35)


def test_joint_permutation_is_invertible():
    take = motion_import.read_take(TAKE)
    perm = motion_import.joint_permutation(take.joint_names)
    x = np.arange(G1.num_joints, dtype=float)
    assert np.array_equal(x[perm][motion_import.invert(perm)], x)


def test_joint_permutation_rejects_a_missing_name():
    names = ("not_a_joint",) + tuple(G1.mujoco_joint_names[1:])
    with pytest.raises(motion_import.MotionImportError, match="not_a_joint"):
        motion_import.joint_permutation(names)


def test_joint_permutation_rejects_a_duplicate():
    names = (G1.mujoco_joint_names[0],) + tuple(G1.mujoco_joint_names[:-1])
    with pytest.raises(motion_import.MotionImportError):
        motion_import.joint_permutation(names)


def test_load_take_shape_and_units():
    qpos = motion_import.load_take(TAKE)
    assert qpos.shape == (591, 36)
    # Pelvis height: 50-107 cm in the corpus becomes 0.50-1.08 m here.
    assert 0.4 < qpos[:, 2].min() < 0.6
    assert 1.0 < qpos[:, 2].max() < 1.2
    # Joints are radians: the corpus's worst magnitude was 157 deg = 2.74 rad.
    assert np.abs(qpos[:, 7:]).max() < np.pi


def test_root_quaternion_is_unit_and_wxyz():
    qpos = motion_import.load_take(TAKE)
    quat = qpos[:, 3:7]
    assert np.allclose(np.linalg.norm(quat, axis=1), 1.0)
    # wxyz, not xyzw: the pelvis is near upright, so |w| dominates for a boxer.
    assert np.abs(quat[:, 0]).mean() > np.abs(quat[:, 1]).mean()


def test_joints_land_in_mujoco_order():
    take = motion_import.read_take(TAKE)
    qpos = motion_import.load_take(TAKE)
    corpus_index = take.joint_names.index("right_elbow_joint")
    mujoco_index = G1.mujoco_joint_names.index("right_elbow_joint")
    # take.frames has already dropped `Frame`, so joint j sits at column 6 + j.
    assert np.allclose(qpos[:, 7 + mujoco_index], np.radians(take.frames[:, 6 + corpus_index]))


def test_recovered_yaw_is_the_corpus_heading_channel():
    """EULER_ORDER was chosen because this holds; assert it so a corpus swap is caught."""
    from scipy.spatial.transform import Rotation

    take = motion_import.read_take(TAKE)
    qpos = motion_import.load_take(TAKE)
    matrices = Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_matrix()
    yaw = np.unwrap(np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]))
    heading = np.unwrap(np.radians(take.frames[:, 5]))
    residual = np.degrees(yaw - heading)
    assert np.abs(residual - residual.mean()).max() < 1e-6


def test_every_take_fits_the_robots_joint_limits():
    """The corpus is in the robot's own joint convention - a flipped sign would break this."""
    import mujoco

    from openroboxing.paths import G1_29DOF_SIM_XML

    model = mujoco.MjModel.from_xml_path(str(G1_29DOF_SIM_XML))
    limits = {}
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(m=model, type=mujoco.mjtObj.mjOBJ_JOINT, id=joint)
        if name in G1.mujoco_joint_names and model.jnt_limited[joint]:
            limits[name] = tuple(model.jnt_range[joint])

    assert len(limits) == G1.num_joints
    for path in sorted(MOTIONS_DIR.glob("*.csv")):
        qpos = motion_import.load_take(path)
        for i, name in enumerate(G1.mujoco_joint_names):
            low, high = limits[name]
            column = qpos[:, 7 + i]
            assert column.min() >= low - 0.02, f"{path.name} {name} below {low}"
            assert column.max() <= high + 0.02, f"{path.name} {name} above {high}"
