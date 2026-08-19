"""Skeleton forward kinematics: 29 joint angles → the generator's 34 global joint transforms.

The load-bearing test is :func:`test_agrees_with_mujoco_forward_kinematics`: upstream's converter and
MuJoCo's own solver are two independent implementations, so their agreement on the *same* model is
real evidence rather than a restatement of one of them.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_skeleton_fk.py -v
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from openroboxing.paths import GENERATOR_SKELETON_XML
from openroboxing.runtime.conventions import G1
from openroboxing.spec.constants import NUM_FRAMES_PER_TOKEN, NUM_JOINTS, QPOS_DIM
from openroboxing.studio.pose_record import PoseRecord, PoseSource
from openroboxing.studio.skeleton_fk import (
    MUJOCO_TO_MOTION,
    NUM_SKELETON_JOINTS,
    SkeletonFK,
    SkeletonFKError,
    skeleton_fk,
)

pytest.importorskip("mujoco")
pytest.importorskip("torch")


@pytest.fixture(scope="module")
def fk() -> SkeletonFK:
    return skeleton_fk()


def _record(**overrides) -> PoseRecord:
    from openroboxing.runtime.obs import default_angles

    angles = dict(zip(G1.mujoco_joint_names, default_angles(G1, "mujoco")))
    base = {
        "name": "guard-high",
        "joint_angles": angles,
        "horizon_tokens": 8,
        "library_version": "v0.1",
        "source": PoseSource(clip="walk_boxing", start_frame=25, end_frame=35),
    }
    base.update(overrides)
    return PoseRecord(**base)


def _mujoco_kinematics_only():
    """The generator's MJCF as a MuJoCo model.

    It ships without its meshes, so geoms and assets are stripped. Only the kinematic tree — bodies,
    joints, offsets, rest quaternions — participates in forward kinematics, so nothing under test is
    removed. `default` stays: bodies reference it through `childclass`.
    """
    import mujoco

    tree = ET.parse(GENERATOR_SKELETON_XML)
    root = tree.getroot()
    for tag in ("asset", "contact", "actuator", "sensor", "keyframe"):
        for element in root.findall(tag):
            root.remove(element)
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ("geom", "site", "camera", "light"):
                parent.remove(child)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


# --- the acceptance criterion ---------------------------------------------------------------------
def test_agrees_with_mujoco_forward_kinematics(fk) -> None:
    """Upstream's converter and MuJoCo's solver must agree on the same model.

    Compared at the joint anchors, which both implementations define identically, across a range of
    joint amplitudes so the check covers more than the rest pose.
    """
    import mujoco

    model = _mujoco_kinematics_only()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)

    worst = 0.0
    for amplitude in (0.0, 0.1, 0.3, 0.8):
        angles = rng.uniform(-amplitude, amplitude, size=NUM_JOINTS)
        qpos = np.zeros(QPOS_DIM)
        qpos[3] = 1.0
        qpos[7:] = angles

        positions, _ = fk.transforms(qpos)

        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        compared = 0
        for index, skeleton_name in enumerate(fk.joint_names):
            robot = (
                skeleton_name[:-5] + "_joint"
                if skeleton_name.endswith("_skel")
                else skeleton_name
            )
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, robot)
            if joint_id < 0:  # a skeleton joint the robot does not have
                continue
            expected = MUJOCO_TO_MOTION @ data.xanchor[joint_id]
            worst = max(worst, float(np.abs(positions[0, index] - expected).max()))
            compared += 1
        assert compared == NUM_JOINTS, f"only cross-checked {compared} of {NUM_JOINTS} joints"

    assert worst < 1e-6, f"upstream FK and MuJoCo FK disagree by {worst:.3e} m"


def test_produces_the_shapes_patch_p0_requires(fk) -> None:
    positions, rotations = fk.target_transforms_for_frames(_record())
    assert tuple(positions.shape) == (1, NUM_FRAMES_PER_TOKEN, NUM_SKELETON_JOINTS, 3)
    assert tuple(rotations.shape) == (1, NUM_FRAMES_PER_TOKEN, NUM_SKELETON_JOINTS, 3, 3)


# --- the transform ---------------------------------------------------------------------------------
def test_shapes_and_finiteness(fk) -> None:
    qpos = np.zeros((5, QPOS_DIM))
    qpos[:, 3] = 1.0
    positions, rotations = fk.transforms(qpos)
    assert positions.shape == (5, NUM_SKELETON_JOINTS, 3)
    assert rotations.shape == (5, NUM_SKELETON_JOINTS, 3, 3)
    assert np.isfinite(positions).all() and np.isfinite(rotations).all()


def test_a_single_frame_may_be_passed_unbatched(fk) -> None:
    qpos = np.zeros(QPOS_DIM)
    qpos[3] = 1.0
    positions, rotations = fk.transforms(qpos)
    assert positions.shape == (1, NUM_SKELETON_JOINTS, 3)
    assert rotations.shape == (1, NUM_SKELETON_JOINTS, 3, 3)


def test_rotations_are_rotation_matrices(fk) -> None:
    rng = np.random.default_rng(3)
    qpos = np.zeros(QPOS_DIM)
    qpos[3] = 1.0
    qpos[7:] = rng.uniform(-0.5, 0.5, size=NUM_JOINTS)
    _, rotations = fk.transforms(qpos)

    identity = np.einsum("jab,jcb->jac", rotations[0], rotations[0])
    assert np.allclose(identity, np.eye(3), atol=1e-5), "rotations are not orthonormal"
    assert np.allclose(np.linalg.det(rotations[0]), 1.0, atol=1e-5), "rotations are not proper"


def test_root_translation_moves_every_joint_equally(fk) -> None:
    base = np.zeros(QPOS_DIM)
    base[3] = 1.0
    moved = base.copy()
    moved[0:3] = (1.0, 2.0, 3.0)

    at_origin, _ = fk.transforms(base)
    displaced, _ = fk.transforms(moved)
    offset = displaced[0] - at_origin[0]

    assert np.allclose(offset, offset[0], atol=1e-5), "translation is not rigid"
    assert np.allclose(offset[0], MUJOCO_TO_MOTION @ np.array([1.0, 2.0, 3.0]), atol=1e-5)


def test_bad_qpos_shape_raises(fk) -> None:
    with pytest.raises(SkeletonFKError, match="expected"):
        fk.transforms(np.zeros((4, 20)))


# --- construction asserts what it depends on ---------------------------------------------------------
def test_joint_order_mismatch_raises(tmp_path) -> None:
    """The check that stops a scrambled qpos: rename a joint and construction must fail."""
    tree = ET.parse(GENERATOR_SKELETON_XML)
    joints = tree.getroot().find("worldbody").findall(".//joint")
    joints[0].set("name", "not_the_joint_you_are_looking_for")
    broken = tmp_path / "renamed.xml"
    tree.write(broken)

    with pytest.raises(SkeletonFKError, match="orders its joints differently"):
        SkeletonFK(xml_path=broken)


def test_missing_model_raises(tmp_path) -> None:
    with pytest.raises(SkeletonFKError, match="generator MJCF not found"):
        SkeletonFK(xml_path=tmp_path / "absent.xml")


def test_missing_skeleton_rest_pose_raises(tmp_path) -> None:
    with pytest.raises(SkeletonFKError, match="skeleton rest pose not found"):
        SkeletonFK(skeleton_dir=tmp_path / "absent")


def test_skeleton_has_the_joints_the_record_expands_to(fk) -> None:
    from openroboxing.studio.pose_record import to_skeleton_angles

    expanded = to_skeleton_angles(_record(), list(fk.joint_names))
    assert set(expanded) == set(fk.joint_names)
    assert len(fk.joint_names) == NUM_SKELETON_JOINTS


# --- pose → qpos -------------------------------------------------------------------------------------
def test_pose_qpos_carries_the_authored_angles_and_a_neutral_root(fk) -> None:
    record = _record()
    qpos = fk.pose_qpos(record)
    assert np.allclose(qpos[7:], record.to_array())
    assert np.allclose(qpos[0:3], 0.0), "the pose must not carry a position; placement owns it"
    assert np.allclose(qpos[3:7], (1.0, 0.0, 0.0, 0.0)), "the pelvis must be upright and unrotated"


def test_different_poses_give_different_transforms(fk) -> None:
    reach = _record()
    angles = dict(reach.joint_angles)
    angles["left_elbow_joint"] += 1.0
    extended = _record(joint_angles=angles)

    a, _ = fk.pose_transforms(reach)
    b, _ = fk.pose_transforms(extended)
    assert np.abs(a - b).max() > 0.05, "bending an elbow by 1 rad must move the arm"


# --- placement -----------------------------------------------------------------------------------------
def test_placement_puts_the_pelvis_where_it_was_asked(fk) -> None:
    positions, _ = fk.target_transforms_for_frames(_record(), root_position=(1.0, 2.0, 0.8))
    pelvis = positions[0, 0, 0].numpy()
    assert np.allclose(pelvis, MUJOCO_TO_MOTION @ np.array([1.0, 2.0, 0.8]), atol=1e-5)


def test_every_frame_of_a_key_pose_is_the_same_pose(fk) -> None:
    positions, rotations = fk.target_transforms_for_frames(_record())
    assert np.allclose(positions[0, 0].numpy(), positions[0, -1].numpy())
    assert np.allclose(rotations[0, 0].numpy(), rotations[0, -1].numpy())


def test_heading_rotates_the_pose_about_the_up_axis(fk) -> None:
    """A half-turn must negate the horizontal axes and leave height alone (motion space is y-up)."""
    straight, _ = fk.target_transforms_for_frames(_record(), root_heading=0.0)
    turned, _ = fk.target_transforms_for_frames(_record(), root_heading=np.pi)

    a = straight[0, 0].numpy()
    b = turned[0, 0].numpy()
    assert np.allclose(a[:, 1], b[:, 1], atol=1e-5), "a yaw must not change any joint's height"
    assert np.allclose(a[:, [0, 2]], -b[:, [0, 2]], atol=1e-5)


def test_frames_must_be_positive(fk) -> None:
    with pytest.raises(SkeletonFKError, match="frames must be at least 1"):
        fk.target_transforms_for_frames(_record(), frames=0)


# --- overriding a generated target ------------------------------------------------------------------------
def _target_stub(fk, heading: float = 0.0, pelvis_pitch: float = 0.0, frames: int = 4):
    """Stand-in for `input['target_global_joint_*']`: a placed pose, optionally pitched."""
    import torch

    positions, rotations = fk.target_transforms_for_frames(
        _record(), root_position=(3.0, 1.0, 0.75), root_heading=heading, frames=frames
    )
    if pelvis_pitch:
        cos, sin = np.cos(pelvis_pitch), np.sin(pelvis_pitch)
        # About x. Motion space is y-up, so a rotation about y would be a *yaw* — which the override
        # is supposed to keep. This has to be a genuine lean for the test to mean anything.
        lean = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]], dtype=rotations.dtype
        )
        rotations = torch.matmul(lean, rotations)
    return positions, rotations


def test_override_keeps_the_placement_it_was_given(fk) -> None:
    current_positions, current_rotations = _target_stub(fk, heading=0.7)
    positions, rotations = fk.target_transforms(_record(), current_positions, current_rotations)

    assert positions.shape == current_positions.shape
    assert rotations.shape == current_rotations.shape
    assert np.allclose(
        positions[..., 0, :].numpy(), current_positions[..., 0, :].numpy(), atol=1e-5
    ), "the pelvis moved; placement must be preserved exactly"


def test_override_replaces_the_body(fk) -> None:
    current_positions, current_rotations = _target_stub(fk, heading=0.4)

    angles = dict(_record().joint_angles)
    angles["right_elbow_joint"] += 1.2
    punch = _record(joint_angles=angles)

    positions, _ = fk.target_transforms(punch, current_positions, current_rotations)
    assert (
        np.abs(positions.numpy() - current_positions.numpy()).max() > 0.05
    ), "the authored body did not replace the sampled one"


def test_override_is_heading_only(fk) -> None:
    """A sampled pelvis lean must not change the authored *pose* — the lean is seed-dependent.

    It does shift the pose's yaw a little, and that is not a bug to fix: the heading comes from
    ``atan2(R[0,2], R[2,2])``, which is upstream's own definition (``full_agent.py:431``), and it
    genuinely reads slightly differently off a tilted frame. Adopting a lean-invariant definition
    instead would rotate our pose away from where the generator believes the fighter faces.

    So the assertion is on the property that matters — the body configuration is untouched, and the
    residual is a rigid yaw. Pairwise distances pin the shape; heights pin the axis.
    """
    upright = _target_stub(fk, heading=0.4, pelvis_pitch=0.0)
    pitched = _target_stub(fk, heading=0.4, pelvis_pitch=0.35)

    a = fk.target_transforms(_record(), *upright)[0].numpy()[0, 0]
    b = fk.target_transforms(_record(), *pitched)[0].numpy()[0, 0]

    distances_a = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    distances_b = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=-1)
    assert np.allclose(
        distances_a, distances_b, atol=1e-6
    ), "the sampled pelvis lean deformed the authored pose"
    assert np.allclose(a[:, 1], b[:, 1], atol=1e-6), "the residual is not a rotation about up"


def test_override_rejects_a_wrongly_shaped_target(fk) -> None:
    import torch

    positions, rotations = _target_stub(fk)
    with pytest.raises(SkeletonFKError, match="current_positions must end"):
        fk.target_transforms(_record(), torch.zeros(1, 4, 29, 3), rotations)
    with pytest.raises(SkeletonFKError, match="current_rotations must end"):
        fk.target_transforms(_record(), positions, torch.zeros(1, 4, 29, 3, 3))
