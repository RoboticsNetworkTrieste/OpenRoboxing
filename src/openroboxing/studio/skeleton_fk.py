"""Skeleton forward kinematics: 29 joint angles → the generator's 34 global joint transforms.

This is the step that lets an authored :class:`~openroboxing.studio.pose_record.PoseRecord` drive
generation. Patch P0 (`spec/upstream_patches.md`) accepts ``specific_target_joint_positions`` and
``specific_target_joint_rotations``; this module produces them.

We do not implement forward kinematics
--------------------------------------
Upstream already has it: :class:`motionbricks.helper.mujoco_helper.mujoco_qpos_converter` turns a
36-dim MuJoCo ``qpos`` into ``g1skel34`` global transforms, and it is the *same* call the agent makes
on its own context frames (``full_agent.py:182``). Reusing it means an authored pose lands in exactly
the space the model was trained on, and it keeps `CLAUDE.md` invariant 3 intact. A reimplementation
would have to rediscover the joint-axis resolution, the rest-pose body quaternions, the z-up→y-up
change of basis and the dead-joint fill — four chances to be silently wrong.

Two different robots
--------------------
The converter is built against :data:`~openroboxing.paths.GENERATOR_SKELETON_XML`, **not** the deploy
model. They are different revisions of the G1: the waist and shoulder offsets differ by 9–19 mm, so
their Cartesian frames disagree by up to ~9 mm at the same joint angles (measured; see
``spec/upstream_notes.md`` §Skeleton). The rule that follows is simple and this module holds to it:

- anything the **generator consumes** is computed on the generator's model — that is this file;
- anything describing the **physical robot** is computed on the deploy model — that is
  ``studio/telegraph.py``.

Joint angles themselves carry across exactly, which is why the discrepancy is cosmetic rather than
fatal.

Conventions
-----------
- **Input** is a ``(N, 36)`` qpos in MuJoCo order and MuJoCo frame: 3 root position, 4 root
  quaternion ``wxyz``, 29 joints.
- **Output** is in *motion* space — y-up, z-forward — with 34 joints in ``g1skel34`` order. This is
  the generator's frame, not MuJoCo's; the change of basis is
  ``motion = M @ mujoco``, ``M = [[0,1,0],[0,0,1],[1,0,0]]``.
- Rotations are ``3x3`` matrices, never quaternions, matching the tensors P0 replaces.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import numpy as np

from openroboxing.paths import (
    GENERATOR_SKELETON_DIR,
    GENERATOR_SKELETON_XML,
    MOTIONBRICKS_ROOT,
)
from openroboxing.runtime.conventions import G1, G1Conventions
from openroboxing.spec.constants import NUM_FRAMES_PER_TOKEN, QPOS_DIM

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openroboxing.studio.pose_record import PoseRecord

#: Skeleton joints the generator carries but the 29-DOF robot does not drive. Their transforms are
#: filled by upstream's dead-joint scheme, not by us.
NUM_SKELETON_JOINTS = 34

#: Change of basis, MuJoCo (z-up, x-forward) → motion (y-up, z-forward). Mirrors
#: ``mujoco_qpos_converter.mujoco_to_motion_matrix``; asserted equal at construction.
MUJOCO_TO_MOTION = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


class SkeletonFKError(RuntimeError):
    """Forward kinematics could not be built or run. Never recovered from silently."""


def _import_motionbricks():
    """Import the upstream pieces, with an error that says what is missing rather than ImportError."""
    if str(MOTIONBRICKS_ROOT) not in sys.path:
        sys.path.insert(0, str(MOTIONBRICKS_ROOT))
    try:
        from motionbricks.helper.mujoco_helper import mujoco_qpos_converter
        from motionbricks.motionlib.core.skeletons.g1 import G1Skeleton34
        from motionbricks.motionlib.core.utils.rotations import angle_to_Y_rotation_matrix
        import torch
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SkeletonFKError(
            f"cannot import MotionBricks from {MOTIONBRICKS_ROOT}: {exc}"
        ) from exc
    return torch, mujoco_qpos_converter, G1Skeleton34, angle_to_Y_rotation_matrix


class _SkeletonOnly:
    """The only attribute :class:`mujoco_qpos_converter` needs from a motion representation.

    Passing the real one would mean loading a checkpoint onto a GPU. The converter touches
    ``motion_rep.skeleton`` and nothing else on the paths used here, so a skeleton is enough — and
    that keeps forward kinematics runnable on a laptop with no weights present.
    """

    def __init__(self, skeleton) -> None:
        self.skeleton = skeleton


class SkeletonFK:
    """Forward kinematics onto the generator's ``g1skel34`` skeleton.

    Construction parses the generator's MJCF and loads the skeleton's rest offsets; it needs no
    checkpoint, no GPU and no network. Use :func:`skeleton_fk` for the shared instance.
    """

    def __init__(
        self,
        xml_path: Path = GENERATOR_SKELETON_XML,
        skeleton_dir: Path = GENERATOR_SKELETON_DIR,
        conventions: G1Conventions = G1,
    ) -> None:
        for path, what in ((xml_path, "generator MJCF"), (skeleton_dir, "skeleton rest pose")):
            if not Path(path).exists():
                raise SkeletonFKError(f"{what} not found at {path}")

        torch, converter_class, skeleton_class, _ = _import_motionbricks()
        self._torch = torch
        self.xml_path = Path(xml_path)

        # Before anything else: the converter reads its joint order out of its own MJCF, and would
        # raise a bare KeyError deep inside its constructor if a name did not resolve. Checking here
        # turns that into a message that says which joint and why it matters.
        self._check_joint_order(conventions)

        skeleton = skeleton_class(folder=str(skeleton_dir), name="g1skel34", t_pose="capture")
        # Built directly rather than through `get_mujoco_converter`, which caches one converter in a
        # module-level global: constructing ours through it would hand our XML to the next agent that
        # asks for one.
        self._converter = converter_class(_SkeletonOnly(skeleton), str(xml_path))
        self._skeleton = skeleton

        self.joint_names: tuple[str, ...] = tuple(skeleton.bone_order_names)
        self._check_agrees_with()

    def _check_joint_order(self, conventions: G1Conventions) -> None:
        """Assert the qpos we are handed is the qpos this converter expects.

        If the generator's joint order ever diverges from the order our qpos vectors are built in,
        every joint is silently misassigned — `CLAUDE.md` invariant 4. Checked by name, at
        construction, so it cannot go wrong at run time.
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(self.xml_path)
        hinge = [j.get("name") for j in tree.getroot().find("worldbody").findall(".//joint")]
        if hinge != list(conventions.mujoco_joint_names):
            differing = [
                f"[{i}] generator={a!r} ours={b!r}"
                for i, (a, b) in enumerate(zip(hinge, conventions.mujoco_joint_names))
                if a != b
            ] or [f"generator has {len(hinge)} joints, we have {len(conventions.mujoco_joint_names)}"]
            raise SkeletonFKError(
                f"{self.xml_path} orders its joints differently from our MuJoCo convention, so a "
                f"qpos built here would be scrambled: {differing[:5]}"
            )

    def _check_agrees_with(self) -> None:
        """Assert the skeleton and change of basis are the ones this module documents."""
        if len(self.joint_names) != NUM_SKELETON_JOINTS:
            raise SkeletonFKError(
                f"expected {NUM_SKELETON_JOINTS} skeleton joints, got {len(self.joint_names)}"
            )
        if not np.allclose(self._converter.mujoco_to_motion_matrix.numpy(), MUJOCO_TO_MOTION):
            raise SkeletonFKError(
                "upstream's mujoco→motion change of basis is not the one this module documents"
            )

    # -- the transform ----------------------------------------------------------------------------
    def transforms(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Global joint transforms for a qpos stream.

        Args:
            qpos: ``(N, 36)`` MuJoCo order and frame.

        Returns:
            ``(positions, rotations)`` of shape ``(N, 34, 3)`` and ``(N, 34, 3, 3)``, in motion
            space.
        """
        arr = np.asarray(qpos, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != QPOS_DIM:
            raise SkeletonFKError(f"expected (N, {QPOS_DIM}) qpos, got shape {arr.shape}")

        with self._torch.no_grad():
            positions, rotations = self._converter.convert_mujoco_qpos_to_motion_transforms(
                self._torch.from_numpy(arr[None, ...])
            )
        positions = positions[0].numpy().astype(np.float64)
        rotations = rotations[0].numpy().astype(np.float64)

        if not (np.isfinite(positions).all() and np.isfinite(rotations).all()):
            raise SkeletonFKError("forward kinematics produced a non-finite transform")
        return positions, rotations

    def pose_qpos(self, record: PoseRecord, conventions: G1Conventions = G1) -> np.ndarray:
        """The record's angles as a ``(36,)`` qpos: pelvis at the origin, upright, heading zero.

        A pose record describes a *body configuration*; where that body stands is placement's job
        (`spec/pose_record.md`). The root is therefore neutral here, and
        :meth:`target_transforms` puts the pose where the generator decided it goes.
        """
        qpos = np.zeros(QPOS_DIM)
        qpos[3] = 1.0  # identity quaternion, wxyz
        qpos[7:] = record.to_array(conventions)
        return qpos

    def pose_transforms(
        self, record: PoseRecord, conventions: G1Conventions = G1
    ) -> tuple[np.ndarray, np.ndarray]:
        """The record's transforms with the pelvis at the origin. ``(34, 3)`` and ``(34, 3, 3)``."""
        positions, rotations = self.transforms(self.pose_qpos(record, conventions))
        return positions[0], rotations[0]

    # -- what patch P0 consumes -------------------------------------------------------------------
    def target_transforms(
        self,
        record: PoseRecord,
        current_positions,
        current_rotations,
        conventions: G1Conventions = G1,
    ):
        """Replace the clip-sampled target pose with an authored one, keeping its placement.

        Patch P0 runs *after* ``_generate_target_joint_transforms``, so the tensors handed in are
        already placed: the spring model chose where the fighter will be and which way it will face,
        and ``_override_target_transforms`` may have overridden that with an explicit target. None of
        that is the pose's business, so this keeps it and swaps only the body.

        **Heading only.** The pose is re-rooted onto the target's position and *yaw*, discarding the
        sampled pelvis pitch and roll. Those come from whichever clip frame the seed happened to
        land on, so honouring them would make an authored pose render differently run to run. A lean
        is expressible through the three waist joints, which the record does own. This mirrors
        ``compute_apply_delta_heading`` in the bridge, which is yaw-only for the same reason.

        Args:
            record: the authored pose.
            current_positions: ``[batch, frames, 34, 3]`` — ``input['target_global_joint_positions']``.
            current_rotations: ``[batch, frames, 34, 3, 3]`` — the matching rotations.

        Returns:
            ``(positions, rotations)`` as torch tensors shaped and typed like the inputs, ready to
            assign back into the input dict.
        """
        torch, _, _, angle_to_Y_rotation_matrix = _import_motionbricks()

        if current_positions.shape[-2:] != (NUM_SKELETON_JOINTS, 3):
            raise SkeletonFKError(
                f"current_positions must end in ({NUM_SKELETON_JOINTS}, 3), got "
                f"{tuple(current_positions.shape)}"
            )
        if current_rotations.shape[-3:] != (NUM_SKELETON_JOINTS, 3, 3):
            raise SkeletonFKError(
                f"current_rotations must end in ({NUM_SKELETON_JOINTS}, 3, 3), got "
                f"{tuple(current_rotations.shape)}"
            )

        device, dtype = current_positions.device, current_positions.dtype
        positions, rotations = self.pose_transforms(record, conventions)
        pose_pos = torch.as_tensor(positions, device=device, dtype=dtype)
        pose_rot = torch.as_tensor(rotations, device=device, dtype=dtype)

        # Relative to the pelvis, so the pose can be carried to wherever the root goes.
        pose_pos = pose_pos - pose_pos[0:1]

        root_rot = current_rotations[..., 0, :, :]  # [batch, frames, 3, 3]
        # y-axis rotation of the target root, read the way upstream reads it (full_agent.py:431).
        heading = torch.atan2(root_rot[..., 0, 2], root_rot[..., 2, 2])
        yaw = angle_to_Y_rotation_matrix(heading).to(device=device, dtype=dtype)

        placed_positions = current_positions[..., 0:1, :] + torch.matmul(
            yaw[..., None, :, :], pose_pos[..., None]
        )[..., 0]
        placed_rotations = torch.matmul(yaw[..., None, :, :], pose_rot)

        return placed_positions, placed_rotations

    def target_transforms_for_frames(
        self,
        record: PoseRecord,
        root_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        root_heading: float = 0.0,
        frames: int = NUM_FRAMES_PER_TOKEN,
        conventions: G1Conventions = G1,
    ):
        """Standalone target tensors, for driving the patch without a live agent.

        Placement is given explicitly here rather than inherited from a generated target, which is
        what the tools and tests need. Arguments are in **MuJoCo** convention — ``root_position`` is
        ``(x, y, z)`` with z up, ``root_heading`` is yaw about z — because that is what a caller
        holding a robot state has; the change of basis happens inside.

        Returns:
            ``(positions, rotations)`` of shape ``[1, frames, 34, 3]`` and ``[1, frames, 34, 3, 3]``.
        """
        torch, _, _, _ = _import_motionbricks()

        if frames < 1:
            raise SkeletonFKError(f"frames must be at least 1, got {frames}")

        qpos = self.pose_qpos(record, conventions)
        qpos[0:3] = root_position
        half = 0.5 * float(root_heading)
        qpos[3:7] = (np.cos(half), 0.0, 0.0, np.sin(half))  # wxyz, yaw about MuJoCo's z

        positions, rotations = self.transforms(qpos)  # (1, 34, 3) — one frame in, one frame out
        # A key pose is one configuration held across the token, so every frame is the same pose.
        positions = np.repeat(positions[None, 0:1, ...], frames, axis=1)
        rotations = np.repeat(rotations[None, 0:1, ...], frames, axis=1)
        return (
            torch.from_numpy(positions).float(),
            torch.from_numpy(rotations).float(),
        )


@lru_cache(maxsize=1)
def skeleton_fk(
    xml_path: Path = GENERATOR_SKELETON_XML, skeleton_dir: Path = GENERATOR_SKELETON_DIR
) -> SkeletonFK:
    """The shared :class:`SkeletonFK`. Parsing the MJCF twice is pure waste; it never changes."""
    return SkeletonFK(xml_path, skeleton_dir)
