"""Ingest the mocap corpus in ``motions/`` (M5-T2).

Implements the input half of ``spec/combination.md`` 0.1.

Conventions
-----------
- **Input** is a Maya-style CSV export: ``Frame``, ``root_translate{X,Y,Z}`` in **centimetres**,
  ``root_rotate{X,Y,Z}`` in **degrees**, then 29 ``<joint>_dof`` columns in **degrees**. Z is up,
  and the corpus is sampled at :data:`~openroboxing.spec.constants.GENERATOR_HZ` (30 fps).
- **Output** is ``(N, 36)`` MuJoCo qpos: root position (3, **metres**), root quaternion
  (4, **wxyz**), 29 joint angles (**radians**) in **MuJoCo joint order**.
- The joint permutation is derived **by name** and asserted invertible. The corpus order happens to
  match MuJoCo's — measured 2026-08-27 — and is never assumed to (`CLAUDE.md` invariant 4).
- Nothing is coerced and nothing is defaulted: a column that cannot be placed raises.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openroboxing.runtime.conventions import G1, G1Conventions

#: Suffix the corpus puts on every joint column.
JOINT_SUFFIX = "_dof"

#: The root columns, in the order the corpus writes them.
ROOT_COLUMNS = (
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
)

#: Centimetres to metres. The corpus writes pelvis height as 50-107, which is centimetres.
CM_TO_M = 0.01

#: How ``root_rotate{X,Y,Z}`` compose, in scipy's spelling: lower case is extrinsic, so the Z
#: rotation is applied last in world.
#:
#: **Measured, not assumed** — see ``tools/pin_euler_order.py``. Determined 2026-08-27 as the unique
#: convention whose recovered yaw *is* the corpus's own heading channel ``root_rotateZ``, to 0.00
#: degrees over all 38 takes; every other candidate deviates by 2.1-2.9 degrees. ``"ZYX"`` scores
#: identically because it is the same rotation spelled intrinsically.
#:
#: It agrees with the corpus's provenance: Maya's default rotate order XYZ composes as
#: ``Rz . Ry . Rx``, which is exactly scipy's extrinsic ``"xyz"``.
#:
#: This affects the root quaternion only, never the joint angles. Re-run the tool if the corpus is
#: replaced.
EULER_ORDER = "xyz"

#: Value columns per row once ``Frame`` is dropped: 6 root + 29 joints. Not ``QPOS_DIM``, which is
#: 36 because a qpos carries a 4-component quaternion where the corpus carries 3 Euler angles.
CORPUS_COLUMNS = len(ROOT_COLUMNS) + G1.num_joints


class MotionImportError(RuntimeError):
    """A take could not be read or placed. Never recovered from silently."""


@dataclass(frozen=True)
class Take:
    """One mocap take, as written, before any conversion."""

    name: str
    raw_joint_columns: tuple[str, ...]
    joint_names: tuple[str, ...]
    frames: np.ndarray  # (N, 35): 6 root columns then 29 joints, in the corpus's own order


def invert(perm: np.ndarray) -> np.ndarray:
    """The inverse of a permutation, as an index array."""
    out = np.empty_like(perm)
    out[perm] = np.arange(len(perm))
    return out


def joint_permutation(
    joint_names: tuple[str, ...], conventions: G1Conventions = G1
) -> np.ndarray:
    """Indices that gather corpus joint order into MuJoCo joint order.

    Raises:
        MotionImportError: if a MuJoCo joint has no corpus column, or a name repeats.
    """
    if len(set(joint_names)) != len(joint_names):
        raise MotionImportError(f"corpus joint names repeat: {joint_names}")
    index = {name: i for i, name in enumerate(joint_names)}
    missing = [n for n in conventions.mujoco_joint_names if n not in index]
    if missing:
        mujoco_names = set(conventions.mujoco_joint_names)
        unrecognised = [n for n in joint_names if n not in mujoco_names]
        raise MotionImportError(
            f"corpus has no column for MuJoCo joints {missing}; "
            f"unrecognised corpus columns: {unrecognised}"
        )
    perm = np.array([index[n] for n in conventions.mujoco_joint_names], dtype=int)
    if not np.array_equal(perm[invert(perm)], np.arange(len(perm))):
        raise MotionImportError("joint permutation is not invertible")
    return perm


def read_take(path: Path) -> Take:
    """Read a corpus CSV as written. No units are converted here."""
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise MotionImportError(f"{path} has no frames")
    header = rows[0]
    if tuple(header[1:7]) != ROOT_COLUMNS:
        raise MotionImportError(f"{path} root columns are {header[1:7]}, expected {ROOT_COLUMNS}")
    raw_joint_columns = tuple(header[7:])
    bad = [c for c in raw_joint_columns if not c.endswith(JOINT_SUFFIX)]
    if bad:
        raise MotionImportError(f"{path} joint columns without a {JOINT_SUFFIX!r} suffix: {bad}")
    joint_names = tuple(c[: -len(JOINT_SUFFIX)] for c in raw_joint_columns)
    if len(joint_names) != G1.num_joints:
        raise MotionImportError(
            f"{path} has {len(joint_names)} joint columns, expected {G1.num_joints}"
        )
    frames = np.array([[float(v) for v in row[1:]] for row in rows[1:]], dtype=np.float64)
    if frames.shape[1] != CORPUS_COLUMNS:
        raise MotionImportError(
            f"{path} has {frames.shape[1]} value columns, expected {CORPUS_COLUMNS}"
        )
    return Take(
        name=path.stem,
        raw_joint_columns=raw_joint_columns,
        joint_names=joint_names,
        frames=frames,
    )
