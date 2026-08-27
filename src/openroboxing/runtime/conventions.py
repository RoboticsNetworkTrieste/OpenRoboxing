"""MuJoCo ↔ IsaacLab index mappings for the 29-DOF G1, derived by name.

``CLAUDE.md`` invariant 4: every index mapping is derived from the authoritative name lists and
asserted invertible at construction. Never hard-code a permutation, never copy one from
documentation. This is the single most likely source of a silent catastrophic bug — a wrong
permutation does not crash, it quietly drives the wrong joints.

The gather convention — read this before using anything here
------------------------------------------------------------
Both this module and the C++ reference use the **gather** convention. A permutation named
``a_to_b`` is indexed by the **destination** (``b``) slot and holds the **source** (``a``) index::

    mujoco_vec[i]  = isaac_vec[ISAACLAB_TO_MUJOCO[i]]
    isaac_vec[i]   = mujoco_vec[MUJOCO_TO_ISAACLAB[i]]

which in numpy is simply ``isaac_vec = mujoco_vec[MUJOCO_TO_ISAACLAB]``.

The intuitive reading — "``isaaclab_to_mujoco[isaac_index]`` gives me the MuJoCo index" — is the
**inverse** of what these arrays hold, and using it silently scrambles every joint. Verified against
the reference's own usage at ``g1_deploy_onnx_ref.cpp:3120`` (actions, IsaacLab → motor order) and
``:2827`` (joint state, motor → IsaacLab order).

Sources of truth
----------------
- **MuJoCo joint/body order**: read from the MJCF at import via ``mujoco.mj_id2name``.
- **IsaacLab joint order**: ``gear_sonic.envs.env_utils.joint_utils.G1_ISAACLab_ORDER``. That module
  imports only ``torch``, so it works without Isaac Lab installed.
- **IsaacLab body order**: derived structurally — IsaacLab enumerates articulation links
  **breadth-first** from the root, where MuJoCo is depth-first. ``gear_sonic/.../robots/g1.py`` holds
  equivalent arrays but cannot be imported without Isaac Lab.

Both mappings are cross-checked three ways at import: name-derived vs. breadth-first-derived vs. the
C++ arrays recorded in ``spec/rates.md``. Disagreement raises.

Quaternions
-----------
MuJoCo is ``wxyz``; most other code is ``xyzw``. Helpers are provided; they are pure reorderings and
do not normalise or otherwise touch the values.
"""

from __future__ import annotations

import collections
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from openroboxing.paths import G1_29DOF_XML
from openroboxing.spec.constants import NUM_JOINTS

# The MJCF's free joint for the floating base. It is not an actuated DOF and is excluded from the
# 29-joint vectors the policy consumes.
_FLOATING_BASE_JOINT = "floating_base_joint"

# MuJoCo body 0 is always the static `world` body; the articulation root (pelvis) is body 1.
_WORLD_BODY_ID = 0

# --- Assertion targets, NOT sources of truth ----------------------------------------------------
# Transcribed from gear_sonic_deploy/.../include/policy_parameters.hpp:100-104 and recorded in
# spec/rates.md. The mappings below are *derived*; these exist only so a derivation that disagrees
# with the deployed C++ fails loudly at import instead of producing plausible, wrong motion.
_CPP_ISAACLAB_TO_MUJOCO = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)
_CPP_MUJOCO_TO_ISAACLAB = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)


class ConventionError(RuntimeError):
    """A mapping could not be derived, or a derivation disagreed with its cross-check."""


def _invert(perm: tuple[int, ...]) -> tuple[int, ...]:
    """Inverse of a permutation, under the gather convention."""
    out = [0] * len(perm)
    for dst, src in enumerate(perm):
        out[src] = dst
    return tuple(out)


def _check_permutation(perm: tuple[int, ...], n: int, what: str) -> None:
    if len(perm) != n:
        raise ConventionError(f"{what}: expected length {n}, got {len(perm)}")
    if sorted(perm) != list(range(n)):
        dupes = [i for i, c in collections.Counter(perm).items() if c > 1]
        raise ConventionError(
            f"{what}: not a permutation of range({n}); duplicated indices {dupes or '<none>'}"
        )


def _mujoco_joint_names(model: mujoco.MjModel) -> list[str]:
    """Actuated joint names in MuJoCo order, excluding the floating base."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    return [n for n in names if n and n != _FLOATING_BASE_JOINT]


def _mujoco_body_names(model: mujoco.MjModel) -> list[str]:
    """Body names in MuJoCo order, excluding the static `world` body."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    return [n for n in names[1:] if n]


def _breadth_first_body_order(model: mujoco.MjModel) -> tuple[int, ...]:
    """Articulation bodies in breadth-first order, as MuJoCo indices relative to the root.

    IsaacLab enumerates links breadth-first from the articulation root; MuJoCo stores them
    depth-first. Returns the gather array for MuJoCo → IsaacLab.
    """
    children: dict[int, list[int]] = collections.defaultdict(list)
    for body in range(1, model.nbody):
        children[int(model.body_parentid[body])].append(body)

    order: list[int] = []
    queue = collections.deque([_WORLD_BODY_ID + 1])
    while queue:
        body = queue.popleft()
        order.append(body)
        queue.extend(children[body])
    return tuple(b - 1 for b in order)


def _load_isaaclab_joint_names() -> list[str]:
    """The authoritative IsaacLab joint-name order.

    Imported from ``gear_sonic.envs.env_utils.joint_utils``, which depends only on ``torch``. The
    richer ``robots/g1.py`` cannot be used: it imports ``isaaclab``.

    ``gear_sonic`` is an upstream package living at ``GR00T_ROOT/gear_sonic/``, not inside
    OpenRoboxing, so it is never on ``sys.path`` by default — put it there explicitly. This used to
    work by accident: before the extraction, OpenRoboxing lived *inside* GR00T-WholeBodyControl, so
    ``gear_sonic/`` sat next to the code as a sibling top-level package, and running from the
    repository root put it on ``sys.path`` for free. The split broke that silently. The fix mirrors
    ``runtime/generator.py``'s shim for ``motionbricks`` — same idea, different root: ``motionbricks``
    lives at ``MOTIONBRICKS_ROOT/motionbricks/`` so ``MOTIONBRICKS_ROOT`` goes on the path, but
    ``gear_sonic`` lives directly at ``GR00T_ROOT/gear_sonic/`` so it is ``GR00T_ROOT`` itself that
    has to go on the path here.

    ``GR00T_ROOT`` is imported inside this function, not at module scope, so a test that reloads
    ``openroboxing.paths`` (to point ``OPENROBOXING_GR00T_ROOT`` elsewhere) is picked up on the next
    call rather than seeing whatever value happened to be current when this module was first
    imported.
    """
    from openroboxing.paths import GR00T_ROOT

    if str(GR00T_ROOT) not in sys.path:
        sys.path.insert(0, str(GR00T_ROOT))

    try:
        from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise ConventionError(
            "cannot import G1_ISAACLab_ORDER from gear_sonic.envs.env_utils.joint_utils "
            f"under {GR00T_ROOT}; is the GR00T-WholeBodyControl checkout complete?"
        ) from exc
    return list(G1_ISAACLab_ORDER)


@dataclass(frozen=True)
class G1Conventions:
    """Name-derived index mappings for one G1 model.

    All arrays follow the gather convention documented in the module docstring.
    """

    mujoco_joint_names: tuple[str, ...]
    isaaclab_joint_names: tuple[str, ...]
    mujoco_body_names: tuple[str, ...]

    #: gather array: ``isaac_joint_vec = mujoco_joint_vec[mujoco_to_isaaclab]``
    mujoco_to_isaaclab: tuple[int, ...]
    #: gather array: ``mujoco_joint_vec = isaac_joint_vec[isaaclab_to_mujoco]``
    isaaclab_to_mujoco: tuple[int, ...]

    #: gather array over bodies: ``isaac_body_arr = mujoco_body_arr[mujoco_to_isaaclab_body]``
    mujoco_to_isaaclab_body: tuple[int, ...]
    isaaclab_to_mujoco_body: tuple[int, ...]

    @property
    def num_joints(self) -> int:
        return len(self.mujoco_joint_names)

    @property
    def num_bodies(self) -> int:
        return len(self.mujoco_body_names)

    @property
    def isaaclab_body_names(self) -> tuple[str, ...]:
        """Body names in IsaacLab order."""
        return tuple(self.mujoco_body_names[i] for i in self.mujoco_to_isaaclab_body)

    # -- joint vectors ---------------------------------------------------------------------------
    def to_isaaclab(self, x: np.ndarray) -> np.ndarray:
        """Reorder a MuJoCo-ordered joint array into IsaacLab order along the last axis."""
        return self._gather(x, self.mujoco_to_isaaclab, self.num_joints, "joint")

    def to_mujoco(self, x: np.ndarray) -> np.ndarray:
        """Reorder an IsaacLab-ordered joint array into MuJoCo order along the last axis."""
        return self._gather(x, self.isaaclab_to_mujoco, self.num_joints, "joint")

    # -- body arrays -----------------------------------------------------------------------------
    def bodies_to_isaaclab(self, x: np.ndarray) -> np.ndarray:
        """Reorder a MuJoCo-ordered body array into IsaacLab order along axis ``-1`` or ``-2``.

        Accepts ``(..., num_bodies)`` or ``(..., num_bodies, k)`` — the latter covers per-body
        vectors such as positions ``(30, 3)`` and quaternions ``(30, 4)``.
        """
        return self._gather_bodies(x, self.mujoco_to_isaaclab_body)

    def bodies_to_mujoco(self, x: np.ndarray) -> np.ndarray:
        """Reorder an IsaacLab-ordered body array into MuJoCo order."""
        return self._gather_bodies(x, self.isaaclab_to_mujoco_body)

    def _gather(self, x: np.ndarray, perm: tuple[int, ...], n: int, what: str) -> np.ndarray:
        arr = np.asarray(x)
        if arr.shape[-1] != n:
            raise ConventionError(f"expected {n} {what}s on the last axis, got shape {arr.shape}")
        return arr[..., np.asarray(perm)]

    def _gather_bodies(self, x: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
        arr = np.asarray(x)
        n = self.num_bodies
        idx = np.asarray(perm)
        if arr.shape[-1] == n:
            return arr[..., idx]
        if arr.ndim >= 2 and arr.shape[-2] == n:
            return arr[..., idx, :]
        raise ConventionError(f"expected {n} bodies on axis -1 or -2, got shape {arr.shape}")


def build_conventions(
    xml_path: Path | str = G1_29DOF_XML,
    *,
    isaaclab_joint_names: list[str] | None = None,
    verify_against_cpp: bool = True,
) -> G1Conventions:
    """Derive the mappings from a MuJoCo model and validate them.

    Args:
        xml_path: MJCF to read the MuJoCo joint/body order from.
        isaaclab_joint_names: override the IsaacLab joint order. Used by tests to prove that a
            mismatched name list is rejected; leave ``None`` in production.
        verify_against_cpp: cross-check the joint mapping against the arrays deployed in the C++
            reference. Only meaningful for the 29-DOF G1.

    Raises:
        ConventionError: on any name-set mismatch, duplicate, non-permutation, non-invertible
            mapping, or disagreement with a cross-check.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    mj_joints = _mujoco_joint_names(model)
    isaac_joints = (
        list(isaaclab_joint_names)
        if isaaclab_joint_names is not None
        else _load_isaaclab_joint_names()
    )

    if len(set(mj_joints)) != len(mj_joints):
        dupes = [n for n, c in collections.Counter(mj_joints).items() if c > 1]
        raise ConventionError(f"duplicate MuJoCo joint names: {dupes}")
    if len(set(isaac_joints)) != len(isaac_joints):
        dupes = [n for n, c in collections.Counter(isaac_joints).items() if c > 1]
        raise ConventionError(f"duplicate IsaacLab joint names: {dupes}")

    missing = set(isaac_joints) - set(mj_joints)
    extra = set(mj_joints) - set(isaac_joints)
    if missing or extra:
        raise ConventionError(
            "MuJoCo and IsaacLab joint name sets differ; "
            f"missing from MuJoCo: {sorted(missing) or '<none>'}; "
            f"absent from IsaacLab: {sorted(extra) or '<none>'}"
        )

    mj_index = {name: i for i, name in enumerate(mj_joints)}
    mujoco_to_isaaclab = tuple(mj_index[name] for name in isaac_joints)
    isaaclab_to_mujoco = _invert(mujoco_to_isaaclab)

    n_joints = len(mj_joints)
    _check_permutation(mujoco_to_isaaclab, n_joints, "mujoco_to_isaaclab")
    _check_permutation(isaaclab_to_mujoco, n_joints, "isaaclab_to_mujoco")
    for k in range(n_joints):
        if mujoco_to_isaaclab[isaaclab_to_mujoco[k]] != k:
            raise ConventionError(f"joint mapping is not invertible at index {k}")

    mj_bodies = _mujoco_body_names(model)
    mujoco_to_isaaclab_body = _breadth_first_body_order(model)
    isaaclab_to_mujoco_body = _invert(mujoco_to_isaaclab_body)
    n_bodies = len(mj_bodies)
    _check_permutation(mujoco_to_isaaclab_body, n_bodies, "mujoco_to_isaaclab_body")
    _check_permutation(isaaclab_to_mujoco_body, n_bodies, "isaaclab_to_mujoco_body")

    if verify_against_cpp:
        _verify_against_cpp(mujoco_to_isaaclab, isaaclab_to_mujoco, n_joints)
        _verify_joint_order_is_breadth_first(model, mj_joints, mujoco_to_isaaclab)

    return G1Conventions(
        mujoco_joint_names=tuple(mj_joints),
        isaaclab_joint_names=tuple(isaac_joints),
        mujoco_body_names=tuple(mj_bodies),
        mujoco_to_isaaclab=mujoco_to_isaaclab,
        isaaclab_to_mujoco=isaaclab_to_mujoco,
        mujoco_to_isaaclab_body=mujoco_to_isaaclab_body,
        isaaclab_to_mujoco_body=isaaclab_to_mujoco_body,
    )


def _verify_against_cpp(
    mujoco_to_isaaclab: tuple[int, ...],
    isaaclab_to_mujoco: tuple[int, ...],
    n_joints: int,
) -> None:
    """Fail if the derived joint mapping differs from the deployed C++ arrays."""
    if n_joints != NUM_JOINTS:
        raise ConventionError(
            f"C++ cross-check is defined for {NUM_JOINTS} joints, model has {n_joints}; "
            "pass verify_against_cpp=False for a different embodiment"
        )
    if mujoco_to_isaaclab != _CPP_MUJOCO_TO_ISAACLAB:
        raise ConventionError(
            "derived mujoco_to_isaaclab disagrees with policy_parameters.hpp; "
            f"derived={mujoco_to_isaaclab} cpp={_CPP_MUJOCO_TO_ISAACLAB}"
        )
    if isaaclab_to_mujoco != _CPP_ISAACLAB_TO_MUJOCO:
        raise ConventionError(
            "derived isaaclab_to_mujoco disagrees with policy_parameters.hpp; "
            f"derived={isaaclab_to_mujoco} cpp={_CPP_ISAACLAB_TO_MUJOCO}"
        )


def _verify_joint_order_is_breadth_first(
    model: mujoco.MjModel, mj_joints: list[str], mujoco_to_isaaclab: tuple[int, ...]
) -> None:
    """Second, independent derivation: IsaacLab joint order is breadth-first over the tree.

    Agreeing with the name-derived mapping means the two disagree only if both the name list and the
    kinematic tree changed consistently — which is the point of a cross-check.
    """
    body_rank = {body: rank for rank, body in enumerate(_breadth_first_body_order(model))}
    joint_body: dict[str, int] = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and name != _FLOATING_BASE_JOINT:
            joint_body[name] = int(model.jnt_bodyid[jid]) - 1

    derived = tuple(
        sorted(range(len(mj_joints)), key=lambda j: body_rank[joint_body[mj_joints[j]]])
    )
    if derived != mujoco_to_isaaclab:
        raise ConventionError(
            "breadth-first joint derivation disagrees with the name-derived mapping; "
            f"bfs={derived} name={mujoco_to_isaaclab}"
        )


# Built at import so a bad model or a renamed joint fails immediately and loudly.
G1: G1Conventions = build_conventions()


def to_isaaclab(x: np.ndarray) -> np.ndarray:
    """Reorder a MuJoCo-ordered joint array into IsaacLab order (default 29-DOF G1)."""
    return G1.to_isaaclab(x)


def to_mujoco(x: np.ndarray) -> np.ndarray:
    """Reorder an IsaacLab-ordered joint array into MuJoCo order (default 29-DOF G1)."""
    return G1.to_mujoco(x)


# --- Quaternion conventions ---------------------------------------------------------------------
def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """MuJoCo ``wxyz`` → ``xyzw``, on the last axis. Pure reordering; no normalisation."""
    arr = np.asarray(q)
    if arr.shape[-1] != 4:
        raise ConventionError(f"expected a quaternion on the last axis, got shape {arr.shape}")
    return arr[..., [1, 2, 3, 0]]


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """``xyzw`` → MuJoCo ``wxyz``, on the last axis. Pure reordering; no normalisation."""
    arr = np.asarray(q)
    if arr.shape[-1] != 4:
        raise ConventionError(f"expected a quaternion on the last axis, got shape {arr.shape}")
    return arr[..., [3, 0, 1, 2]]


def quat_wxyz_to_yaw(quat: np.ndarray) -> float:
    """Heading in radians from a MuJoCo ``wxyz`` quaternion: rotation about world Z.

    The full extraction, not the pure-yaw shortcut a product of two yaw-only quaternions allows
    (``runtime.fight.apply_yaw``): valid for any orientation, including a body that is also pitched
    or rolled, because it reads only the Z-axis component of the rotation rather than assuming there
    is no other one.

    Shared so a live fighter's heading (``runtime.fight``) and a recorded take's heading
    (``studio.combination_record``) are read off a quaternion by the identical formula — two
    independent derivations of the same convention are exactly how a sign or axis error would go
    unnoticed (`CLAUDE.md`: most bugs here are convention bugs).
    """
    arr = np.asarray(quat)
    if arr.shape[-1] != 4:
        raise ConventionError(f"expected a quaternion on the last axis, got shape {arr.shape}")
    w, x, y, z = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
