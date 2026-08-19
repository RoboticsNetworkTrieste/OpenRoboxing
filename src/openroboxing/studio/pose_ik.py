"""Solve arm joint angles from a wanted hand position (M2-T5).

Authoring a boxing pose by typing shoulder angles does not work: the sign conventions are not
guessable, and the G1's arm is short enough that intuition from a human boxer is misleading — the
hand reaches at most **0.30 m** forward of the pelvis, measured. So poses are stated as *where the
hand should be* and the angles are solved. That also satisfies `CLAUDE.md` standing rule 3: the
numbers in the library come from a target and a solver, not from taste.

This is a four-DOF-per-arm problem (shoulder pitch/roll/yaw, elbow) and deliberately a small,
dependency-light solve: a coarse grid to find the basin, then Nelder–Mead to polish. Redundancy is
resolved by a regulariser pulling toward the guard, so a solution stays a *boxing* pose rather than
whatever contortion happens to touch the target.

Conventions
-----------
- Targets are **pelvis-relative, in the robot's own frame**: ``+x`` forward, ``+y`` the robot's left,
  ``+z`` up. This is MuJoCo's frame with the root at the origin and no yaw.
- Angles come back keyed by MuJoCo joint **name**, ready to merge into a pose record.
- Reachability is checked and a miss **raises**: silently returning the closest possible pose would
  put an unreachable target into the library with no sign that it was never met.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from openroboxing.paths import G1_29DOF_XML
from openroboxing.runtime.conventions import G1
from openroboxing.spec.constants import QPOS_DIM

#: Joints the solver may move, per side. Wrists are excluded: they barely move the wrist-yaw link
#: and would only add redundancy for the solver to abuse.
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")

#: How hard the solver is pulled back toward the reference pose, per radian of deviation, in metres.
#: Large enough to pick a natural solution out of the redundancy, small enough not to bias the hand
#: away from its target: at 0.02, a whole radian of extra deviation costs the same as 20 mm of miss.
REGULARISER = 0.02

#: A solve must land this close, in metres, or it raises. About a finger's width — tighter than the
#: 2-3 deg the generator will reproduce the pose to anyway.
TOLERANCE_M = 0.015


class PoseIKError(RuntimeError):
    """A hand target could not be reached. Never silently approximated."""


@dataclass(frozen=True)
class ArmTarget:
    """Where a hand should be, pelvis-relative in the robot's frame."""

    forward: float
    left: float
    up: float

    def as_array(self) -> np.ndarray:
        return np.array([self.forward, self.left, self.up])


@lru_cache(maxsize=1)
def _model_and_data():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(G1_29DOF_XML))
    return mujoco, model, mujoco.MjData(model)


@lru_cache(maxsize=4)
def _hand_body(side: str) -> int:
    mujoco, model, _ = _model_and_data()
    name = f"{side}_wrist_yaw_link"
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        raise PoseIKError(f"body {name!r} is not in {G1_29DOF_XML.name}")
    return body


def joint_names(side: str) -> tuple[str, ...]:
    """The four joints this solver moves for one arm, as MuJoCo names."""
    if side not in ("left", "right"):
        raise PoseIKError(f"side must be 'left' or 'right', got {side!r}")
    names = tuple(f"{side}_{joint}_joint" for joint in ARM_JOINTS)
    unknown = [n for n in names if n not in G1.mujoco_joint_names]
    if unknown:
        raise PoseIKError(f"joints not in the model: {unknown}")
    return names


def hand_position(angles: dict[str, float], side: str) -> np.ndarray:
    """Where the hand is, pelvis-relative, for a full or partial set of joint angles."""
    from openroboxing.runtime.obs import default_angles

    mujoco, model, data = _model_and_data()
    qpos = np.zeros(QPOS_DIM)
    qpos[3] = 1.0
    qpos[7:] = default_angles(G1, "mujoco")
    for joint, value in angles.items():
        qpos[7 + G1.mujoco_joint_names.index(joint)] = value

    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return data.xpos[_hand_body(side)] - data.xpos[pelvis]


def _limits(names: tuple[str, ...]) -> np.ndarray:
    mujoco, model, _ = _model_and_data()
    bounds = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        bounds.append(model.jnt_range[jid] if model.jnt_limited[jid] else (-np.pi, np.pi))
    return np.array(bounds)


def reach_envelope(side: str = "left", samples: int = 12) -> dict[str, float]:
    """The furthest the hand goes in each direction. Reported so targets can be set honestly."""
    names = joint_names(side)
    bounds = _limits(names)
    grids = [np.linspace(lo, hi, samples) for lo, hi in bounds]
    best = {"forward": -np.inf, "up": -np.inf, "reach": -np.inf}
    for pitch in grids[0]:
        for roll in grids[1]:
            for elbow in grids[3]:
                pos = hand_position(
                    dict(zip(names, (pitch, roll, 0.0, elbow))), side
                )
                best["forward"] = max(best["forward"], float(pos[0]))
                best["up"] = max(best["up"], float(pos[2]))
                best["reach"] = max(best["reach"], float(np.linalg.norm(pos)))
    return best


def solve_arm(
    target: ArmTarget,
    side: str,
    *,
    reference: dict[str, float] | None = None,
    tolerance_m: float = TOLERANCE_M,
    regulariser: float = REGULARISER,
) -> dict[str, float]:
    """Angles putting one hand at ``target``.

    Args:
        reference: the pose the solver is pulled back toward when the target underdetermines the
            arm — normally the guard, so a strike stays recognisably a boxer's.

    Raises:
        PoseIKError: if the closest reachable hand position misses by more than ``tolerance_m``. The
            message reports where the hand got to, so a target can be moved somewhere reachable.
    """
    from scipy.optimize import minimize

    names = joint_names(side)
    bounds = _limits(names)
    goal = target.as_array()
    anchor = np.array([(reference or {}).get(name, 0.0) for name in names])

    def cost(values: np.ndarray) -> float:
        clipped = np.clip(values, bounds[:, 0], bounds[:, 1])
        miss = np.linalg.norm(hand_position(dict(zip(names, clipped)), side) - goal)
        penalty = regulariser * float(np.abs(clipped - anchor).sum())
        # Leaving the limits must cost more than any pose inside them can save.
        excursion = float(np.abs(values - clipped).sum())
        return miss + penalty + 10.0 * excursion

    # Coarse grid first: the shoulder is a 3-sphere of solutions and gradient-free polishing from a
    # single guess lands in whichever basin it started in.
    grids = [np.linspace(lo, hi, 7) for lo, hi in bounds]
    best_start, best_cost = anchor, np.inf
    for pitch in grids[0]:
        for roll in grids[1]:
            for elbow in grids[3]:
                start = np.array([pitch, roll, anchor[2], elbow])
                value = cost(start)
                if value < best_cost:
                    best_start, best_cost = start, value

    result = minimize(cost, best_start, method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 4000})
    angles = np.clip(result.x, bounds[:, 0], bounds[:, 1])
    solution = dict(zip(names, (float(a) for a in angles)))

    reached = hand_position(solution, side)
    miss = float(np.linalg.norm(reached - goal))
    if miss > tolerance_m:
        raise PoseIKError(
            f"{side} hand cannot reach fwd{goal[0]:+.2f} left{goal[1]:+.2f} up{goal[2]:+.2f}: "
            f"closest is fwd{reached[0]:+.2f} left{reached[1]:+.2f} up{reached[2]:+.2f}, "
            f"{miss * 1e3:.0f} mm short of the {tolerance_m * 1e3:.0f} mm tolerance"
        )
    return solution
