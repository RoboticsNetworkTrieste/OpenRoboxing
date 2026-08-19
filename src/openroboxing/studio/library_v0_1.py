"""The authored pose library, v0.1 (M2-T5).

Poses are defined here as *where the hands go*, and the joint angles are solved
(:mod:`openroboxing.studio.pose_ik`). Two reasons, both learned the hard way:

- The G1's joint conventions are not guessable. Its elbow is **negative** in a boxing guard, the
  opposite of the intuition a human boxer brings. Hand-written angles produced poses that rendered
  as waving.
- The arm is short. The hand reaches at most **0.38 m** forward of the pelvis, so a boxer's sense of
  distance does not transfer and targets have to be set against measured numbers.

The guard is not authored at all: it is taken from the ``walk_boxing`` clip, which is real boxing
motion and therefore the most defensible statement of what this robot's guard looks like. Every
strike is solved as a displacement of that guard's hands, with the guard as the solver's reference so
the redundant shoulder resolves to something a boxer would recognise.

`CLAUDE.md` standing rule 3 is satisfied throughout: the guard is measured, the strikes are solved
from stated targets, and the horizons are the one free choice — declared as such below.

Conventions
-----------
Hand targets are pelvis-relative in the robot's frame: ``+x`` forward, ``+y`` the robot's left,
``+z`` up. The guard's hands sit at about ``fwd +0.27, up +0.27``; the head is around ``up +0.45``.
"""

from __future__ import annotations

from openroboxing.runtime.conventions import G1
from openroboxing.spec.constants import MIN_TOKENS
from openroboxing.studio.pose_ik import ArmTarget, solve_arm

LIBRARY_VERSION = "v0.1"

#: The guard, read off frame 323 of `walk_boxing` (seed 1234) — see `poses/candidates`. Taken from
#: the clip rather than authored: this is what the model's own boxing looks like, so a fighter that
#: returns to it looks like it belongs in the motion it is generated from.
GUARD_ARMS: dict[str, float] = {
    "left_shoulder_pitch_joint": -1.08,
    "left_shoulder_roll_joint": 0.34,
    "left_shoulder_yaw_joint": -0.47,
    "left_elbow_joint": -0.57,
    "right_shoulder_pitch_joint": -1.08,
    "right_shoulder_roll_joint": -0.34,
    "right_shoulder_yaw_joint": 0.47,
    "right_elbow_joint": -0.57,
}

#: A settled stance under the guard. Small: the legs carry balance, and the policy has to keep the
#: robot upright while the arms do the work.
GUARD_STANCE: dict[str, float] = {
    "left_knee_joint": 0.30,
    "right_knee_joint": 0.30,
    "left_hip_pitch_joint": -0.18,
    "right_hip_pitch_joint": -0.18,
    "left_ankle_pitch_joint": -0.12,
    "right_ankle_pitch_joint": -0.12,
}

GUARD: dict[str, float] = {**GUARD_ARMS, **GUARD_STANCE}

#: Where each strike puts its hand, and what the rest of the body does. Targets are set against the
#: measured envelope (0.38 m of forward reach), so "fully extended" means 0.34, not a boxer's arm.
STRIKES: dict[str, tuple[str, ArmTarget, dict[str, float]]] = {
    # A straight left: the hand goes out and slightly in toward the centre line, at chin height.
    "jab-left": ("left", ArmTarget(forward=0.34, left=0.02, up=0.30), {"waist_yaw_joint": -0.25}),
    # A left hook: the hand crosses the centre line, the waist turns into it. The turn is where a
    # hook's power comes from, so it is part of the pose rather than something the policy invents.
    "hook-left": (
        "left",
        ArmTarget(forward=0.24, left=-0.10, up=0.31),
        {"waist_yaw_joint": -0.50, "waist_roll_joint": 0.15},
    ),
    # A right uppercut: the only strike with a vertical line, so the only one that beats a high
    # guard. Ends high and close, with the torso extending under it.
    "uppercut-right": (
        "right",
        ArmTarget(forward=0.22, left=-0.05, up=0.40),
        {"waist_yaw_joint": 0.40, "waist_pitch_joint": -0.22},
    ),
}

#: Defensive shapes, which move the body rather than a hand and so are stated as angles.
#: A slip leans off the centre line without the feet moving; a cover gives up punching to take one.
DEFENCES: dict[str, dict[str, float]] = {
    "slip-left": {
        "waist_yaw_joint": 0.65,
        "waist_roll_joint": -0.35,
        "waist_pitch_joint": -0.22,
        "left_knee_joint": 0.52,
        "right_knee_joint": 0.42,
    },
    "cover": {
        "left_shoulder_pitch_joint": -1.45,
        "left_shoulder_roll_joint": 0.15,
        "left_shoulder_yaw_joint": -0.35,
        "left_elbow_joint": -0.30,
        "right_shoulder_pitch_joint": -1.45,
        "right_shoulder_roll_joint": -0.15,
        "right_shoulder_yaw_joint": 0.35,
        "right_elbow_joint": -0.30,
        "waist_pitch_joint": -0.18,
    },
}

#: How long each move runs. **The one genuinely free choice in this file**, and the main lever on how
#: a move feels: a jab is the shortest the model will produce, a hook and an uppercut travel further,
#: defence is quick. First choices, to be tuned against measured telegraph windows — that tuning is
#: what `tools/build_library.py --measure` exists to inform.
HORIZONS: dict[str, int] = {
    "guard": 8,
    "jab-left": MIN_TOKENS,
    "jab-right": MIN_TOKENS,
    "hook-left": 8,
    "hook-right": 8,
    "uppercut-right": 8,
    "uppercut-left": 8,
    "slip-left": MIN_TOKENS,
    "slip-right": MIN_TOKENS,
    "cover": MIN_TOKENS,
}


#: How far a player may steer each pose live, per joint, in radians. Deliberately small and only on
#: the joints that *aim* a strike — the envelope's corners must be admitted too, so every radian here
#: costs measurement (`spec/intent.md` §Feasibility). Defence has no envelope: a slip is a commitment,
#: and letting it be steered mid-flight would make it a dodge with no cost.
ENVELOPES: dict[str, dict[str, float]] = {
    "jab-left": {"left_shoulder_pitch_joint": 0.20, "waist_yaw_joint": 0.20},
    "jab-right": {"right_shoulder_pitch_joint": 0.20, "waist_yaw_joint": 0.20},
    "hook-left": {"left_shoulder_pitch_joint": 0.20, "waist_yaw_joint": 0.20},
    "hook-right": {"right_shoulder_pitch_joint": 0.20, "waist_yaw_joint": 0.20},
    "uppercut-left": {"left_shoulder_pitch_joint": 0.20, "waist_pitch_joint": 0.15},
    "uppercut-right": {"right_shoulder_pitch_joint": 0.20, "waist_pitch_joint": 0.15},
}


def mirror(pose: dict[str, float]) -> dict[str, float]:
    """Swap left and right, negating the joints whose sign is handed.

    ``roll`` and ``yaw`` axes point the same way for both arms, so mirroring negates them; ``pitch``
    axes are shared and keep their sign. Waist roll and yaw flip for the same reason. Done here
    rather than by hand so a left/right pair cannot silently drift apart.
    """
    flipped: dict[str, float] = {}
    for joint, value in pose.items():
        if joint.startswith("left_"):
            name = "right_" + joint[len("left_") :]
        elif joint.startswith("right_"):
            name = "left_" + joint[len("right_") :]
        else:
            name = joint
        flipped[name] = -value if ("roll" in joint or "yaw" in joint) else value

    unknown = set(flipped) - set(G1.mujoco_joint_names)
    if unknown:
        raise ValueError(f"mirroring produced unknown joints: {sorted(unknown)}")
    return flipped


def build() -> dict[str, dict[str, float]]:
    """Solve the library. Returns name → joint-angle overrides on the default pose."""
    poses: dict[str, dict[str, float]] = {"guard": dict(GUARD)}

    for name, (side, target, body) in STRIKES.items():
        arm = solve_arm(target, side, reference=GUARD_ARMS)
        poses[name] = {**GUARD, **arm, **body}
        # Every strike gets its mirror, so a loadout can be built either-handed.
        mirrored = "-".join(
            {"left": "right", "right": "left"}.get(part, part) for part in name.split("-")
        )
        poses[mirrored] = mirror(poses[name])

    for name, angles in DEFENCES.items():
        poses[name] = {**GUARD, **angles}
        if name.endswith("-left"):
            poses[name.replace("-left", "-right")] = mirror(poses[name])

    missing = set(poses) - set(HORIZONS)
    if missing:
        raise ValueError(f"no horizon declared for {sorted(missing)}")
    return poses
