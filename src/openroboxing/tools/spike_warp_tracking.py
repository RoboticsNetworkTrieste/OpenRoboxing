"""Measure whether a warped combination is trackable at all (M5-T12).

The design's phase-1 checkpoint, and not a formality. Poses authored in joint space are reached to
2-3 degrees when they are the target of an unhurried plan. A combination asks for something harder:
a **forced** leg as short as ``MIN_TOKENS``, aimed at an authored pose, while the root is dragged
along a drift toward the ghost. That combination of constraints is unmeasured, and phases 2-4 are
built on it.

This drives the generator directly — no physics, no policy. It answers *"does MotionBricks reach the
pose it is aimed at, on time, under a drift"*, which is the question phase 2 needs settled. Tracking
the result under physics is a separate question and belongs to phase 3.

Two placements are measured per combination:

- ``recorded`` — the ghost sits exactly where the recording ends, so the residual drift is zero and
  the fighter is asked only for the motion the take contains. This isolates *"can it reach the
  pose"*.
- ``drifted`` — the ghost is pushed further along the same direction, so every leg carries drift on
  top of the recorded footwork. The difference between the two columns is the price of the drift.

Run: ``.venv_mb/bin/python -m openroboxing.tools.spike_warp_tracking``
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from openroboxing.paths import COMBINATION_DIR
from openroboxing.runtime.conventions import G1
from openroboxing.runtime.generator import GeneratorIntent, MotionBricksGenerator
from openroboxing.runtime.warp import Leg, WarpError, warp
from openroboxing.spec.constants import GENERATOR_DT, NUM_FRAMES_PER_TOKEN
from openroboxing.studio import combination_record as cr
from openroboxing.studio.pose_record import PoseRecord

#: One combination of each character, so a failure says which kind failed. Travel differs by an
#: order of magnitude between them, which is exactly what the drift column is probing.
DEFAULT_PREFIXES = ("shadow-boxing", "ib-dodge", "ib-combat-turn-jog")

#: How far past the recorded endpoint the ``drifted`` placement puts the ghost, in metres.
DRIFT_EXTRA_M = 1.0


def pick_records(prefixes: tuple[str, ...]) -> list[cr.CombinationRecord]:
    """One record per prefix, chosen deterministically so the spike is repeatable."""
    files = sorted(COMBINATION_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"no combinations in {COMBINATION_DIR}; run tools.import_motions first")
    chosen = []
    for prefix in prefixes:
        match = next((p for p in files if p.stem.startswith(prefix)), None)
        if match is None:
            raise SystemExit(f"no combination starting {prefix!r} in {COMBINATION_DIR}")
        chosen.append(cr.load(match))
    return chosen


def leg_pose(leg: Leg, name: str) -> PoseRecord:
    """A leg's target as the ``PoseRecord`` the override consumes.

    ``generator._install_pose_override`` hands ``intent.pose`` to
    ``skeleton_fk.target_transforms``, which reads a ``PoseRecord``, not a bare mapping.
    """
    return PoseRecord(
        name=name,
        joint_angles=dict(leg.joint_angles),
        horizon_tokens=leg.horizon_tokens,
        library_version="v0.2",
    )


def run_combination(
    generator: MotionBricksGenerator,
    record: cr.CombinationRecord,
    ghost: tuple[float, float],
    *,
    seed: int,
) -> list[tuple[int, int, float, int]]:
    """Drive one combination leg by leg. Returns ``(leg, tokens, worst error deg, frames)``."""
    generator.reset(seed)
    legs = warp(record, (0.0, 0.0), 0.0, ghost)
    rows = []
    for index, leg in enumerate(legs):
        intent = GeneratorIntent(
            style="walk_boxing",
            movement_angle=leg.movement_angle,
            facing_angle=leg.facing_angle,
            target_position=leg.target_position,
            target_heading=leg.target_heading,
            pose=leg_pose(leg, f"{record.name}-leg{index}"),
            horizon_tokens=leg.horizon_tokens,
        )
        # force=True so each leg plans immediately rather than waiting for the ambient cadence:
        # this measures the plan, not the replan schedule.
        generator.generate(intent, generator.context_qpos(), GENERATOR_DT, force=True)
        plan = generator.plan()
        wanted = np.array([leg.joint_angles[n] for n in G1.mujoco_joint_names])
        error = float(np.degrees(np.abs(plan[-1, 7:] - wanted)).max())
        rows.append((index, leg.horizon_tokens, error, len(plan)))
        # Consume the plan so the next leg's context is where this one left off, which is what a
        # real sequence would see.
        for _ in range(len(plan)):
            generator.next_frame()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefixes", nargs="*", default=list(DEFAULT_PREFIXES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drift-extra", type=float, default=DRIFT_EXTRA_M)
    args = parser.parse_args()

    records = pick_records(tuple(args.prefixes))
    generator = MotionBricksGenerator()

    print(
        f"{'combination':30s} {'placement':>9s} {'leg':>4s} {'tok':>4s} "
        f"{'pose err deg':>13s} {'frames':>7s} {'asked':>6s}"
    )
    summary: dict[str, list[float]] = {}
    for record in records:
        dx, dy = record.recorded_displacement
        length = math.hypot(dx, dy)
        direction = (dx / length, dy / length) if length > 1e-6 else (1.0, 0.0)
        placements = {
            "recorded": (dx, dy),
            "drifted": (
                dx + direction[0] * args.drift_extra,
                dy + direction[1] * args.drift_extra,
            ),
        }
        for label, ghost in placements.items():
            try:
                rows = run_combination(generator, record, ghost, seed=args.seed)
            except WarpError as exc:
                print(f"{record.name[:30]:30s} {label:>9s}  unreachable: {exc}")
                continue
            for index, tokens, error, frames in rows:
                asked = tokens * NUM_FRAMES_PER_TOKEN
                flag = "" if frames == asked else "  <- plan length differs"
                print(
                    f"{record.name[:30]:30s} {label:>9s} {index:4d} {tokens:4d} "
                    f"{error:13.1f} {frames:7d} {asked:6d}{flag}"
                )
            summary.setdefault(label, []).extend(r[2] for r in rows)

    print("\nworst-joint error at the plan endpoint, degrees:")
    for label, errors in summary.items():
        arr = np.array(errors)
        print(
            f"  {label:>9s}  n={len(arr):3d}  median {np.median(arr):6.1f}  "
            f"mean {arr.mean():6.1f}  worst {arr.max():6.1f}"
        )
    print(
        "\nReading: authored poses are reached to 2-3 deg when they are the target of an unhurried\n"
        "plan. Errors far above that on the shortest legs mean the forced-leg assumption does not\n"
        "hold, and D2's leg length or D4's drift is what to revisit - not the design."
    )


if __name__ == "__main__":
    main()
