"""Run one G1 under physics, tracking a generated reference motion (M1-T6).

Acceptance criterion from WORKPLAN.md M1-T6:
  `python -m openroboxing.tools.run_single --style walk_boxing --seconds 30` runs a G1 that walks and
  shadow-boxes under physics without falling, and writes a run log with tracking error per body.

Usage
-----
    python -m openroboxing.tools.run_single --style walk_boxing --seconds 30
    python -m openroboxing.tools.run_single --style walk --seconds 10 --out run.npz

Headless by design: nothing here opens a display, and the generator never imports the upstream
keyboard controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from openroboxing.runtime.conventions import G1
from openroboxing.runtime.generator import GeneratorIntent
from openroboxing.runtime.world import SingleFighterWorld
from openroboxing.spec.constants import POSE_DWELL_TICKS, TICK_HZ


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_single",
        description="Run a single G1 fighter under MuJoCo physics (M1-T6).",
    )
    parser.add_argument("--style", default="walk_boxing", help="clip name driving the generator")
    parser.add_argument("--seconds", type=float, default=30.0, help="simulated seconds to run")
    parser.add_argument("--seed", type=int, default=1234, help="generator seed")
    parser.add_argument(
        "--movement-angle", type=float, default=0.0, help="direction of travel, radians"
    )
    parser.add_argument("--facing-angle", type=float, default=0.0, help="facing, radians")
    parser.add_argument(
        "--fall-height", type=float, default=0.4, help="root height counted as a fall, metres"
    )
    parser.add_argument(
        "--loadout",
        type=Path,
        default=None,
        help="drive the fighter from a loadout, committing each slot in turn (M2-T5)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the run log to this npz")
    args = parser.parse_args(argv)

    print(f"building world (style={args.style}, seed={args.seed})...")
    build_start = time.perf_counter()
    world = SingleFighterWorld(style=args.style, seed=args.seed, fall_height=args.fall_height)
    world.reset(seed=args.seed)
    print(f"  ready in {time.perf_counter() - build_start:.1f}s")
    print(f"  physics timestep {world.model.opt.timestep}s, {world.substeps} substeps per tick")

    intent = GeneratorIntent(
        style=args.style, movement_angle=args.movement_angle, facing_angle=args.facing_angle
    )

    timeline = None
    if args.loadout is not None:
        from openroboxing.runtime.intents import IntentError, IntentTimeline, Loadout

        loadout = Loadout.load(args.loadout)
        timeline = IntentTimeline(loadout, context=args.style)
        order = sorted(loadout.slots)
        print(f"loadout {loadout.name!r} ({loadout.version}): " + ", ".join(
            f"{slot}={loadout.slots[slot].name}" for slot in order
        ))

        fired: list[tuple[int, str]] = []

        def intent_for_tick(tick: int):
            """Commit the next slot the moment nothing is scheduled, one at a time.

            Deliberately *not* the queue a match allows: this tool exists to look at one move at a
            time under physics, and overlapping them would make the tracking log unreadable.
            """
            if not timeline.scheduled(tick) and len(fired) < len(order):
                slot = order[len(fired)]
                timeline.stage(pose_slot=slot)
                try:
                    commit = timeline.commit(now=tick)
                except IntentError as exc:  # a rejected commit is a bug here, not a fallback
                    raise SystemExit(f"commit rejected at tick {tick}: {exc}") from exc
                fired.append((tick, commit.pose.name))
                # No span to print: a commit's start and end are settled as it runs
                # (`spec/intent.md` 2.0). These carry no placement, so there is nothing to walk to
                # and each fires as soon as its readable window elapses — and then stands in its
                # pose for the dwell, which is the only length a commit has since 2.0.
                print(
                    f"  tick {tick:>5}: slot {slot} -> {commit.pose.name} "
                    f"({POSE_DWELL_TICKS} ticks of dwell)"
                )
            return timeline.generator_intent(tick)
    else:
        intent_for_tick = None
        fired = []

    run_start = time.perf_counter()
    log = world.run(seconds=args.seconds, intent=intent, intent_for_tick=intent_for_tick)
    wall = time.perf_counter() - run_start

    summary = log.summary()
    simulated = summary["seconds"]
    print()
    print(
        f"ran {simulated:.1f}s simulated in {wall:.1f}s wall  "
        f"(real-time factor {simulated / wall:.2f}x)"
    )
    print(f"  fell               : {'YES at tick ' + str(log.fell_at_tick) if log.fell else 'no'}")
    print(f"  root height min    : {summary['min_root_height_m']:.3f} m")
    print(f"  root height final  : {summary['final_root_height_m']:.3f} m")
    print(f"  joint error mean   : {summary['mean_joint_error_rad']:.4f} rad")
    print(f"  joint error max    : {summary['max_joint_error_rad']:.4f} rad")

    travelled = (
        float(np.linalg.norm(log.root_position[-1][:2] - log.root_position[0][:2]))
        if len(log.root_position) > 1
        else 0.0
    )
    print(f"  distance travelled : {travelled:.2f} m")
    if args.loadout is not None:
        print(f"  moves committed    : {len(fired)} of {len(order)}")
        if len(fired) < len(order):
            print(f"    not reached      : {[s for s in order[len(fired):]]}")

    print()
    print("per-joint mean |tracking error| (rad), worst 8:")
    per_joint = log.per_joint_error()
    for idx in np.argsort(per_joint)[::-1][:8]:
        print(f"  {G1.mujoco_joint_names[idx]:32s} {per_joint[idx]:.4f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out,
            tick=np.array(log.tick),
            root_height=np.array(log.root_height),
            joint_tracking_error=np.array(log.joint_tracking_error),
            root_position=np.array(log.root_position),
            reference_root_position=np.array(log.reference_root_position),
            per_joint_error=per_joint,
            joint_names=np.array(G1.mujoco_joint_names),
            summary=np.array(json.dumps(summary)),
            tick_hz=np.array(TICK_HZ),
        )
        print(f"\nwrote run log to {args.out}")

    return 1 if log.fell else 0


if __name__ == "__main__":
    raise SystemExit(main())
