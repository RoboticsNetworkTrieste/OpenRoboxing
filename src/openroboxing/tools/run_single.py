"""Run one G1 under physics, tracking a generated reference motion (M1-T6; ported for combinations, B2).

Acceptance criterion from WORKPLAN.md M1-T6:
  `python -m openroboxing.tools.run_single --style walk_boxing --seconds 30` runs a G1 that walks and
  shadow-boxes under physics without falling, and writes a run log with tracking error per body.

``--library`` optionally drives the fighter from a combination library instead of the ambient style
alone, committing every combination in it in turn — a combination name where `spec/intent.md` 1.0-2.2
took a loadout's pose slot. The on-disk library is all-draft today (telegraph and tracking error have
not been measured), so driving from it needs ``--allow-draft``.

Headless by design: nothing here opens a display, and the generator never imports the upstream
keyboard controller.

Run: ``.venv_mb/bin/python -m openroboxing.tools.run_single --style walk_boxing --seconds 10``
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from openroboxing.runtime.conventions import G1, quat_wxyz_to_yaw
from openroboxing.runtime.generator import GeneratorIntent
from openroboxing.runtime.world import SingleFighterWorld
from openroboxing.spec.constants import TICK_HZ


def _ghost_at_recorded_endpoint(world: SingleFighterWorld, record) -> tuple[float, float]:
    """Where ``record``'s own recorded displacement lands from the fighter's position right now.

    Zero residual by construction (`runtime/warp.py`'s baseline case): the fighter is asked to run
    exactly the recording, nothing more, which is what this tool exists to look at — a *drifted*
    placement is `tools/measure_drift_gain.py`'s question, not this one's. The rotation is
    `runtime/warp.py::warp`'s own convention (recorded offsets are in the take's frame, rotated by the
    fighter's current heading on the way out), duplicated here rather than imported because `warp()`
    always also needs a ghost to warp *toward* — there is no standalone "rotate an offset" helper to
    reuse.
    """
    heading = quat_wxyz_to_yaw(world.data.qpos[3:7])
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    dx, dy = record.recorded_displacement
    return (
        float(world.data.qpos[0]) + cos_h * dx - sin_h * dy,
        float(world.data.qpos[1]) + sin_h * dx + cos_h * dy,
    )


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
        "--library",
        type=Path,
        default=None,
        help="drive the fighter from a combination library, committing each move in turn",
    )
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help=(
            "commit draft (unmeasured) combinations from --library. The on-disk library is "
            "entirely draft today, so results are NOT admissible; for exercising the loop only."
        ),
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
    if args.library is not None:
        from openroboxing.runtime.intents import IntentError, IntentTimeline
        from openroboxing.studio import combination_record as cr

        library = {p.stem: cr.load(p) for p in sorted(args.library.glob("*.json"))}
        if not library:
            raise SystemExit(f"no combinations in {args.library}")
        if args.allow_draft:
            print(
                "WARNING: --allow-draft is set. This library has not been measured (telegraph, "
                "tracking error) and results below are not admissible."
            )
        timeline = IntentTimeline(library, require_admitted=not args.allow_draft)
        order = sorted(library)
        print(
            f"library {args.library} ({library[order[0]].library_version}): "
            f"{len(library)} combinations"
        )

        fired: list[tuple[int, str]] = []

        def intent_for_tick(tick: int):
            """Commit the next combination the moment nothing is scheduled, one at a time.

            Deliberately *not* the queue a match allows: this tool exists to look at one move at a
            time under physics, and overlapping them would make the tracking log unreadable.
            """
            if not timeline.scheduled(tick) and len(fired) < len(order):
                name = order[len(fired)]
                record = timeline.library[name]
                ghost = _ghost_at_recorded_endpoint(world, record)
                timeline.stage(combination=name, ghost=ghost)
                try:
                    commit = timeline.commit(now=tick)
                except IntentError as exc:  # a rejected commit is a bug here, not a fallback
                    raise SystemExit(f"commit rejected at tick {tick}: {exc}") from exc
                fired.append((tick, name))
                # A commit's length is exact arithmetic the moment it starts (`spec/intent.md` "A
                # commit's span"), so there is a real number to print here — unlike 2.0-2.2's dwell,
                # which was only known once the body settled.
                print(
                    f"  tick {tick:>5}: -> {name} "
                    f"({commit.record.duration_ticks} ticks = "
                    f"{commit.record.duration_ticks / TICK_HZ:.2f}s)"
                )

            def anchor():
                return (
                    (float(world.data.qpos[0]), float(world.data.qpos[1])),
                    quat_wxyz_to_yaw(world.data.qpos[3:7]),
                )

            return timeline.generator_intent(tick, anchor=anchor)
    else:
        intent_for_tick = None
        order = []
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
    if args.library is not None:
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
