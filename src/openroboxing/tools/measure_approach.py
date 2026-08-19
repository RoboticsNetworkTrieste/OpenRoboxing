"""Does an approach close? Body against plan, over a sweep of bearings.

`spec/intent.md` 2.0 ends an approach when the **body** reaches the placement, and gives it
``approach_timeout_ticks`` before the pose is thrown wherever the fighter stands. Whether that
timeout is a guard or the actual mechanism is a measurement, and this is it: one placement at a
fixed distance, taken from several directions, scored on how near the body ever got against how
near the *plan* ever got.

Reading the two columns
-----------------------
- **plan** is the reference frame the encoder is chasing, mapped to the world by the sparring
  bench's own viz transform. MotionBricks is kinematic: its plan arrives every time.
- **body** is the pelvis under physics, which is what
  :meth:`~openroboxing.runtime.fight.FightWorld.has_arrived` reads and therefore what actually
  decides the commit.

The gap between them is the diagnosis. Plan short means steering or target conversion; body short
with the plan in means the policy cannot track what it was handed - and if the body is still short
when ``approach_timeout_ticks`` expires, the strike is fired by a clock rather than by an arrival.

First run, 2026-08-17 (seed 1234, orthodox slot 1, 1.5 m, radius 0.40 m): the plan closed to
0.02-0.19 m at all seven bearings; the body closed to 0.007 m straight ahead and 0.38-0.54 m
off-axis, so four of seven approaches ended on the timeout with ``arrived=False``.

Usage
-----
    .venv_mb/bin/python -m openroboxing.tools.measure_approach
    .venv_mb/bin/python -m openroboxing.tools.measure_approach --distance 2.5 --bearings 0 90 180
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from openroboxing.paths import LOADOUT_DIR
from openroboxing.runtime.intents import Loadout, Placement
from openroboxing.runtime.reference import LOOKAHEAD_TICKS
from openroboxing.server.host import QueuedPilot
from openroboxing.server.sparring_app import SparringWorld
from openroboxing.server.sparring_tap import viz_world_path
from openroboxing.spec.constants import ARRIVAL_RADIUS_M, TICK_HZ

#: Bearings swept by default, in degrees from the generator's forward axis. Chosen to cover the
#: turn: straight ahead, both diagonals, both sides, and behind.
DEFAULT_BEARINGS = (0, 45, 90, 135, 180, -135, -90, -45)

#: Hard stop per bearing. Longer than the ring-derived timeout plus a dwell, so every run ends
#: because the commit ended and not because the loop ran out.
MAX_TICKS = 900


def measure_bearing(world, slot: str, distance: float, bearing_deg: float) -> dict:
    """Commit one placement at ``bearing_deg`` and run it to the end of the commit."""
    world.reset_round(0)
    red = world.fighters["red"]
    timeline = red.timeline

    start = np.asarray(world.data.xpos[red.pelvis_body][:2], dtype=np.float64)
    axis = float(red.apply_yaw)
    heading = axis + math.radians(bearing_deg)
    target = start + distance * np.array([math.cos(heading), math.sin(heading)])

    timeline.stage(pose_slot=slot, placement=Placement(position=tuple(target), heading=heading))
    commit = timeline.commit(0)

    best_body, best_plan = math.inf, math.inf
    for tick in range(MAX_TICKS):
        world.step(tick)
        here = np.asarray(world.data.xpos[red.pelvis_body][:2], dtype=np.float64)
        best_body = min(best_body, float(np.linalg.norm(target - here)))

        motion = np.asarray(red.stream.motion)
        if len(motion) > tick + LOOKAHEAD_TICKS:
            path = viz_world_path(motion, tick, here, axis)
            best_plan = min(best_plan, float(np.linalg.norm(target - path[LOOKAHEAD_TICKS])))

        if commit.end_tick is not None and tick > commit.end_tick:
            break

    approach = (
        None
        if commit.strike_at is None or commit.commit_at is None
        else commit.strike_at - commit.commit_at
    )
    dwell = (
        None
        if commit.ended_at is None or commit.strike_at is None
        else commit.ended_at - commit.strike_at
    )
    return {
        "bearing": bearing_deg,
        "best_body": best_body,
        "best_plan": best_plan,
        "arrived": commit.arrived,
        "approach_ticks": approach,
        "dwell_ticks": dwell,
        "completed_by": commit.completed_by,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="measure_approach",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--distance", type=float, default=1.5, help="metres to the placement")
    parser.add_argument(
        "--bearings",
        type=float,
        nargs="+",
        default=list(DEFAULT_BEARINGS),
        help="degrees from the generator's forward axis",
    )
    parser.add_argument("--slot", default="1", help="loadout slot to commit")
    parser.add_argument("--loadout", default="orthodox")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--context",
        default=None,
        help="clip to travel in; default is the runtime's own (`intents.TRAVEL_CONTEXT`)",
    )
    parser.add_argument(
        "--leg",
        type=float,
        default=None,
        help="metres the generator is aimed ahead; 0 aims at the whole placement",
    )
    parser.add_argument(
        "--travel-angle",
        choices=("on", "off"),
        default="on",
        help="off pins the movement direction to the generator's forward axis, as before 2026-08-17",
    )
    args = parser.parse_args(argv)

    loadout = Loadout.load(LOADOUT_DIR / f"{args.loadout}.json")
    print("building the ring (two generators + the policy; this loads checkpoints)...")
    build: dict = {}
    if args.context is not None:
        build["context"] = args.context
    world = SparringWorld(
        loadouts={"red": loadout, "blue": loadout},
        pilots={"red": QueuedPilot()},
        match_seed=args.seed,
        require_admitted=False,
        **build,
    )
    if args.leg is not None:
        world.approach_leg_m = args.leg
    if args.travel_angle == "off":
        # The ablation, not a mode: before 2026-08-17 the runtime never set a movement direction,
        # so upstream read a constant "straight ahead" and its lateral blendspace never fired.
        world.travel_angle = lambda fighter, target_gen: 0.0

    timeout = world.fighters["red"].timeline.approach_timeout_ticks
    context = world.fighters["red"].timeline.staged.context
    print(
        f"\n{args.distance:.2f} m, arrival radius {ARRIVAL_RADIUS_M} m, "
        f"timeout {timeout} ticks ({timeout / TICK_HZ:.1f} s), pose {loadout.slots[args.slot].name}"
        f"\ncontext {context!r}, leg {world.approach_leg_m:.2f} m, "
        f"travel angle {args.travel_angle}"
    )
    print(
        f"\n{'bearing':>8} {'body':>9} {'plan':>9} {'arrived':>8} {'approach':>9} "
        f"{'dwell':>6} {'ended by':>9}"
    )

    results = []
    for bearing in args.bearings:
        result = measure_bearing(world, args.slot, args.distance, bearing)
        results.append(result)
        ticks = result["approach_ticks"]
        note = "" if ticks is None or ticks < timeout else "  <- timed out"
        print(
            f"{result['bearing']:>8.0f} {result['best_body']:>9.3f} {result['best_plan']:>9.3f} "
            f"{str(result['arrived']):>8} {'—' if ticks is None else ticks:>9} "
            f"{'—' if result['dwell_ticks'] is None else result['dwell_ticks']:>6} "
            f"{str(result['completed_by']):>9}{note}"
        )

    landed = sum(1 for r in results if r["arrived"])
    print(
        f"\n{landed} of {len(results)} arrived; "
        f"body {min(r['best_body'] for r in results):.3f}-{max(r['best_body'] for r in results):.3f} m, "
        f"plan {min(r['best_plan'] for r in results):.3f}-{max(r['best_plan'] for r in results):.3f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
