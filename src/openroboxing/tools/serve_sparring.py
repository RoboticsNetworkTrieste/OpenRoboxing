"""The sparring bench: free-space debugging of the core motion stack.

One player fighter (red, your browser) and a passive sacco (blue). No rounds, no scoring — commits
run until you reset. Everything is recorded and scrubbable, and the runtime's knobs are live.

Usage
-----
    .venv_mb/bin/python -m openroboxing.tools.serve_sparring
    .venv_mb/bin/python -m openroboxing.tools.serve_sparring --loadout orthodox --seed 7

Open http://localhost:8081/. Keys: 1-6 pose, WASD ghost, SPACE commit, Q unstage, P pause, R reset,
arrows scrub. The manual acceptance check is `spec/sparring_protocol.md` + the design spec's Task 12
list: queue ten commits, watch the plan ghost lead the robot, scrub back, turn a knob, replay a
script.
"""

from __future__ import annotations

import argparse
import asyncio

from openroboxing.paths import LOADOUT_DIR
from openroboxing.runtime.intents import Loadout
from openroboxing.server.host import QueuedPilot
from openroboxing.server.sparring_app import (
    SPARRING_MAX_OUTSTANDING,
    SparringHost,
    SparringWorld,
    serve_sparring,
)
from openroboxing.server.sparring_tap import DebugTap
from openroboxing.spec.constants import TICK_HZ


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="serve_sparring",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--loadout", default="orthodox", help="loadout for both fighters")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--admitted-only",
        action="store_true",
        help="refuse draft poses, as a match does. Default: drafts are welcome — this is a bench",
    )
    parser.add_argument(
        "--record-minutes",
        type=float,
        default=10.0,
        help="recording window; older ticks fall off the scrubber",
    )
    args = parser.parse_args(argv)

    loadout = Loadout.load(LOADOUT_DIR / f"{args.loadout}.json")
    entries = ", ".join(f"{s}={loadout.slots[s].name}" for s in sorted(loadout.slots))
    print(f"loadout  {entries}")

    print("building the ring (two generators + the policy; this loads checkpoints)...")
    world = SparringWorld(
        loadouts={"red": loadout, "blue": loadout},
        pilots={"red": QueuedPilot()},  # blue defaults to IdlePilot: the sacco
        match_seed=args.seed,
        max_outstanding=SPARRING_MAX_OUTSTANDING,
        require_admitted=args.admitted_only,
    )
    print(f"  ready: nq={world.model.nq}, {world.substeps} substeps per tick")

    host = SparringHost(
        world, tap=DebugTap(max_ticks=int(args.record_minutes * 60 * TICK_HZ))
    )
    host.reset(seed=args.seed)

    try:
        asyncio.run(serve_sparring(host, port=args.port))
    except KeyboardInterrupt:
        pass

    print(f"\nsession over: {host.tick} ticks, {host.dropped} dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
