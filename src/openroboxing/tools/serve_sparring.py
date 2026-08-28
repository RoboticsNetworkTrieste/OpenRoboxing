"""The sparring bench: free-space debugging of the core motion stack.

One player fighter (red, your browser) and a passive sacco (blue). No rounds, no scoring — commits
run until you reset. Everything is recorded and scrubbable, and the runtime's knobs are live.

Usage
-----
    .venv_mb/bin/python -m openroboxing.tools.serve_sparring
    .venv_mb/bin/python -m openroboxing.tools.serve_sparring --library poses/v0.2/combinations --seed 7

Open http://localhost:8081/. Keys: pick a combination, drag the ghost, commit, SPACE to fire.

No ``--loadout`` any more (B3, ported for `spec/intent.md` 3.0)
------------------------------------------------------------------
`D6` retired the per-seat loadout — both fighters read the same shared combination library, paged
rather than dealt — so what this tool loads is a **library directory**, exactly like
`tools/serve_match.py`, not a named loadout file. Unlike `serve_match.py` this tool defaults to
*allowing* drafts: the on-disk library is entirely `admission="draft"` today (telegraph and tracking
error unmeasured), and a bench exists to rehearse exactly that
(`spec/intent.md`: "The Studio passes ``require_admitted=False``... a match never does"). Pass
``--admitted-only`` to hold the bench to the match's own rule instead.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from openroboxing.paths import COMBINATION_DIR
from openroboxing.server.host import QueuedPilot
from openroboxing.server.sparring_app import (
    SPARRING_MAX_OUTSTANDING,
    SparringHost,
    SparringWorld,
    serve_sparring,
)
from openroboxing.server.sparring_tap import DebugTap
from openroboxing.spec.constants import TICK_HZ
from openroboxing.studio import combination_record as cr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="serve_sparring",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--library", type=Path, default=COMBINATION_DIR, help="combination library directory"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--admitted-only",
        action="store_true",
        help="refuse draft combinations, as a match does. Default: drafts are welcome — this is a bench",
    )
    parser.add_argument(
        "--record-minutes",
        type=float,
        default=10.0,
        help="recording window; older ticks fall off the scrubber",
    )
    args = parser.parse_args(argv)

    library = {p.stem: cr.load(p) for p in sorted(args.library.glob("*.json"))}
    if not library:
        raise SystemExit(f"no combinations in {args.library}")
    admitted = sum(1 for r in library.values() if r.admission == "admitted")
    print(
        f"combination library {args.library}: {len(library)} moves, {admitted} admitted"
        + ("" if args.admitted_only else " (drafts welcome on this bench)")
    )

    print("building the ring (two generators + the policy; this loads checkpoints)...")
    world = SparringWorld(
        libraries={"red": library, "blue": library},
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
