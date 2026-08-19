"""Host a live match for browser clients (M4-T1, M4-T2).

Acceptance criteria:
  M4-T1 — a hotseat match is playable end to end by two people on one machine.
  M4-T2 — two browsers on the LAN play a full match; injecting 200 ms latency does not change match
          outcomes systematically.

Usage
-----
    python -m openroboxing.tools.serve_match
    python -m openroboxing.tools.serve_match --port 8080 --round-seconds 60
    python -m openroboxing.tools.serve_match --no-wait      # start immediately, watch it alone

Open http://localhost:8080/. **One page holds both seats** — red plays 1-6 and SPACE, blue plays
U I O J K L and ENTER — which is what makes it a hotseat game on one keyboard.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from openroboxing.paths import LOADOUT_DIR
from openroboxing.runtime.arena import FIGHTERS
from openroboxing.runtime.intents import Loadout
from openroboxing.runtime.match import MatchFormat
from openroboxing.server.app import serve
from openroboxing.server.host import MatchHost
from openroboxing.spec.constants import TICK_HZ


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="serve_match", description="Host one live match for browser clients (M4-T2)."
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--red", default="orthodox", help="red's loadout")
    parser.add_argument("--blue", default="orthodox", help="blue's loadout")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--round-seconds", type=float, default=None)
    parser.add_argument("--match-id", default="live")
    parser.add_argument("--out", type=Path, default=None, help="write the record here when done")
    parser.add_argument(
        "--no-wait", action="store_true", help="do not wait for both seats before starting"
    )
    args = parser.parse_args(argv)

    default = MatchFormat()
    match_format = MatchFormat(
        rounds=args.rounds if args.rounds is not None else default.rounds,
        round_ticks=(
            int(round(args.round_seconds * TICK_HZ))
            if args.round_seconds is not None
            else default.round_ticks
        ),
        get_up_window_ticks=default.get_up_window_ticks,
    )

    loadouts = {
        "red": Loadout.load(LOADOUT_DIR / f"{args.red}.json"),
        "blue": Loadout.load(LOADOUT_DIR / f"{args.blue}.json"),
    }
    for fighter in FIGHTERS:
        entries = ", ".join(
            f"{s}={loadouts[fighter].slots[s].name}" for s in sorted(loadouts[fighter].slots)
        )
        print(f"{fighter:<5} {entries}")

    print(f"\nformat: {match_format.rounds} x {match_format.round_seconds:.0f}s")
    print("building the ring...")
    host = MatchHost(
        loadouts=loadouts,
        match_format=match_format,
        match_seed=args.seed,
        match_id=args.match_id,
    )
    print(f"  ready: nq={host.world.model.nq}, {host.world.substeps} substeps per tick")

    record = asyncio.run(serve(host, port=args.port, wait_for_players=not args.no_wait))

    stats = host.stats.summary()
    print("\nmatch over")
    for round_record in record.rounds:
        print(
            f"  round {round_record.index + 1}: {round_record.ticks:>5} ticks  "
            f"{round_record.ended_by:<9} hits {len(round_record.hits):>3}  "
            f"commits {len(round_record.commits)}"
        )
    print(
        f"\n  {stats['ticks']:.0f} ticks, {stats['dropped']:.0f} dropped "
        f"({100 * stats['dropped'] / max(1, stats['ticks']):.2f}%)"
    )
    print(
        f"  step {stats['mean_step_ms']:.2f} ms mean, {stats['p95_step_ms']:.2f} ms p95 "
        f"(budget {1000 / TICK_HZ:.0f} ms)"
    )
    print(f"  {stats['frames']:.0f} frames, {stats['mean_render_ms']:.2f} ms each")

    if args.out is not None:
        trace_path = record.save(args.out)
        print(f"\nwrote {args.out} and {trace_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
