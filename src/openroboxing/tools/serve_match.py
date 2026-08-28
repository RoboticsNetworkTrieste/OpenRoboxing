"""Host a live match for browser clients (M4-T1, M4-T2; ported for combinations, B2).

Acceptance criteria:
  M4-T1 — a hotseat match is playable end to end by two people on one machine.
  M4-T2 — two browsers on the LAN play a full match; injecting 200 ms latency does not change match
          outcomes systematically.

Open http://localhost:8080/ (or ``--port``). **One page holds both seats** — red plays 1-6 and SPACE,
blue plays U I O J K L and ENTER — which is what makes it a hotseat game on one keyboard.

No ``--allow-draft`` here, unlike this phase's other tools
------------------------------------------------------------
`spec/intent.md` is explicit that admission is a match's own rule with no override: "The Studio passes
``require_admitted=False`` so a draft combination can be rehearsed before it has been measured; **a
match never does**." :class:`~openroboxing.server.host.MatchHost` — the class this tool serves —
does not accept ``require_admitted`` at all, and that is by design, not an omission this tool works
around: a live, servable match is exactly the thing the rule exists to protect. The on-disk
combination library is entirely draft today — telegraph and tracking error have not been measured for
any of it, which is scheduled work this tool does not do — so it will refuse to build a match until at
least one combination is admitted. That is correct behaviour, not a bug in the port. Rehearse a draft
combination under physics with `tools/run_single.py --library ... --allow-draft` instead, which builds
:class:`~openroboxing.runtime.intents.IntentTimeline` directly and is explicitly the tool the Studio's
own escape hatch is for.

Run: ``.venv_mb/bin/python -m openroboxing.tools.serve_match --no-wait``
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from openroboxing.paths import COMBINATION_DIR
from openroboxing.runtime.arena import FIGHTERS
from openroboxing.runtime.match import MatchFormat
from openroboxing.server.app import serve
from openroboxing.server.host import MatchHost
from openroboxing.spec.constants import TICK_HZ
from openroboxing.studio import combination_record as cr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="serve_match", description="Host one live match for browser clients (M4-T2)."
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--library", type=Path, default=COMBINATION_DIR, help="combination library directory"
    )
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

    library = {p.stem: cr.load(p) for p in sorted(args.library.glob("*.json"))}
    if not library:
        raise SystemExit(f"no combinations in {args.library}")
    admitted = sum(1 for r in library.values() if r.admission == "admitted")
    print(
        f"combination library {args.library}: {len(library)} moves, {admitted} admitted "
        "(a match refuses the rest)"
    )

    print(f"\nformat: {match_format.rounds} x {match_format.round_seconds:.0f}s")
    print("building the ring...")
    host = MatchHost(
        libraries={f: library for f in FIGHTERS},
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
