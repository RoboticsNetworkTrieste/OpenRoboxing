"""Try a setting, see what changes (M4-T4 preparation).

`WORKPLAN` M4-T4 is a playtest with people in a room. This is the step before that: a way for **one
person** to change a parameter and see, in numbers, what it did to the fight — so you arrive at the
playtest with a shortlist instead of a shrug.

It sweeps one knob at a time, runs agent-vs-agent matches at each value, and reports the metrics
M4-T4 asks for: match length, commits per minute, hit rate, and **how often the fighters just circle
each other**, which is the passivity failure mode the whole commit design exists to prevent.

Since `spec/intent.md` 1.1 it also reports what a commit *costs* — how long a move holds a fighter,
and how much of that is walking. Two of M4-T4's open questions turn on those numbers:
`MAX_OUTSTANDING_COMMITS` (five moves used to be 4-10 s of commitment and is now 20-30 s) and
`TARGET_COMMIT_RATE` in `spec/scoring.md`. Both were derived against the old rule and neither has
been changed on a guess.

Usage
-----
    python -m openroboxing.tools.tune --list
    python -m openroboxing.tools.tune --knob commit_horizon --values 15 30 60
    python -m openroboxing.tools.tune --knob ring_size --values 3.5 4.9 --repeats 3 --out tune.json

What it cannot tell you
-----------------------
Whether it is *fun*. Every number here is a proxy: "engaged 40 % of the round" is not "felt tense".
Use it to find settings worth putting in front of people, not to decide instead of them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

from openroboxing.league.scoring import CONTACT_RANGE_M, engagement
from openroboxing.paths import LOADOUT_DIR
from openroboxing.runtime.arena import FIGHTERS, ArenaConfig
from openroboxing.runtime.intents import Loadout
from openroboxing.runtime.match import MatchFormat
from openroboxing.server.agent import BaselineAgent
from openroboxing.server.client import play_match
from openroboxing.server.host import MatchHost
from openroboxing.spec.constants import (
    COMMIT_HORIZON_TICKS,
    MAX_OUTSTANDING_COMMITS,
    TICK_HZ,
)

#: The knobs worth sweeping, and where each one lives. Every one of these is a number the playtest is
#: expected to move (`WORKPLAN` M4-T4 names commit horizon, commit window, loadout, telegraph floor).
KNOBS: dict[str, dict[str, Any]] = {
    "commit_horizon": {
        "help": "ticks between committing and the move executing — the readable windup",
        "default": COMMIT_HORIZON_TICKS,
        "unit": "ticks",
    },
    "round_seconds": {
        "help": "how long a round lasts",
        "default": 60.0,
        "unit": "s",
    },
    "ring_size": {
        "help": "side of the ring inside the ropes — smaller means nowhere to hide",
        "default": ArenaConfig().ring_size,
        "unit": "m",
    },
    "start_separation": {
        "help": "how far apart the fighters start, each side of centre",
        "default": ArenaConfig().start_separation,
        "unit": "m",
    },
    "glove_radius": {
        "help": "glove size — bigger gloves land more often and hit softer",
        "default": ArenaConfig().glove_radius,
        "unit": "m",
    },
    "queue_depth": {
        "help": "commits a fighter may have unfinished at once — how far ahead you are committed",
        "default": MAX_OUTSTANDING_COMMITS,
        "unit": "commits",
    },
}

#: Separation beyond which the fighters are not really fighting. Reuses the scorer's own number
#: rather than inventing a second idea of "close" (`spec/scoring.md`).
CIRCLING_SEPARATION_M = CONTACT_RANGE_M + 1.0


def _configure(knob: str, value: float) -> tuple[ArenaConfig, MatchFormat, int, int]:
    """Turn one knob into the four things a match is built from."""
    arena = ArenaConfig()
    horizon = COMMIT_HORIZON_TICKS
    queue_depth = MAX_OUTSTANDING_COMMITS
    round_seconds = 60.0

    if knob == "commit_horizon":
        horizon = int(value)
    elif knob == "queue_depth":
        queue_depth = int(value)
    elif knob == "round_seconds":
        round_seconds = float(value)
    elif knob in ("ring_size", "start_separation", "glove_radius"):
        arena = ArenaConfig(**{**ArenaConfig().__dict__, knob: float(value)})
    else:
        raise SystemExit(f"unknown knob {knob!r}; try --list")

    match_format = MatchFormat(
        rounds=1,
        round_ticks=int(round(round_seconds * TICK_HZ)),
        get_up_window_ticks=MatchFormat().get_up_window_ticks,
    )
    return arena, match_format, horizon, queue_depth


async def _one(loadouts, arena, match_format, horizon, queue_depth, seed: int) -> dict[str, Any]:
    # Every knob goes in at construction. The arena in particular *has* to: `build_arena` compiles
    # ring size and glove radius into the model, so assigning `world.config` afterwards changes the
    # record and nothing a fighter can touch — which is what this tool used to do, sweeping two
    # knobs that did nothing and reporting the difference as noise.
    host = MatchHost(
        loadouts=loadouts,
        match_format=match_format,
        match_seed=seed,
        match_id=f"tune-{seed}",
        render=False,
        config=arena,
        horizon_ticks=horizon,
        max_outstanding=queue_depth,
    )

    record, _ = await play_match(
        host, {"red": BaselineAgent(seed=0), "blue": BaselineAgent(seed=1)}
    )
    round_record = record.rounds[0]

    from openroboxing.runtime.contact import ContactTracker, FightTrace
    from openroboxing.runtime.replay import RecordedMatch, ReplayWorld

    recorded = RecordedMatch(
        record=record.to_dict(), traces={r.index: r.trace for r in record.rounds}
    )
    world = ReplayWorld(recorded)
    world.reset_round(0)
    trace = FightTrace()
    for tick in range(world.ticks):
        world.step(tick)
        world.observe(ContactTracker(), trace, tick)

    minutes = round_record.ticks / TICK_HZ / 60.0
    separations = np.array(trace.separation_m) if trace.separation_m else np.zeros(1)

    # What a commit *costs*, which since `spec/intent.md` 1.1 is the number a sweep turns on: a move
    # is its walk plus its pose, so "commits per minute" no longer describes how busy a fighter is
    # without knowing how long each one held it.
    commits = round_record.commits
    finished = [c for c in commits if c["strike_at"] is not None]
    walks = [(c["strike_at"] - c["commit_at"]) / TICK_HZ for c in finished]
    spans = [(c["end_tick"] - c["commit_at"]) / TICK_HZ for c in finished if c["end_tick"]]

    return {
        "ticks": round_record.ticks,
        "seconds": round_record.ticks / TICK_HZ,
        "hits": len(round_record.hits),
        "hits_per_minute": len(round_record.hits) / minutes if minutes else 0.0,
        "commits": len(round_record.commits),
        "commits_per_minute": len(round_record.commits) / minutes if minutes else 0.0,
        "knockdowns": len(round_record.knockdowns),
        "mean_walk_s": float(np.mean(walks)) if walks else 0.0,
        "mean_commit_s": float(np.mean(spans)) if spans else 0.0,
        # A move that timed out was thrown from wherever the fighter got stuck. A sweep where this
        # climbs is a sweep making the ring unwalkable, not one making the fight livelier.
        "timed_out_fraction": (
            sum(1 for c in commits if c["arrived"] is False) / len(commits) if commits else 0.0
        ),
        "unfinished_at_bell": sum(1 for c in commits if c["end_tick"] is None),
        "mean_separation_m": float(separations.mean()),
        # The passivity metric M4-T4 names: the fraction of the round spent too far apart to fight.
        "circling_fraction": float((separations > CIRCLING_SEPARATION_M).mean()),
        # And its positive form, using the scorer's own falloff rather than a second definition.
        "engagement": float(np.mean([engagement(s) for s in separations])),
    }


def sweep(knob: str, values: list[float], repeats: int, seed: int) -> dict[str, Any]:
    loadouts = {f: Loadout.load(LOADOUT_DIR / "orthodox.json") for f in FIGHTERS}
    results: dict[str, Any] = {"knob": knob, "unit": KNOBS[knob]["unit"], "values": {}}

    for value in values:
        arena, match_format, horizon, queue_depth = _configure(knob, value)
        runs = []
        for index in range(repeats):
            runs.append(
                asyncio.run(
                    _one(loadouts, arena, match_format, horizon, queue_depth, seed + index)
                )
            )
            print(
                f"  {knob}={value:<8} run {index + 1}/{repeats}: "
                f"{runs[-1]['hits']} hits, {runs[-1]['commits']} commits "
                f"({runs[-1]['mean_commit_s']:.1f}s each, {runs[-1]['mean_walk_s']:.1f}s walking), "
                f"circling {runs[-1]['circling_fraction']:.0%}",
                flush=True,
            )
        results["values"][str(value)] = {
            key: float(np.mean([r[key] for r in runs]))
            for key in runs[0]
        }
        results["values"][str(value)]["repeats"] = repeats
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tune", description="Sweep one setting and see what it does to the fight (M4-T4 prep)."
    )
    parser.add_argument("--list", action="store_true", help="show the knobs and their defaults")
    parser.add_argument("--knob", default=None, choices=sorted(KNOBS))
    parser.add_argument("--values", type=float, nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=2, help="matches per value")
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.list or args.knob is None:
        print("knobs you can sweep:\n")
        for name, meta in KNOBS.items():
            print(f"  {name:<18} default {meta['default']:<8} {meta['unit']:<6} {meta['help']}")
        print("\n  python -m openroboxing.tools.tune --knob commit_horizon --values 15 30 60")
        return 0

    if not args.values:
        parser.error("--values is required with --knob")

    print(f"sweeping {args.knob} over {args.values} ({args.repeats} matches each)")
    results = sweep(args.knob, args.values, args.repeats, args.seed)

    print()
    header = (
        f"{args.knob + ' (' + results['unit'] + ')':<18} {'len s':>7} {'hits/min':>9} "
        f"{'commits/min':>12} {'commit s':>9} {'walk s':>7} {'late':>6} "
        f"{'circling':>9} {'engaged':>8} {'sep m':>7}"
    )
    print(header)
    print("-" * len(header))
    for value, row in results["values"].items():
        print(
            f"{value:<18} {row['seconds']:>7.1f} {row['hits_per_minute']:>9.1f} "
            f"{row['commits_per_minute']:>12.1f} {row['mean_commit_s']:>9.1f} "
            f"{row['mean_walk_s']:>7.1f} {row['unfinished_at_bell']:>6.1f} "
            f"{row['circling_fraction']:>8.0%} "
            f"{row['engagement']:>8.2f} {row['mean_separation_m']:>7.2f}"
        )

    print(
        "\n  commit s = how long a move held the fighter; walk s = how much of that was walking "
        "there.\n  late     = commits still unfinished at the bell — a queue deeper than the round.\n"
        "  circling = fraction of the round spent more than "
        f"{CIRCLING_SEPARATION_M:.1f} m apart — the passivity failure mode M4-T4 names."
        "\n  engaged  = the scorer's own closeness weighting, 1.0 = in range the whole round."
        "\n\n  These are proxies. They find settings worth showing people; they do not decide"
        "\n  whether it is fun."
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
