"""Is one seat worth points? (docs/ASSUMPTIONS.md §A12; ported for combinations, B2)

The M4-T2 latency A/B incidentally measured red winning 5 of 16 *baseline* matches. The interval
contained 0.5, so nothing was established — but if the seat itself is worth points, then Swiss
pairing plus Glicko-2 will faithfully rate which side of the ring somebody stood on, and Season 0
would be measuring the arena instead of the players.

The experiment
--------------
Two conditions, identical except for **which agent sits where**:

- ``as-is``  : red = BaselineAgent(seed=0), blue = BaselineAgent(seed=1)
- ``swapped``: red = BaselineAgent(seed=1), blue = BaselineAgent(seed=0)

Read it like this:

- red's win rate **follows the agent seed** (high in one condition, low in the other) -> the agents
  differ, the seats do not. Harmless: they open by reading the same shared combination library
  differently (`server/agent.py::BaselineAgent`'s own cycling, seeded).
- red's win rate **stays with the seat** (similar in both) -> the *seat* is worth points. That is a
  fairness bug and it matters before anybody is paired.

Both seats read the same shared combination library from ``welcome`` (`spec/intent.md`'s `D6` — there
is no per-seat loadout left to swap), so what varies between conditions is only which agent
:class:`~openroboxing.server.agent.BaselineAgent` seed sits where. Built on
:class:`~openroboxing.server.host.MatchHost`, which never accepts a draft combination
(`spec/intent.md` "Admission is enforced at construction"); there is no ``--allow-draft`` here for the
same reason `tools/serve_match.py` has none.

Run: ``.venv_mb/bin/python -m openroboxing.tools.seat_fairness --matches 4 --round-seconds 10``
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from openroboxing.league.scoring import score_match, traces_from_replay
from openroboxing.paths import COMBINATION_DIR
from openroboxing.runtime.arena import FIGHTERS
from openroboxing.runtime.match import MatchFormat
from openroboxing.server.agent import BaselineAgent
from openroboxing.server.client import play_match
from openroboxing.server.host import MatchHost
from openroboxing.spec.constants import TICK_HZ
from openroboxing.studio import combination_record as cr
from openroboxing.tools.latency_ab import _wilson_halfwidth


async def _one(libraries, match_format, seed: int, red_seed: int, blue_seed: int) -> str | None:
    host = MatchHost(
        libraries=libraries,
        match_format=match_format,
        match_seed=seed,
        match_id=f"seat-{seed}",
        render=False,
    )
    agents = {"red": BaselineAgent(seed=red_seed), "blue": BaselineAgent(seed=blue_seed)}
    record, _ = await play_match(host, agents)

    from openroboxing.runtime.replay import RecordedMatch

    recorded = RecordedMatch(
        record=record.to_dict(), traces={r.index: r.trace for r in record.rounds}
    )
    return score_match(recorded.record, traces_from_replay(recorded)).winner


def run(matches: int, round_seconds: float, rounds: int, seed: int, library_dir: Path) -> dict:
    library = {p.stem: cr.load(p) for p in sorted(library_dir.glob("*.json"))}
    if not library:
        raise SystemExit(f"no combinations in {library_dir}")
    libraries = {f: library for f in FIGHTERS}
    match_format = MatchFormat(
        rounds=rounds,
        round_ticks=int(round(round_seconds * TICK_HZ)),
        get_up_window_ticks=MatchFormat().get_up_window_ticks,
    )

    results: dict[str, dict] = {}
    for label, (red_seed, blue_seed) in (("as-is", (0, 1)), ("swapped", (1, 0))):
        wins = {f: 0 for f in FIGHTERS}
        draws = 0
        for index in range(matches):
            # The same match seeds in both conditions, so physics is not a confounder.
            winner = asyncio.run(
                _one(libraries, match_format, seed + index, red_seed, blue_seed)
            )
            if winner is None:
                draws += 1
            else:
                wins[winner] += 1
            print(f"  {label:<9} match {index + 1:>3}/{matches}: {winner or 'draw'}", flush=True)

        played = matches - draws
        rate = wins["red"] / played if played else 0.0
        low, high = _wilson_halfwidth(wins["red"], played)
        results[label] = {
            "red_agent_seed": red_seed,
            "matches": matches,
            "draws": draws,
            "wins": wins,
            "red_win_rate": rate,
            "red_win_interval": [low, high],
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="seat_fairness", description="Is red or blue worth points? (ASSUMPTIONS §A12)"
    )
    parser.add_argument("--matches", type=int, default=20, help="matches per condition")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--round-seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument(
        "--library", type=Path, default=COMBINATION_DIR, help="combination library directory"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    print(f"seat fairness: {args.matches} matches per condition, same match seeds in both")
    results = run(args.matches, args.round_seconds, args.rounds, args.seed, args.library)

    print("\ncondition   red agent   red wins   draws   red win rate   95% interval")
    for label, data in results.items():
        low, high = data["red_win_interval"]
        print(
            f"  {label:<9} seed {data['red_agent_seed']}      "
            f"{data['wins']['red']:>3}/{data['matches']:<3} {data['draws']:>6}   "
            f"{data['red_win_rate']:>11.2f}   [{low:.2f}, {high:.2f}]"
        )

    as_is, swapped = results["as-is"], results["swapped"]
    shift = swapped["red_win_rate"] - as_is["red_win_rate"]
    overlap = not (
        as_is["red_win_interval"][1] < swapped["red_win_interval"][0]
        or swapped["red_win_interval"][1] < as_is["red_win_interval"][0]
    )

    print(f"\n  red's rate moved {shift:+.2f} when the agents swapped seats.")
    if not overlap:
        print(
            "  The intervals are DISJOINT: the result follows the AGENT, not the seat.\n"
            "  The seats are fine; the two baseline agents are simply not equally good."
        )
    else:
        combined = (
            as_is["wins"]["red"] + swapped["wins"]["red"],
            (as_is["matches"] - as_is["draws"]) + (swapped["matches"] - swapped["draws"]),
        )
        low, high = _wilson_halfwidth(*combined)
        print(
            f"  The intervals OVERLAP: the result stays with the SEAT.\n"
            f"  Pooled across both conditions red wins {combined[0]}/{combined[1]} "
            f"= {combined[0] / max(1, combined[1]):.2f}, interval [{low:.2f}, {high:.2f}]."
        )
        if high < 0.5 or low > 0.5:
            print("  That interval EXCLUDES 0.5: one seat is worth points. This is a fairness bug.")
        else:
            print("  That interval still contains 0.5: no seat advantage established at this power.")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
