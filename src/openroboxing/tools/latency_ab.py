"""Does injected latency change who wins? (M4-T2; ported for combinations, B2)

Acceptance criterion from WORKPLAN.md M4-T2:
  artificially injecting 200 ms latency does not change match outcomes systematically (run a
  scripted-agent A/B and compare win rates).

Two identical baseline agents play N matches with no latency, then N more with one seat delayed. If
the handicapped seat's win rate moves outside what the sample size can explain, the host is giving
one side an advantage and that is a bug in the host, not in the network.

Why this is a fair test: the host applies queued intents on its **own** 30 Hz tick and never waits
for a client (`spec/protocol.md` §Latency). Latency should therefore delay *when* a commit lands and
nothing else — the claim being checked.

Both agents read the shared combination library from ``welcome`` (`spec/intent.md`'s `D6`;
:class:`~openroboxing.server.agent.BaselineAgent` already speaks it — see its module docstring), so
there is nothing loadout-shaped left for this tool to choose per seat, only which library to serve.
Built on :class:`~openroboxing.server.host.MatchHost`, which never accepts a draft combination
(`spec/intent.md` "Admission is enforced at construction"); there is no ``--allow-draft`` here for the
same reason `tools/serve_match.py` has none.

Run: ``.venv_mb/bin/python -m openroboxing.tools.latency_ab --matches 4 --round-seconds 10``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
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


def _wilson_halfwidth(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a win rate. Small samples need it; the normal approximation lies."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


async def _one_match(libraries, match_format, seed: int, latency: dict[str, float]) -> str | None:
    """Run one agent-vs-agent match and return the winner, or ``None`` for a draw."""
    host = MatchHost(
        libraries=libraries,
        match_format=match_format,
        match_seed=seed,
        match_id=f"ab-{seed}",
        render=False,  # nobody is watching; rendering would only slow the sweep
    )
    # Different seeds per seat so the two agents do not mirror each other exactly.
    agents = {"red": BaselineAgent(seed=0), "blue": BaselineAgent(seed=1)}
    record, _ = await play_match(host, agents, latency_ms=latency)

    from openroboxing.runtime.replay import RecordedMatch

    recorded = RecordedMatch(record=record.to_dict(), traces={r.index: r.trace for r in record.rounds})
    score = score_match(recorded.record, traces_from_replay(recorded))
    return score.winner


def run(
    matches: int, latency_ms: float, round_seconds: float, rounds: int, seed: int, library_dir: Path
) -> dict:
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
    for label, latency in (
        ("baseline", {}),
        (f"red +{latency_ms:.0f}ms", {"red": latency_ms}),
    ):
        wins = {f: 0 for f in FIGHTERS}
        draws = 0
        for index in range(matches):
            winner = asyncio.run(
                _one_match(libraries, match_format, seed + index, latency)
            )
            if winner is None:
                draws += 1
            else:
                wins[winner] += 1
            print(f"  {label:<16} match {index + 1:>3}/{matches}: {winner or 'draw'}")

        played = matches - draws
        rate = wins["red"] / played if played else 0.0
        low, high = _wilson_halfwidth(wins["red"], played)
        results[label] = {
            "matches": matches,
            "draws": draws,
            "wins": wins,
            "red_win_rate": rate,
            "red_win_interval": [low, high],
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="latency_ab",
        description="Scripted-agent A/B: does injected latency change outcomes? (M4-T2)",
    )
    parser.add_argument("--matches", type=int, default=12, help="matches per condition")
    parser.add_argument("--latency-ms", type=float, default=200.0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--round-seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--library", type=Path, default=COMBINATION_DIR, help="combination library directory"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    print(
        f"A/B: {args.matches} matches per condition, {args.rounds} x {args.round_seconds:.0f}s, "
        f"latency {args.latency_ms:.0f} ms on red"
    )
    results = run(args.matches, args.latency_ms, args.round_seconds, args.rounds, args.seed, args.library)

    print("\ncondition           red wins   draws   red win rate   95% interval")
    for label, data in results.items():
        low, high = data["red_win_interval"]
        print(
            f"  {label:<17} {data['wins']['red']:>3}/{data['matches']:<3} "
            f"{data['draws']:>6}   {data['red_win_rate']:>11.2f}   [{low:.2f}, {high:.2f}]"
        )

    conditions = list(results.values())
    a, b = conditions[0], conditions[1]
    overlap = not (
        a["red_win_interval"][1] < b["red_win_interval"][0]
        or b["red_win_interval"][1] < a["red_win_interval"][0]
    )
    print()
    if overlap:
        print("  intervals overlap: no systematic effect detected at this sample size.")
    else:
        print("  intervals are DISJOINT: latency shifted outcomes. That is a host bug.")
    print(
        "  Note: a null result at this sample size bounds the effect, it does not prove absence.\n"
        f"  With {a['matches']} matches the interval is about "
        f"+-{(a['red_win_interval'][1] - a['red_win_interval'][0]) / 2:.2f} wide."
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.out}")

    return 0 if overlap else 1


if __name__ == "__main__":
    raise SystemExit(main())
