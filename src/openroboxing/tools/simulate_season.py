"""Simulate a full season and print the table (M5-T1).

Acceptance criterion from WORKPLAN.md M5-T1:
  a simulated 32-player, 10-week season runs end to end from scripted clients and produces a sane
  table; ratings converge and the 8-match threshold behaves as specified.

Usage
-----
    python -m openroboxing.tools.simulate_season
    python -m openroboxing.tools.simulate_season --entrants 32 --weeks 10 --seed 7
    python -m openroboxing.tools.simulate_season --out seasons/season-0.json

Each entrant is given a hidden **true strength**, and a match is decided by the Bradley-Terry
probability that strength implies. Ratings never see the strength — so "ratings converge" is a real
claim: the correlation between the final table and the hidden ranking is measured and printed.

This does not simulate physics. Running 32 x 10 real matches at 2x real time would take five hours;
`M5-T4` and `M4-T2` are what put real matches behind this, and the interface is `MatchResult` either
way.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from openroboxing.league.rating import DEFAULT_RATING
from openroboxing.league.season import (
    MATCHES_TO_RANK,
    MatchResult,
    Season,
    format_table,
)

#: Spread of hidden strengths, in rating points. 300 is about two Glicko-2 classes: wide enough that
#: a table can be wrong, narrow enough that upsets happen.
STRENGTH_SPREAD = 300.0

#: Chance a match is drawn when the fighters are evenly matched. `spec/scoring.md` makes a draw
#: possible (rounds within DRAW_MARGIN), and a league with no draws is not testing the draw path.
DRAW_RATE = 0.06


def _true_win_probability(strength_a: float, strength_b: float) -> float:
    """Bradley-Terry on the hidden strengths, on the Elo/Glicko 400-point scale."""
    return 1.0 / (1.0 + 10.0 ** ((strength_b - strength_a) / 400.0))


def simulate(entrants: int, weeks: int, seed: int, verbose: bool = True) -> tuple[Season, dict]:
    rng = np.random.default_rng(seed)
    season = Season(name=f"season-0-sim-{seed}", weeks=weeks)

    strengths = {}
    for index in range(entrants):
        handle = f"fighter{index:02d}"
        season.register(handle)
        strengths[handle] = float(rng.normal(DEFAULT_RATING, STRENGTH_SPREAD))

    for week in range(weeks):
        fixtures = season.pair_week()
        results = []
        for fixture in fixtures:
            if fixture.is_bye:
                results.append(MatchResult(fixture=fixture))
                continue
            probability = _true_win_probability(
                strengths[fixture.home], strengths[fixture.away]
            )
            draw = rng.random() < DRAW_RATE
            if draw:
                winner = None
            else:
                winner = fixture.home if rng.random() < probability else fixture.away
            results.append(MatchResult(fixture=fixture, winner=winner))
        season.report_week(results)

        if verbose:
            ranked = len(season.table())
            byes = sum(1 for f in fixtures if f.is_bye)
            print(
                f"  week {week + 1:>2}: {len(fixtures) - byes:>2} matches, {byes} bye  "
                f"-> {ranked} ranked, {len(season.provisional())} provisional"
            )

    return season, strengths


def _convergence(season: Season, strengths: dict) -> dict:
    """How well the final table recovers the hidden ranking. Spearman, computed without scipy."""
    table = season.table()
    if len(table) < 3:
        return {"ranked": len(table), "spearman": float("nan")}

    by_rating = [e.handle for e in table]
    by_strength = sorted(by_rating, key=lambda h: -strengths[h])

    rank_rating = {h: i for i, h in enumerate(by_rating)}
    rank_strength = {h: i for i, h in enumerate(by_strength)}
    n = len(by_rating)
    d_squared = sum((rank_rating[h] - rank_strength[h]) ** 2 for h in by_rating)
    spearman = 1.0 - 6.0 * d_squared / (n * (n**2 - 1))

    errors = [e.rating.rating - strengths[e.handle] for e in table]
    return {
        "ranked": n,
        "spearman": spearman,
        "mean_abs_rating_error": float(np.mean(np.abs(errors))),
        "mean_rd": float(np.mean([e.rating.rd for e in table])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="simulate_season",
        description="Run a simulated Swiss season and print the table (M5-T1).",
    )
    parser.add_argument("--entrants", type=int, default=32, help="how many fighters register")
    parser.add_argument("--weeks", type=int, default=10, help="season length")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=None, help="write the season json here")
    args = parser.parse_args(argv)

    print(f"simulating {args.entrants} entrants over {args.weeks} weeks (seed {args.seed})")
    season, strengths = simulate(args.entrants, args.weeks, args.seed)

    print()
    print(format_table(season.table(), title=f"TABLE  ({MATCHES_TO_RANK}+ matches)"))
    provisional = season.provisional()
    if provisional:
        print()
        print(format_table(provisional, title=f"PROVISIONAL  (< {MATCHES_TO_RANK} matches)"))

    stats = _convergence(season, strengths)
    print()
    print("convergence against the hidden true strengths (ratings never see these):")
    print(f"  ranked fighters       : {stats['ranked']}")
    print(f"  Spearman rank corr.   : {stats['spearman']:.3f}   (1.0 = perfect order)")
    print(f"  mean |rating - truth| : {stats['mean_abs_rating_error']:.0f} points")
    print(f"  mean RD               : {stats['mean_rd']:.0f}   (started at 350)")

    try:
        semis = season.playoff()
        print(f"\nplayoff: {semis[0][0]} v {semis[0][1]}   |   {semis[1][0]} v {semis[1][1]}")
    except Exception as exc:
        print(f"\nno playoff: {exc}")

    if args.out is not None:
        print(f"\nwrote {season.save(args.out)}")

    return 0 if not math.isnan(stats["spearman"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
