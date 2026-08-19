"""Score a recorded match (M5-T2).

Acceptance criterion from WORKPLAN.md M5-T2:
  replaying ten recorded matches produces scores that a human watching the replays agrees with in at
  least eight cases; disagreements are documented as rule bugs, not code bugs.

That criterion needs a human with ten replays. This tool is the half that can be automated: it scores
a record and shows **why**, dimension by dimension, so a disagreement can be traced to a rule in
`spec/scoring.md` rather than argued about in the abstract.

Usage
-----
    python -m openroboxing.tools.score_match matches/match-1234.json
    python -m openroboxing.tools.score_match matches/*.json --out scores/
    python -m openroboxing.tools.score_match matches/a.json --no-trace   # skip ring control

Scoring never re-simulates. ``--no-trace`` skips even the replay, at the cost of ring control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openroboxing.league.scoring import (
    DIMENSION_WEIGHTS,
    SPEC_VERSION,
    MatchScore,
    score_match,
    traces_from_replay,
)
from openroboxing.runtime.replay import RecordedMatch


def _report(recorded: RecordedMatch, score: MatchScore) -> None:
    fighters = list(score.rounds[0].dimensions) if score.rounds else []
    print(f"\n{score.match_id}   (scoring spec {SPEC_VERSION})")

    header = f"{'round':>6} " + "".join(f"{f:>34}" for f in fighters)
    print(header)
    print(f"{'':>6} " + "".join(f"{'dmg   ctrl  aggr   pts':>34}" for _ in fighters))

    for round_score in score.rounds:
        cells = ""
        for fighter in fighters:
            d = round_score.dimensions[fighter]
            cells += (
                f"{d.damage:>10.3f} {d.control:>5.2f} {d.aggression:>5.2f} "
                f"{round_score.points.get(fighter, 0):>5}   "
            )
        won = round_score.winner or "even"
        print(f"{round_score.index + 1:>6} {cells}  {won}")

    print()
    totals = score.points
    for fighter in fighters:
        won = score.rounds_won(fighter)
        print(f"  {fighter:<6} {totals[fighter]:>4} points, {won} round(s) won")
    print(f"  -> {score.winner or 'draw'}")

    weights = ", ".join(f"{k} {v}" for k, v in DIMENSION_WEIGHTS.items())
    print(f"\n  weights: {weights}  (spec/scoring.md)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="score_match", description="Score recorded matches from their records (M5-T2)."
    )
    parser.add_argument("records", type=Path, nargs="+", help="match record json files")
    parser.add_argument("--out", type=Path, default=None, help="write <match_id>.score.json here")
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="skip the replay; faster, but ring control scores zero for both",
    )
    parser.add_argument("--quiet", action="store_true", help="only print the verdict line")
    args = parser.parse_args(argv)

    for path in args.records:
        recorded = RecordedMatch.load(path)
        traces = None if args.no_trace else traces_from_replay(recorded)
        score = score_match(recorded.record, traces)

        if args.quiet:
            print(f"{score.match_id}: {score.points} -> {score.winner or 'draw'}")
        else:
            _report(recorded, score)

        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            out = args.out / f"{score.match_id}.score.json"
            out.write_text(json.dumps(score.to_dict(), indent=2) + "\n")
            print(f"  wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
