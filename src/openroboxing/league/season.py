"""A season: entrants, weeks, the table (M5-T1).

Implements ``spec/season.md`` v0.1 — Season 0's format from
``docs/OpenRoboxing_project_definition_v0.8.md`` §8: ten weeks, one division, Swiss pairing, Glicko-2,
eight matches to appear on the table, top four to the playoff.

A season is a pure function of its fixtures and results
--------------------------------------------------------
Nothing here simulates a fight. A week is *paired*, results are *reported*, and ratings are applied at
the end of the week as one Glicko-2 rating period — which is how Glicko-2 is defined and why matches
are batched rather than rated one at a time.

That makes a season replayable from its results alone, and makes "ratings converge" a claim a test
can check (`tools/simulate_season.py`).

Conventions
-----------
- Fighters are addressed by **handle**, a string, everywhere.
- A **result** is the outcome as the scorer produced it (`league/scoring.py`), reduced to
  ``winner | None``. The league does not re-litigate a score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from openroboxing.league.pairing import Fixture, PairingState, pair_round
from openroboxing.league.rating import Glicko2, Rating, Result

SPEC_VERSION = "0.1"

#: Season 0, from the project definition §8.
SEASON_WEEKS = 10
MATCHES_TO_RANK = 8
PLAYOFF_SIZE = 4

#: Swiss scores.
SCORE_WIN = 1.0
SCORE_DRAW = 0.5
SCORE_LOSS = 0.0
SCORE_BYE = 1.0


class SeasonError(RuntimeError):
    """A season could not be run. Never recovered from silently."""


@dataclass
class Entrant:
    """One fighter's season."""

    handle: str
    loadout: str = "orthodox"
    rating: Rating = field(default_factory=Rating)
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    byes: int = 0
    score: float = 0.0
    opponents: set[str] = field(default_factory=set)

    @property
    def is_ranked(self) -> bool:
        """Whether this fighter appears on the table.

        Counts **matches played**, not wins, and byes do not count — a fighter cannot be ranked on a
        match that never happened (`spec/season.md`).
        """
        return self.played >= MATCHES_TO_RANK

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle,
            "loadout": self.loadout,
            "rating": round(self.rating.rating, 1),
            "rd": round(self.rating.rd, 1),
            "volatility": round(self.rating.volatility, 5),
            "conservative": round(self.rating.conservative, 1),
            "interval": [round(v, 1) for v in self.rating.interval],
            "played": self.played,
            "won": self.won,
            "drawn": self.drawn,
            "lost": self.lost,
            "byes": self.byes,
            "score": self.score,
            "ranked": self.is_ranked,
        }


@dataclass(frozen=True)
class MatchResult:
    """One fixture's outcome, as the scorer decided it."""

    fixture: Fixture
    winner: str | None = None
    match_id: str | None = None

    def __post_init__(self) -> None:
        if self.fixture.is_bye:
            if self.winner not in (None, self.fixture.home):
                raise SeasonError(f"a bye cannot be won by {self.winner!r}")
            return
        if self.winner is not None and self.winner not in self.fixture.fighters():
            raise SeasonError(
                f"{self.winner!r} won a fixture between {self.fixture.fighters()}"
            )


class Season:
    """Season 0. Register entrants, pair each week, report results, read the table."""

    def __init__(self, name: str = "season-0", weeks: int = SEASON_WEEKS, tau: float | None = None):
        if weeks < 1:
            raise SeasonError(f"a season needs at least one week, got {weeks}")
        self.name = name
        self.weeks = weeks
        self.glicko = Glicko2() if tau is None else Glicko2(tau)
        self.entrants: dict[str, Entrant] = {}
        self.fixtures: list[Fixture] = []
        self.results: list[MatchResult] = []
        self.week = 0

    # -- registration --------------------------------------------------------------------------------
    def register(self, handle: str, loadout: str = "orthodox") -> Entrant:
        """Enter a fighter. Open entry, so this is allowed at any point in the season."""
        if handle in self.entrants:
            raise SeasonError(f"{handle!r} is already registered")
        if not handle.strip():
            raise SeasonError("a fighter needs a handle")
        entrant = Entrant(handle=handle, loadout=loadout)
        self.entrants[handle] = entrant
        return entrant

    def __getitem__(self, handle: str) -> Entrant:
        if handle not in self.entrants:
            raise SeasonError(f"{handle!r} is not registered")
        return self.entrants[handle]

    # -- running a week -------------------------------------------------------------------------------
    def _pairing_state(self) -> PairingState:
        return PairingState(
            scores={h: e.score for h, e in self.entrants.items()},
            conservative={h: e.rating.conservative for h, e in self.entrants.items()},
            played={h: set(e.opponents) for h, e in self.entrants.items()},
            byes={h for h, e in self.entrants.items() if e.byes > 0},
        )

    def pair_week(self) -> list[Fixture]:
        """Pair the next week. Does not advance the clock — :meth:`report_week` does."""
        if self.week >= self.weeks:
            raise SeasonError(f"{self.name} is over; it ran {self.weeks} weeks")
        if not self.entrants:
            raise SeasonError("nobody is registered")
        return pair_round(self.week + 1, list(self.entrants), self._pairing_state())

    def report_week(self, results: Sequence[MatchResult]) -> None:
        """Apply a week's results and rate everybody, as one Glicko-2 rating period.

        Every entrant is rated, including those who did not fight: Glicko-2 grows the RD of a fighter
        who sat out, which is how the system says it is less sure about them.
        """
        if self.week >= self.weeks:
            raise SeasonError(f"{self.name} is over; it ran {self.weeks} weeks")

        # Snapshot the ratings first. Everyone in a period is rated against the ratings their
        # opponents held at the *start* of it — rating in sequence would make the order matter.
        before = {h: e.rating for h, e in self.entrants.items()}
        period: dict[str, list[Result]] = {h: [] for h in self.entrants}

        for result in results:
            fixture = result.fixture
            for handle in fixture.fighters():
                if handle not in self.entrants:
                    raise SeasonError(f"result names {handle!r}, who is not registered")

            if fixture.is_bye:
                entrant = self[fixture.home]
                entrant.byes += 1
                entrant.score += SCORE_BYE
                continue

            home, away = self[fixture.home], self[fixture.away]
            for a, b in ((home, away), (away, home)):
                a.played += 1
                a.opponents.add(b.handle)

            if result.winner is None:
                home.drawn += 1
                away.drawn += 1
                home.score += SCORE_DRAW
                away.score += SCORE_DRAW
                period[home.handle].append(Result(before[away.handle], SCORE_DRAW))
                period[away.handle].append(Result(before[home.handle], SCORE_DRAW))
            else:
                winner = self[result.winner]
                loser = away if result.winner == home.handle else home
                winner.won += 1
                loser.lost += 1
                winner.score += SCORE_WIN
                period[winner.handle].append(Result(before[loser.handle], SCORE_WIN))
                period[loser.handle].append(Result(before[winner.handle], SCORE_LOSS))

        for handle, entrant in self.entrants.items():
            entrant.rating = self.glicko.rate(before[handle], period[handle])

        self.fixtures.extend(r.fixture for r in results)
        self.results.extend(results)
        self.week += 1

    # -- reading it -----------------------------------------------------------------------------------
    def table(self) -> list[Entrant]:
        """Ranked fighters, best first, ordered by **conservative rating** (`spec/season.md`)."""
        return sorted(
            (e for e in self.entrants.values() if e.is_ranked),
            key=lambda e: (-e.rating.conservative, e.handle),
        )

    def provisional(self) -> list[Entrant]:
        """Everyone not yet at the 8-match threshold. Hidden from the table, not excluded from it."""
        return sorted(
            (e for e in self.entrants.values() if not e.is_ranked),
            key=lambda e: (-e.rating.conservative, e.handle),
        )

    def playoff(self) -> list[tuple[str, str]]:
        """Semi-finals for the top four: 1v4 and 2v3. Only ranked fighters are eligible.

        An unbeaten fighter with three matches has not earned a title shot (`spec/season.md`).
        """
        top = self.table()[:PLAYOFF_SIZE]
        if len(top) < PLAYOFF_SIZE:
            raise SeasonError(
                f"only {len(top)} ranked fighter(s); a playoff needs {PLAYOFF_SIZE}. "
                f"{len(self.provisional())} are still provisional"
            )
        return [(top[0].handle, top[3].handle), (top[1].handle, top[2].handle)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": SPEC_VERSION,
            "name": self.name,
            "weeks": self.weeks,
            "weeks_run": self.week,
            "table": [e.to_dict() for e in self.table()],
            "provisional": [e.to_dict() for e in self.provisional()],
            "fixtures": [asdict(f) for f in self.fixtures],
            "results": [
                {
                    "week": r.fixture.week,
                    "home": r.fixture.home,
                    "away": r.fixture.away,
                    "winner": r.winner,
                    "match_id": r.match_id,
                }
                for r in self.results
            ],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


def format_table(entrants: Sequence[Entrant], title: str = "table") -> str:
    """The table as text. Used by the simulator and by anything without a browser."""
    lines = [
        f"{title}",
        f"{'#':>3} {'handle':<16} {'rating':>7} {'rd':>6} {'interval':>17} "
        f"{'P':>3} {'W':>3} {'D':>3} {'L':>3} {'bye':>4} {'score':>6}",
    ]
    for rank, e in enumerate(entrants, start=1):
        low, high = e.rating.interval
        lines.append(
            f"{rank:>3} {e.handle:<16} {e.rating.rating:>7.0f} {e.rating.rd:>6.0f} "
            f"{f'[{low:.0f}, {high:.0f}]':>17} {e.played:>3} {e.won:>3} {e.drawn:>3} "
            f"{e.lost:>3} {e.byes:>4} {e.score:>6.1f}"
        )
    return "\n".join(lines)


def results_from_scores(
    fixtures: Sequence[Fixture], winners: Mapping[int, str | None], match_ids: Mapping[int, str]
) -> list[MatchResult]:
    """Turn ``fixture index -> winner`` into results. The bridge from `league/scoring.py`."""
    return [
        MatchResult(fixture=f, winner=winners.get(i), match_id=match_ids.get(i))
        for i, f in enumerate(fixtures)
    ]
