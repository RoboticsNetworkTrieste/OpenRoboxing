"""Swiss pairing (M5-T1).

Implements ``spec/season.md`` §Pairing. Each week every registered fighter is paired with somebody on
a similar score, preferring an opponent they have not met.

Deterministic on purpose
------------------------
Pairing sorts by score, then conservative rating, then handle — the last of which is a total order,
so two runs of the same season produce the same fixtures. A league whose fixtures depend on dict
ordering cannot be audited, and "the pairing was unlucky" must never be answerable with "the pairing
was arbitrary".

Conventions
-----------
- **Score** is the Swiss score: 1 win, 0.5 draw, 0 loss, 1 for a bye.
- A **bye** is recorded as a fixture with no opponent. It scores but does not rate and does not count
  towards the 8-match threshold (`spec/season.md`) — a fighter cannot be rated on a match that never
  happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


class PairingError(RuntimeError):
    """A round could not be paired. Never recovered from silently."""


@dataclass(frozen=True)
class Fixture:
    """One scheduled match, or a bye when ``away`` is ``None``."""

    week: int
    home: str
    away: str | None = None

    @property
    def is_bye(self) -> bool:
        return self.away is None

    def fighters(self) -> tuple[str, ...]:
        return (self.home,) if self.is_bye else (self.home, self.away)

    def __str__(self) -> str:
        return f"w{self.week}: {self.home} bye" if self.is_bye else (
            f"w{self.week}: {self.home} v {self.away}"
        )


@dataclass
class PairingState:
    """What pairing needs to know about the season so far."""

    scores: Mapping[str, float]
    conservative: Mapping[str, float]
    played: Mapping[str, set[str]] = field(default_factory=dict)
    byes: Iterable[str] = field(default_factory=set)

    def have_met(self, a: str, b: str) -> bool:
        return b in self.played.get(a, set())


def _ordered(handles: Sequence[str], state: PairingState) -> list[str]:
    """Strongest first: score, then conservative rating, then handle as the deterministic tiebreak."""
    return sorted(
        handles,
        key=lambda h: (-state.scores.get(h, 0.0), -state.conservative.get(h, 0.0), h),
    )


def _choose_bye(order: Sequence[str], state: PairingState) -> str:
    """The lowest-placed fighter who has not had one; byes spread out rather than repeat."""
    had = set(state.byes)
    for handle in reversed(order):
        if handle not in had:
            return handle
    return order[-1]  # everybody has had one; the bottom of the table gets the next


def pair_round(week: int, handles: Sequence[str], state: PairingState) -> list[Fixture]:
    """Pair one week.

    Walks the standings from the top, pairing each unpaired fighter with the nearest unpaired fighter
    below them they have not already played. **A rematch is allowed rather than leaving anybody
    unpaired**: a fighter with no match gets nothing from the week, which is worse than a repeat
    (`spec/season.md`).
    """
    if len(set(handles)) != len(handles):
        raise PairingError(f"duplicate handles in the entry list: {sorted(handles)}")
    if not handles:
        return []

    order = _ordered(list(handles), state)
    fixtures: list[Fixture] = []

    bye: str | None = None
    if len(order) % 2 == 1:
        bye = _choose_bye(order, state)
        order = [h for h in order if h != bye]

    unpaired = list(order)
    while unpaired:
        home = unpaired.pop(0)
        opponent = next((h for h in unpaired if not state.have_met(home, h)), None)
        if opponent is None:
            # Everyone left has already been played. A repeat beats a blank week.
            opponent = unpaired[0]
        unpaired.remove(opponent)
        fixtures.append(Fixture(week=week, home=home, away=opponent))

    if bye is not None:
        fixtures.append(Fixture(week=week, home=bye, away=None))
    return fixtures


def rematch_count(fixtures: Sequence[Fixture], state: PairingState) -> int:
    """How many of these fixtures repeat a pairing. Reported, never hidden."""
    return sum(1 for f in fixtures if not f.is_bye and state.have_met(f.home, f.away))
