"""Scoring a recorded match (M5-T2).

Implements ``spec/scoring.md`` v0.1.

Reads a :class:`~openroboxing.runtime.match.MatchRecord` and nothing else — no simulator, no
checkpoint, no policy. That is the whole design: a rule change re-scores the entire archive in
seconds, and two people running the same version over the same record get the same table.

The four dimensions are ``WORKPLAN`` M5-T2's: landed impulses, knockdowns, ring control, aggression.
**Ring control and aggression had no agreed definition when this was written**; the ones used here
are `spec/scoring.md`'s and are logged in `docs/ASSUMPTIONS.md` §A1 as decisions taken without the
project owner.

Conventions
-----------
- Every dimension is a **rate or a fraction**, never a raw count, so a round cut short by a knockout
  is not penalised for being short.
- A score is a *derivation*, written beside a record and never into it. Re-scoring never rewrites
  history.
- Input is the record as a dict — what ``MatchRecord.to_dict()`` produces and what a JSON archive
  holds — so a score can be taken of a match this process did not run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from openroboxing.runtime.arena import FIGHTERS
from openroboxing.spec.constants import TICK_HZ

SPEC_VERSION = "0.1"

#: Region weights for landed impulse. Arms are zero on purpose: a punch stopped by the guard is the
#: guard working, and paying for it would reward throwing at a guard.
REGION_WEIGHTS: Mapping[str, float] = {
    "head": 1.0,
    "body": 0.7,
    "arm": 0.0,
    "leg": 0.0,
    "other": 0.0,
}

#: Pelvis separation at which two fighters can exchange, metres. Measured, not chosen: the G1's hand
#: reaches 0.38 m forward of its own pelvis (`studio/pose_ik.py`), so two of them reach ~0.76 m.
CONTACT_RANGE_M = 0.80

#: Separation beyond ``CONTACT_RANGE_M`` at which engagement decays to nothing, metres. Reaches zero
#: at 2.0 m, about the distance the fighters start at.
ENGAGEMENT_FALLOFF_M = 1.20

#: How much further than contact range a commit still counts as thrown *at* someone.
AGGRESSION_REACH = 1.6

#: Commits per minute that score 1.0 on aggression. Derived: a fighter always committing manages
#: 28-75/min (`spec/rates.md`), so this is about a fifth of the maximum.
TARGET_COMMIT_RATE = 12.0

#: Aggression cannot be farmed past this, so commit-spam does not outscore boxing.
MAX_AGGRESSION = 1.5

#: What each dimension is worth in the round score.
DIMENSION_WEIGHTS: Mapping[str, float] = {
    "damage": 0.50,
    "control": 0.25,
    "aggression": 0.25,
}

#: Round scores are equal within this margin, rather than decided by floating-point noise.
DRAW_MARGIN = 0.02

#: The 10-point must: what the loser of a round gets.
POINTS_WINNER = 10
POINTS_ON_POINTS = 9
POINTS_ONE_KNOCKDOWN = 8
POINTS_HEAVY = 7


class ScoringError(RuntimeError):
    """A record could not be scored. Never recovered from silently."""


@dataclass(frozen=True)
class Dimensions:
    """One fighter's four numbers for one round, before they are weighed against the opponent's."""

    damage: float
    knockdowns_against_opponent: int
    knockouts_against_opponent: int
    control: float
    aggression: float
    commits: int
    commits_in_range: int


@dataclass
class RoundScore:
    """One round, judged."""

    index: int
    points: dict[str, int]
    dimensions: dict[str, Dimensions]
    winner: str | None
    margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "points": dict(self.points),
            "winner": self.winner,
            "margin": self.margin,
            "dimensions": {f: asdict(d) for f, d in self.dimensions.items()},
        }


@dataclass
class MatchScore:
    """A whole match, judged. Written beside a record, never into it."""

    match_id: str
    spec_version: str = SPEC_VERSION
    rounds: list[RoundScore] = field(default_factory=list)

    @property
    def points(self) -> dict[str, int]:
        return {f: sum(r.points.get(f, 0) for r in self.rounds) for f in FIGHTERS}

    @property
    def winner(self) -> str | None:
        """The fighter with more round points, or ``None`` for a draw.

        No countback. A tiebreak hierarchy would be a rule nobody has agreed to, and a draw is an
        honest answer (`spec/scoring.md`).
        """
        totals = self.points
        best = max(totals.values())
        leaders = [f for f, p in totals.items() if p == best]
        return leaders[0] if len(leaders) == 1 else None

    def rounds_won(self, fighter: str) -> int:
        return sum(1 for r in self.rounds if r.winner == fighter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "match_id": self.match_id,
            "points": self.points,
            "winner": self.winner,
            "rounds": [r.to_dict() for r in self.rounds],
        }


# -- the dimensions -------------------------------------------------------------------------------------
def damage(hits: Sequence[Mapping[str, Any]], fighter: str) -> float:
    """Weighted landed impulse for one fighter.

    Impulse, not peak force: a peak is one substep's spike and is dominated by contact stiffness, so
    a glancing touch at an unlucky timestep can out-peak a real punch.
    """
    total = 0.0
    for hit in hits:
        if hit.get("attacker") != fighter:
            continue
        weight = REGION_WEIGHTS.get(hit.get("region", "other"))
        if weight is None:
            raise ScoringError(
                f"hit on region {hit.get('region')!r} has no weight; "
                f"spec/scoring.md knows {sorted(REGION_WEIGHTS)}"
            )
        total += float(hit.get("impulse_ns", 0.0)) * weight
    return total


def engagement(separation_m: float) -> float:
    """How much two fighters at this separation are actually fighting. 1 in range, 0 far apart."""
    if separation_m <= CONTACT_RANGE_M:
        return 1.0
    decayed = 1.0 - (separation_m - CONTACT_RANGE_M) / ENGAGEMENT_FALLOFF_M
    return max(0.0, min(1.0, decayed))


def ring_control(trace, fighter: str, opponent: str) -> float:
    """Fraction of the round spent holding the middle *while pressuring the opponent*.

    Both halves are needed. Centre advantage is a comparison rather than a distance, so it means the
    same thing in any ring size; the engagement weight is what stops standing alone in the middle
    from counting.
    """
    if not trace.tick:
        return 0.0
    mine = trace.centre_distance_m.get(fighter)
    theirs = trace.centre_distance_m.get(opponent)
    if mine is None or theirs is None:
        raise ScoringError(f"the trace has no centre distance for {fighter!r} or {opponent!r}")

    total = 0.0
    for index in range(len(trace.tick)):
        nearer = 1.0 if mine[index] < theirs[index] else 0.0
        total += nearer * engagement(trace.separation_m[index])
    return total / len(trace.tick)


def aggression(
    commits: Sequence[Mapping[str, Any]],
    fighter: str,
    ticks: int,
    separation_at: Sequence[float] | None,
) -> tuple[float, int, int]:
    """``(score, commits, commits in range)`` for one fighter.

    A commit counts if the fighters were within reach **when the punch was thrown** — `strike_at`,
    the moment the fighter arrived and the pose fired, not when the move was issued and not when it
    started walking. Committing at range and closing is boxing; committing at range and staying there
    is not.

    Since `spec/intent.md` 1.1 that distinction does real work: `commit_at` is now the start of a walk
    that may run for seconds, so scoring on it would credit a fighter for the distance it was still
    covering. A commit whose `strike_at` is ``None`` never threw before the bell and scores nothing.

    Records written before 1.1 have no `strike_at` at all; for those `commit_at` **was** the moment
    the move fired, so it is read instead. Absent and null mean different things here and are not
    conflated.

    Without a separation series (a record whose trace is unavailable) every commit counts, and the
    score is an upper bound. That is stated rather than silently assumed.
    """
    if ticks <= 0:
        raise ScoringError(f"a round must have run for at least one tick, got {ticks}")

    mine = [c for c in commits if c.get("fighter") == fighter]
    reach = CONTACT_RANGE_M * AGGRESSION_REACH

    in_range = 0
    for commit in mine:
        thrown_at = commit["strike_at"] if "strike_at" in commit else commit.get("commit_at")
        if thrown_at is None:
            continue  # walked until the bell and never threw
        if separation_at is None:
            in_range += 1
            continue
        at = int(thrown_at)
        if 0 <= at < len(separation_at) and separation_at[at] <= reach:
            in_range += 1

    minutes = ticks / TICK_HZ / 60.0
    rate = in_range / minutes if minutes > 0 else 0.0
    return min(MAX_AGGRESSION, rate / TARGET_COMMIT_RATE), len(mine), in_range


# -- putting a round together ----------------------------------------------------------------------------
def _knockdowns(knockdowns: Sequence[Mapping[str, Any]], against: str) -> tuple[int, int]:
    episodes = [k for k in knockdowns if k.get("fighter") == against]
    return len(episodes), sum(1 for k in episodes if k.get("became_knockout"))


def _loser_points(knockdowns_against: int, knocked_out: bool) -> int:
    """The 10-point must, from `spec/scoring.md`."""
    if knocked_out or knockdowns_against >= 2:
        return POINTS_HEAVY
    if knockdowns_against == 1:
        return POINTS_ONE_KNOCKDOWN
    return POINTS_ON_POINTS


def score_round(
    round_data: Mapping[str, Any],
    trace,
    fighters: Sequence[str] = FIGHTERS,
) -> RoundScore:
    """Judge one round.

    Args:
        round_data: a ``RoundRecord`` as a dict.
        trace: a :class:`~openroboxing.runtime.contact.FightTrace` for the round, or ``None``.
            Without it, ring control is zero for both and aggression is an upper bound.
    """
    if len(fighters) != 2:
        raise ScoringError(f"a round is scored between two fighters, got {list(fighters)}")

    ticks = int(round_data.get("ticks", 0))
    hits = round_data.get("hits", [])
    commits = round_data.get("commits", [])
    knockdowns = round_data.get("knockdowns", [])
    separation = list(trace.separation_m) if trace is not None and trace.tick else None

    dimensions: dict[str, Dimensions] = {}
    for fighter in fighters:
        opponent = next(f for f in fighters if f != fighter)
        against, knocked = _knockdowns(knockdowns, against=opponent)
        score, issued, in_range = aggression(commits, fighter, ticks, separation)
        dimensions[fighter] = Dimensions(
            damage=damage(hits, fighter),
            knockdowns_against_opponent=against,
            knockouts_against_opponent=knocked,
            control=ring_control(trace, fighter, opponent) if trace is not None else 0.0,
            aggression=score,
            commits=issued,
            commits_in_range=in_range,
        )

    # Damage is normalised within the round, so a cagey round and a brawl are scored out of the same
    # total. Neither landing anything is 0.5/0.5 and the round is decided on the other two.
    total_damage = sum(d.damage for d in dimensions.values())
    weighted = {}
    for fighter, d in dimensions.items():
        share = d.damage / total_damage if total_damage > 0 else 0.5
        weighted[fighter] = (
            DIMENSION_WEIGHTS["damage"] * share
            + DIMENSION_WEIGHTS["control"] * d.control
            + DIMENSION_WEIGHTS["aggression"] * min(1.0, d.aggression)
        )

    first, second = fighters
    margin = weighted[first] - weighted[second]
    if abs(margin) <= DRAW_MARGIN:
        winner = None
        points = {first: POINTS_WINNER, second: POINTS_WINNER}
    else:
        winner = first if margin > 0 else second
        loser = second if margin > 0 else first
        knocked_out = round_data.get("knocked_out") == loser
        against, _ = _knockdowns(knockdowns, against=loser)
        points = {
            winner: POINTS_WINNER,
            loser: _loser_points(against, knocked_out),
        }

    return RoundScore(
        index=int(round_data.get("index", 0)),
        points=points,
        dimensions=dimensions,
        winner=winner,
        margin=float(margin),
    )


def score_match(record: Mapping[str, Any], traces: Mapping[int, Any] | None = None) -> MatchScore:
    """Judge a whole match.

    Args:
        record: a ``MatchRecord`` as a dict.
        traces: round index -> :class:`~openroboxing.runtime.contact.FightTrace`. Optional, but ring
            control needs it; see :func:`traces_from_replay`.
    """
    fighters = tuple(record.get("fighters", {})) or FIGHTERS
    if len(fighters) != 2:
        raise ScoringError(f"a match is scored between two fighters, got {list(fighters)}")

    score = MatchScore(match_id=str(record.get("match_id", "match")))
    for round_data in record.get("rounds", []):
        trace = (traces or {}).get(int(round_data.get("index", 0)))
        score.rounds.append(score_round(round_data, trace, fighters))
    return score


def traces_from_replay(recorded) -> dict[int, Any]:
    """Rebuild each round's :class:`FightTrace` by replaying the state trace.

    Ring control is defined on positions, which the record's JSON does not carry — only the state
    trace does. Replaying is cheap (no policy, no generator) and exact for positions, which is
    exactly what this dimension needs (`spec/match_record.md` §"What replays").
    """
    from openroboxing.runtime.contact import ContactTracker, FightTrace
    from openroboxing.runtime.replay import ReplayWorld

    world = ReplayWorld(recorded)
    traces: dict[int, Any] = {}
    for index in sorted(recorded.traces):
        world.reset_round(index)
        trace = FightTrace()
        tracker = ContactTracker()
        for tick in range(world.ticks):
            world.step(tick)
            world.observe(tracker, trace, tick)
        traces[index] = trace
    return traces
