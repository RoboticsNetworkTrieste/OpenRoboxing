"""The match loop: rounds, the clock, knockdowns, and the record (M3-T4).

Implements ``spec/match_record.md`` v0.1. The rules live here and nowhere else — the world only
supplies physics, contacts and a trace.

The rule that shapes everything
-------------------------------
**A knockout ends the round, not the match.** Both fighters reset and the next round starts, so a
match always runs its full three rounds. That removes count-outs, the three-knockdown rule and any
special case for a knockout in the final round.

Driven through a world, not tied to one
---------------------------------------
:class:`Match` takes anything satisfying :class:`MatchWorld`. Physics is one implementation; a
recorded trace replayed back through the same loop is another, and that is how a disputed match gets
re-scored without re-simulating it. It also means every rule below is testable without a GPU.

Scoring hooks only
------------------
This emits events and computes no winner (`WORKPLAN` M3-T4). A round knows it ended by bell or by
knockout; what that is *worth* is M5-T2's.

Conventions
-----------
- **All times are ticks** at ``TICK_HZ``, matching every other tick in the project.
- **A knockdown is an episode**: contiguous ticks during which a fighter is down. It becomes a
  knockout when that run reaches ``get_up_window_ticks``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from openroboxing.runtime.arena import FIGHTERS
from openroboxing.runtime.contact import ContactTracker, FightTrace, HitEvent
from openroboxing.spec.constants import TICK_HZ

SCHEMA_VERSION = "0.2"

#: How a round finished.
ROUND_ENDINGS = ("bell", "knockout")


class MatchError(RuntimeError):
    """A match could not be run or recorded. Never recovered from silently."""


@dataclass(frozen=True)
class MatchFormat:
    """The numbers a match is fought under. Travels in the record, so an old match stays readable.

    Defaults are `spec/match_record.md` v0.1: three 60 s rounds and boxing's eight-count.
    """

    rounds: int = 3
    round_ticks: int = int(60 * TICK_HZ)
    get_up_window_ticks: int = int(8 * TICK_HZ)
    tick_hz: float = float(TICK_HZ)

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise MatchError(f"a match needs at least one round, got {self.rounds}")
        if self.round_ticks < 1:
            raise MatchError(f"round_ticks must be positive, got {self.round_ticks}")
        if self.get_up_window_ticks < 1:
            raise MatchError(
                f"get_up_window_ticks must be positive, got {self.get_up_window_ticks}"
            )
        if self.get_up_window_ticks > self.round_ticks:
            raise MatchError(
                f"a get-up window of {self.get_up_window_ticks} ticks cannot fit in a "
                f"{self.round_ticks}-tick round; no count could ever complete"
            )

    @property
    def round_seconds(self) -> float:
        return self.round_ticks / self.tick_hz

    @property
    def get_up_seconds(self) -> float:
        return self.get_up_window_ticks / self.tick_hz


@dataclass(frozen=True)
class KnockdownEvent:
    """One trip to the canvas. ``became_knockout`` is what separates a slip from a count."""

    fighter: str
    start_tick: int
    end_tick: int
    lowest_torso_height_m: float
    min_upright: float
    became_knockout: bool

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick + 1


class MatchWorld(Protocol):
    """What the match loop needs from a world. Physics is one implementation; a replay is another."""

    def reset_round(self, index: int) -> None:
        """Put both fighters back in their starting stance for round ``index``."""

    def step(self, tick: int) -> None:
        """Advance one control tick."""

    def observe(self, tracker: ContactTracker, trace: FightTrace, tick: int) -> None:
        """Record this tick into the tracker and the trace."""

    def qpos(self) -> np.ndarray:
        """Both fighters' qpos, concatenated, for the state trace."""

    def commits(self) -> list[dict[str, Any]]:
        """Commits issued in the current round, as plain dicts. Read once, at the bell."""


class KnockdownDetector:
    """Turns per-tick down/up into knockdown episodes, and a long one into a knockout.

    Separate from the match loop because it is the one piece of rules with real state, and because a
    disputed knockdown should be re-derivable from a trace alone.
    """

    def __init__(self, get_up_window_ticks: int) -> None:
        if get_up_window_ticks < 1:
            raise MatchError(f"get_up_window_ticks must be positive, got {get_up_window_ticks}")
        self.window = get_up_window_ticks
        self._open: dict[str, dict[str, Any]] = {}
        self.events: list[KnockdownEvent] = []
        self.knocked_out: str | None = None

    def observe(self, trace: FightTrace, index: int, tick: int) -> str | None:
        """Read one tick of the trace. Returns the fighter knocked out at this tick, if any."""
        for fighter in FIGHTERS:
            if fighter not in trace.torso_height_m:
                continue
            down = trace.is_down(fighter, index)
            episode = self._open.get(fighter)

            if down:
                height = trace.torso_height_m[fighter][index]
                upright = trace.torso_upright[fighter][index]
                if episode is None:
                    episode = {
                        "start": tick,
                        "last": tick,
                        "lowest": height,
                        "min_upright": upright,
                    }
                    self._open[fighter] = episode
                episode["last"] = tick
                episode["lowest"] = min(episode["lowest"], height)
                episode["min_upright"] = min(episode["min_upright"], upright)

                if episode["last"] - episode["start"] + 1 >= self.window:
                    self._close(fighter, became_knockout=True)
                    self.knocked_out = fighter
                    return fighter
            elif episode is not None:
                # Back up inside the window: a knockdown, not a knockout.
                self._close(fighter, became_knockout=False)
        return None

    def flush(self) -> list[KnockdownEvent]:
        """Close any knockdown still open.

        A fighter down when the bell goes has **not** been counted out — the round ended first — so
        the episode closes with ``became_knockout = False``. `spec/match_record.md` says so
        explicitly because it is the case the rule as stated does not cover.
        """
        for fighter in list(self._open):
            self._close(fighter, became_knockout=False)
        return self.events

    def _close(self, fighter: str, became_knockout: bool) -> None:
        episode = self._open.pop(fighter)
        self.events.append(
            KnockdownEvent(
                fighter=fighter,
                start_tick=episode["start"],
                end_tick=episode["last"],
                lowest_torso_height_m=float(episode["lowest"]),
                min_upright=float(episode["min_upright"]),
                became_knockout=became_knockout,
            )
        )


@dataclass
class RoundRecord:
    """One round, as `spec/match_record.md` defines it."""

    index: int
    ticks: int
    ended_by: str
    knocked_out: str | None
    hits: list[HitEvent] = field(default_factory=list)
    knockdowns: list[KnockdownEvent] = field(default_factory=list)
    commits: list[dict[str, Any]] = field(default_factory=list)
    trace: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    def to_dict(self, include_trace: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "index": self.index,
            "ticks": self.ticks,
            "ended_by": self.ended_by,
            "knocked_out": self.knocked_out,
            "hits": [asdict(h) for h in self.hits],
            "knockdowns": [asdict(k) for k in self.knockdowns],
            "commits": list(self.commits),
        }
        if include_trace:
            out["trace"] = self.trace.tolist()
        return out


@dataclass
class MatchRecord:
    """The only output of a match. Everything downstream is resolved from this."""

    match_id: str
    format: MatchFormat
    fighters: dict[str, dict[str, Any]]
    versions: dict[str, str]
    seeds: dict[str, int]
    rounds: list[RoundRecord] = field(default_factory=list)
    #: The ring this was fought in, as ``ArenaConfig``. A trace is ``qpos`` and nothing else, so a
    #: replay has to rebuild the ring around it — see `spec/match_record.md` 0.2.
    arena: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self, include_trace: bool = False) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "match_id": self.match_id,
            "format": asdict(self.format),
            "arena": self.arena,
            "fighters": self.fighters,
            "versions": self.versions,
            "seeds": self.seeds,
            "rounds": [r.to_dict(include_trace) for r in self.rounds],
        }

    def knockouts(self) -> list[tuple[int, str]]:
        """``(round index, fighter)`` for every round that ended in a knockout."""
        return [(r.index, r.knocked_out) for r in self.rounds if r.ended_by == "knockout"]

    def hits_by(self, fighter: str) -> list[HitEvent]:
        return [h for r in self.rounds for h in r.hits if h.attacker == fighter]

    def save(self, path: Path) -> Path:
        """Write the record. The trace goes beside it as ``.npz``, being far too big for JSON.

        `spec/match_record.md` makes the trace authoritative, so it is written first: a record whose
        JSON exists without its trace would look complete and replay wrong.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        trace_path = path.with_suffix(".trace.npz")
        np.savez_compressed(
            trace_path, **{f"round_{r.index}": r.trace for r in self.rounds}
        )
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return trace_path


def _arena_of(world: Any) -> dict[str, Any]:
    """The world's ``ArenaConfig`` as plain data, or ``{}`` for a world that has no ring.

    A stub world in a rules test has no arena and does not need one; a record from it simply says so
    rather than inventing dimensions. :meth:`RecordedMatch.arena_config` reads an absent ``arena`` as
    the defaults, which is what such a record was fought in.
    """
    config = getattr(world, "config", None)
    if config is None:
        return {}
    try:
        return asdict(config)
    except TypeError:  # not a dataclass; nothing meaningful to record
        return {}


class Match:
    """Runs a match to `spec/match_record.md`."""

    def __init__(
        self,
        world: MatchWorld,
        match_id: str = "match",
        match_format: MatchFormat | None = None,
        fighters: dict[str, dict[str, Any]] | None = None,
        versions: dict[str, str] | None = None,
        seeds: dict[str, int] | None = None,
        arena: dict[str, Any] | None = None,
    ) -> None:
        self.world = world
        self.format = match_format or MatchFormat()
        self.record = MatchRecord(
            match_id=match_id,
            format=self.format,
            fighters=fighters or {f: {"handle": f} for f in FIGHTERS},
            versions=versions or {},
            seeds=seeds or {},
            # Asked of the world rather than passed in: the world is what actually built the ring,
            # so a record cannot claim a config the fight was not held in.
            arena=arena if arena is not None else _arena_of(world),
        )

    def run_round(self, index: int) -> RoundRecord:
        """One round: to the bell, or to a knockout, whichever comes first."""
        self.world.reset_round(index)

        tracker = ContactTracker()
        trace = FightTrace()
        detector = KnockdownDetector(self.format.get_up_window_ticks)
        states: list[np.ndarray] = []

        ended_by, knocked_out, ticks = "bell", None, self.format.round_ticks
        for tick in range(self.format.round_ticks):
            self.world.step(tick)
            self.world.observe(tracker, trace, tick)
            states.append(np.asarray(self.world.qpos(), dtype=np.float32))

            if detector.observe(trace, len(trace.tick) - 1, tick) is not None:
                ended_by, knocked_out, ticks = "knockout", detector.knocked_out, tick + 1
                break

        tracker.flush()
        detector.flush()

        return RoundRecord(
            index=index,
            ticks=ticks,
            ended_by=ended_by,
            knocked_out=knocked_out,
            hits=list(tracker.events),
            knockdowns=list(detector.events),
            commits=self.world.commits(),
            trace=np.asarray(states, dtype=np.float32),
        )

    def run(self) -> MatchRecord:
        """Every round, always. A knockout ends its round and the match carries on."""
        for index in range(self.format.rounds):
            self.record.rounds.append(self.run_round(index))
        return self.record
