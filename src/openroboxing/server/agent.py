"""The agent API: a program in a seat (M5-T4; :class:`BaselineAgent` rewritten for combinations).

An agent is **just a client**. It speaks ``spec/protocol.md`` over the same websocket a browser uses,
sees exactly what a human sees, and is subject to the same rules. `WORKPLAN` M5-T4: "same intent
structure over the same transport", which is what makes human-vs-agent exhibitions and imitation
learning from human matches free rather than a separate integration.

:class:`BaselineAgent` is a placeholder opponent, not a trained one
---------------------------------------------------------------------
It exists so a human — or another script — has something to hit while the real opponent stays a
research track (`WORKPLAN` M5): nothing here is learned, every choice is a hand-written heuristic
read straight off ``welcome``'s combination library, and it should be obvious from reading
:meth:`BaselineAgent.decide` what it does and why. Beating it is a low bar on purpose.

Out of process, on purpose
--------------------------
"Sandboxed" is achieved by the agent being a **separate process holding a socket**, not a plugin. It
cannot touch the simulation, cannot read the opponent's staged move (the protocol never sends it),
and cannot stall a tick — the host applies queued intents on its own clock and never waits.

Two limits the host enforces
-----------------------------
- **Rate**: :data:`MAX_MESSAGES_PER_SECOND` client messages. Over that, messages are dropped and the
  client is told. A human peaks around 5/s; the limit is far above that and far below what a loop
  could send.
- **Compute budget**: an agent that thinks for longer than :data:`DECISION_BUDGET_S` on a frame
  simply misses it. Not enforced *on* the agent — it cannot be, across a socket — but measured and
  reported, because an agent that habitually misses frames is losing to the clock and should know.

Exhibition, not the table
-------------------------
`WORKPLAN` M5-T4 and the project definition §8 put agents outside the Season 0 table.
:func:`is_exhibition` is the one place that decides it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Mapping, Protocol, Sequence

#: Client messages per second before the host starts dropping them. A human peaks near 5/s.
MAX_MESSAGES_PER_SECOND = 60

#: How long an agent may take to decide on one frame before it has missed it. One 30 Hz frame.
DECISION_BUDGET_S = 1.0 / 30.0

#: Handles with this prefix are agents. Crude and explicit, which beats a flag a client can lie about
#: — the host assigns it, and `M5-T1`'s registration is where a real identity would come from.
AGENT_PREFIX = "agent:"


class AgentError(RuntimeError):
    """An agent could not be run. Never recovered from silently."""


def is_exhibition(handle: str) -> bool:
    """Whether a result belongs in the exhibition list rather than the Season 0 table."""
    return handle.startswith(AGENT_PREFIX)


@dataclass
class RateLimiter:
    """A sliding-window limiter. Drops, never disconnects.

    Disconnecting a noisy agent mid-match would hand its opponent a walkover, which is a worse
    failure than ignoring some of its input.
    """

    limit: int = MAX_MESSAGES_PER_SECOND
    window_s: float = 1.0
    _times: list[float] = field(default_factory=list)
    dropped: int = 0

    def allow(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        cutoff = now - self.window_s
        self._times = [t for t in self._times if t > cutoff]
        if len(self._times) >= self.limit:
            self.dropped += 1
            return False
        self._times.append(now)
        return True


@dataclass
class AgentStats:
    """What an agent cost. Reported so a slow agent knows it is losing to the clock."""

    frames: int = 0
    decisions: int = 0
    over_budget: int = 0
    decision_ms: list[float] = field(default_factory=list)

    @property
    def mean_decision_ms(self) -> float:
        return sum(self.decision_ms) / len(self.decision_ms) if self.decision_ms else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "frames": float(self.frames),
            "decisions": float(self.decisions),
            "over_budget": float(self.over_budget),
            "mean_decision_ms": self.mean_decision_ms,
        }


class Agent(Protocol):
    """What a program in a seat must provide.

    Args to :meth:`decide` are exactly what the protocol delivers — no privileged state.
    """

    def reset(self) -> None:
        """A new round is starting."""

    def decide(self, state: dict[str, Any], seat: str, slots: Sequence[str]) -> list[dict]:
        """Zero or more client messages, given one ``state`` frame.

        ``slots`` is what the loadout era called a fighter's dealt pose names; a combination has
        none to name any more (`spec/intent.md` `D6`). It survives here only so `run_decision` and
        `server/client.py`'s ``AgentConnection`` — which still pass it — do not need a second call
        shape; an :class:`Agent` that has moved on to reading ``welcome``'s library, as
        :class:`BaselineAgent` has, is free to ignore it.
        """


#: A combination's own name, as `tools/import_motions.py::build_from_take` slugs it from the take
#: that produced it — e.g. ``shadow-boxing-r-001-a359-00``, ``ib-dodge-270-r-001-a437-00``,
#: ``ib-combat-turn-jog-start-270-r-003-a437-00``. These are the three families in the corpus under
#: `motions/` today (`CLAUDE.md`); a name outside all three is classified ``"strike"``, both because
#: that family is the corpus majority and because an agent that cannot classify a move should still
#: throw it rather than freeze up over a name it does not recognise.
_DEFENSIVE_PREFIXES = ("ib-dodge",)
_TRAVEL_PREFIXES = ("ib-combat-turn-jog",)


def combination_kind(name: str) -> str:
    """``"defensive"``, ``"travel"`` or ``"strike"`` — read from a combination's own name.

    A move's name is honest about what it is (the corpus is authored, not generated, so the prefix
    is not a guess); reading it is a legitimate, cheap way to pick sensibly and is as far as
    :class:`BaselineAgent` goes with it — anything cleverer than "prefer not to throw a haymaker
    while stuck in a clinch" is out of scope for a placeholder opponent.
    """
    if name.startswith(_DEFENSIVE_PREFIXES):
        return "defensive"
    if name.startswith(_TRAVEL_PREFIXES):
        return "travel"
    return "strike"


class BaselineAgent:
    """The reference opponent: it commits, it varies, and it never asks for a ghost it cannot reach.

    Deliberately simple and deliberately *legible* — this is the thing a submitted agent must beat,
    so it should be obvious what it does:

    - It aims every commit at striking distance from the opponent, judged from the same standoff a
      human would want.
    - It cycles through the combinations that can actually cover that distance, rather than spamming
      one, because a repeated move is the easiest thing in the game to read.
    - It waits a beat between commits, so it is not merely rate-limited into looking patient.

    Choosing under `spec/intent.md` 3.0's reach guard
    ----------------------------------------------------
    A commit is "play this recording, landing its last pose on the ghost" (`spec/intent.md` 3.0), and
    a combination's own recorded duration bounds how far its ghost may sit from the anchor
    (``reach_m``, `spec/protocol.md` §"welcome's combination library"). The host refuses a ghost
    beyond that bound outright — there is no partial credit, no "it gets as close as it can" — so an
    agent that ignored ``reach_m`` would spend its whole match being told "can't get there"
    (`server/protocol.py::check_reach`), which is a much worse opponent than one that simply cannot
    reach as far as a human might place a ghost.

    :meth:`decide` therefore always filters the library to combinations whose own ``reach_m`` covers
    the distance it wants to travel before choosing one, and never asks for more than the one it
    picked can give — see :meth:`_choose`.

    It measures from its **anchor** — where its queue leaves it — rather than from where it is
    standing now, because that is where its next move actually starts (unchanged in spirit since the
    loadout era; `spec/protocol.md`'s ``anchor`` is exactly this).
    """

    #: Ticks to wait after a commit before offering another. Not a cooldown on the fighter — the
    #: queue would accept them — but on the *decision*, so it does not fill the queue in one frame
    #: and then spend four seconds unable to react.
    RECOVERY_TICKS = 25

    #: The standoff this agent aims for, in metres. Two fighters standing settle at 0.99 m and the
    #: scorer's contact range is 0.80 m, so this is just inside reach.
    STRIKE_RANGE_M = 0.75

    #: Back off when closer than this: at under half a metre the fighters are inside each other's
    #: arms and it stops being boxing.
    CLINCH_RANGE_M = 0.45

    #: Shaved off a combination's advertised ``reach_m`` before it is trusted. ``reach_m`` arrives
    #: over the wire already rounded to 3 d.p. (`server/protocol.py::welcome`); this agent recomputes
    #: nothing server-side, so a ghost placed at the rounded boundary could round the wrong way and
    #: trip the host's own, unrounded check. A 2 cm margin costs nothing a human would notice and
    #: means this agent's own placements never test that edge.
    REACH_MARGIN_M = 0.02

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._library: dict[str, dict[str, Any]] = {}
        self.reset()

    def reset(self) -> None:
        """A new round. Forgets its cooldown and its place in the cycle — **not** the library, which
        came from ``welcome`` once per connection and does not change round to round."""
        self._index = self.seed
        self._last_commit_tick = -10_000

    def on_welcome(self, message: Mapping[str, Any]) -> None:
        """Learn the whole shared combination library (`spec/protocol.md` 0.6's ``welcome``).

        Optional in the :class:`Agent` protocol — an agent that does not care never defines it. This
        one does, because every choice it makes reads straight off what a client would show a human:
        ``name`` and ``reach_m`` per entry (`spec/intent.md`'s `D6`: the whole library, not a
        per-seat loadout, so there is nothing to be dealt). Not ``heading_delta`` — since
        `spec/intent.md` 3.1 a fighter faces its opponent whatever the recording turns by, so a
        combination's recorded turn is not something to choose on.
        """
        self._library = {
            str(c["name"]): dict(c) for c in message.get("combinations") or []
        }

    def _reach(self, state: Mapping[str, Any], seat: str):
        """``(distance, (ux, uy), (ax, ay))`` from this fighter's anchor to the opponent, or ``None``.

        Measured from the **anchor**, not the live position: under a queue the next move starts where
        the last one leaves off, so aiming at where the fighter *is* aims at the past.
        """
        seats = state.get("seats") or {}
        me = seats.get(seat) or {}
        others = [name for name in seats if name != seat]
        if len(others) != 1:
            return None
        them = seats[others[0]] or {}

        origin = me.get("anchor") or me.get("position")
        target = them.get("position")
        if not origin or not target:
            return None

        ax, ay = float(origin["x"]), float(origin["y"])
        dx = float(target["x"]) - ax
        dy = float(target["y"]) - ay
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return distance, (1.0, 0.0), (ax, ay)
        return distance, (dx / distance, dy / distance), (ax, ay)

    def _choose(self, needed_m: float, *, prefer: tuple[str, ...] = ()) -> tuple[str, float] | None:
        """The next combination to play, and the reach it actually offers.

        Only ever returns one whose ``reach_m`` can cover ``needed_m`` — unless *nothing* in the
        library can, in which case it returns whichever reaches furthest rather than refusing to act
        at all, so a fighter placed well outside every combination's reach still throws something
        instead of standing there (the ghost is then clamped short in :meth:`decide`, never past what
        was chosen). ``prefer`` narrows the pool to a `combination_kind` first, when at least one
        candidate in it still reaches — used to favour stepping out of a clinch over swinging into it.
        """
        if not self._library:
            return None

        names = sorted(self._library)
        eligible = [n for n in names if self._library[n]["reach_m"] >= needed_m]
        pool = eligible or names

        if prefer:
            narrowed = [n for n in pool if combination_kind(n) in prefer]
            if narrowed:
                pool = narrowed

        if not eligible:
            # Nothing reaches: take whichever goes furthest, out of the (possibly `prefer`-narrowed)
            # pool, so the fallback is still the best available answer rather than an arbitrary one.
            name = max(pool, key=lambda n: self._library[n]["reach_m"])
        else:
            name = pool[self._index % len(pool)]
        self._index += 1
        return name, float(self._library[name]["reach_m"])

    def decide(self, state: dict[str, Any], seat: str, slots: Sequence[str]) -> list[dict]:
        me = state.get("seats", {}).get(seat)
        if not me or not self._library or state.get("phase") != "fighting":
            return []
        if not me.get("can_commit", False):
            return []

        tick = int(state.get("tick", 0))
        if tick - self._last_commit_tick < self.RECOVERY_TICKS:
            return []

        reach = self._reach(state, seat)
        if reach is None:
            # No positions (an older protocol, or a malformed state). Never guess a ghost from
            # nothing (`CLAUDE.md` "fail loudly", read the other way round).
            return []
        distance, (ux, uy), (ax, ay) = reach

        # One signed target serves both jobs, because a commit is "go here **and** do this": end up
        # at striking distance from the opponent. Further away that closes the gap; inside a clinch
        # the same expression is negative and steps back out.
        travel = distance - self.STRIKE_RANGE_M
        clinched = distance < self.CLINCH_RANGE_M

        choice = self._choose(
            abs(travel), prefer=("defensive", "travel") if clinched else ()
        )
        if choice is None:
            return []
        name, reach_m = choice

        # Never ask for more than the chosen combination can give, even when nothing in the library
        # could cover the original target — this is what keeps `check_reach` from ever refusing this
        # agent's own commit (`spec/intent.md` "Feasibility").
        budget = max(0.0, reach_m - self.REACH_MARGIN_M)
        clamped = max(-budget, min(budget, travel))
        ghost = (ax + ux * clamped, ay + uy * clamped)

        self._last_commit_tick = tick
        return [{"type": "intent", "combination": name, "ghost": list(ghost)}, {"type": "commit"}]


class IdleAgent:
    """Commits nothing. The control case for any claim that an agent caused something."""

    def reset(self) -> None:
        return

    def decide(self, state: dict[str, Any], seat: str, slots: Sequence[str]) -> list[dict]:
        return []


def run_decision(agent: Agent, state: dict, seat: str, slots: Sequence[str], stats: AgentStats):
    """Call an agent and time it. Over-budget decisions are counted, not discarded.

    Discarding would be dishonest in the other direction: the message was produced, and the host
    applies it whenever it arrives. What the budget measures is whether the agent is keeping up.
    """
    start = time.perf_counter()
    messages = agent.decide(state, seat, slots)
    elapsed = time.perf_counter() - start

    stats.decisions += 1
    stats.decision_ms.append(elapsed * 1e3)
    if elapsed > DECISION_BUDGET_S:
        stats.over_budget += 1
    return messages or []
