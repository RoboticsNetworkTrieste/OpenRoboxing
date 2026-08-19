"""The agent API: a program in a seat (M5-T4).

An agent is **just a client**. It speaks ``spec/protocol.md`` over the same websocket a browser uses,
sees exactly what a human sees, and is subject to the same rules. `WORKPLAN` M5-T4: "same intent
structure over the same transport", which is what makes human-vs-agent exhibitions and imitation
learning from human matches free rather than a separate integration.

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
        """Zero or more client messages, given one ``state`` frame."""


#: Pose names that move a fighter without throwing anything. A commit needs *some* pose, and closing
#: distance means committing a stance at a spot rather than a punch (`spec/intent.md` 1.0).
STANCE_POSES = ("guard", "cover")


class BaselineAgent:
    """The reference opponent: it walks in, it throws, it varies, and it does not punch the air.

    Deliberately simple and deliberately *legible* — this is the thing a submitted agent must beat,
    so it should be obvious what it does:

    - It aims every punch at striking distance, judged from the same ``CONTACT_RANGE_M`` scoring uses.
    - It cycles its strikes rather than spamming one slot, because a repeated move is the easiest
      thing in the game to read.
    - It waits a beat between commits, so it is not merely rate-limited into looking patient.

    Distance under the queued model
    -------------------------------
    There is no steering to do: **a commit is "go here and do this"**, and since ``spec/intent.md``
    1.1 it walks the whole way rather than one plan's worth. So the agent does not close and then
    punch as two commits — it aims every punch at striking distance from the opponent and lets the
    walk happen inside the commit. Only backing out of a clinch spends a commit on a stance, because
    throwing while retreating is not a punch anybody lands.

    It measures from its **anchor** — where its queue leaves it — rather than from where it is
    standing now, because that is where its next move actually starts. ``separation_m`` is the
    fallback for a state that carries no positions.
    """

    #: Ticks to wait after a commit before offering another. Not a cooldown on the fighter — the
    #: queue would accept them — but on the *decision*, so it does not fill the queue in one frame
    #: and then spend four seconds unable to react.
    RECOVERY_TICKS = 25

    #: Close to this, in metres. Two fighters standing settle at 0.99 m and the scorer's contact
    #: range is 0.80 m, so this is just inside reach.
    STRIKE_RANGE_M = 0.75

    #: Close enough that a punch is worth throwing at the scalar-range fallback, where no positions
    #: are available to aim a placement with.
    STEP_DEADBAND_M = 0.25

    #: Back off when closer than this: at a third of a metre the fighters are inside each other's
    #: arms and it stops being boxing.
    CLINCH_RANGE_M = 0.45

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._stance_slot: str | None = None
        self._strike_slots: list[str] = []
        self.reset()

    def reset(self) -> None:
        self._index = self.seed
        self._last_commit_tick = -10_000

    def on_welcome(self, message: Mapping[str, Any]) -> None:
        """Learn which slot is a stance and which are strikes, from the loadout it was dealt.

        Optional in the :class:`Agent` protocol — an agent that does not care never defines it. This
        one does, because "walk over there" and "hit them" are different poses and it has to tell
        them apart without hard-coding a slot number.
        """
        loadout = dict(message.get("loadout") or {})
        self._stance_slot = next(
            (slot for slot, name in sorted(loadout.items()) if name in STANCE_POSES), None
        )
        self._strike_slots = [
            slot for slot, name in sorted(loadout.items()) if name not in STANCE_POSES
        ]

    def _reach(self, state: Mapping[str, Any], seat: str):
        """``(distance, (ux, uy), (ox, oy))`` from this fighter's anchor to the opponent, or ``None``.

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

        ox, oy = float(origin["x"]), float(origin["y"])
        dx = float(target["x"]) - ox
        dy = float(target["y"]) - oy
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return distance, (1.0, 0.0), (ox, oy)
        return distance, (dx / distance, dy / distance), (ox, oy)

    def _next_strike(self, slots: Sequence[str]) -> str | None:
        """The next punch in the cycle.

        Falls back to every slot when no welcome told it which are stances — an agent that has not
        been dealt a loadout should still fight, rather than stand there because a hook it never
        heard about might be a guard.
        """
        choices = self._strike_slots or list(slots)
        if not choices:
            return None
        slot = choices[self._index % len(choices)]
        self._index += 1
        return slot

    def decide(self, state: dict[str, Any], seat: str, slots: Sequence[str]) -> list[dict]:
        me = state.get("seats", {}).get(seat)
        if not me or not slots or state.get("phase") != "fighting":
            return []
        if not me.get("can_commit", False):
            return []

        tick = int(state.get("tick", 0))
        if tick - self._last_commit_tick < self.RECOVERY_TICKS:
            return []

        reach = self._reach(state, seat)
        if reach is None:
            # No positions (an older protocol, or a replay). Fall back to the scalar range and only
            # throw when it says the opponent is reachable; never guess a placement from nothing.
            separation = state.get("separation_m")
            if separation is None or separation > self.STRIKE_RANGE_M + self.STEP_DEADBAND_M:
                return []
            return self._commit(tick, self._next_strike(slots), placement=None)

        distance, (ux, uy), (ox, oy) = reach

        # One target serves both jobs, because a commit is "go here **and** do this": end up at
        # striking distance from the opponent. Further away that closes the gap; inside a clinch the
        # same expression is negative and steps back out.
        travel = distance - self.STRIKE_RANGE_M
        placement = (ox + ux * travel, oy + uy * travel, math.atan2(uy, ux))

        # No "close first, punch second". The commit walks there and throws on arrival, however far
        # that is, so spending a separate commit on the approach would only cost a queue slot.
        clinched = distance < self.CLINCH_RANGE_M
        slot = self._stance_slot if clinched else self._next_strike(slots)
        return self._commit(tick, slot, placement)

    def _commit(self, tick: int, slot: str | None, placement) -> list[dict]:
        """Place, stage, commit. A missing slot means it cannot act, and it says so by doing nothing.

        With no stance in the loadout it genuinely cannot walk — and committing a hook to cover
        ground would score as aggression it never earned, so refusing is the honest answer.
        """
        if slot is None:
            return []
        self._last_commit_tick = tick

        messages: list[dict] = []
        if placement is not None:
            x, y, heading = placement
            messages.append({"type": "place", "x": x, "y": y, "heading": heading})
        return messages + [{"type": "stage", "slot": slot}, {"type": "commit"}]


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
