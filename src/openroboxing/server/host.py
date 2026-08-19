"""The match host: one process, one match, two clients (M4-T2).

Implements ``spec/protocol.md`` v0.4 over aiohttp websockets. The host owns the match; a client is a
view and a keyboard and is never trusted.

Three clocks, deliberately different
------------------------------------
- **Simulation** at 50 Hz (`spec/rates.md`), paced against the wall clock because a human is playing.
- **Intent service** at 30 Hz (`WORKPLAN` M4-T2): queued keypresses are applied on this tick, so a
  press lands within 33 ms and the simulation never blocks on a socket.
- **Streaming** at 30 FPS: a binary frame of body transforms and the ``state`` describing it.

Falling behind is reported, not hidden
--------------------------------------
If a tick overruns its 20 ms the host does **not** try to catch up by running two — that would make
the fight speed up after a stall, which is worse than a stutter. It drops the deficit and counts it.
:attr:`MatchHost.stats` carries the count, and the acceptance run prints it.

Every viewer gets its own state
-------------------------------
`spec/protocol.md` 0.4 makes the **JSON** per viewer, because it carries things one seat may see and
another may not: your staged slot and your queued-but-unstarted commits. Broadcasting one message to
every socket would hand the opponent a readable list of your next four moves, which is precisely the
risk queueing is supposed to be. So the subscriber map records the seat each socket is watching as,
and :meth:`MatchHost.broadcast_state` composes the JSON per recipient.

The **binary frame is not** per viewer: it holds the two real fighters and nothing else, so it is
packed once and sent to the whole room. The shadow a player aims with is drawn in the browser and the
host never sees it — it learns a placement only when one is committed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from openroboxing.runtime.arena import FIGHTERS, ArenaConfig
from openroboxing.runtime.contact import ContactTracker, FightTrace
from openroboxing.runtime.fight import FightWorld, Pilot
from openroboxing.runtime.intents import IntentError, IntentTimeline, Loadout, Placement
from openroboxing.runtime.match import (
    KnockdownDetector,
    MatchFormat,
    MatchRecord,
    RoundRecord,
)
from openroboxing.server import protocol
from openroboxing.server.agent import RateLimiter
from openroboxing.server.scene import Scene
from openroboxing.spec.constants import (
    COMMIT_HORIZON_TICKS,
    MAX_OUTSTANDING_COMMITS,
    TICK_DT,
    TICK_HZ,
)

#: Client-facing rates, from `WORKPLAN` M4-T2.
STREAM_FPS = 30
INTENT_HZ = 30

#: How long a round-over pause lasts before the next round, in seconds. There is no rest period in
#: the rules (`spec/match_record.md` leaves it Open); this is only long enough to read the result.
ROUND_BREAK_S = 4.0

#: A tick this far behind schedule is counted as dropped rather than chased.
LATE_TICK_S = TICK_DT * 2

#: How often the live score is recomputed, in ticks. Twice a second: ring control integrates over the
#: whole trace, and nobody can read a number that changes 25 times a second (`spec/protocol.md` 0.2).
SCORE_INTERVAL_TICKS = TICK_HZ // 2


class HostError(RuntimeError):
    """The host could not run a match. Never recovered from silently."""


@dataclass
class HostStats:
    """What the host cost. Printed at the end of a match; a stutter must be visible."""

    ticks: int = 0
    dropped: int = 0
    frames: int = 0
    step_ms: list[float] = field(default_factory=list)
    render_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        return {
            "ticks": float(self.ticks),
            "dropped": float(self.dropped),
            "frames": float(self.frames),
            "mean_step_ms": float(np.mean(self.step_ms)) if self.step_ms else 0.0,
            "p95_step_ms": float(np.percentile(self.step_ms, 95)) if self.step_ms else 0.0,
            "mean_render_ms": float(np.mean(self.render_ms)) if self.render_ms else 0.0,
        }


class QueuedPilot:
    """A :class:`~openroboxing.runtime.fight.Pilot` fed by a websocket.

    Keypresses land in a queue and are applied on the intent tick, never inside the socket handler —
    so a client cannot make the simulation wait, and two clients cannot interleave into the same
    timeline mid-step.
    """

    def __init__(self) -> None:
        self._pending: list[dict[str, Any]] = []
        self.last_error: str | None = None
        self.staged: str | None = None
        #: Where the client last put its shadow, in world coordinates. ``None`` means it has not
        #: moved the shadow since its last commit, which is a real answer: "do this where I will be".
        self.placement: Placement | None = None
        #: Where the queue leaves this fighter. Written by the host each tick before it steps, so a
        #: commit made without touching the shadow lands at the anchor rather than at whatever
        #: absolute point the *previous* commit happened to use.
        self.anchor: Placement | None = None

    def queue(self, message: dict[str, Any]) -> None:
        self._pending.append(message)

    def reset(self) -> None:
        self._pending.clear()
        self.staged = None
        self.placement = None
        self.anchor = None
        self.last_error = None

    def act(self, timeline: IntentTimeline, tick: int) -> None:
        """Apply everything queued since the last tick. Errors are recorded, never raised."""
        pending, self._pending = self._pending, []
        for message in pending:
            try:
                if message["type"] == "stage":
                    timeline.stage(pose_slot=message["slot"])
                    self.staged = message["slot"]
                elif message["type"] == "clear":
                    timeline.clear_pose()
                    self.staged = None
                elif message["type"] == "place":
                    placement = Placement(
                        position=(message["x"], message["y"]), heading=message["heading"]
                    )
                    timeline.stage(placement=placement)
                    self.placement = placement
                elif message["type"] == "commit":
                    if timeline.staged.pose_slot is None:
                        raise IntentError("nothing is staged")
                    # An untouched shadow commits at the anchor. Resolving it here rather than
                    # leaving the previous commit's absolute placement staged is what makes
                    # "commit without aiming" mean "do it where I will be" instead of "do it again
                    # over there".
                    if self.placement is None and self.anchor is not None:
                        timeline.stage(placement=self.anchor)
                    timeline.commit(tick)
                    self.placement = None
            except IntentError as exc:
                self.last_error = str(exc)


class MatchHost:
    """One match, driven live. Built once, run once.

    Args:
        loadouts: what each seat brought.
        match_format: rounds and clock. Defaults to `spec/match_record.md`.
        match_seed: the number a match reproduces from.
        render: whether to produce frames. False for a headless A/B (`WORKPLAN` M4-T2 latency test).
        config: ring geometry and timestep. Must be passed **here** rather than assigned afterwards —
            `build_arena` compiles ring size, rope heights and glove radius into the model, so a
            config set after construction changes the record and nothing a fighter can touch.
        horizon_ticks: the readable window between committing and executing. A `M4-T4` knob.
        max_outstanding: how many commits may be unfinished at once. A `M4-T4` knob, and since
            `spec/intent.md` 1.1 a much heavier one: a move is now its walk plus its pose.
    """

    def __init__(
        self,
        loadouts: dict[str, Loadout],
        *,
        match_format: MatchFormat | None = None,
        match_seed: int = 1234,
        match_id: str = "live",
        render: bool = True,
        pilots: dict[str, Pilot] | None = None,
        config: ArenaConfig | None = None,
        horizon_ticks: int = COMMIT_HORIZON_TICKS,
        max_outstanding: int = MAX_OUTSTANDING_COMMITS,
    ) -> None:
        self.format = match_format or MatchFormat()
        self.match_id = match_id
        self.loadouts = loadouts
        self.pilots: dict[str, Any] = pilots or {f: QueuedPilot() for f in FIGHTERS}
        self.world = FightWorld(
            loadouts=loadouts,
            pilots=self.pilots,
            match_seed=match_seed,
            config=config,
            horizon_ticks=horizon_ticks,
            max_outstanding=max_outstanding,
        )
        self.stats = HostStats()
        self.handles = {f: f for f in FIGHTERS}
        self.phase = protocol.PHASE_FIGHTING
        self.round_index = 0
        self.tick = 0

        self.record = MatchRecord(
            match_id=match_id,
            format=self.format,
            fighters={f: {"handle": f, "loadout": {"name": loadouts[f].name}} for f in FIGHTERS},
            versions={},
            seeds={"match_seed": match_seed},
            arena=self.world.config.__dict__.copy(),
        )

        self._render = render
        self._scene: Scene | None = None
        self._tracker = ContactTracker()
        self._trace = FightTrace()
        self._detector = KnockdownDetector(self.format.get_up_window_ticks)
        self._states: list[np.ndarray] = []
        #: socket -> the seat it watches as, or ``None`` for a spectator. A dict rather than a set
        #: because 0.4 composes state per recipient; see the module docstring.
        self._subscribers: dict[Any, str | None] = {}
        self._limiters: dict[str, RateLimiter] = {}
        self._score_cache: dict[str, Any] | None = None
        self._scored_at = -10_000
        #: Points and rounds won from **completed** rounds only. The round in progress adds nothing
        #: until the bell, because the 10-point must depends on how it finished.
        self._points: dict[str, int] = {f: 0 for f in FIGHTERS}
        self._rounds_won: dict[str, int] = {f: 0 for f in FIGHTERS}

    # -- the scene and its frames --------------------------------------------------------------------
    @property
    def scene(self) -> Scene:
        """The arena, described for a browser. Built from the model the fight is actually running."""
        if self._scene is None:
            self._scene = Scene(self.world.model, self.record.arena)
        return self._scene

    def frame(self) -> bytes | None:
        """One binary frame: every body's world transform. ``None`` when streaming is off.

        The same bytes for everybody. The shadow is drawn in the browser from the pose angles in
        ``welcome``, so a frame holds nothing private and one pack serves the whole room.
        """
        if not self._render:
            return None

        start = time.perf_counter()
        packed = self.scene.pack(self.tick, self.world.data)
        self.stats.render_ms.append((time.perf_counter() - start) * 1e3)
        self.stats.frames += 1
        return packed

    # -- state ---------------------------------------------------------------------------------------
    def seat_states(self, viewer: str | None = None) -> dict[str, dict]:
        """Every seat, as ``viewer`` is allowed to see it.

        A seat sees its own staging, shadow and full queue. Of anybody else it sees only what is
        already visible in the ring: the commit currently executing.
        """
        seats = {}
        for name, fighter in self.world.fighters.items():
            timeline = fighter.timeline
            own = name == viewer
            scheduled = timeline.scheduled(self.tick)
            visible = protocol.visible_queue(scheduled, self.tick, own)
            pilot = self.pilots.get(name)
            index = len(self._trace.tick) - 1
            seats[name] = protocol.seat_state(
                handle=self.handles.get(name, name),
                staged=getattr(pilot, "staged", None) if own else None,
                placement=timeline.staged.placement if own else None,
                anchor=self.world.anchor(name, self.tick) if own else None,
                position=self.world.root_pose(name),
                queue=[protocol.queue_entry(c, self.tick) for c in visible],
                can_commit=len(scheduled) < timeline.max_outstanding,
                hits_landed=sum(1 for e in self._tracker.events if e.attacker == name),
                torso_height_m=(
                    self._trace.torso_height_m[name][index]
                    if index >= 0 and name in self._trace.torso_height_m
                    else 0.0
                ),
                down=(
                    self._trace.is_down(name, index)
                    if index >= 0 and name in self._trace.torso_height_m
                    else False
                ),
            )
        return seats

    def live_score(self) -> dict | None:
        """The round so far, scored by `spec/scoring.md`'s own code (`spec/protocol.md` 0.2).

        Deliberately not a second scoring system. A bespoke "who is winning" number invented for the
        UI would disagree with the official one at the worst possible moment — the bell. So this runs
        :func:`~openroboxing.league.scoring.score_round` over the events so far and reports its
        weighted share, which is provisional and says so.

        Cached between recomputes because ring control integrates over the whole trace, and a client
        cannot perceive 25 recomputes a second in any case.
        """
        from dataclasses import asdict

        from openroboxing.league.scoring import (
            DIMENSION_WEIGHTS,
            DRAW_MARGIN,
            score_round,
        )

        if not self._trace.tick:
            return self._score_cache

        if self._score_cache is not None and self.tick - self._scored_at < SCORE_INTERVAL_TICKS:
            return self._score_cache

        partial = {
            "index": self.round_index,
            "ticks": max(1, self.tick),
            "ended_by": "bell",
            "knocked_out": None,
            "hits": [asdict(h) for h in self._tracker.events],
            "knockdowns": [asdict(k) for k in self._detector.events],
            "commits": self.world.commits(),
        }
        judged = score_round(partial, self._trace, FIGHTERS)

        # The weighted share each fighter holds, normalised so the pair sums to 1 — which is what a
        # scoreboard can actually draw. The underlying dimensions travel too, so a client can explain
        # *why* somebody is ahead rather than just asserting it.
        totals = {}
        for fighter, d in judged.dimensions.items():
            damage_total = sum(x.damage for x in judged.dimensions.values())
            share = d.damage / damage_total if damage_total > 0 else 0.5
            totals[fighter] = (
                DIMENSION_WEIGHTS["damage"] * share
                + DIMENSION_WEIGHTS["control"] * d.control
                + DIMENSION_WEIGHTS["aggression"] * min(1.0, d.aggression)
            )
        overall = sum(totals.values()) or 1.0

        self._score_cache = protocol.live_score(
            share={f: v / overall for f, v in totals.items()},
            dimensions={
                f: {"damage": d.damage, "control": d.control, "aggression": d.aggression}
                for f, d in judged.dimensions.items()
            },
            points=self._points,
            rounds_won=self._rounds_won,
            draw_margin=DRAW_MARGIN,
        )
        self._scored_at = self.tick
        return self._score_cache

    def state_message(self, viewer: str | None = None) -> dict:
        return protocol.state(
            tick=self.tick,
            round_index=self.round_index,
            clock_ticks=max(0, self.format.round_ticks - self.tick),
            seats=self.seat_states(viewer),
            phase=self.phase,
            score=self.live_score(),
            separation_m=self.world.separation_m(),
        )

    def spectator_welcome(self) -> dict:
        """A watcher's welcome: the format and the ring, but no loadout.

        A spectator is told nothing a player would not already have shown by fighting. Sending them
        both loadouts would leak, at a projector, what each fighter has available — and anybody in
        the room can read a projector.
        """
        message = protocol.welcome(
            seat="spectator",
            loadout=type("_Empty", (), {"slots": {}})(),
            match_format=self.format,
            arena=self.record.arena,
            match_id=self.match_id,
        )
        message["handles"] = dict(self.handles)
        return message

    def welcome_message(self, seat: str) -> dict:
        return protocol.welcome(
            seat=seat,
            loadout=self.loadouts[seat],
            match_format=self.format,
            arena=self.record.arena,
            match_id=self.match_id,
        )

    # -- the round -------------------------------------------------------------------------------------
    def start_round(self, index: int) -> None:
        self.round_index = index
        self.tick = 0
        self.phase = protocol.PHASE_FIGHTING
        self.world.reset_round(index)
        self._tracker = ContactTracker()
        self._trace = FightTrace()
        self._detector = KnockdownDetector(self.format.get_up_window_ticks)
        self._states = []
        self._score_cache = None
        self._scored_at = -10_000
        for pilot in self.pilots.values():
            if hasattr(pilot, "reset"):
                pilot.reset()

    def step_once(self) -> str | None:
        """One simulation tick. Returns the fighter knocked out at this tick, if any."""
        start = time.perf_counter()
        # Tell each pilot where its queue leaves it, before the world asks it to act. This is what a
        # commit made without touching the shadow lands on.
        for name, pilot in self.pilots.items():
            if isinstance(pilot, QueuedPilot) and name in self.world.fighters:
                pilot.anchor = self.world.anchor(name, self.tick)
        self.world.step(self.tick)
        self.world.observe(self._tracker, self._trace, self.tick)
        self._states.append(np.asarray(self.world.qpos(), dtype=np.float32))
        knocked_out = self._detector.observe(self._trace, len(self._trace.tick) - 1, self.tick)
        self.stats.step_ms.append((time.perf_counter() - start) * 1e3)
        self.stats.ticks += 1
        self.tick += 1
        return knocked_out

    def finish_round(self, ended_by: str, knocked_out: str | None) -> RoundRecord:
        self._tracker.flush()
        self._detector.flush()
        record = RoundRecord(
            index=self.round_index,
            ticks=self.tick,
            ended_by=ended_by,
            knocked_out=knocked_out,
            hits=list(self._tracker.events),
            knockdowns=list(self._detector.events),
            commits=self.world.commits(),
            trace=np.asarray(self._states, dtype=np.float32),
        )
        self.record.rounds.append(record)

        # Score the finished round with the official scorer and fold it into the running total. This
        # is the number that stops being provisional: the 10-point must depends on how a round ended,
        # so it can only be applied now.
        from openroboxing.league.scoring import score_round

        judged = score_round(record.to_dict(), self._trace, FIGHTERS)
        for fighter, points in judged.points.items():
            self._points[fighter] = self._points.get(fighter, 0) + points
        if judged.winner:
            self._rounds_won[judged.winner] = self._rounds_won.get(judged.winner, 0) + 1
        self._score_cache = None

        return record

    # -- clients ---------------------------------------------------------------------------------------
    def subscribe(self, socket, viewer: str | None = None) -> None:
        """Register a socket and the seat it watches as. ``None`` is a spectator."""
        self._subscribers[socket] = viewer

    def unsubscribe(self, socket) -> None:
        self._subscribers.pop(socket, None)

    async def broadcast_event(self, message: dict) -> None:
        """Send one message to everybody. Events are public — a bell rings for the whole room."""
        for socket in list(self._subscribers):
            try:
                await socket.send_json(message)
            except Exception:
                self._subscribers.pop(socket, None)

    async def broadcast_state(self, with_frame: bool = True) -> None:
        """Send each client the state **it** is allowed to see, and the ring as everyone sees it.

        The JSON is composed per recipient, because a seat's staging and its queued-but-unstarted
        commits are private (`spec/protocol.md` §"Seat state"). The binary frame is not: it holds
        only the two real fighters, so it is packed once and sent to everybody. A dead socket is
        dropped, never allowed to stall a tick.
        """
        frame = self.frame() if with_frame else None
        for socket, viewer in list(self._subscribers.items()):
            try:
                await socket.send_json(self.state_message(viewer))
                if frame is not None:
                    await socket.send_bytes(frame)
            except Exception:
                self._subscribers.pop(socket, None)

    # -- the loop ---------------------------------------------------------------------------------------
    async def run(self) -> MatchRecord:
        """Run the whole match in real time, streaming as it goes."""
        stream_every = max(1, round(TICK_HZ / STREAM_FPS))
        intent_every = max(1, round(TICK_HZ / INTENT_HZ))

        for index in range(self.format.rounds):
            self.start_round(index)
            deadline = time.perf_counter()
            ended_by, knocked_out = "bell", None

            for _ in range(self.format.round_ticks):
                deadline += TICK_DT
                knocked_out = self.step_once()

                if self.tick % stream_every == 0 or knocked_out:
                    await self.broadcast_state()

                if knocked_out is not None:
                    ended_by = "knockout"
                    break

                now = time.perf_counter()
                if now < deadline:
                    await asyncio.sleep(deadline - now)
                elif now - deadline > LATE_TICK_S:
                    # Behind. Drop the deficit rather than run two ticks: a fight that speeds up
                    # after a stall is worse than one that stutters.
                    self.stats.dropped += 1
                    deadline = now

            _ = intent_every  # intents are applied inside world.step via the pilots
            record = self.finish_round(ended_by, knocked_out)
            self.phase = (
                protocol.PHASE_MATCH_OVER
                if index == self.format.rounds - 1
                else protocol.PHASE_ROUND_OVER
            )
            await self.broadcast_event(
                protocol.event(
                    "round_end",
                    round=index + 1,
                    ended_by=record.ended_by,
                    knocked_out=record.knocked_out,
                    hits={f: sum(1 for h in record.hits if h.attacker == f) for f in FIGHTERS},
                )
            )
            await self.broadcast_state()
            if self.phase == protocol.PHASE_ROUND_OVER:
                await asyncio.sleep(ROUND_BREAK_S)

        await self.broadcast_event(protocol.event("match_end", match_id=self.match_id))
        return self.record

    # -- client messages ---------------------------------------------------------------------------------
    def handle(self, seat: str, message: dict) -> dict | None:
        """Apply one validated client message. Returns a reply, or ``None``.

        Rate-limited per seat. A client over the limit has messages **dropped**, never its socket
        closed: disconnecting a noisy agent mid-match would hand its opponent a walkover, which is a
        worse failure than ignoring some input (`server/agent.py`).
        """
        limiter = self._limiters.setdefault(seat, RateLimiter())
        if not limiter.allow():
            return protocol.error(
                f"rate limit: more than {limiter.limit} messages per second",
                rejected=message["type"],
            )

        if message["type"] == "ping":
            return protocol.pong(message.get("t"))
        if message["type"] == "join":
            self.handles[seat] = message["handle"]
            return None

        pilot = self.pilots.get(seat)
        if not isinstance(pilot, QueuedPilot):
            return protocol.error(f"seat {seat} is not player-controlled", rejected=message["type"])

        if message["type"] == "stage" and message["slot"] not in self.loadouts[seat].slots:
            return protocol.error(
                f"slot {message['slot']!r} is not in this loadout", rejected="stage"
            )

        # Answered here rather than left to surface as a pilot error, because a full queue is the one
        # rejection a player acts on immediately and `can_commit` in the next state is 33 ms away.
        if message["type"] == "commit":
            timeline = self.world.fighters[seat].timeline
            if len(timeline.scheduled(self.tick)) >= timeline.max_outstanding:
                return protocol.error(
                    f"{timeline.max_outstanding} moves are already queued; no cancellation",
                    rejected="commit",
                )

        pilot.queue(message)
        return None
