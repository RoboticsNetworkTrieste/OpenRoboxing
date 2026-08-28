"""The client/host message protocol (M4-T2; rewritten for combinations, M6-T8).

Implements ``spec/protocol.md`` v0.6. Message construction and validation live here so the host and
any future client library agree by construction rather than by both being careful.

**The client is a view and a keyboard.** Every function that reads a client message treats it as
hostile: unknown types, an unknown combination, a malformed or non-finite ghost, a ghost beyond a
combination's own reach, and a commit into a full queue all produce a :class:`ProtocolError` (or, for
the two that need the host's own state — the library and the fighter's position — a plain
:class:`ProtocolError` raised from :func:`check_combination` / :func:`check_reach` and turned into an
``error`` message by ``server/host.py``). Nothing a client sends ever reaches the match unvalidated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from openroboxing.runtime.arena import FIGHTERS
from openroboxing.spec.constants import APPROACH_SPEED_M_S, TICK_HZ

if TYPE_CHECKING:  # keep `studio` out of this module's eager import graph, as `runtime/` does.
    from openroboxing.studio.combination_record import CombinationRecord

SPEC_VERSION = "0.6"

#: Everything a client may send. `spec/intent.md` 3.0 (`D6`) collapses 0.4's separate ``stage``
#: (which slot) and ``place`` (where, with a player-set heading) into one ``intent`` message: a
#: combination has no slot to name and its ghost carries no heading any more (`D5` — derived, never
#: chosen), so there is exactly one thing left to stage, not two.
CLIENT_MESSAGES = ("join", "intent", "commit", "clear", "ping")

#: How far outside the ring a ghost may sit, metres. A ghost is a *request* :func:`check_reach` may
#: still refuse on its own combination's much tighter `reach_m`; this is only a sanity bound against
#: a client sending nonsense, generous because the ring's own size is a match parameter this module
#: does not know.
MAX_GHOST_M = 50.0

#: Match phases, as `spec/protocol.md` defines them.
PHASE_FIGHTING = "fighting"
PHASE_ROUND_OVER = "round_over"
PHASE_MATCH_OVER = "match_over"


class ProtocolError(RuntimeError):
    """A client message was malformed or illegal. Answered, never crashed on."""


def parse(message: Any) -> dict[str, Any]:
    """Validate one decoded client message. Raises :class:`ProtocolError` on anything unexpected."""
    if not isinstance(message, Mapping):
        raise ProtocolError("a message must be a JSON object")

    kind = message.get("type")
    if kind not in CLIENT_MESSAGES:
        raise ProtocolError(f"unknown message type {kind!r}; expected one of {CLIENT_MESSAGES}")

    if kind == "join":
        seat = message.get("seat")
        if seat is not None and seat not in FIGHTERS:
            raise ProtocolError(f"seat must be one of {FIGHTERS}, got {seat!r}")
        handle = str(message.get("handle", "")).strip()
        if not handle:
            raise ProtocolError("join needs a handle")
        return {"type": "join", "handle": handle[:32], "seat": seat}

    if kind == "intent":
        # Which combination, and where its last keyframe should land. The library itself is not
        # known here — this function only checks shape — so an unknown name is caught downstream,
        # where the host holds the library (`server/host.py::MatchHost.handle`, `check_combination`).
        combination = message.get("combination")
        if not isinstance(combination, str) or not combination:
            raise ProtocolError(f"intent needs a combination, got {combination!r}")

        ghost = message.get("ghost")
        if not isinstance(ghost, (list, tuple)) or len(ghost) != 2:
            raise ProtocolError(f"intent needs a ghost [x, y], got {ghost!r}")
        values: list[float] = []
        for raw in ghost:
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                raise ProtocolError(f"ghost coordinates must be numeric, got {raw!r}")
            value = float(raw)
            if not math.isfinite(value):
                raise ProtocolError(f"ghost coordinates must be finite, got {value!r}")
            if abs(value) > MAX_GHOST_M:
                raise ProtocolError(
                    f"ghost is {value:.1f} m from the origin, beyond the "
                    f"{MAX_GHOST_M:.0f} m sanity bound"
                )
            values.append(value)
        return {"type": "intent", "combination": combination, "ghost": (values[0], values[1])}

    if kind == "ping":
        return {"type": "ping", "t": message.get("t")}

    return {"type": kind}


# -- feasibility: the one place the speed ceiling is enforced ---------------------------------------
def reach_m(duration_ticks: int) -> float:
    """The furthest a ghost may sit from the anchor a combination this long can still reach, at
    :data:`~openroboxing.spec.constants.APPROACH_SPEED_M_S` of drift, inside its own recorded
    duration.

    Per combination, not per match (`spec/intent.md` "Feasibility"): under 1.0-1.1 anywhere in the
    ring was reachable and distance only cost time; a combination's duration is fixed by its
    recording, so its reach is bounded and differs move to move — roughly 1.6-6.6 m across the
    library. ``welcome`` ships this number per entry so a client can show it before a placement is
    ever attempted; :func:`check_reach` enforces the identical arithmetic when a commit is issued, so
    the two can never disagree.
    """
    return APPROACH_SPEED_M_S * duration_ticks / TICK_HZ


def check_combination(combination: str, library: Mapping[str, CombinationRecord]) -> CombinationRecord:
    """Resolve a combination the client named, or raise naming exactly what is wrong.

    Called when an ``intent`` message is handled — staging time, not commit time — so an unknown
    name is refused before it can ever reach :func:`check_reach`.
    """
    if combination not in library:
        raise ProtocolError(f"combination {combination!r} is not in the library")
    return library[combination]


def check_reach(
    record: CombinationRecord, anchor: tuple[float, float], ghost: tuple[float, float]
) -> None:
    """Refuse a ghost ``record`` cannot reach from ``anchor`` inside its own recorded duration.

    The one place `spec/intent.md`'s speed ceiling is enforced (:func:`reach_m`) — checked here,
    against the *projected* anchor the caller passes in (`server/host.py`'s
    ``MatchHost._anchor_position``: the last queued commit's ghost, or the fighter's live position
    when the queue is empty), and nowhere else. Execution re-warps for real from wherever the fighter
    actually is and never refuses (``runtime/intents.py`` calls ``runtime/warp.py::warp`` with
    ``speed_ceiling=None``) — this is strictly an issue-time guard, so a player learns "can't get
    there" before paying for the commit rather than after.

    Deliberately a straight-line distance against a position-and-heading-independent ceiling, not
    ``warp()``'s exact residual (which additionally rotates the combination's own recorded footwork
    by the anchor heading before measuring what is left over): the client was told exactly this
    ceiling in ``welcome`` (``reach_m`` there is computed the same way, with no heading either), so
    this guard is written to match precisely what the client was promised rather than a tighter one
    it was never shown.

    Raises:
        ProtocolError: naming the distance the ghost needed and the combination's own reach.
    """
    reach = reach_m(record.duration_ticks)
    distance = math.hypot(ghost[0] - anchor[0], ghost[1] - anchor[1])
    if distance > reach:
        raise ProtocolError(
            f"{record.name!r} cannot reach that far: {distance:.2f} m needed, its reach is "
            f"{reach:.2f} m"
        )


# -- host -> client --------------------------------------------------------------------------------------
def welcome(
    seat: str,
    library: Mapping[str, CombinationRecord],
    match_format,
    arena: Mapping[str, Any],
    match_id: str,
) -> dict:
    """Everything a client needs before the first frame: its seat, the whole library, the ring.

    ``combinations`` carries the **entire shared library**, not a per-seat loadout
    (`spec/intent.md`'s `D6`: no loadout — both fighters have identical, complete access to every
    move, and the library is not secret, so a spectator gets it too).

    Each entry carries its final keyframe's joint angles because the **shadow is drawn in the
    browser** (0.4, unchanged in principle at 3.0): a ghost that had to ask the server where its
    elbow goes could not be aimed with, so the client is given the angles once and runs them through
    the kinematic tree in ``/scene.json`` — now for the pose the *combination* ends in, rather than a
    single authored strike.

    ``reach_m`` is carried **per combination**, not once for the whole match: since a combination's
    duration is fixed by its recording (unlike 1.1-2.2's open-ended approach, where anywhere in the
    ring was reachable and distance only cost time), how far its ghost may be placed differs move to
    move — see :func:`reach_m`.
    """
    combinations = [
        {
            "name": name,
            "seconds": round(record.duration_ticks / TICK_HZ, 3),
            "heading_delta": round(float(record.recorded_heading_delta), 5),
            "reach_m": round(reach_m(record.duration_ticks), 3),
            "pose": {
                joint: round(float(angle), 5)
                for joint, angle in record.keyframes[-1].joint_angles.items()
            },
        }
        for name, record in sorted(library.items())
    ]
    return {
        "type": "welcome",
        "spec_version": SPEC_VERSION,
        "seat": seat,
        "match_id": match_id,
        "combinations": combinations,
        # Still the ceiling `check_reach` validates a placement against — repurposed, not retired
        # (`spec/intent.md` "Removed at 3.0" § `APPROACH_SPEED_M_S`).
        "approach_speed_m_s": APPROACH_SPEED_M_S,
        "format": asdict(match_format),
        "arena": dict(arena),
    }


def xy_message(point: tuple[float, float] | None) -> dict | None:
    """A world ``(x, y)`` point as JSON, or ``None``.

    Used for a ghost, an anchor, or a fighter's own position — all bare positions since
    `spec/intent.md` 3.0's `D5`: a ghost's heading is derived, never staged, so none of these carry
    the ``heading`` field 0.4-0.5's ``Placement`` did.
    """
    if point is None:
        return None
    return {"x": round(float(point[0]), 4), "y": round(float(point[1]), 4)}


def queue_entry(commit, tick: int) -> dict:
    """One scheduled commit, as a client sees it.

    ``commit_at`` and ``end_tick`` are **null until the commit becomes current** (the readable
    window, `spec/intent.md` "A commit's span") — never "unknown": the instant ``commit_at`` is
    stamped, ``end_tick`` is exact arithmetic (``commit_at + record.duration_ticks``), fixed for
    good. Unlike 0.5 there is no separate ``strike_at`` / ``approaching`` any more — 3.0 deleted the
    approach and the pose-versus-walk distinction it existed to draw, so a commit is simply not
    started, running, or finished, which ``is_executing`` alone answers.
    """
    return {
        "combination": commit.record.name,
        "ghost": xy_message(commit.ghost),
        "issued_at": commit.issued_at,
        "commit_at": commit.commit_at,
        "end_tick": commit.end_tick,
        "executing": commit.is_executing(tick),
    }


def visible_queue(scheduled, tick: int, own: bool) -> list:
    """Which of a fighter's scheduled commits a given viewer may see.

    Your own queue in full; of anybody else, only what is **executing** — which is already visible in
    the ring. A queued-but-unstarted commit has been paid for and not yet shown, and sending it would
    hand the opponent a readable list of your next four moves, which is exactly the risk queueing is
    supposed to be (`spec/protocol.md` §"Seat state").

    A rule rather than a branch in the host, because it is the one place a leak could happen and it
    should be checkable on its own.
    """
    if own:
        return list(scheduled)
    return [commit for commit in scheduled if commit.is_executing(tick)]


def seat_state(
    handle: str,
    staged: str | None,
    queue: list[dict],
    can_commit: bool,
    hits_landed: int,
    torso_height_m: float,
    down: bool,
    ghost: tuple[float, float] | None = None,
    anchor: tuple[float, float] | None = None,
    position: tuple[float, float] | None = None,
) -> dict:
    """One fighter, as `spec/protocol.md` §"Seat state" defines it.

    ``staged`` is a combination name now, not a slot (`D6` — there is nothing left to slot).

    ``can_commit`` is here so a client can grey out a key **without knowing the rule**. If the queue
    bound changes, no client changes.

    ``queue`` is passed already filtered by the caller, because *what* a seat may see depends on
    whose seat it is: your own queue in full, the opponent's only where it is executing.

    ``ghost`` is what a seat has most recently staged — world ``(x, y)``, heading derived and never
    part of it (`spec/intent.md` "Ghost heading is derived, not staged") — echoed back so a
    reconnecting client can recover it, the same reason 0.4's ``placement`` was.

    ``anchor`` is where a commit issued *right now* would start from — the last queued commit's
    ghost, or the fighter's live position when the queue is empty — the same value
    :func:`check_reach` measures a new ghost against.

    ``position`` is **public for every seat**. It is where a fighter is standing right now, which the
    binary frame already carries for anybody who draws it — so withholding it from the JSON would
    only handicap agents, which see what a human sees and no more.
    """
    return {
        "handle": handle,
        "staged": staged,
        "position": xy_message(position),
        "ghost": xy_message(ghost),
        "anchor": xy_message(anchor),
        "queue": list(queue),
        "queue_depth": len(queue),
        "can_commit": can_commit,
        "hits_landed": hits_landed,
        "torso_height_m": round(float(torso_height_m), 3),
        "down": bool(down),
    }


def live_score(
    share: Mapping[str, float],
    dimensions: Mapping[str, Mapping[str, float]],
    points: Mapping[str, int],
    rounds_won: Mapping[str, int],
    draw_margin: float,
) -> dict:
    """The round so far, as `spec/protocol.md` 0.2 defines it.

    ``leading`` is ``None`` inside ``draw_margin``, using the *same* margin the official scorer uses
    to call a round even. A UI that showed a leader the scorer would call level would be lying by a
    rounding error.
    """
    fighters = list(share)
    leading = None
    if len(fighters) == 2:
        first, second = fighters
        if abs(share[first] - share[second]) > draw_margin:
            leading = first if share[first] > share[second] else second

    return {
        "share": {f: round(float(v), 4) for f, v in share.items()},
        "leading": leading,
        "dimensions": {
            f: {k: round(float(v), 4) for k, v in d.items()} for f, d in dimensions.items()
        },
        "points": dict(points),
        "rounds_won": dict(rounds_won),
    }


def state(
    tick: int,
    round_index: int,
    clock_ticks: int,
    seats: Mapping[str, dict],
    phase: str,
    score: Mapping[str, Any] | None = None,
    separation_m: float | None = None,
) -> dict:
    """One frame's worth of match state. Sent at 30 Hz beside the binary frame it describes."""
    return {
        "type": "state",
        "tick": tick,
        "round": round_index + 1,
        "clock_ticks": clock_ticks,
        "phase": phase,
        "seats": dict(seats),
        "score": dict(score) if score else None,
        "separation_m": None if separation_m is None else round(float(separation_m), 3),
    }


def event(name: str, **fields: Any) -> dict:
    return {"type": "event", "event": name, **fields}


def error(message: str, rejected: str | None = None) -> dict:
    return {"type": "error", "message": message, "rejected": rejected}


def pong(t: Any) -> dict:
    return {"type": "pong", "t": t}
