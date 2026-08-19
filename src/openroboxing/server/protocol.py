"""The client/host message protocol (M4-T2).

Implements ``spec/protocol.md`` v0.4. Message construction and validation live here so the host and
any future client library agree by construction rather than by both being careful.

**The client is a view and a keyboard.** Every function that reads a client message treats it as
hostile: unknown types, bad slots, non-finite coordinates and commits into a full queue produce an
:class:`ProtocolError` that the host turns into an ``error`` message. Nothing a client sends ever
reaches the match unvalidated.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Mapping

from openroboxing.runtime.arena import FIGHTERS
from openroboxing.spec.constants import APPROACH_SPEED_M_S, SECONDS_PER_TOKEN

SPEC_VERSION = "0.4"

#: Everything a client may send. ``move`` was removed at 0.4 — steering is gone, and a player moves
#: by placing the shadow (`spec/intent.md` §"What happened to walking").
CLIENT_MESSAGES = ("join", "stage", "commit", "clear", "place", "ping")

#: How far outside the ring a shadow may be placed, metres. A placement is a *request* the generator
#: may refuse, so this is only a sanity bound against a client sending nonsense; it is generous
#: because the ring's own size is a match parameter and this module does not know it.
MAX_PLACEMENT_M = 50.0

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

    if kind == "stage":
        slot = message.get("slot")
        if not isinstance(slot, str) or not slot:
            raise ProtocolError("stage needs a slot")
        return {"type": "stage", "slot": slot}

    if kind == "place":
        # Absolute MuJoCo world coordinates. The client owns where its shadow is *drawn* — a preview
        # that round-trips before it moves is unusable — but the host commits only what it is sent,
        # so every field is checked as if it came from an adversary.
        values = {}
        for field in ("x", "y", "heading"):
            raw = message.get(field)
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                raise ProtocolError(f"place needs a numeric {field!r}, got {raw!r}")
            value = float(raw)
            if not math.isfinite(value):
                raise ProtocolError(f"place {field!r} must be finite, got {value!r}")
            if field != "heading" and abs(value) > MAX_PLACEMENT_M:
                raise ProtocolError(
                    f"place {field!r} is {value:.1f} m from the origin, beyond the "
                    f"{MAX_PLACEMENT_M:.0f} m sanity bound"
                )
            values[field] = value
        return {"type": "place", **values}

    if kind == "ping":
        return {"type": "ping", "t": message.get("t")}

    return {"type": kind}


# -- host -> client --------------------------------------------------------------------------------------
def welcome(seat: str, loadout, match_format, arena: Mapping[str, Any], match_id: str) -> dict:
    """Everything a client needs before the first frame: its seat, its keys, the format, the ring.

    ``poses`` carries each slot's joint angles because the **shadow is drawn in the browser** (0.4).
    A ghost that had to ask the server where its elbow goes could not be aimed with, so the client
    gets the angles once and runs them through the kinematic tree in ``/scene.json``.

    This is a seat's own loadout only. A spectator's welcome has no slots, so it carries no poses.
    """
    slots = sorted(loadout.slots.items())
    return {
        "type": "welcome",
        "spec_version": SPEC_VERSION,
        "seat": seat,
        "match_id": match_id,
        "loadout": {slot: pose.name for slot, pose in slots},
        "horizons": {slot: pose.horizon_tokens for slot, pose in slots},
        # Everywhere in the ring is reachable since `spec/intent.md` 1.1 — a distant placement
        # costs *time*, not failure — so a client is given the two numbers it needs to say how much:
        # the pose's own length, and how fast a fighter closes on a placement.
        "pose_seconds": {
            slot: round(pose.horizon_tokens * SECONDS_PER_TOKEN, 3) for slot, pose in slots
        },
        "approach_speed_m_s": APPROACH_SPEED_M_S,
        "poses": {
            slot: {name: round(float(angle), 5) for name, angle in pose.joint_angles.items()}
            for slot, pose in slots
        },
        "format": asdict(match_format),
        "arena": dict(arena),
    }


def placement_message(placement) -> dict | None:
    """A :class:`~openroboxing.runtime.intents.Placement` as JSON, or ``None``."""
    if placement is None:
        return None
    return {
        "x": round(float(placement.position[0]), 4),
        "y": round(float(placement.position[1]), 4),
        "heading": round(float(placement.heading), 4),
    }


def queue_entry(commit, tick: int) -> dict:
    """One scheduled commit, as a client sees it.

    ``commit_at``, ``strike_at`` and ``end_tick`` are **null until the move reaches each stage**
    (`spec/intent.md` 1.1): a commit runs until it arrives, so its span is not known when it is
    issued. A client must read null as "not yet", never as zero — `spec/protocol.md` §"Seat state".
    """
    return {
        "slot": commit.slot,
        "pose": commit.pose.name,
        "issued_at": commit.issued_at,
        "commit_at": commit.commit_at,
        "strike_at": commit.strike_at,
        "end_tick": commit.end_tick,
        "executing": commit.is_executing(tick),
        "approaching": commit.is_approaching(tick),
        "placement": placement_message(commit.placement),
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
    placement=None,
    anchor=None,
    position=None,
) -> dict:
    """One fighter, as `spec/protocol.md` §"Seat state" defines it.

    ``can_commit`` is here so a client can grey out a key **without knowing the rule**. If the queue
    bound changes, no client changes.

    ``queue`` is passed already filtered by the caller, because *what* a seat may see depends on
    whose seat it is: your own queue in full, the opponent's only where it is executing. A
    queued-but-unstarted commit has been paid for and not yet shown, and leaking it would hand the
    opponent a readable list of your next four moves (`spec/protocol.md` §"Seat state").

    ``position`` is **public for every seat**. It is where a fighter is standing right now, which the
    binary frame already carries for anybody who draws it — so withholding it from the JSON would
    only handicap agents, which see what a human sees and no more.
    """
    return {
        "handle": handle,
        "staged": staged,
        "position": placement_message(position),
        "placement": placement_message(placement),
        "anchor": placement_message(anchor),
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
