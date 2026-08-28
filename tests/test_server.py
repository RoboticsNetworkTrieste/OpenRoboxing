"""M4-T2: the match host and its protocol (rewritten for combinations, M6-T8).

Acceptance criterion from WORKPLAN.md M4-T2:
  two browsers on the LAN play a full match; artificially injecting 200 ms latency does not change
  match outcomes systematically.

A browser cannot be driven from pytest. What is tested here is everything underneath it: the
protocol validates hostile input, the host applies queued intents on its own tick, and a real
websocket client can join, play and be streamed to. The browser half is `docs/playtest/` work.

Rewritten against `spec/intent.md` 3.0 and `spec/protocol.md` 0.6: a commit now carries a
**combination** (`spec/combination.md`) and a **ghost** — world ``(x, y)`` only, no heading — instead
of a loadout slot and a placement with a player-set heading. The loadout (deleted, task A6),
``Placement`` and the ``stage`` / ``place`` messages they were staged through are gone from the
protocol; see this file's own commit message for exactly which tests that retired, and why.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_server.py -v
    .venv_mb/bin/python -m pytest tests/test_server.py -v -m slow   # needs a GPU
"""

from __future__ import annotations

import pytest

from openroboxing.runtime.arena import FIGHTERS
from openroboxing.runtime.conventions import G1
from openroboxing.runtime.intents import IntentTimeline
from openroboxing.server import protocol
from openroboxing.server.host import MatchHost, QueuedPilot
from openroboxing.spec.constants import APPROACH_SPEED_M_S, MAX_OUTSTANDING_COMMITS, TICK_HZ
from openroboxing.studio.combination_record import CombinationRecord, CombinationSource, Keyframe

pytest.importorskip("aiohttp")

ANGLES = {name: 0.0 for name in G1.mujoco_joint_names}


def _combination(
    name: str,
    *,
    offsets=((0.3, 0.0), (0.6, 0.0)),
    headings=(0.0, 0.0),
    tokens=(6, 6),
    admitted: bool = True,
) -> CombinationRecord:
    """A small, admitted combination — cheap enough to drive a timeline through fully.

    Each test file in this repo keeps its own copy of this builder rather than sharing a fixtures
    module — see e.g. `test_fight.py`, `test_intents_combinations.py`, `test_warp.py`.
    """
    keyframes = [Keyframe(dict(ANGLES), None, (0.0, 0.0), 0.0)]
    for offset, heading, token in zip(offsets, headings, tokens, strict=True):
        keyframes.append(Keyframe(dict(ANGLES), token, offset, heading))
    return CombinationRecord(
        name=name,
        library_version="v0.2",
        source=CombinationSource("t", 0, 100, False),
        keyframes=keyframes,
        telegraph_ms=180.0 if admitted else None,
        tracking_error_rad=0.1 if admitted else None,
        admission="admitted" if admitted else "draft",
    )


@pytest.fixture(scope="module")
def library() -> dict[str, CombinationRecord]:
    return {
        "combo-a": _combination("combo-a"),
        "combo-b": _combination("combo-b", offsets=((0.1, 0.0), (0.2, 0.0))),
    }


# --- the protocol treats clients as hostile ---------------------------------------------------------
@pytest.mark.parametrize(
    "message, error",
    [
        ("not an object", "must be a JSON object"),
        ({}, "unknown message type"),
        ({"type": "explode"}, "unknown message type"),
        ({"type": "join"}, "join needs a handle"),
        ({"type": "join", "handle": "   "}, "join needs a handle"),
        ({"type": "join", "handle": "a", "seat": "green"}, "seat must be one of"),
        ({"type": "intent"}, "needs a combination"),
        ({"type": "intent", "combination": ""}, "needs a combination"),
        ({"type": "intent", "combination": 3, "ghost": [0.0, 0.0]}, "needs a combination"),
        ({"type": "intent", "combination": "jab"}, "needs a ghost"),
        ({"type": "intent", "combination": "jab", "ghost": [0.0]}, "needs a ghost"),
        ({"type": "intent", "combination": "jab", "ghost": [0.0, 0.0, 0.0]}, "needs a ghost"),
        ({"type": "intent", "combination": "jab", "ghost": "nope"}, "needs a ghost"),
        ({"type": "intent", "combination": "jab", "ghost": ["x", 0.0]}, "must be numeric"),
        ({"type": "intent", "combination": "jab", "ghost": [True, 0.0]}, "must be numeric"),
        ({"type": "intent", "combination": "jab", "ghost": [float("nan"), 0.0]}, "must be finite"),
        ({"type": "intent", "combination": "jab", "ghost": [float("inf"), 0.0]}, "must be finite"),
        ({"type": "intent", "combination": "jab", "ghost": [1e6, 0.0]}, "sanity bound"),
    ],
)
def test_a_malformed_message_is_refused(message, error) -> None:
    """A client sends these straight into the generator's target. NaN would surface as a fighter
    that vanishes, which is a very hard bug to read backwards."""
    with pytest.raises(protocol.ProtocolError, match=error):
        protocol.parse(message)


def test_a_valid_message_comes_back_normalised() -> None:
    assert protocol.parse({"type": "commit"}) == {"type": "commit"}
    assert protocol.parse({"type": "clear"}) == {"type": "clear"}
    assert protocol.parse({"type": "intent", "combination": "jab", "ghost": [1.5, -0.5]}) == {
        "type": "intent",
        "combination": "jab",
        "ghost": (1.5, -0.5),
    }
    assert protocol.parse({"type": "ping", "t": 7}) == {"type": "ping", "t": 7}


def test_a_handle_is_trimmed_and_bounded() -> None:
    parsed = protocol.parse({"type": "join", "handle": "  " + "x" * 100 + "  "})
    assert len(parsed["handle"]) == 32


def test_the_stage_and_place_messages_are_gone() -> None:
    """Retired at `spec/protocol.md` 0.6: a combination has no slot for ``stage`` to name and its
    ghost carries no heading for ``place`` to set any more (`spec/intent.md` `D5`/`D6`). Accepting
    either silently would let an old client think it was steering while nothing moved."""
    assert "stage" not in protocol.CLIENT_MESSAGES
    assert "place" not in protocol.CLIENT_MESSAGES
    with pytest.raises(protocol.ProtocolError, match="unknown message type"):
        protocol.parse({"type": "stage", "slot": "1"})
    with pytest.raises(protocol.ProtocolError, match="unknown message type"):
        protocol.parse({"type": "place", "x": 0.0, "y": 0.0, "heading": 0.0})


def test_the_movement_message_is_gone() -> None:
    """Retired with the channel at 0.4. Accepting it silently would let an old client think it was
    steering while nothing moved."""
    assert "move" not in protocol.CLIENT_MESSAGES
    with pytest.raises(protocol.ProtocolError, match="unknown message type"):
        protocol.parse({"type": "move", "direction": "in"})


# --- feasibility: reach_m and the checks that use it (spec/protocol.md 0.6) ------------------------
def test_reach_m_scales_with_a_combinations_own_duration(library) -> None:
    combo = library["combo-a"]
    assert protocol.reach_m(combo.duration_ticks) == pytest.approx(
        APPROACH_SPEED_M_S * combo.duration_ticks / TICK_HZ
    )


def test_check_combination_resolves_a_known_name(library) -> None:
    assert protocol.check_combination("combo-a", library) is library["combo-a"]


def test_check_combination_rejects_an_unknown_name(library) -> None:
    with pytest.raises(protocol.ProtocolError, match="not in the library"):
        protocol.check_combination("does-not-exist", library)


def test_check_reach_accepts_a_ghost_inside_the_combinations_reach(library) -> None:
    combo = library["combo-a"]
    reach = protocol.reach_m(combo.duration_ticks)
    protocol.check_reach(combo, (0.0, 0.0), (reach * 0.5, 0.0))  # must not raise


def test_check_reach_rejects_a_ghost_beyond_the_combinations_reach(library) -> None:
    """`spec/intent.md` "Feasibility": the one place the speed ceiling is enforced, and the error
    must carry the number a client can show, not just a bare refusal."""
    combo = library["combo-a"]
    reach = protocol.reach_m(combo.duration_ticks)
    with pytest.raises(protocol.ProtocolError) as caught:
        protocol.check_reach(combo, (0.0, 0.0), (reach * 5, 0.0))
    assert f"{reach:.2f}" in str(caught.value)


def test_check_reach_is_measured_from_the_anchor_not_the_origin(library) -> None:
    """A ghost the fighter is already standing next to must not be refused just because it is far
    from the world origin — the check is relative to `anchor`, not absolute."""
    combo = library["combo-a"]
    reach = protocol.reach_m(combo.duration_ticks)
    protocol.check_reach(combo, (10.0, 10.0), (10.0 + reach * 0.5, 10.0))  # must not raise


# --- welcome carries the whole library (D6, spec/protocol.md 0.6) ----------------------------------
def test_welcome_carries_the_combination_library(library) -> None:
    from openroboxing.runtime.match import MatchFormat

    message = protocol.welcome("red", library, MatchFormat(), {}, "m1")
    assert message["type"] == "welcome"
    assert message["seat"] == "red"
    assert {c["name"] for c in message["combinations"]} == set(library)

    names = [c["name"] for c in message["combinations"]]
    assert names == sorted(names), "sorted by name"

    for entry in message["combinations"]:
        record = library[entry["name"]]
        assert set(entry["pose"]) == set(G1.mujoco_joint_names), f"{entry['name']} cannot be posed"
        assert entry["pose"] == {
            joint: round(angle, 5) for joint, angle in record.keyframes[-1].joint_angles.items()
        }
        assert entry["seconds"] == round(record.duration_ticks / TICK_HZ, 3)
        assert entry["reach_m"] == round(protocol.reach_m(record.duration_ticks), 3)
        assert entry["heading_delta"] == round(record.recorded_heading_delta, 5)

    for key in ("loadout", "horizons", "pose_seconds", "poses"):
        assert key not in message, f"1.0-2.2's {key!r} did not survive D6"

    assert message["approach_speed_m_s"] == APPROACH_SPEED_M_S


def test_a_spectator_receives_the_same_library_as_a_seat(library) -> None:
    """`D6`: the library is not secret — both fighters already have identical, complete access — so
    a spectator sees exactly what a seat sees."""
    from openroboxing.runtime.match import MatchFormat

    seat_message = protocol.welcome("red", library, MatchFormat(), {}, "m1")
    spectator_message = protocol.welcome("spectator", library, MatchFormat(), {}, "m1")
    assert spectator_message["combinations"] == seat_message["combinations"]


def test_the_protocol_version_and_the_spec_agree() -> None:
    import re

    from openroboxing.paths import OPENROBOXING_ROOT

    spec = (OPENROBOXING_ROOT / "spec/protocol.md").read_text()
    assert re.search(rf"Version \*\*{re.escape(protocol.SPEC_VERSION)}\*\*", spec), (
        f"protocol.py's SPEC_VERSION ({protocol.SPEC_VERSION!r}) is not the version stamped at the "
        "top of spec/protocol.md"
    )


# --- the queue: shape and visibility (spec/protocol.md 0.6 "Seat state") ---------------------------
def test_the_opponents_staged_move_is_never_in_a_state_message(library) -> None:
    """`WORKPLAN` M4-T1: no HUD on the fighters — the windup is the only cue.

    A `state` carries each seat's own staged combination for its own client to draw. Nothing in the
    schema lets one seat learn what the other has *staged but not committed*, which is the
    information the whole design withholds.
    """
    timeline = IntentTimeline(library)
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    commit = timeline.commit(0)

    seat = protocol.seat_state(
        "blue", staged=None, queue=[], can_commit=True, hits_landed=0, torso_height_m=0.84,
        down=False,
    )
    message = protocol.state(0, 0, 3000, {"blue": seat}, protocol.PHASE_FIGHTING)

    assert "qpos" not in message, "a modified client would see the opponent exactly"
    assert message["seats"]["blue"]["staged"] is None
    assert message["seats"]["blue"]["ghost"] is None, "nor where they are aiming"
    assert set(protocol.queue_entry(commit, 0)) == {
        "combination", "ghost", "issued_at", "commit_at", "end_tick", "executing",
    }


def test_a_seat_sees_its_own_queue_and_of_another_only_what_is_running(library) -> None:
    """A queued-but-unstarted commit has been paid for and not yet shown. Leaking it would hand the
    opponent a readable list of the next four moves, which is exactly the risk queueing is meant to
    be."""
    timeline = IntentTimeline(library)
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    running = timeline.commit(0)
    # Simulate "started" directly rather than driving `generator_intent` through a warp — `Commit`'s
    # span is plain, settable fields once a commit has begun (`spec/intent.md` "A commit's span"),
    # and this test is about queue visibility, not about warping.
    running.commit_at = 0
    running.ended_at = 100

    timeline.stage(combination="combo-b", ghost=(0.2, 0.0))
    timeline.commit(1)

    tick = 5
    scheduled = timeline.scheduled(tick)
    assert len(scheduled) == 2

    assert protocol.visible_queue(scheduled, tick, own=True) == list(scheduled)
    hidden = protocol.visible_queue(scheduled, tick, own=False)
    assert [c.record.name for c in hidden] == ["combo-a"], "only the running move is public"
    assert hidden[0].is_executing(tick)


def test_nothing_of_another_seat_is_visible_before_its_move_starts(library) -> None:
    """The readable window is the point: a commit becomes public only once it starts."""
    timeline = IntentTimeline(library)
    timeline.stage(combination="combo-a", ghost=(1.0, 0.0))
    commit = timeline.commit(0)

    tick = 5  # well past issue, but `commit_at` is still None: nothing has started it
    scheduled = timeline.scheduled(tick)
    assert protocol.visible_queue(scheduled, tick, own=False) == []
    assert protocol.visible_queue(scheduled, tick, own=True) == [commit]


def test_a_seat_sees_its_own_shadow_and_anchor() -> None:
    seat = protocol.seat_state(
        "red", None, [], True, 0, 0.84, False, ghost=(1.25, -0.5), anchor=(0.9, 0.0),
    )
    assert seat["ghost"] == {"x": 1.25, "y": -0.5}
    assert seat["anchor"] == {"x": 0.9, "y": 0.0}
    assert protocol.seat_state("red", None, [], True, 0, 0.84, False)["ghost"] is None


# --- the live score (spec/protocol.md 0.2) -----------------------------------------------------------
def test_a_state_carries_a_score_slot() -> None:
    """Added in 0.2 by the project owner's decision. v0.1 withheld it."""
    message = protocol.state(0, 0, 3000, {}, protocol.PHASE_FIGHTING)
    assert "score" in message
    assert message["score"] is None, "no score until there is a round to score"


def test_a_state_carries_the_range() -> None:
    """Range is not secret — both fighters can see it by looking — and without it a player cannot
    manage distance, which is most of boxing."""
    message = protocol.state(0, 0, 3000, {}, protocol.PHASE_FIGHTING, separation_m=1.4237)
    assert message["separation_m"] == 1.424


def test_the_leader_is_only_named_outside_the_scorer_s_own_draw_margin() -> None:
    """A UI that showed a leader the official scorer would call level would be lying by a rounding
    error, so the same margin is used."""
    from openroboxing.league.scoring import DRAW_MARGIN

    level = protocol.live_score(
        share={"red": 0.5, "blue": 0.5},
        dimensions={"red": {}, "blue": {}},
        points={"red": 0, "blue": 0},
        rounds_won={"red": 0, "blue": 0},
        draw_margin=DRAW_MARGIN,
    )
    assert level["leading"] is None

    ahead = protocol.live_score(
        share={"red": 0.7, "blue": 0.3},
        dimensions={"red": {}, "blue": {}},
        points={"red": 0, "blue": 0},
        rounds_won={"red": 0, "blue": 0},
        draw_margin=DRAW_MARGIN,
    )
    assert ahead["leading"] == "red"


def test_a_hair_s_difference_is_still_level() -> None:
    from openroboxing.league.scoring import DRAW_MARGIN

    barely = protocol.live_score(
        share={"red": 0.5 + DRAW_MARGIN / 4, "blue": 0.5 - DRAW_MARGIN / 4},
        dimensions={"red": {}, "blue": {}},
        points={"red": 0, "blue": 0},
        rounds_won={"red": 0, "blue": 0},
        draw_margin=DRAW_MARGIN,
    )
    assert barely["leading"] is None


def test_the_score_carries_why_not_just_who() -> None:
    """A bar with no reason behind it is just a colour."""
    score = protocol.live_score(
        share={"red": 0.6, "blue": 0.4},
        dimensions={
            "red": {"damage": 3.25, "control": 0.2, "aggression": 0.8},
            "blue": {"damage": 1.0, "control": 0.1, "aggression": 0.4},
        },
        points={"red": 10, "blue": 9},
        rounds_won={"red": 1, "blue": 0},
        draw_margin=0.02,
    )
    assert set(score) == {"share", "leading", "dimensions", "points", "rounds_won"}
    assert set(score["dimensions"]["red"]) == {"damage", "control", "aggression"}
    assert score["points"] == {"red": 10, "blue": 9}


def test_can_commit_carries_the_rule_so_the_client_does_not() -> None:
    free = protocol.seat_state("red", None, [], True, 0, 0.84, False)
    full = protocol.seat_state(
        "red", "combo-a", [{"combination": "combo-a"}], False, 0, 0.84, False
    )
    assert free["can_commit"] is True and full["can_commit"] is False
    assert free["queue_depth"] == 0 and full["queue_depth"] == 1


def test_can_commit_follows_the_bound_not_a_single_active_move(library) -> None:
    timeline = IntentTimeline(library)
    for index in range(MAX_OUTSTANDING_COMMITS):
        timeline.stage(combination="combo-a", ghost=(0.1, 0.0))
        timeline.commit(0)
        free = len(timeline.scheduled(0)) < timeline.max_outstanding
        assert free is (index < MAX_OUTSTANDING_COMMITS - 1)


# --- the queued pilot: intent, commit, clear (spec/intent.md 3.0) ----------------------------------
def test_keypresses_are_applied_on_the_tick_not_when_they_arrive(library) -> None:
    """A client must not be able to make the simulation wait, nor interleave into a step."""
    pilot = QueuedPilot()
    timeline = IntentTimeline(library)

    pilot.queue({"type": "intent", "combination": "combo-a", "ghost": (1.0, 0.0)})
    pilot.queue({"type": "commit"})
    assert timeline.commits == (), "nothing happens until the tick"

    pilot.act(timeline, 100)
    assert len(timeline.commits) == 1
    assert timeline.commits[0].record.name == "combo-a"
    assert timeline.commits[0].ghost == (1.0, 0.0)
    assert timeline.commits[0].issued_at == 100


def test_an_intent_naming_an_unknown_combination_is_rejected_not_a_crash(library) -> None:
    """`spec/protocol.md`: an unknown combination is a typed error, never a dropped connection."""
    pilot = QueuedPilot()
    timeline = IntentTimeline(library)

    pilot.queue({"type": "intent", "combination": "does-not-exist", "ghost": (1.0, 0.0)})
    pilot.act(timeline, 0)

    assert timeline.staged.combination is None, "the bad name never reached the stage"
    assert pilot.last_error is not None and "not in the library" in pilot.last_error


def test_committing_with_nothing_staged_is_an_error_not_a_crash(library) -> None:
    pilot = QueuedPilot()
    timeline = IntentTimeline(library)
    pilot.queue({"type": "commit"})
    pilot.act(timeline, 0)

    assert timeline.commits == ()
    assert "no combination is staged" in pilot.last_error


def test_a_second_commit_is_queued_rather_than_refused(library) -> None:
    """The 1.0 rule, unchanged at 3.0: commits stack up to the bound and run back to back."""
    pilot = QueuedPilot()
    timeline = IntentTimeline(library)

    pilot.queue({"type": "intent", "combination": "combo-a", "ghost": (1.0, 0.0)})
    pilot.queue({"type": "commit"})
    pilot.act(timeline, 0)

    pilot.queue({"type": "intent", "combination": "combo-b", "ghost": (0.2, 0.0)})
    pilot.queue({"type": "commit"})
    pilot.act(timeline, 5)

    assert len(timeline.commits) == 2
    assert pilot.last_error is None


def test_committing_into_a_full_queue_is_rejected(library) -> None:
    """The existing error, unchanged by the rewrite: `IntentTimeline.commit` itself refuses, and the
    pilot records it rather than the socket handler raising."""
    pilot = QueuedPilot()
    timeline = IntentTimeline(library, max_outstanding=1)

    pilot.queue({"type": "intent", "combination": "combo-a", "ghost": (0.1, 0.0)})
    pilot.queue({"type": "commit"})
    pilot.act(timeline, 0)
    assert len(timeline.commits) == 1

    pilot.queue({"type": "intent", "combination": "combo-a", "ghost": (0.1, 0.0)})
    pilot.queue({"type": "commit"})
    pilot.act(timeline, 0)

    assert len(timeline.commits) == 1, "the queue did not grow past its bound"
    assert "already queued" in pilot.last_error


def test_staging_stays_free_while_a_commit_runs(library) -> None:
    """`spec/intent.md`: staging is unbounded and happens during play."""
    pilot = QueuedPilot()
    timeline = IntentTimeline(library)
    pilot.queue({"type": "intent", "combination": "combo-a", "ghost": (1.0, 0.0)})
    pilot.queue({"type": "commit"})
    pilot.act(timeline, 0)

    pilot.queue({"type": "intent", "combination": "combo-b", "ghost": (0.3, 0.0)})
    pilot.act(timeline, 10)
    assert timeline.staged.combination == "combo-b"
    assert pilot.staged == "combo-b"


def test_clearing_unstages_without_cancelling(library) -> None:
    pilot = QueuedPilot()
    timeline = IntentTimeline(library)
    pilot.queue({"type": "intent", "combination": "combo-a", "ghost": (1.0, 0.0)})
    pilot.act(timeline, 0)
    pilot.queue({"type": "clear"})
    pilot.act(timeline, 1)

    assert timeline.staged.combination is None
    assert pilot.staged is None


def test_a_reset_pilot_forgets_the_queue(library) -> None:
    pilot = QueuedPilot()
    pilot.queue({"type": "intent", "combination": "combo-a", "ghost": (1.0, 0.0)})
    pilot.reset()

    timeline = IntentTimeline(library)
    pilot.act(timeline, 0)
    assert timeline.staged.combination is None, "a queue survived into the next round"


# --- the host, live -------------------------------------------------------------------------------------
@pytest.mark.slow
def test_the_arena_config_reaches_the_compiled_ring(library) -> None:
    """`build_arena` compiles ring size into the model, so a config assigned *after* construction
    changes the record and nothing a fighter can touch.

    `tools/tune.py` did exactly that and swept `ring_size` and `glove_radius` for nothing, reporting
    the absence of any difference as noise. The knob has to go in at construction, and this is what
    says so.
    """
    import mujoco

    from openroboxing.runtime.arena import ArenaConfig

    small = MatchHost(
        libraries={f: library for f in FIGHTERS},
        config=ArenaConfig(ring_size=3.0),
        max_outstanding=2,
        render=False,
    )
    post = mujoco.mj_name2id(small.world.model, mujoco.mjtObj.mjOBJ_GEOM, "post_00")
    assert post >= 0, "the ring has no corner posts"
    corner = small.world.model.geom_pos[post][:2]
    assert abs(abs(corner[0]) - 1.5) < 1e-6, f"the ring was not built at 3.0 m: post at {corner}"

    assert small.record.arena["ring_size"] == 3.0, "and the record must agree with the ring"
    for fighter in small.world.fighters.values():
        assert fighter.timeline.max_outstanding == 2, "the queue bound must reach the timeline"


@pytest.mark.slow
def test_the_host_answers_a_full_queue_immediately(library) -> None:
    """`can_commit` in the next state is 33 ms away, and a full queue is the one rejection a player
    reacts to at once."""
    host = MatchHost(libraries={f: library for f in FIGHTERS}, render=False)
    timeline = host.world.fighters["red"].timeline
    for _ in range(MAX_OUTSTANDING_COMMITS):
        timeline.stage(combination="combo-a", ghost=(0.1, 0.0))
        timeline.commit(0)

    reply = host.handle("red", {"type": "commit"})
    assert reply is not None and reply["rejected"] == "commit"
    assert "already queued" in reply["message"]


@pytest.mark.slow
def test_the_host_rejects_an_unknown_combination_immediately(library) -> None:
    host = MatchHost(libraries={f: library for f in FIGHTERS}, render=False)
    reply = host.handle(
        "red", {"type": "intent", "combination": "does-not-exist", "ghost": (0.1, 0.0)}
    )
    assert reply is not None and reply["rejected"] == "intent"
    assert "not in the library" in reply["message"]


@pytest.mark.slow
def test_the_host_rejects_a_ghost_beyond_its_combinations_reach(library) -> None:
    """The one place the speed ceiling is enforced (`spec/intent.md` "Feasibility"), wired end to
    end: `MatchHost.handle` refuses before the commit ever reaches the queue, and the error names the
    reach the client was already shown in `welcome`."""
    host = MatchHost(libraries={f: library for f in FIGHTERS}, render=False)
    combo = library["combo-a"]
    reach = protocol.reach_m(combo.duration_ticks)

    host.handle("red", {"type": "intent", "combination": "combo-a", "ghost": (reach * 10, 0.0)})
    reply = host.handle("red", {"type": "commit"})

    assert reply is not None and reply["rejected"] == "commit"
    assert f"{reach:.2f}" in reply["message"]
    assert host.world.fighters["red"].timeline.commits == (), "the commit never queued"


@pytest.mark.slow
def test_a_client_can_join_stage_commit_and_be_streamed_to(library) -> None:
    """The whole stack in one go: aiohttp, the protocol, the host, physics, the renderer."""
    import asyncio
    import json

    from aiohttp.test_utils import TestClient, TestServer

    from openroboxing.runtime.match import MatchFormat
    from openroboxing.server.app import build_app

    async def run() -> dict:
        host = MatchHost(
            libraries={f: library for f in FIGHTERS},
            match_format=MatchFormat(rounds=1, round_ticks=120, get_up_window_ticks=100),
            match_id="test",
        )
        client = TestClient(TestServer(build_app(host)))
        await client.start_server()

        red = await client.ws_connect("/ws?seat=red")
        blue = await client.ws_connect("/ws?seat=blue")

        welcome = await red.receive_json()
        await red.receive_json()  # the first state
        await blue.receive_json()
        await blue.receive_json()

        await red.send_json({"type": "join", "handle": "carlo"})
        await red.send_json({"type": "intent", "combination": "combo-a", "ghost": [0.5, 0.0]})
        await red.send_json({"type": "commit"})

        match_task = asyncio.create_task(host.run())
        frames, states = 0, 0
        deadline = asyncio.get_event_loop().time() + 25.0
        while not match_task.done() and asyncio.get_event_loop().time() < deadline:
            try:
                message = await asyncio.wait_for(red.receive(), timeout=5.0)
            except asyncio.TimeoutError:
                break
            if message.type.name == "BINARY":
                frames += 1
            elif message.type.name == "TEXT":
                payload = json.loads(message.data)
                if payload.get("type") == "state":
                    states += 1

        record = await match_task
        await red.close()
        await blue.close()
        await client.close()
        return {
            "welcome": welcome,
            "frames": frames,
            "states": states,
            "record": record,
            "stats": host.stats.summary(),
            "commits": record.rounds[0].commits,
        }

    result = asyncio.run(run())

    assert result["welcome"]["type"] == "welcome"
    assert result["welcome"]["seat"] == "red"
    assert {c["name"] for c in result["welcome"]["combinations"]} == set(library)

    assert result["states"] > 10, "the host did not stream state"
    assert result["frames"] > 10, "the host did not stream video"

    assert len(result["record"].rounds) == 1
    assert result["commits"], "the client's commit never reached the match"
    assert result["commits"][0]["fighter"] == "red"
    assert result["commits"][0]["combination"] == "combo-a"


@pytest.mark.slow
def test_a_spectator_watches_and_cannot_play(library) -> None:
    """M5-T3's screen is a spectator. It must not occupy a seat, must not be able to fight, and
    (`D6`) sees the same combination library a fighter does."""
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer

    from openroboxing.runtime.match import MatchFormat
    from openroboxing.server.app import build_app

    async def run() -> dict:
        host = MatchHost(
            libraries={f: library for f in FIGHTERS},
            match_format=MatchFormat(rounds=1, round_ticks=60, get_up_window_ticks=50),
            render=False,
        )
        client = TestClient(TestServer(build_app(host)))
        await client.start_server()

        watcher = await client.ws_connect("/ws?seat=spectator")
        welcome = await watcher.receive_json()
        state = await watcher.receive_json()

        await watcher.send_json({"type": "commit"})
        refusal = await watcher.receive_json()

        # Both seats must still be free: a spectator occupies nothing.
        red = await client.ws_connect("/ws?seat=red")
        red_welcome = await red.receive_json()

        await watcher.close()
        await red.close()
        await client.close()
        return {"welcome": welcome, "state": state, "refusal": refusal, "red": red_welcome}

    result = asyncio.run(run())

    assert result["welcome"]["type"] == "welcome"
    assert result["welcome"]["seat"] == "spectator"
    assert result["welcome"]["combinations"] == result["red"]["combinations"], (
        "a spectator must see exactly what a fighter sees (D6) — there is no loadout to leak"
    )
    assert result["state"]["type"] == "state"
    assert result["refusal"]["type"] == "error"
    assert "cannot play" in result["refusal"]["message"]
    assert result["red"]["type"] == "welcome", "the spectator took a fighter's seat"
    assert result["red"]["seat"] == "red"


@pytest.mark.slow
def test_a_taken_seat_is_refused(library) -> None:
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer

    from openroboxing.runtime.match import MatchFormat
    from openroboxing.server.app import build_app

    async def run() -> dict:
        host = MatchHost(
            libraries={f: library for f in FIGHTERS},
            match_format=MatchFormat(rounds=1, round_ticks=60, get_up_window_ticks=50),
            render=False,
        )
        client = TestClient(TestServer(build_app(host)))
        await client.start_server()

        first = await client.ws_connect("/ws?seat=red")
        await first.receive_json()
        await first.receive_json()

        second = await client.ws_connect("/ws?seat=red")
        message = await second.receive_json()

        await first.close()
        await second.close()
        await client.close()
        return message

    assert asyncio.run(run())["type"] == "error"
