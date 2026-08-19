"""M4-T2: the match host and its protocol.

Acceptance criterion from WORKPLAN.md M4-T2:
  two browsers on the LAN play a full match; artificially injecting 200 ms latency does not change
  match outcomes systematically.

A browser cannot be driven from pytest. What is tested here is everything underneath it: the
protocol validates hostile input, the host applies queued intents on its own tick, and a real
websocket client can join, play and be streamed to. The browser half is `docs/playtest/` work.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_server.py -v
    .venv_mb/bin/python -m pytest tests/test_server.py -v -m slow   # needs a GPU
"""

from __future__ import annotations

import pytest

from openroboxing.paths import LOADOUT_DIR
from openroboxing.runtime.arena import FIGHTERS
from openroboxing.runtime.intents import IntentTimeline, Loadout
from openroboxing.server import protocol
from openroboxing.server.host import MatchHost, QueuedPilot
from openroboxing.spec.constants import COMMIT_HORIZON_TICKS, MAX_OUTSTANDING_COMMITS


def _drive(timeline, through: int, *, start: int = 0) -> None:
    """Run a timeline's commit queue forward, arriving instantly.

    A commit's span is settled as it runs (`spec/intent.md` 1.1), so a test that needs a started
    move has to start it. Arriving instantly keeps the walk out of tests that are about something
    else.
    """
    for tick in range(start, through):
        timeline.generator_intent(tick, has_arrived=lambda _commit: True)

pytest.importorskip("aiohttp")


@pytest.fixture(scope="module")
def loadout() -> Loadout:
    return Loadout.load(LOADOUT_DIR / "orthodox.json")


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
        ({"type": "stage"}, "stage needs a slot"),
        ({"type": "stage", "slot": 3}, "stage needs a slot"),
    ],
)
def test_a_malformed_message_is_refused(message, error) -> None:
    with pytest.raises(protocol.ProtocolError, match=error):
        protocol.parse(message)


def test_a_valid_message_comes_back_normalised() -> None:
    assert protocol.parse({"type": "commit"}) == {"type": "commit"}
    assert protocol.parse({"type": "stage", "slot": "4"}) == {"type": "stage", "slot": "4"}
    assert protocol.parse({"type": "ping", "t": 7}) == {"type": "ping", "t": 7}


def test_a_handle_is_trimmed_and_bounded() -> None:
    parsed = protocol.parse({"type": "join", "handle": "  " + "x" * 100 + "  "})
    assert len(parsed["handle"]) == 32


def test_the_opponents_staged_move_is_never_in_a_state_message(loadout) -> None:
    """`WORKPLAN` M4-T1: no HUD on the fighters — the windup is the only cue.

    A `state` carries each seat's own staged slot for its own client to draw. Nothing in the schema
    lets one seat learn what the other has *staged but not committed*, which is the information the
    whole design withholds.
    """
    timeline = IntentTimeline(loadout)
    timeline.stage(pose_slot="3")
    commit = timeline.commit(0)

    seat = protocol.seat_state("blue", staged=None, queue=[], can_commit=True,
                               hits_landed=0, torso_height_m=0.84, down=False)
    message = protocol.state(0, 0, 3000, {"blue": seat}, protocol.PHASE_FIGHTING)

    assert "qpos" not in message, "a modified client would see the opponent exactly"
    assert message["seats"]["blue"]["staged"] is None
    assert message["seats"]["blue"]["placement"] is None, "nor where they are aiming"
    assert set(protocol.queue_entry(commit, 0)) == {
        "slot", "pose", "issued_at", "commit_at", "strike_at", "end_tick",
        "executing", "approaching", "placement",
    }


def test_a_seat_sees_its_own_queue_and_of_another_only_what_is_running(loadout) -> None:
    """A queued-but-unstarted commit has been paid for and not yet shown. Leaking it would hand the
    opponent a readable list of the next four moves, which is the risk queueing is meant to be."""
    timeline = IntentTimeline(loadout)
    timeline.stage(pose_slot="1")
    running = timeline.commit(0)
    timeline.stage(pose_slot="2")
    timeline.commit(1)
    _drive(timeline, COMMIT_HORIZON_TICKS + 1)

    tick = running.commit_at
    scheduled = timeline.scheduled(tick)
    assert len(scheduled) == 2

    assert protocol.visible_queue(scheduled, tick, own=True) == list(scheduled)
    hidden = protocol.visible_queue(scheduled, tick, own=False)
    assert [c.slot for c in hidden] == ["1"], "only the running move is public"
    assert hidden[0].is_executing(tick)


def test_nothing_of_another_seat_is_visible_before_its_move_starts(loadout) -> None:
    """The readable window is the point: a commit becomes public when it becomes visible."""
    timeline = IntentTimeline(loadout)
    timeline.stage(pose_slot="1")
    commit = timeline.commit(0)

    # Inside the readable window nothing has started, so `commit_at` is still None — the queue is not
    # a schedule any more (`spec/intent.md` 1.1) and a viewer's rule cannot be written against one.
    inside = COMMIT_HORIZON_TICKS - 1
    during_window = timeline.scheduled(inside)
    assert protocol.visible_queue(during_window, inside, own=False) == []
    assert protocol.visible_queue(during_window, inside, own=True) == [commit]


# --- a match's knobs reach the match (M4-T4) ------------------------------------------------------
@pytest.mark.slow
def test_the_arena_config_reaches_the_compiled_ring(loadout) -> None:
    """`build_arena` compiles ring size into the model, so a config assigned *after* construction
    changes the record and nothing a fighter can touch.

    `tools/tune.py` did exactly that and swept `ring_size` and `glove_radius` for nothing, reporting
    the absence of any difference as noise. The knob has to go in at construction, and this is what
    says so.
    """
    import mujoco

    from openroboxing.runtime.arena import ArenaConfig

    small = MatchHost(
        loadouts={f: loadout for f in FIGHTERS},
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


# --- placement (spec/protocol.md 0.4) -------------------------------------------------------------
def test_a_placement_is_accepted() -> None:
    assert protocol.parse({"type": "place", "x": 1.5, "y": -0.25, "heading": 0.3}) == {
        "type": "place", "x": 1.5, "y": -0.25, "heading": 0.3
    }


@pytest.mark.parametrize(
    "message, error",
    [
        ({"type": "place", "y": 0.0, "heading": 0.0}, "needs a numeric 'x'"),
        ({"type": "place", "x": "far", "y": 0.0, "heading": 0.0}, "needs a numeric 'x'"),
        ({"type": "place", "x": True, "y": 0.0, "heading": 0.0}, "needs a numeric 'x'"),
        ({"type": "place", "x": float("nan"), "y": 0.0, "heading": 0.0}, "must be finite"),
        ({"type": "place", "x": 0.0, "y": 0.0, "heading": float("inf")}, "must be finite"),
        ({"type": "place", "x": 1e6, "y": 0.0, "heading": 0.0}, "sanity bound"),
    ],
)
def test_a_malformed_placement_is_refused(message, error) -> None:
    """A client sends these straight into the generator's target. NaN would surface as a fighter
    that vanishes, which is a very hard bug to read backwards."""
    with pytest.raises(protocol.ProtocolError, match=error):
        protocol.parse(message)


def test_the_movement_message_is_gone() -> None:
    """Retired with the channel at 0.4. Accepting it silently would let an old client think it was
    steering while nothing moved."""
    assert "move" not in protocol.CLIENT_MESSAGES
    with pytest.raises(protocol.ProtocolError, match="unknown message type"):
        protocol.parse({"type": "move", "direction": "in"})


def test_a_state_carries_the_range() -> None:
    """Range is not secret — both fighters can see it by looking — and without it a player cannot
    manage distance, which is most of boxing."""
    message = protocol.state(0, 0, 3000, {}, protocol.PHASE_FIGHTING, separation_m=1.4237)
    assert message["separation_m"] == 1.424


def test_a_seat_sees_its_own_shadow_and_anchor() -> None:
    from openroboxing.runtime.intents import Placement

    seat = protocol.seat_state(
        "red", None, [], True, 0, 0.84, False,
        placement=Placement((1.25, -0.5), 0.75), anchor=Placement((0.9, 0.0), 0.0),
    )
    assert seat["placement"] == {"x": 1.25, "y": -0.5, "heading": 0.75}
    assert seat["anchor"] == {"x": 0.9, "y": 0.0, "heading": 0.0}
    assert protocol.seat_state("red", None, [], True, 0, 0.84, False)["placement"] is None


def test_a_placement_reaches_the_timeline(loadout) -> None:
    from openroboxing.runtime.intents import Placement

    pilot = QueuedPilot()
    timeline = IntentTimeline(loadout)

    pilot.queue({"type": "place", "x": 1.5, "y": -0.5, "heading": 0.4})
    pilot.act(timeline, 0)
    assert timeline.staged.placement == Placement((1.5, -0.5), 0.4)
    assert pilot.placement == Placement((1.5, -0.5), 0.4)


def test_an_untouched_shadow_commits_at_the_anchor(loadout) -> None:
    """"Commit without aiming" must mean "do it where I will be", not "do it again over there"."""
    from openroboxing.runtime.intents import Placement

    pilot = QueuedPilot()
    timeline = IntentTimeline(loadout)
    pilot.anchor = Placement((0.8, 0.1), 0.2)

    pilot.queue({"type": "stage", "slot": "1"})
    pilot.queue({"type": "commit"})
    pilot.act(timeline, 0)
    assert timeline.commits[0].placement == Placement((0.8, 0.1), 0.2)

    # The next commit defaults to the anchor again rather than reusing the first one's placement.
    pilot.anchor = Placement((1.9, 0.0), 0.0)
    pilot.queue({"type": "stage", "slot": "2"})
    pilot.queue({"type": "commit"})
    pilot.act(timeline, 1)
    assert timeline.commits[1].placement == Placement((1.9, 0.0), 0.0)


# --- the live score (spec/protocol.md 0.2) -----------------------------------------------------------
def test_a_state_carries_a_score_slot() -> None:
    """Added in 0.2 by the project owner's decision. v0.1 withheld it."""
    message = protocol.state(0, 0, 3000, {}, protocol.PHASE_FIGHTING)
    assert "score" in message
    assert message["score"] is None, "no score until there is a round to score"


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


def test_can_commit_carries_the_rule_so_the_client_does_not(loadout) -> None:
    free = protocol.seat_state("red", None, [], True, 0, 0.84, False)
    full = protocol.seat_state("red", "1", [{"slot": "1"}], False, 0, 0.84, False)
    assert free["can_commit"] is True and full["can_commit"] is False
    assert free["queue_depth"] == 0 and full["queue_depth"] == 1


def test_can_commit_follows_the_bound_not_a_single_active_move(loadout) -> None:
    timeline = IntentTimeline(loadout)
    for index in range(MAX_OUTSTANDING_COMMITS):
        timeline.stage(pose_slot="1")
        timeline.commit(0)
        free = len(timeline.scheduled(0)) < timeline.max_outstanding
        assert free is (index < MAX_OUTSTANDING_COMMITS - 1)


def test_welcome_carries_the_keys_and_their_durations(loadout) -> None:
    from openroboxing.runtime.match import MatchFormat

    message = protocol.welcome("red", loadout, MatchFormat(), {}, "m1")
    assert set(message["loadout"]) == set(loadout.slots)
    assert all(isinstance(v, int) for v in message["horizons"].values())
    assert message["format"]["rounds"] == 3


def test_welcome_carries_the_pose_angles_the_client_draws_its_shadow_from(loadout) -> None:
    """0.4: the shadow is posed in the browser, so the angles have to get there. A ghost that had to
    ask the server where its elbow goes could not be aimed with."""
    from openroboxing.runtime.conventions import G1
    from openroboxing.runtime.match import MatchFormat

    message = protocol.welcome("red", loadout, MatchFormat(), {}, "m1")
    assert set(message["poses"]) == set(loadout.slots)
    for slot, angles in message["poses"].items():
        assert set(angles) == set(G1.mujoco_joint_names), f"slot {slot} cannot be posed"


# --- the queued pilot -----------------------------------------------------------------------------------
def test_keypresses_are_applied_on_the_tick_not_when_they_arrive(loadout) -> None:
    """A client must not be able to make the simulation wait, nor interleave into a step."""
    pilot = QueuedPilot()
    timeline = IntentTimeline(loadout)

    pilot.queue({"type": "stage", "slot": "2"})
    pilot.queue({"type": "commit"})
    assert timeline.commits == (), "nothing happens until the tick"

    pilot.act(timeline, 100)
    assert len(timeline.commits) == 1
    assert timeline.commits[0].slot == "2"
    assert timeline.commits[0].issued_at == 100


def test_committing_with_nothing_staged_is_an_error_not_a_crash(loadout) -> None:
    pilot = QueuedPilot()
    timeline = IntentTimeline(loadout)
    pilot.queue({"type": "commit"})
    pilot.act(timeline, 0)

    assert timeline.commits == ()
    assert "nothing is staged" in pilot.last_error


def test_a_second_commit_is_queued_rather_than_refused(loadout) -> None:
    """The 1.0 rule: commits stack up to the bound and run back to back (`spec/intent.md`)."""
    pilot = QueuedPilot()
    timeline = IntentTimeline(loadout)

    pilot.queue({"type": "stage", "slot": "1"})
    pilot.queue({"type": "commit"})
    pilot.act(timeline, 0)

    pilot.queue({"type": "stage", "slot": "2"})
    pilot.queue({"type": "commit"})
    pilot.act(timeline, 5)

    assert len(timeline.commits) == 2
    assert pilot.last_error is None
    assert timeline.commits[1].commit_at == timeline.commits[0].end_tick


def test_a_commit_past_the_bound_is_refused_by_the_host(loadout) -> None:
    """The queue bound is enforced here, not in the UI (`spec/intent.md`)."""
    pilot = QueuedPilot()
    timeline = IntentTimeline(loadout)

    for _ in range(MAX_OUTSTANDING_COMMITS + 1):
        pilot.queue({"type": "stage", "slot": "1"})
        pilot.queue({"type": "commit"})
    pilot.act(timeline, 0)

    assert len(timeline.commits) == MAX_OUTSTANDING_COMMITS
    assert "already queued" in pilot.last_error


@pytest.mark.slow
def test_the_host_answers_a_full_queue_immediately(loadout) -> None:
    """`can_commit` in the next state is 33 ms away, and a full queue is the one rejection a player
    reacts to at once."""
    from openroboxing.server.host import MatchHost

    host = MatchHost(loadouts={f: loadout for f in FIGHTERS}, render=False)
    timeline = host.world.fighters["red"].timeline
    for _ in range(MAX_OUTSTANDING_COMMITS):
        timeline.stage(pose_slot="1")
        timeline.commit(0)

    reply = host.handle("red", {"type": "commit"})
    assert reply is not None and reply["rejected"] == "commit"
    assert "already queued" in reply["message"]


def test_staging_stays_free_while_a_commit_runs(loadout) -> None:
    """`spec/intent.md`: staging is unbounded and happens during play."""
    pilot = QueuedPilot()
    timeline = IntentTimeline(loadout)
    pilot.queue({"type": "stage", "slot": "1"})
    pilot.queue({"type": "commit"})
    pilot.act(timeline, 0)

    pilot.queue({"type": "stage", "slot": "5"})
    pilot.act(timeline, 10)
    assert timeline.staged.pose_slot == "5"
    assert pilot.staged == "5"


def test_clearing_unstages_without_cancelling(loadout) -> None:
    pilot = QueuedPilot()
    timeline = IntentTimeline(loadout)
    pilot.queue({"type": "stage", "slot": "1"})
    pilot.act(timeline, 0)
    pilot.queue({"type": "clear"})
    pilot.act(timeline, 1)

    assert timeline.staged.pose_slot is None
    assert pilot.staged is None


def test_a_reset_pilot_forgets_the_queue(loadout) -> None:
    pilot = QueuedPilot()
    pilot.queue({"type": "stage", "slot": "1"})
    pilot.reset()

    timeline = IntentTimeline(loadout)
    pilot.act(timeline, 0)
    assert timeline.staged.pose_slot is None, "a queue survived into the next round"


# --- the host, live -------------------------------------------------------------------------------------
@pytest.mark.slow
async def _play(host, aiohttp_client) -> dict:
    from openroboxing.server.app import build_app

    client = await aiohttp_client(build_app(host))
    socket = await client.ws_connect("/ws?seat=red")
    welcome = await socket.receive_json()
    return {"client": client, "socket": socket, "welcome": welcome}


@pytest.mark.slow
def test_a_client_can_join_stage_commit_and_be_streamed_to() -> None:
    """The whole stack in one go: aiohttp, the protocol, the host, physics, the renderer."""
    import asyncio
    import json

    from aiohttp.test_utils import TestClient, TestServer

    from openroboxing.runtime.match import MatchFormat
    from openroboxing.server.app import build_app
    from openroboxing.server.host import MatchHost

    loadout = Loadout.load(LOADOUT_DIR / "orthodox.json")

    async def run() -> dict:
        host = MatchHost(
            loadouts={f: loadout for f in FIGHTERS},
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
        await red.send_json({"type": "stage", "slot": "1"})
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
    assert set(result["welcome"]["loadout"]) == set(loadout.slots)

    assert result["states"] > 10, "the host did not stream state"
    assert result["frames"] > 10, "the host did not stream video"

    assert len(result["record"].rounds) == 1
    assert result["commits"], "the client's commit never reached the match"
    assert result["commits"][0]["fighter"] == "red"
    assert result["commits"][0]["slot"] == "1"


@pytest.mark.slow
def test_a_spectator_watches_and_cannot_play() -> None:
    """M5-T3's screen is a spectator. It must not occupy a seat, and must not be able to fight."""
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer

    from openroboxing.runtime.match import MatchFormat
    from openroboxing.server.app import build_app
    from openroboxing.server.host import MatchHost

    loadout = Loadout.load(LOADOUT_DIR / "orthodox.json")

    async def run() -> dict:
        host = MatchHost(
            loadouts={f: loadout for f in FIGHTERS},
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
    assert result["welcome"]["loadout"] == {}, "a projector must not show what each fighter holds"
    assert result["state"]["type"] == "state"
    assert result["refusal"]["type"] == "error"
    assert "cannot play" in result["refusal"]["message"]
    assert result["red"]["type"] == "welcome", "the spectator took a fighter's seat"
    assert result["red"]["seat"] == "red"


@pytest.mark.slow
def test_a_taken_seat_is_refused() -> None:
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer

    from openroboxing.runtime.match import MatchFormat
    from openroboxing.server.app import build_app
    from openroboxing.server.host import MatchHost

    loadout = Loadout.load(LOADOUT_DIR / "orthodox.json")

    async def run() -> dict:
        host = MatchHost(
            loadouts={f: loadout for f in FIGHTERS},
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
