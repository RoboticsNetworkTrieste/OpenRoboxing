"""The sparring host: knobs, the step/pause/reset loop, the debug message, scrub repacking.

Everything here runs against a fake world (no GPU, no checkpoints); the scrub test uses the real
arena model, which is CPU-only mujoco.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_sparring_app.py -v
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from openroboxing.runtime.conventions import G1
import openroboxing.runtime.intents as intents_module
from openroboxing.runtime.intents import IntentTimeline, Loadout, Placement
from openroboxing.server.host import QueuedPilot
from openroboxing.server.sparring_app import (
    PLAN_HORIZON_TICKS,
    SPARRING_MAX_OUTSTANDING,
    TRAIL_STRIDE,
    SparringError,
    SparringHost,
    knob_values,
    repack_frame,
    set_knobs,
)
from openroboxing.spec.constants import (
    APPROACH_LEG_M,
    ARRIVAL_RADIUS_M,
    COMMIT_HORIZON_TICKS,
    QPOS_DIM,
)
from openroboxing.studio.pose_record import PoseRecord


# -- fixtures ---------------------------------------------------------------------------------------
def _pose(name: str) -> PoseRecord:
    from openroboxing.runtime.obs import default_angles

    return PoseRecord(
        name=name,
        joint_angles=dict(zip(G1.mujoco_joint_names, default_angles(G1, "mujoco"))),
        horizon_tokens=8,
        library_version="v0.1",
        admission="admitted",
        telegraph_ms=180.0,
        generator_error_rad=0.1,
    )


def _loadout() -> Loadout:
    return Loadout(
        name="orthodox", version="v0.1", slots={"1": _pose("jab-left"), "2": _pose("hook-right")}
    )


def _standing_qpos() -> np.ndarray:
    qpos = np.zeros(QPOS_DIM)
    qpos[2] = 0.79
    qpos[3] = 1.0
    return qpos


class FakeStream:
    def __init__(self, ticks: int = 400) -> None:
        self.motion = np.tile(_standing_qpos(), (ticks, 1))
        self.motion[:, 0] = np.arange(ticks) * 0.01  # drifts +x so the ghost is ahead
        self.replan_dt = 0.5


class FakeFighter(SimpleNamespace):
    pass


def _fake_fighter(loadout: Loadout, pilot) -> FakeFighter:
    agent = SimpleNamespace(frames={"mujoco_qpos": np.zeros((1, 64, QPOS_DIM))})
    generator = SimpleNamespace(agent=agent, generate=lambda *a, **k: None)
    fighter = FakeFighter(
        timeline=IntentTimeline(loadout, max_outstanding=SPARRING_MAX_OUTSTANDING),
        stream=FakeStream(),
        generator=generator,
        pilot=pilot,
        loadout=loadout,
        last_action=np.zeros(29),
        root_qpos=np.arange(0, 7),
        apply_yaw=0.0,
    )
    fighter.robot_state = lambda data: SimpleNamespace(
        joint_pos=np.asarray(data.qpos[7:36], dtype=np.float64)
    )
    return fighter


class FakeWorld:
    def __init__(self) -> None:
        loadout = _loadout()
        self.red_pilot = QueuedPilot()
        self.fighters = {
            "red": _fake_fighter(loadout, self.red_pilot),
            "blue": _fake_fighter(loadout, SimpleNamespace(reset=lambda: None)),
        }
        qpos = np.zeros(79)
        qpos[0:QPOS_DIM] = _standing_qpos()
        self.data = SimpleNamespace(qpos=qpos)
        self.pool = SimpleNamespace(match_seed=1234)
        self.config = SimpleNamespace(ring_size=4.9)
        self.arrival_radius_m = ARRIVAL_RADIUS_M
        self.approach_leg_m = APPROACH_LEG_M
        self.stepped: list[int] = []
        self.resets: list[int] = []

    def step(self, tick: int) -> None:
        self.stepped.append(tick)
        # The real world applies queued pilot messages inside step; mirror that.
        self.red_pilot.act(self.fighters["red"].timeline, tick)

    def reset_round(self, index: int) -> None:
        self.resets.append(index)

    def separation_m(self) -> float:
        return 1.0

    def anchor(self, fighter: str, tick: int) -> Placement:
        return Placement(position=(0.0, 0.0), heading=0.0)

    def root_pose(self, fighter: str) -> Placement:
        return Placement(position=(0.0, 0.0), heading=0.0)


def _host(**kwargs) -> SparringHost:
    return SparringHost(FakeWorld(), render=False, **kwargs)


# -- knobs ------------------------------------------------------------------------------------------
class TestKnobs:
    def test_every_knob_reports_current_and_canonical(self) -> None:
        values = knob_values(FakeWorld())
        assert set(values) == {
            "replan_dt",
            "horizon_ticks",
            "max_outstanding",
            "arrival_radius_m",
            "approach_leg_m",
            "approach_timeout_ticks",
            "pose_dwell_ticks",
        }
        for entry in values.values():
            assert set(entry) == {"current", "canonical"}
        assert values["replan_dt"]["current"] == pytest.approx(0.5)
        assert values["horizon_ticks"]["canonical"] == COMMIT_HORIZON_TICKS
        # The bench's deeper queue is a *deliberate* deviation, and always reads as one.
        assert values["max_outstanding"]["current"] == SPARRING_MAX_OUTSTANDING

    def test_setting_a_knob_mutates_the_world(self) -> None:
        world = FakeWorld()
        result = set_knobs(world, {"replan_dt": 0.3, "arrival_radius_m": 0.25})
        assert world.fighters["red"].stream.replan_dt == pytest.approx(0.3)
        assert world.arrival_radius_m == pytest.approx(0.25)
        assert result["replan_dt"]["current"] == pytest.approx(0.3)

    def test_an_unknown_knob_is_refused(self) -> None:
        with pytest.raises(SparringError, match="unknown knob"):
            set_knobs(FakeWorld(), {"warp_speed": 9})

    def test_a_non_positive_value_is_refused(self) -> None:
        with pytest.raises(SparringError, match="positive"):
            set_knobs(FakeWorld(), {"replan_dt": 0.0})
        with pytest.raises(SparringError, match="finite"):
            set_knobs(FakeWorld(), {"replan_dt": float("nan")})

    def test_the_dwell_knob_reaches_the_next_completion(self) -> None:
        """The counted dwell is now the rule for a caller with **no settle test**, and the knob
        still moves it. It no longer rewrites a span already stamped: since `spec/intent.md` 2.2
        `end_tick` is a record of when a move ended, and turning a knob cannot change the past."""
        world = FakeWorld()
        original = intents_module.POSE_DWELL_TICKS
        try:
            timeline = world.fighters["red"].timeline
            timeline.stage(pose_slot="1")
            commit = timeline.commit(0)
            set_knobs(world, {"pose_dwell_ticks": 10})

            for tick in range(COMMIT_HORIZON_TICKS, COMMIT_HORIZON_TICKS + 12):
                timeline.generator_intent(tick, has_arrived=lambda _c: True)
            assert commit.completed_by == "dwell"
            assert commit.end_tick == commit.strike_at + 10
        finally:
            intents_module.POSE_DWELL_TICKS = original


# -- the host ---------------------------------------------------------------------------------------
class TestHost:
    def test_step_once_advances_and_records(self) -> None:
        host = _host()
        host.step_once()
        assert host.tick == 1
        assert host.world.stepped == [0]
        assert len(host.tap) == 1
        row = host.tap.at(0)
        assert row["machine"] == 0  # OPENING
        assert math.isnan(float(row["dist_target"]))
        assert float(row["step_ms"]) >= 0.0

    def test_a_replan_is_recorded_and_a_cadence_noop_is_not(self) -> None:
        """The watcher compares tensor *identity with a held reference* — an id() comparison let
        the freed tensor's address be recycled and swallowed the event (measured live)."""
        host = _host()
        generator = host.world.fighters["red"].generator

        def replanning_generate(intent, context_qpos, dt, *, force=False):
            generator.agent.frames["mujoco_qpos"] = np.zeros((1, 44, QPOS_DIM))

        # The wrapper was installed at construction around the fake's original no-op generate;
        # swap the *inner* behaviour by reinstalling around a replanning one.
        generator.generate = replanning_generate
        host._watch_replans()

        host._filling_tick = 7
        generator.generate(None, None, 0.5)          # reassigns -> one replan event
        assert host.tap.replans == [(7, False, 44)]
        generator.generate(None, None, 0.5)          # reassigns again -> a second event
        assert len(host.tap.replans) == 2

    def test_a_pilot_error_surfaces_in_the_debug_message(self) -> None:
        """QueuedPilot.act records errors instead of raising; the bench must not let them rot
        there. A commit with nothing staged is the canonical case: the host precheck passes
        (the queue is not full), the pilot refuses it a tick later, and before this the player
        saw nothing at all."""
        host = _host()
        assert host.handle({"type": "commit"}) is None  # precheck passes; queued
        host.step_once()  # the pilot applies it and records the refusal
        message = host.debug_message()
        assert message["pilot_error"] is not None
        assert "staged" in message["pilot_error"]["message"]
        assert message["pilot_error"]["tick"] == 0

    def test_pause_makes_step_a_noop(self) -> None:
        host = _host()
        host.step_once()
        host.paused = True
        host.step_once()
        assert host.tick == 1
        assert len(host.tap) == 1

    def test_reset_reseeds_and_clears(self) -> None:
        host = _host()
        host.step_once()
        host.reset(seed=7)
        assert host.world.pool.match_seed == 7
        assert host.world.resets == [0]
        assert host.tick == 0
        assert len(host.tap) == 0

    def test_the_debug_message_carries_the_panel(self) -> None:
        host = _host()
        timeline = host.world.fighters["red"].timeline
        timeline.stage(pose_slot="1", placement=Placement(position=(1.0, 0.0), heading=0.0))
        timeline.commit(0)
        host.step_once()

        message = host.debug_message()
        assert message["type"] == "debug"
        assert message["machine"] in ("OPENING", "WAITING", "APPROACH", "DWELL", "HOLD")
        assert len(message["queue"]) == 1
        entry = message["queue"][0]
        assert entry["slot"] == "1" and entry["commit_at"] is None
        ghost = message["ghost"]
        assert set(ghost) == {"x", "y", "z", "heading", "angles"}
        assert len(ghost["angles"]) == 29
        assert len(message["trail"]) >= 2
        assert message["knobs"]["replan_dt"]["canonical"] == pytest.approx(0.5)
        assert message["recording"] == {"start_tick": 0, "end_tick": 0}

    def test_scrub_shows_the_plan_tail_not_the_rest_of_the_session(self) -> None:
        """A scrubbed trail is the plan the reference held **then**, bounded like the live one.

        0.1 built the scrub reference from every recorded row between the scrubbed tick and the end
        of the recording, so scrubbing one minute back drew a minute of future robot path across the
        whole ring — the "strange trail" the owner reported, 2026-08-17. Live and scrub must cover
        the same horizon or the two modes disagree about what a plan is.
        """
        host = _host()
        for _ in range(200):
            host.step_once()

        live = host.debug_message()["trail"]
        scrubbed = host.scrub_payload(10)["trail"]
        assert len(scrubbed) == len(live)
        assert len(scrubbed) <= math.ceil(PLAN_HORIZON_TICKS / TRAIL_STRIDE) + 1

    def test_the_panel_separates_the_body_from_the_plan(self) -> None:
        """Two distances to one placement: what the body reached, and what the plan reached.

        Measured 2026-08-17 over seven bearings, this pair is the whole diagnosis of an approach —
        the plan closed to 0.02-0.19 m every time while the body only closed straight ahead, so
        four of seven commits ended on the timeout. One distance cannot say that.
        """
        host = _host()
        timeline = host.world.fighters["red"].timeline
        timeline.stage(pose_slot="1", placement=Placement(position=(1.0, 0.0), heading=0.0))
        commit = timeline.commit(0)
        commit.commit_at = 0  # the fake world has no generator to open the approach for us
        for _ in range(5):
            host.step_once()

        head = host.debug_message()["series_head"]
        assert head["dist"] is not None and head["dist_plan"] is not None
        assert "dist_plan" in host.tap.series(0, 4)

    def test_a_timed_out_commit_says_so_in_the_queue(self) -> None:
        host = _host()
        timeline = host.world.fighters["red"].timeline
        timeline.stage(pose_slot="1", placement=Placement(position=(9.0, 9.0), heading=0.0))
        commit = timeline.commit(0)
        host.step_once()
        commit.commit_at, commit.strike_at, commit.arrived = 0, 1, False

        entry = host.debug_message()["queue"][0]
        assert entry["arrived"] is False

    def test_scrub_carries_the_queue(self) -> None:
        """The panel must not empty out when you scrub — a blank queue reads as "nothing happened"."""
        host = _host()
        timeline = host.world.fighters["red"].timeline
        timeline.stage(pose_slot="1", placement=Placement(position=(1.0, 0.0), heading=0.0))
        timeline.commit(0)
        for _ in range(30):
            host.step_once()

        payload = host.scrub_payload(10)
        assert len(payload["queue"]) == 1
        assert payload["queue"][0]["slot"] == "1"

    def test_handle_routes_sparring_controls(self) -> None:
        host = _host()
        assert host.handle({"type": "pause"}) is None
        assert host.paused is True
        assert host.handle({"type": "resume"}) is None
        assert host.paused is False
        assert host.handle({"type": "reset", "seed": 9}) is None
        assert host.world.pool.match_seed == 9

    def test_handle_refuses_a_bad_teleport(self) -> None:
        reply = _host().handle({"type": "teleport_sacco", "x": float("nan"), "y": 0, "heading": 0})
        assert reply["type"] == "error"

    def test_handle_stages_and_commits_through_the_pilot(self) -> None:
        host = _host()
        assert host.handle({"type": "stage", "slot": "1"}) is None
        assert host.handle({"type": "commit"}) is None
        host.step_once()  # the fake world applies pilot messages on step, as the real one does
        assert len(host.world.fighters["red"].timeline.commits) == 1

    def test_handle_rejects_the_eleventh_commit(self) -> None:
        host = _host()
        timeline = host.world.fighters["red"].timeline
        timeline.stage(pose_slot="1")
        for _ in range(SPARRING_MAX_OUTSTANDING):
            timeline.commit(0)
        reply = host.handle({"type": "commit"})
        assert reply["type"] == "error" and "queued" in reply["message"]

    def test_handle_rejects_garbage(self) -> None:
        reply = _host().handle({"type": "warp"})
        assert reply["type"] == "error"


# -- scrub repacking (real arena model, CPU mujoco) --------------------------------------------------
class TestRepack:
    def test_a_repacked_frame_is_byte_identical(self) -> None:
        mujoco = pytest.importorskip("mujoco")
        from openroboxing.runtime.arena import ArenaConfig, build_arena
        from openroboxing.server.scene import Scene

        model = build_arena(ArenaConfig())
        data = mujoco.MjData(model)
        data.qpos[0] += 0.3  # move red so the frame is not the default pose
        mujoco.mj_forward(model, data)
        scene = Scene(model, {})
        live = scene.pack(5, data)
        recorded = np.asarray(data.qpos, dtype=np.float64).copy()  # the tap's qpos dtype

        scratch = mujoco.MjData(model)
        assert repack_frame(scene, model, scratch, 5, recorded) == live


# -- sockets ----------------------------------------------------------------------------------------
class TestSockets:
    def test_every_socket_may_drive(self) -> None:
        """A refreshed page must never lose control of the bench.

        The 0.1 gate gave control to the first socket and made everyone after it a viewer. On a
        page refresh the browser can open the new socket before the server processes the old one's
        close, and nothing ever promoted a viewer — so the live page was locked out forever:
        stage/commit rejected, the queue empty, the robot standing still while the (local) aim
        ghost still moved. Reproduced live, 2026-08-17. The bench is one human on localhost; the
        gate defended nothing, so it is gone: every socket drives.
        """
        import asyncio

        pytest.importorskip("aiohttp")
        from aiohttp.test_utils import TestClient, TestServer

        from openroboxing.server.sparring_app import build_sparring_app

        async def run() -> None:
            host = _host()
            client = TestClient(TestServer(build_sparring_app(host)))
            await client.start_server()
            try:
                first = await client.ws_connect("/ws")
                second = await client.ws_connect("/ws")  # the page after a refresh
                for socket in (first, second):
                    for _ in range(3):  # welcome, state, debug
                        await socket.receive_json()

                await second.send_json({"type": "stage", "slot": "1"})
                await second.send_json({"type": "place", "x": 1.0, "y": 0.5, "heading": 0.0})
                await second.send_json({"type": "commit"})
                await asyncio.sleep(0.05)  # let the handler run

                host.step_once()  # the world applies the pilot's queue on step
                assert len(host.world.fighters["red"].timeline.commits) == 1, (
                    "the second socket's commit never reached the timeline; is a controller "
                    "gate back?"
                )
            finally:
                await client.close()

        asyncio.run(run())
