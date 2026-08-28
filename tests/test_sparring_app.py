"""The sparring host: knobs, the step/pause/reset loop, the debug message, scrub repacking.

Ported for `spec/intent.md` 3.0 (B3): a commit carries a combination and a ghost, not a loadout
slot and a placement — see this file's own commit message for exactly which tests that retired
(the approach's `arrived`/`completed_by`, `arrival_radius_m`, `approach_leg_m`,
`approach_timeout_ticks`, `pose_dwell_ticks`) and what replaced them (`leg_index`/`leg_count`,
`drift_speed_m_s`, the `drift_gain` knob, `keyframe_events`, `drafts_allowed`).

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

from openroboxing.runtime import sequence, warp
from openroboxing.runtime.conventions import G1
from openroboxing.runtime.intents import IntentTimeline
from openroboxing.server.host import QueuedPilot
from openroboxing.server.sparring_app import (
    KNOBS,
    PLAN_HORIZON_TICKS,
    SPARRING_MAX_OUTSTANDING,
    TRAIL_STRIDE,
    SparringError,
    SparringHost,
    bench_queue_entry,
    knob_values,
    repack_frame,
    set_knobs,
)
from openroboxing.server.sparring_tap import MACHINE_STATES
from openroboxing.spec.constants import COMMIT_HORIZON_TICKS, DRIFT_GAIN, QPOS_DIM
from openroboxing.studio.combination_record import CombinationRecord, CombinationSource, Keyframe

ANGLES = {name: 0.0 for name in G1.mujoco_joint_names}


# -- fixtures ---------------------------------------------------------------------------------------
def _combination(name: str, *, tokens=(6, 6)) -> CombinationRecord:
    """A small, admitted combination. Mirrors the builder every other test file in this repo keeps
    its own copy of (`test_fight.py`, `test_intents_combinations.py`, `test_warp.py`)."""
    keyframes = [Keyframe(dict(ANGLES), None, (0.0, 0.0), 0.0)]
    for i, token in enumerate(tokens, start=1):
        keyframes.append(Keyframe(dict(ANGLES), token, (0.1 * i, 0.0), 0.0))
    return CombinationRecord(
        name=name,
        library_version="v0.2",
        source=CombinationSource("t", 0, 100, False),
        keyframes=keyframes,
        telegraph_ms=180.0,
        tracking_error_rad=0.1,
        admission="admitted",
    )


def _library() -> dict[str, CombinationRecord]:
    return {
        "jab-cross": _combination("jab-cross"),
        "hook-body": _combination("hook-body", tokens=(7, 9)),
    }


def _standing_qpos() -> np.ndarray:
    qpos = np.zeros(QPOS_DIM)
    qpos[2] = 0.79
    qpos[3] = 1.0
    return qpos


class FakeStream:
    def __init__(self, ticks: int = 400) -> None:
        self.motion = np.tile(_standing_qpos(), (ticks, 1))
        self.motion[:, 0] = np.arange(ticks) * 0.01  # drifts +x so a plan ghost ahead is meaningful
        self.replan_dt = 0.5


class FakeFighter(SimpleNamespace):
    pass


def _fake_fighter(
    library: dict[str, CombinationRecord], pilot, *, pelvis_body: int, require_admitted: bool
) -> FakeFighter:
    agent = SimpleNamespace(frames={"mujoco_qpos": np.zeros((1, 64, QPOS_DIM))})
    generator = SimpleNamespace(agent=agent, generate=lambda *a, **k: None)
    fighter = FakeFighter(
        timeline=IntentTimeline(
            library, max_outstanding=SPARRING_MAX_OUTSTANDING, require_admitted=require_admitted
        ),
        stream=FakeStream(),
        generator=generator,
        pilot=pilot,
        library=library,
        require_admitted=require_admitted,
        last_action=np.zeros(29),
        root_qpos=np.arange(0, 7),
        pelvis_body=pelvis_body,
        apply_yaw=0.0,
    )
    fighter.robot_state = lambda data: SimpleNamespace(
        joint_pos=np.asarray(data.qpos[7:36], dtype=np.float64)
    )
    return fighter


class FakeWorld:
    """Stands in for `SparringWorld`. No mujoco, no physics — just the surface `SparringHost` reads:
    `fighters`, `data.qpos`/`data.xpos`, `pool`, `config`, `step`/`reset_round`/`separation_m`, and
    the private `_drift_speed_m_s` table `bench_queue_entry` reads (`runtime/fight.py`'s own private
    attribute — the bench reaches into it rather than recompute the formula a third time).
    """

    def __init__(self, *, require_admitted: bool = False) -> None:
        library = _library()
        self.red_pilot = QueuedPilot()
        self.fighters = {
            "red": _fake_fighter(
                library, self.red_pilot, pelvis_body=0, require_admitted=require_admitted
            ),
            "blue": _fake_fighter(
                library,
                SimpleNamespace(reset=lambda: None),
                pelvis_body=1,
                require_admitted=require_admitted,
            ),
        }
        self.fighters["blue"].root_qpos = np.arange(36, 43)

        qpos = np.zeros(79)
        qpos[0:QPOS_DIM] = _standing_qpos()
        qpos[36 : 36 + QPOS_DIM] = _standing_qpos()
        xpos = np.zeros((2, 3))
        xpos[:, 2] = 0.79
        self.data = SimpleNamespace(qpos=qpos, xpos=xpos)

        self.pool = SimpleNamespace(match_seed=1234)
        self.config = SimpleNamespace(ring_size=4.9)
        #: Mirrors `FightWorld._drift_speed_m_s` exactly — see the class docstring.
        self._drift_speed_m_s: dict[int, float] = {}
        self.stepped: list[int] = []
        self.resets: list[int] = []
        self.teleports: list[tuple[float, float, float]] = []

    def step(self, tick: int) -> None:
        self.stepped.append(tick)
        # The real world applies queued pilot messages inside step; mirror that.
        self.red_pilot.act(self.fighters["red"].timeline, tick)

    def reset_round(self, index: int) -> None:
        self.resets.append(index)

    def separation_m(self) -> float:
        return 1.0

    def teleport_sacco(self, x: float, y: float, heading: float) -> None:
        self.teleports.append((x, y, heading))


def _host(**kwargs) -> SparringHost:
    return SparringHost(FakeWorld(), render=False, **kwargs)


def _started_commit(timeline: IntentTimeline, ghost: tuple[float, float] = (1.0, 0.0)):
    """A commit staged, committed, and started right now, with a real runner — the fixture every
    leg-progress / keyframe-event test needs, built the same way `runtime/intents.py`'s own
    `generator_intent` would (`warp` then `CombinationRunner`), just without a live anchor callback.
    """
    timeline.stage(combination="jab-cross", ghost=ghost)
    commit = timeline.commit(0)
    commit.commit_at = 0
    legs = warp.warp(commit.record, (0.0, 0.0), 0.0, commit.ghost, speed_ceiling=None)
    commit.runner = sequence.CombinationRunner(commit.record, legs, commit_at=0)
    commit.ended_at = commit.runner.end_tick
    return commit


# -- knobs ------------------------------------------------------------------------------------------
class TestKnobs:
    def test_every_knob_reports_current_and_canonical(self) -> None:
        values = knob_values(FakeWorld())
        assert set(values) == {"replan_dt", "horizon_ticks", "max_outstanding", "drift_gain"}
        for entry in values.values():
            assert set(entry) == {"current", "canonical"}
        assert values["replan_dt"]["current"] == pytest.approx(0.5)
        assert values["horizon_ticks"]["canonical"] == COMMIT_HORIZON_TICKS
        # The bench's deeper queue is a *deliberate* deviation, and always reads as one.
        assert values["max_outstanding"]["current"] == SPARRING_MAX_OUTSTANDING
        assert values["drift_gain"]["current"] == pytest.approx(DRIFT_GAIN)
        assert values["drift_gain"]["canonical"] == pytest.approx(DRIFT_GAIN)

    def test_setting_a_knob_mutates_the_world(self) -> None:
        world = FakeWorld()
        from openroboxing.runtime import warp as warp_module

        try:
            result = set_knobs(world, {"replan_dt": 0.3, "drift_gain": 0.9})
            assert world.fighters["red"].stream.replan_dt == pytest.approx(0.3)
            assert warp_module.DRIFT_GAIN == pytest.approx(0.9)
            assert result["replan_dt"]["current"] == pytest.approx(0.3)
            assert result["drift_gain"]["current"] == pytest.approx(0.9)
        finally:
            warp_module.DRIFT_GAIN = DRIFT_GAIN  # a process-global; must not leak into other tests

    def test_an_unknown_knob_is_refused(self) -> None:
        with pytest.raises(SparringError, match="unknown knob"):
            set_knobs(FakeWorld(), {"warp_speed": 9})

    def test_a_non_positive_value_is_refused(self) -> None:
        with pytest.raises(SparringError, match="positive"):
            set_knobs(FakeWorld(), {"replan_dt": 0.0})
        with pytest.raises(SparringError, match="finite"):
            set_knobs(FakeWorld(), {"replan_dt": float("nan")})

    def test_the_approach_and_dwell_knobs_are_gone(self) -> None:
        """`spec/intent.md` "Removed at 3.0": the approach and the counted dwell are gone, and so
        are the knobs that tuned them."""
        for name in (
            "arrival_radius_m",
            "approach_leg_m",
            "approach_timeout_ticks",
            "pose_dwell_ticks",
        ):
            assert name not in KNOBS


# -- the queue entry: leg progress and drift, in isolation ------------------------------------------
class TestQueueEntry:
    def test_an_unstarted_commit_has_no_leg_progress_or_drift(self) -> None:
        timeline = IntentTimeline(_library(), require_admitted=False)
        timeline.stage(combination="jab-cross", ghost=(1.0, 0.0))
        commit = timeline.commit(0)
        entry = bench_queue_entry(commit, 0, SimpleNamespace(_drift_speed_m_s={}))
        assert entry["leg_index"] is None
        assert entry["leg_count"] is None
        assert entry["drift_speed_m_s"] is None
        assert entry["combination"] == "jab-cross"
        assert entry["ghost"] == {"x": 1.0, "y": 0.0}

    def test_a_started_commit_reports_its_live_leg_and_drift(self) -> None:
        timeline = IntentTimeline(_library(), require_admitted=False)
        commit = _started_commit(timeline)
        world = SimpleNamespace(_drift_speed_m_s={id(commit): 0.512345})
        entry = bench_queue_entry(commit, 0, world)
        assert entry["leg_index"] == 0
        assert entry["leg_count"] == 2
        assert entry["drift_speed_m_s"] == pytest.approx(0.512)


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

    def test_reset_forgets_leg_progress_bookkeeping(self) -> None:
        """`_last_leg_index` is keyed by `id(commit)`; a stale entry surviving a reset could collide
        with a brand-new commit object reusing a freed address."""
        host = _host()
        _started_commit(host.world.fighters["red"].timeline)
        host.step_once()
        assert host._last_leg_index
        host.reset()
        assert host._last_leg_index == {}

    def test_the_debug_message_carries_the_panel(self) -> None:
        host = _host()
        timeline = host.world.fighters["red"].timeline
        commit = _started_commit(timeline)
        host.step_once()

        message = host.debug_message()
        assert message["type"] == "debug"
        assert message["machine"] in MACHINE_STATES
        assert len(MACHINE_STATES) == 4, "no APPROACH/DWELL split survives at 3.0"
        assert len(message["queue"]) == 1
        entry = message["queue"][0]
        assert entry["combination"] == "jab-cross" and entry["commit_at"] == 0
        assert entry["leg_index"] == 0 and entry["leg_count"] == 2
        ghost = message["plan_ghost"]
        assert set(ghost) == {"x", "y", "z", "heading", "angles"}
        assert len(ghost["angles"]) == 29
        assert len(message["trail"]) >= 2
        assert message["knobs"]["replan_dt"]["canonical"] == pytest.approx(0.5)
        assert message["recording"] == {"start_tick": 0, "end_tick": 0}
        assert message["drafts_allowed"] is True  # FakeWorld defaults require_admitted=False
        assert "keyframe_events" in message
        assert commit.record.name == "jab-cross"

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
        """Two distances to one ghost: what the body reached, and what the plan reached.

        `spec/sparring_protocol.md` §"Two distances, not one": under 3.0 this pair is a trend across
        a whole commit rather than an arrival test, but both numbers must still be present the
        instant a commit is executing.
        """
        host = _host()
        timeline = host.world.fighters["red"].timeline
        _started_commit(timeline)
        for _ in range(5):
            host.step_once()

        head = host.debug_message()["series_head"]
        assert head["dist"] is not None and head["dist_plan"] is not None
        assert "dist_plan" in host.tap.series(0, 4)

    def test_a_started_commits_drift_speed_reaches_the_queue(self) -> None:
        """`drift_speed_m_s` is 3.0's replacement for 0.1-0.2's `arrived`: how hard an off-target
        commit had to run to still land on its ghost, read straight off `FightWorld`'s own private
        drift table (`runtime/fight.py::_record_drift`)."""
        host = _host()
        timeline = host.world.fighters["red"].timeline
        commit = _started_commit(timeline, ghost=(9.0, 9.0))
        host.world._drift_speed_m_s[id(commit)] = 4.5
        host.step_once()

        entry = host.debug_message()["queue"][0]
        assert entry["drift_speed_m_s"] == pytest.approx(4.5)

    def test_an_unstarted_commits_drift_speed_and_legs_are_null(self) -> None:
        host = _host()
        timeline = host.world.fighters["red"].timeline
        timeline.stage(combination="jab-cross", ghost=(1.0, 0.0))
        timeline.commit(0)
        host.step_once()

        entry = host.debug_message()["queue"][0]
        assert entry["drift_speed_m_s"] is None
        assert entry["leg_index"] is None and entry["leg_count"] is None

    def test_a_leg_boundary_is_logged_as_a_keyframe_event(self) -> None:
        """3.0's replacement for the settle test: a leg's end is exact arithmetic, and what is
        still worth recording about it is the tracking error at the tick it ended."""
        host = _host()
        timeline = host.world.fighters["red"].timeline
        commit = _started_commit(timeline)
        for _ in range(commit.ended_at + 1):
            host.step_once()

        assert host.tap.keyframe_events, "no leg boundary was ever crossed"
        _tick, ordinal, leg, err_mean, err_max = host.tap.keyframe_events[0]
        assert ordinal == 0
        assert leg == 0  # leg 0 finished; leg 1 is now live
        assert err_mean >= 0.0 and err_max >= err_mean

    def test_scrub_carries_the_queue(self) -> None:
        """The panel must not empty out when you scrub — a blank queue reads as "nothing happened"."""
        host = _host()
        timeline = host.world.fighters["red"].timeline
        _started_commit(timeline)
        for _ in range(30):
            host.step_once()

        payload = host.scrub_payload(10)
        assert len(payload["queue"]) == 1
        assert payload["queue"][0]["combination"] == "jab-cross"

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

    def test_handle_forwards_a_good_teleport(self) -> None:
        host = _host()
        assert host.handle({"type": "teleport_sacco", "x": 1.0, "y": 0.5, "heading": 0.2}) is None
        assert host.world.teleports == [(1.0, 0.5, 0.2)]

    def test_handle_stages_and_commits_through_the_pilot(self) -> None:
        host = _host()
        assert host.handle({"type": "intent", "combination": "jab-cross", "ghost": [1.0, 0.0]}) is None
        assert host.handle({"type": "commit"}) is None
        host.step_once()  # the fake world applies pilot messages on step, as the real one does
        assert len(host.world.fighters["red"].timeline.commits) == 1

    def test_handle_rejects_an_unknown_combination_immediately(self) -> None:
        reply = _host().handle(
            {"type": "intent", "combination": "does-not-exist", "ghost": [0.1, 0.0]}
        )
        assert reply["type"] == "error" and reply["rejected"] == "intent"

    def test_handle_rejects_a_ghost_beyond_its_combinations_reach(self) -> None:
        """The one place `spec/intent.md`'s speed ceiling is enforced, mirrored from
        `server/host.py::MatchHost.handle` exactly — a bench that refused something different from
        what a match refuses would be testing the wrong thing."""
        host = _host()
        host.handle({"type": "intent", "combination": "jab-cross", "ghost": [50.0, 0.0]})
        host.step_once()  # the pilot applies queued messages a tick later; now it is really staged
        reply = host.handle({"type": "commit"})
        assert reply is not None and reply["rejected"] == "commit"
        assert host.world.fighters["red"].timeline.commits == ()

    def test_handle_rejects_the_eleventh_commit(self) -> None:
        host = _host()
        timeline = host.world.fighters["red"].timeline
        timeline.stage(combination="jab-cross", ghost=(0.1, 0.0))
        for _ in range(SPARRING_MAX_OUTSTANDING):
            timeline.commit(0)
        reply = host.handle({"type": "commit"})
        assert reply["type"] == "error" and "queued" in reply["message"]

    def test_handle_rejects_garbage(self) -> None:
        reply = _host().handle({"type": "warp"})
        assert reply["type"] == "error"

    def test_drafts_allowed_reflects_require_admitted(self) -> None:
        """`spec/intent.md`: 'The Studio passes require_admitted=False... a match never does.' A
        bench session must say which one it is rather than let it be silent."""
        strict = SparringHost(FakeWorld(require_admitted=True), render=False)
        assert strict.welcome_message()["drafts_allowed"] is False
        strict.step_once()
        assert strict.debug_message()["drafts_allowed"] is False

        lenient = SparringHost(FakeWorld(require_admitted=False), render=False)
        assert lenient.welcome_message()["drafts_allowed"] is True


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

                await second.send_json(
                    {"type": "intent", "combination": "jab-cross", "ghost": [1.0, 0.5]}
                )
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
