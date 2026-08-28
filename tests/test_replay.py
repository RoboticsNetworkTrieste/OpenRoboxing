"""M3-T5 acceptance: replays.

Acceptance criterion from WORKPLAN.md M3-T5:
  a recorded match replays visually identically from the trace; the intent log alone is under a few
  hundred kilobytes for a full match.

"Visually identically" is earned, not asserted by construction. The trace is stored as **float32**
while the simulation runs in float64, so a replay is *not* the same numbers — the question is whether
that survives to the pixels. :func:`test_a_replayed_frame_is_pixel_identical` renders the same tick
both ways and compares.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_replay.py -v
    .venv_mb/bin/python -m pytest tests/test_replay.py -v -m slow   # needs a GPU
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from openroboxing.runtime.arena import ArenaConfig, build_arena, reset_to_stance
from openroboxing.runtime.contact import ContactTracker, FightTrace
from openroboxing.runtime.match import Match, MatchFormat, MatchRecord
from openroboxing.runtime.replay import (
    DEFAULT_CAMERA,
    RecordedMatch,
    ReplayError,
    ReplayRenderer,
    ReplayWorld,
    replay_frames,
)
from openroboxing.studio.combination_record import CombinationRecord, CombinationSource, Keyframe

pytest.importorskip("mujoco")

#: Rules tests replay a handful of ticks; the numbers are not the point.
SHORT = MatchFormat(rounds=2, round_ticks=30, get_up_window_ticks=10)


@pytest.fixture(scope="module")
def arena():
    return build_arena(ArenaConfig())


def _combination(name: str = "combo-a") -> CombinationRecord:
    """A small, admitted combination — cheap enough to build a `FighterRuntime` from.

    `spec/intent.md` 3.0: a commit carries a combination and a ghost, not a loadout slot. Each
    test file in this repo keeps its own copy of this builder rather than sharing a fixtures module
    (see e.g. `test_server.py`, `test_fight.py`); this replay module only needs one entry, since it
    never actually generates through it — it just needs a valid library to build a `FighterRuntime`
    and read `root_qpos` off it.
    """
    from openroboxing.runtime.conventions import G1

    angles = {joint: 0.0 for joint in G1.mujoco_joint_names}
    keyframes = [
        Keyframe(dict(angles), None, (0.0, 0.0), 0.0),
        Keyframe(dict(angles), 6, (0.15, 0.0), 0.0),
        Keyframe(dict(angles), 6, (0.3, 0.0), 0.0),
    ]
    return CombinationRecord(
        name=name,
        library_version="v0.2",
        source=CombinationSource("t", 0, 100, False),
        keyframes=keyframes,
        telegraph_ms=180.0,
        tracking_error_rad=0.1,
        admission="admitted",
    )


@pytest.fixture(scope="module")
def library() -> dict[str, CombinationRecord]:
    return {"combo-a": _combination("combo-a")}


def _standing_qpos(arena) -> np.ndarray:
    import mujoco

    data = mujoco.MjData(arena)
    reset_to_stance(arena, data, ArenaConfig())
    return data.qpos.copy()


def _synthetic_trace(
    arena, ticks: int, library=None, drop: str | None = None, from_tick: int = 0
) -> np.ndarray:
    """A trace of a fighter standing, optionally with one of them sinking to the canvas."""
    from openroboxing.runtime.fight import FighterRuntime, IdlePilot

    base = _standing_qpos(arena)
    trace = np.tile(base, (ticks, 1)).astype(np.float32)
    if drop is not None:
        if library is None:
            library = {"combo-a": _combination("combo-a")}

        class _Stub:
            pass

        runtime = FighterRuntime(drop, arena, _Stub(), library, IdlePilot())
        trace[from_tick:, runtime.root_qpos[2]] = 0.06  # torso on the canvas
    return trace


#: One commit, in the shape `runtime/fight.py::FightWorld.commits()` actually emits
#: (`spec/match_record.md` 0.3's ``CommitEvent``): a combination and a ghost, not a loadout slot
#: and a placement, plus the achieved drift speed the commit's warp needed to reach that ghost.
DEFAULT_COMMITS = [
    {
        "fighter": "red",
        "combination": "combo-a",
        "ghost": [0.3, 0.0],
        "issued_at": 2,
        "commit_at": 32,
        "end_tick": 44,
        "drift_speed_m_s": 0.05,
    }
]


def _write(tmp_path, traces: dict[int, np.ndarray], commits=None, **overrides) -> object:
    """Write a record + trace pair to disk the way :meth:`MatchRecord.save` does."""
    record = MatchRecord(
        match_id="replay-test",
        format=SHORT,
        fighters={"red": {"handle": "red"}, "blue": {"handle": "blue"}},
        versions={"rules": "0.2"},
        seeds={"match_seed": 1234},
        arena=overrides.get("arena", {}),
    )
    from openroboxing.runtime.match import RoundRecord

    for index, trace in sorted(traces.items()):
        record.rounds.append(
            RoundRecord(
                index=index,
                ticks=trace.shape[0],
                ended_by="bell",
                knocked_out=None,
                commits=list(DEFAULT_COMMITS if commits is None else commits),
                trace=trace,
            )
        )
    path = tmp_path / "replay-test.json"
    record.save(path)
    return path


# --- loading ------------------------------------------------------------------------------------------
def test_a_record_loads_with_its_trace(tmp_path, arena) -> None:
    path = _write(tmp_path, {0: _synthetic_trace(arena, 30), 1: _synthetic_trace(arena, 30)})
    recorded = RecordedMatch.load(path)

    assert recorded.match_id == "replay-test"
    assert recorded.round_count == 2
    assert recorded.trace(0).shape == (30, arena.nq)
    assert recorded.format().round_ticks == SHORT.round_ticks
    assert recorded.commits(0)[0]["fighter"] == "red"


def test_a_record_without_its_trace_refuses_to_load(tmp_path, arena) -> None:
    """The JSON alone is a summary of a replay, not one. Loading it anyway yields an empty fight."""
    path = _write(tmp_path, {0: _synthetic_trace(arena, 10)})
    path.with_suffix(".trace.npz").unlink()

    with pytest.raises(ReplayError, match="no trace at"):
        RecordedMatch.load(path)


def test_an_unreadable_record_raises(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ReplayError, match="cannot read the match record"):
        RecordedMatch.load(bad)


def test_a_record_predating_the_arena_field_reads_as_the_defaults(tmp_path, arena) -> None:
    """`spec/match_record.md` 0.2: a 0.1 record was fought in the defaults, because that is all
    there was."""
    path = _write(tmp_path, {0: _synthetic_trace(arena, 10)})
    data = json.loads(path.read_text())
    del data["arena"]
    path.write_text(json.dumps(data))

    assert RecordedMatch.load(path).arena_config() == ArenaConfig()


def test_the_recorded_arena_comes_back(tmp_path, arena) -> None:
    config = ArenaConfig(ring_size=3.6, glove_radius=0.07)
    from dataclasses import asdict

    path = _write(tmp_path, {0: _synthetic_trace(arena, 10)}, arena=asdict(config))

    assert RecordedMatch.load(path).arena_config() == config


def test_a_trace_from_a_different_ring_is_refused(tmp_path, arena) -> None:
    path = _write(tmp_path, {0: _synthetic_trace(arena, 10)[:, :-4]})
    with pytest.raises(ReplayError, match="different ring"):
        ReplayWorld(RecordedMatch.load(path))


# --- playing it back ------------------------------------------------------------------------------------
def test_the_world_puts_the_fighters_where_the_trace_says(tmp_path, arena) -> None:
    trace = _synthetic_trace(arena, 20)
    trace[7, 0] = -1.75  # move red on one tick only
    world = ReplayWorld(RecordedMatch.load(_write(tmp_path, {0: trace})))

    world.reset_round(0)
    world.step(7)
    assert world.qpos()[0] == pytest.approx(-1.75)
    world.step(8)
    assert world.qpos()[0] != pytest.approx(-1.75)


def test_stepping_past_the_end_of_a_round_raises(tmp_path, arena) -> None:
    world = ReplayWorld(RecordedMatch.load(_write(tmp_path, {0: _synthetic_trace(arena, 12)})))
    world.reset_round(0)
    with pytest.raises(ReplayError, match="asked for tick 12"):
        world.step(12)


def test_stepping_before_a_round_is_loaded_raises(tmp_path, arena) -> None:
    world = ReplayWorld(RecordedMatch.load(_write(tmp_path, {0: _synthetic_trace(arena, 12)})))
    with pytest.raises(ReplayError, match="no round is loaded"):
        world.step(0)


def test_velocity_is_reconstructed_not_left_at_zero(tmp_path, arena) -> None:
    """Contact forces need velocity, and the trace has none. ``mj_differentiatePos`` supplies it."""
    trace = _synthetic_trace(arena, 10)
    trace[:, 0] = np.linspace(-1.2, -0.9, 10)  # red walks forward
    world = ReplayWorld(RecordedMatch.load(_write(tmp_path, {0: trace})))

    world.reset_round(0)
    world.step(3)
    forward = world.data.qvel[0]
    expected = (trace[4, 0] - trace[3, 0]) * 50.0  # TICK_HZ
    assert forward == pytest.approx(expected, rel=1e-4)


def test_the_last_tick_has_no_velocity_to_reconstruct(tmp_path, arena) -> None:
    trace = _synthetic_trace(arena, 6)
    trace[:, 0] = np.linspace(-1.2, -0.9, 6)
    world = ReplayWorld(RecordedMatch.load(_write(tmp_path, {0: trace})))
    world.reset_round(0)
    world.step(5)
    assert np.allclose(world.data.qvel, 0.0), "there is no next frame to difference against"


# --- re-deriving the rules ---------------------------------------------------------------------------------
def test_a_knockdown_re_derives_from_the_trace_alone(tmp_path, arena) -> None:
    """The claim `match.py` makes and `spec/match_record.md` records: a disputed knockdown is
    settleable by anyone, with no GPU, no generator and no policy."""
    down = _synthetic_trace(arena, 30, drop="red", from_tick=5)
    world = ReplayWorld(RecordedMatch.load(_write(tmp_path, {0: down, 1: down})))

    record = Match(world, match_format=SHORT).run()

    assert record.rounds[0].ended_by == "knockout"
    assert record.rounds[0].knocked_out == "red"
    assert record.rounds[0].ticks == 5 + SHORT.get_up_window_ticks
    assert record.knockouts() == [(0, "red"), (1, "red")]


def test_a_clean_round_re_derives_as_a_clean_round(tmp_path, arena) -> None:
    trace = _synthetic_trace(arena, SHORT.round_ticks)
    world = ReplayWorld(RecordedMatch.load(_write(tmp_path, {0: trace, 1: trace})))

    record = Match(world, match_format=SHORT).run()
    assert [r.ended_by for r in record.rounds] == ["bell", "bell"]
    assert all(not r.knockdowns for r in record.rounds)


def test_a_replay_reports_the_commits_that_were_recorded(tmp_path, arena) -> None:
    """A replay does not re-derive what the player did — it reads it back."""
    world = ReplayWorld(RecordedMatch.load(_write(tmp_path, {0: _synthetic_trace(arena, 30)})))
    world.reset_round(0)
    assert world.commits() == DEFAULT_COMMITS


def test_a_replay_carries_the_achieved_drift_speed(tmp_path, arena) -> None:
    """`spec/intent.md` "Off-target execution", owner decision 2026-08-28: a commit that started
    off-target still reaches its ghost, running whatever drift that needs, and that drift speed must
    be visible in the record rather than silent. It is `runtime/fight.py::FightWorld.commits()`'s
    ``drift_speed_m_s``, and a replay reads it back unchanged, like every other commit field."""
    hard_drift = [{**DEFAULT_COMMITS[0], "drift_speed_m_s": 4.75}]
    world = ReplayWorld(
        RecordedMatch.load(_write(tmp_path, {0: _synthetic_trace(arena, 10)}, commits=hard_drift))
    )
    world.reset_round(0)
    assert world.commits()[0]["drift_speed_m_s"] == pytest.approx(4.75)


def test_a_commit_not_yet_started_carries_no_drift_speed(tmp_path, arena) -> None:
    """`FightWorld.commits()`'s own contract: ``drift_speed_m_s`` is ``None`` until a commit starts
    — there is nothing to measure yet, the same reason ``commit_at``/``end_tick`` are ``None`` then
    (`spec/intent.md` "A commit's span"). A replay must not invent a number here."""
    queued = [
        {
            "fighter": "red",
            "combination": "combo-a",
            "ghost": [0.3, 0.0],
            "issued_at": 2,
            "commit_at": None,
            "end_tick": None,
            "drift_speed_m_s": None,
        }
    ]
    world = ReplayWorld(
        RecordedMatch.load(_write(tmp_path, {0: _synthetic_trace(arena, 10)}, commits=queued))
    )
    world.reset_round(0)
    assert world.commits()[0]["drift_speed_m_s"] is None
    assert world.commits()[0]["commit_at"] is None


def test_the_observed_trace_matches_the_recorded_one(tmp_path, arena) -> None:
    trace = _synthetic_trace(arena, 20)
    world = ReplayWorld(RecordedMatch.load(_write(tmp_path, {0: trace})))

    world.reset_round(0)
    observed = FightTrace()
    for tick in range(20):
        world.step(tick)
        world.observe(ContactTracker(), observed, tick)

    assert len(observed.tick) == 20
    assert observed.separation_m[0] == pytest.approx(2 * ArenaConfig().start_separation, abs=0.05)


# --- the picture ---------------------------------------------------------------------------------------
#: What "visually identically" is worth, measured 2026-08-08 at 480x320 on the broadcast camera.
#:
#: Rendering itself is bit-exact — the same ``qpos`` through two separately built arenas gives byte-
#: identical frames. The only difference comes from the trace being **float32** while the simulation
#: runs in float64: that quantises ``qpos`` by up to 4.8e-08, which flipped exactly **one pixel of
#: 153,600** by **1/255 in one channel**. (It flipped none at all until the ring got lights; shadow
#: mapping is what amplifies a 5e-08 joint angle to a least-significant bit.)
#:
#: Recorded rather than asserted away. `spec/match_record.md` chose float32 deliberately, for a
#: 2.4 MB trace instead of 4.8 MB, and this is the price.
MAX_CHANNEL_DIFFERENCE = 1
MAX_DIFFERING_PIXEL_FRACTION = 1e-4


def _staged_frame(arena):
    """A frame with the arms out, so there is something for a replay to get wrong."""
    import mujoco

    live = mujoco.MjData(arena)
    reset_to_stance(arena, live, ArenaConfig())
    live.qpos[arena.joint("red_left_shoulder_pitch_joint").qposadr[0]] = -1.2
    live.qpos[arena.joint("red_left_elbow_joint").qposadr[0]] = -0.6
    live.qpos[arena.joint("blue_right_shoulder_roll_joint").qposadr[0]] = -0.9
    mujoco.mj_forward(arena, live)
    return live


@pytest.mark.slow
def test_rendering_the_same_state_twice_is_bit_exact(arena) -> None:
    """The control for the test below. Without this, a pixel difference could be render jitter and
    the measurement would mean nothing."""
    live = _staged_frame(arena)
    with ReplayRenderer(arena, 480, 320, DEFAULT_CAMERA) as renderer:
        first, second = renderer.frame(live).copy(), renderer.frame(live).copy()

    other = build_arena(ArenaConfig())
    with ReplayRenderer(other, 480, 320, DEFAULT_CAMERA) as renderer:
        rebuilt = renderer.frame(live).copy()

    assert np.array_equal(first, second)
    assert np.array_equal(first, rebuilt), "two builds of the same ring render differently"


@pytest.mark.slow
def test_a_replayed_frame_matches_the_live_render(tmp_path, arena) -> None:
    """The acceptance criterion, measured rather than assumed.

    The trace is float32 and the simulation float64, so a replay renders *different numbers*. This
    pins how far that reaches: see :data:`MAX_CHANNEL_DIFFERENCE`. Loosening these bounds is a
    decision about the format, not a test fix.
    """
    live = _staged_frame(arena)
    with ReplayRenderer(arena, 480, 320, DEFAULT_CAMERA) as renderer:
        direct = renderer.frame(live).astype(int)

    path = _write(tmp_path, {0: np.tile(live.qpos.copy(), (3, 1)).astype(np.float32)})
    frames = list(replay_frames(ReplayWorld(RecordedMatch.load(path)), 0, width=480, height=320))

    assert len(frames) == 3
    assert frames[0].shape == direct.shape == (320, 480, 3)

    difference = np.abs(frames[0].astype(int) - direct)
    differing = int((difference.sum(axis=2) > 0).sum())
    fraction = differing / (direct.shape[0] * direct.shape[1])

    assert difference.max() <= MAX_CHANNEL_DIFFERENCE, (
        f"the replay differs from the live render by {difference.max()}/255 per channel"
    )
    allowed = 100 * MAX_DIFFERING_PIXEL_FRACTION
    assert fraction <= MAX_DIFFERING_PIXEL_FRACTION, (
        f"{differing} pixels differ ({100 * fraction:.4f}%), above the recorded {allowed:.4f}%"
    )


@pytest.mark.slow
def test_every_tick_of_a_round_renders(tmp_path, arena) -> None:
    trace = _synthetic_trace(arena, 12)
    trace[:, 0] = np.linspace(-1.2, -0.8, 12)
    world = ReplayWorld(RecordedMatch.load(_write(tmp_path, {0: trace})))

    frames = list(replay_frames(world, 0, width=240, height=180))
    assert len(frames) == 12
    assert not np.array_equal(frames[0], frames[-1]), "red moved; the frames should differ"


@pytest.mark.slow
def test_a_stride_renders_fewer_frames(tmp_path, arena) -> None:
    world = ReplayWorld(RecordedMatch.load(_write(tmp_path, {0: _synthetic_trace(arena, 12)})))
    assert len(list(replay_frames(world, 0, stride=3, width=240, height=180))) == 4


@pytest.mark.slow
def test_an_unknown_camera_raises(arena) -> None:
    with pytest.raises(ReplayError, match="no camera named"):
        ReplayRenderer(arena, 240, 180, "ringside")


def test_a_zero_stride_raises(tmp_path, arena) -> None:
    world = ReplayWorld(RecordedMatch.load(_write(tmp_path, {0: _synthetic_trace(arena, 6)})))
    with pytest.raises(ReplayError, match="stride must be at least 1"):
        list(replay_frames(world, 0, stride=0))
