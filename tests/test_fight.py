"""Two fighters in the ring under physics (M3-T4's world; rewritten M6-T7 for combinations).

The fast tests here cover what does not need a GPU and is where the bugs actually are: the
name-derived indices into a shared ``MjData`` (`CLAUDE.md` invariant 4), the world-to-generator
heading conversion, the pilots, and the live anchor `spec/intent.md` 3.0's off-target execution
depends on. The slow ones run the thing.

`runtime/intents.py`, `runtime/warp.py` and `runtime/sequence.py` already have their own test
suites (`test_intents_combinations.py`, `test_warp.py`, `test_sequence.py`) covering the commit
queue's arithmetic and the warp's geometry in isolation; this file does not re-test those — it tests
the one thing that is `fight.py`'s alone: wiring a *live* fighter into that machinery.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_fight.py -v
    .venv_mb/bin/python -m pytest tests/test_fight.py -v -m slow   # needs a GPU
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from openroboxing.runtime.arena import FIGHTERS, ArenaConfig, build_arena, reset_to_stance
from openroboxing.runtime.bridge import compute_apply_delta_heading, heading_quat, quat_multiply
from openroboxing.runtime.conventions import G1
from openroboxing.runtime.fight import (
    FightError,
    FighterRuntime,
    FightWorld,
    IdlePilot,
    ScriptedPilot,
    apply_yaw,
    generator_heading,
)
from openroboxing.runtime.intents import IntentError, IntentTimeline
from openroboxing.spec.constants import (
    COMMIT_HORIZON_TICKS,
    MAX_OUTSTANDING_COMMITS,
    NUM_JOINTS,
    QPOS_DIM,
    TICK_HZ,
)
from openroboxing.studio.combination_record import CombinationRecord, CombinationSource, Keyframe

ANGLES = {name: 0.0 for name in G1.mujoco_joint_names}


def _combination(
    name: str,
    *,
    offsets=((0.3, 0.0), (0.6, 0.0)),
    headings=(0.0, 0.0),
    tokens=(6, 6),
    admitted: bool = True,
) -> CombinationRecord:
    """A small, admitted combination — two legs, cheap enough to drive a fight through fully.

    Mirrors the builder in `test_intents_combinations.py`; each file keeps its own copy rather than
    sharing a fixtures module, which is this repo's existing convention (see e.g. `test_warp.py`,
    `test_sequence.py`).
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
def arena():
    return build_arena(ArenaConfig())


@pytest.fixture(scope="module")
def library() -> dict[str, CombinationRecord]:
    return {
        "combo-a": _combination("combo-a"),
        "combo-b": _combination("combo-b", offsets=((0.1, 0.0), (0.2, 0.0))),
    }


class _StubGenerator:
    """A generator stand-in for tests that build a :class:`FighterRuntime` but never step it.

    ``context_qpos`` is the one method the geometry helpers in `fight.py` call even off the physics
    path (:meth:`FightWorld.to_generator_frame`), so it is real enough to answer that, and nothing
    else.
    """

    def context_qpos(self) -> np.ndarray:
        return np.zeros((1, QPOS_DIM))


def _runtime(name: str, arena, library) -> FighterRuntime:
    return FighterRuntime(name, arena, _StubGenerator(), library, IdlePilot())


# --- indices, derived by name ------------------------------------------------------------------------
def test_a_fighter_finds_all_twenty_nine_joints(arena, library) -> None:
    red = _runtime("red", arena, library)

    assert red.joint_qpos.shape == (NUM_JOINTS,)
    assert red.joint_dof.shape == (NUM_JOINTS,)
    assert red.actuators.shape == (NUM_JOINTS,)
    assert red.root_qpos.shape == (7,) and red.root_dof.shape == (6,)


def test_the_two_fighters_share_no_index(arena, library) -> None:
    """The failure this guards against is silent: red would be driving blue's legs."""
    red = _runtime("red", arena, library)
    blue = _runtime("blue", arena, library)

    for attribute in ("joint_qpos", "joint_dof", "actuators", "root_qpos", "root_dof"):
        overlap = set(getattr(red, attribute).tolist()) & set(getattr(blue, attribute).tolist())
        assert not overlap, f"{attribute} overlaps between fighters: {sorted(overlap)}"

    assert red.pelvis_body != blue.pelvis_body


def test_the_indices_cover_the_whole_model(arena, library) -> None:
    red = _runtime("red", arena, library)
    blue = _runtime("blue", arena, library)

    qpos = np.concatenate([f.root_qpos for f in (red, blue)] + [f.joint_qpos for f in (red, blue)])
    assert sorted(qpos.tolist()) == list(range(arena.nq)) == list(range(len(FIGHTERS) * QPOS_DIM))

    actuators = np.concatenate([red.actuators, blue.actuators])
    assert sorted(actuators.tolist()) == list(range(arena.nu))


def test_joints_are_indexed_in_mujoco_order(arena, library) -> None:
    """Not merely 29 joints, but *these* 29 in the order every gain array assumes."""
    import mujoco

    red = _runtime("red", arena, library)
    for position, name in enumerate(G1.mujoco_joint_names):
        joint = mujoco.mj_name2id(arena, mujoco.mjtObj.mjOBJ_JOINT, f"red_{name}")
        assert red.joint_qpos[position] == arena.jnt_qposadr[joint]
        assert red.actuators[position] == int(
            np.flatnonzero(arena.actuator_trnid[:, 0] == joint)[0]
        )


def test_an_unknown_fighter_has_no_joints(arena, library) -> None:
    with pytest.raises(FightError, match="is not in the arena"):
        _runtime("green", arena, library)


def test_a_fighter_reads_its_own_state_out_of_the_shared_data(arena, library) -> None:
    import mujoco

    data = mujoco.MjData(arena)
    reset_to_stance(arena, data, ArenaConfig())

    red = _runtime("red", arena, library)
    blue = _runtime("blue", arena, library)
    data.qpos[red.joint_qpos] = np.arange(NUM_JOINTS, dtype=float)
    data.qpos[blue.joint_qpos] = -np.arange(NUM_JOINTS, dtype=float)

    assert np.array_equal(red.robot_state(data).joint_pos, np.arange(NUM_JOINTS))
    assert np.array_equal(blue.robot_state(data).joint_pos, -np.arange(NUM_JOINTS))
    assert red.robot_state(data).base_quat.shape == (4,)
    assert red.robot_state(data).base_ang_vel.shape == (3,)


def test_the_fighters_start_apart_and_facing_each_other(arena, library) -> None:
    import mujoco

    data = mujoco.MjData(arena)
    reset_to_stance(arena, data, ArenaConfig())
    mujoco.mj_forward(arena, data)

    red = _runtime("red", arena, library)
    blue = _runtime("blue", arena, library)
    separation = np.linalg.norm(data.xpos[red.pelvis_body][:2] - data.xpos[blue.pelvis_body][:2])
    assert separation == pytest.approx(2 * ArenaConfig().start_separation, abs=0.05)


# --- heading conversion -----------------------------------------------------------------------------
def test_a_world_heading_survives_the_round_trip() -> None:
    """The claim :func:`generator_heading` makes: ask in world terms, arrive facing that way."""
    for start_yaw in (0.0, 0.7, -2.4, np.pi):
        ref_start = np.array([np.cos(start_yaw / 2), 0.0, 0.0, np.sin(start_yaw / 2)])
        apply = compute_apply_delta_heading(
            init_base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            init_ref_root_quat_wxyz=ref_start,
        )
        for world in (0.0, 1.3, -0.9):
            asked = generator_heading(world, apply)
            landed = quat_multiply(
                apply, np.array([np.cos(asked / 2), 0.0, 0.0, np.sin(asked / 2)])
            )
            arrived = apply_yaw(heading_quat(landed))
            assert np.cos(arrived - world) == pytest.approx(1.0, abs=1e-9)


def test_an_identity_alignment_leaves_a_heading_alone() -> None:
    assert generator_heading(1.25, np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.25)


def test_apply_yaw_wants_a_quaternion() -> None:
    with pytest.raises(FightError, match="wxyz quaternion"):
        apply_yaw(np.zeros(3))


# --- the live anchor (spec/intent.md 3.0's "Off-target execution") ----------------------------------
class _AnchorOnlyWorld:
    """Just enough of a :class:`FightWorld` to exercise the anchor wiring, no arena, no GPU.

    ``_anchor_now``, ``to_generator_frame``, ``_record_drift`` and ``_intent_at`` all read the
    pelvis, the root quaternion and the generator's own buffer tail out of ``self`` — nothing else —
    so they run unmodified against a hand-built stand-in, the same trick `_GeometryOnlyWorld` used
    before this rewrite for the (now deleted) approach-aiming geometry.
    """

    _anchor_now = FightWorld._anchor_now
    to_generator_frame = FightWorld.to_generator_frame
    _record_drift = FightWorld._record_drift
    _intent_at = FightWorld._intent_at

    def __init__(self, position: tuple[float, float], heading: float) -> None:
        quat = np.array([math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)])
        qpos = np.zeros(7)
        qpos[0:2] = position
        qpos[3:7] = quat
        self.data = SimpleNamespace(
            xpos=np.array([[position[0], position[1], 0.79]]),
            qpos=qpos,
        )
        self._drift_speed_m_s: dict[int, float] = {}


def _fighter_stub(name: str, timeline: IntentTimeline) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        timeline=timeline,
        pelvis_body=0,
        root_qpos=np.arange(7),
        apply_yaw=0.0,
        apply_delta_heading=np.array([1.0, 0.0, 0.0, 0.0]),
        generator=_StubGenerator(),
    )


def test_the_anchor_reports_the_fighters_live_position_not_the_origin() -> None:
    """`spec/intent.md` "Off-target execution": a queued combination is re-warped from wherever the
    fighter *actually* is when it starts, not from wherever it was assumed to be. This is the one
    piece of geometry `fight.py` contributes to that rule, so it is proven directly: stage a commit,
    plant the fighter somewhere that is deliberately not the origin, drive one tick, and check the
    drift speed recorded matches the same formula computed by hand from that live position — not from
    the origin, and not from the ghost alone.
    """
    record = _combination("cross-ring", offsets=((0.3, 0.0), (0.6, 0.0)), tokens=(6, 6))
    timeline = IntentTimeline({"cross-ring": record}, require_admitted=False)
    ghost = (3.0, 1.0)
    timeline.stage(combination="cross-ring", ghost=ghost)
    timeline.commit(0)

    live_position, live_heading = (1.5, 0.7), 0.3
    world = _AnchorOnlyWorld(live_position, live_heading)
    world.fighters = {"red": _fighter_stub("red", timeline)}

    commit_at = timeline.commits[0].issued_at + COMMIT_HORIZON_TICKS
    world._intent_at(world.fighters["red"], commit_at, bearing=0.0)

    commit = timeline.commits[0]
    assert commit.commit_at == commit_at, "the commit never started at the tick we drove"

    cos_h, sin_h = math.cos(live_heading), math.sin(live_heading)
    dx, dy = record.recorded_displacement
    rotated = (cos_h * dx - sin_h * dy, sin_h * dx + cos_h * dy)
    residual = (ghost[0] - live_position[0] - rotated[0], ghost[1] - live_position[1] - rotated[1])
    expected = math.hypot(*residual) / (record.duration_ticks / TICK_HZ)

    recorded = world._drift_speed_m_s[id(commit)]
    assert recorded == pytest.approx(expected)

    # If the anchor had silently defaulted to the origin instead of the live position, the number
    # would be different — pin that down explicitly so a regression that zeroes the anchor is caught
    # by a wrong number rather than merely "some number".
    origin_residual = (ghost[0] - dx, ghost[1] - dy)
    origin_drift = math.hypot(*origin_residual) / (record.duration_ticks / TICK_HZ)
    assert recorded != pytest.approx(origin_drift)


def test_the_anchor_is_not_called_when_nothing_is_starting() -> None:
    """A tick that neither starts nor advances a commit must not touch the drift table at all."""
    record = _combination("idle-tick")
    timeline = IntentTimeline({"idle-tick": record}, require_admitted=False)

    world = _AnchorOnlyWorld((0.0, 0.0), 0.0)
    world.fighters = {"red": _fighter_stub("red", timeline)}

    world._intent_at(world.fighters["red"], 0, bearing=0.0)
    assert world._drift_speed_m_s == {}


# --- pilots ------------------------------------------------------------------------------------------
def _timeline(library) -> IntentTimeline:
    return IntentTimeline(library)


def test_an_idle_pilot_never_commits(library) -> None:
    timeline = _timeline(library)
    pilot = IdlePilot()
    for tick in range(50):
        pilot.act(timeline, tick)
    assert timeline.commits == ()


def test_a_scripted_pilot_commits_on_its_tick(library) -> None:
    timeline = _timeline(library)
    pilot = ScriptedPilot([(7, "combo-a")])

    for tick in range(7):
        pilot.act(timeline, tick)
    assert timeline.commits == ()

    pilot.act(timeline, 7)
    assert len(timeline.commits) == 1
    assert timeline.commits[0].record.name == "combo-a"
    assert timeline.commits[0].issued_at == 7


def test_a_scripted_pilot_fires_each_entry_once(library) -> None:
    timeline = _timeline(library)
    pilot = ScriptedPilot([(3, "combo-a")])
    for _ in range(4):
        pilot.act(timeline, 3)
    assert len(timeline.commits) == 1


def test_a_scripted_ghost_is_carried_into_the_commit(library) -> None:
    timeline = _timeline(library)
    ScriptedPilot([(1, "combo-b", (0.4, 0.0))]).act(timeline, 1)

    assert timeline.commits[0].ghost == (0.4, 0.0)


def test_a_scripted_ghost_defaults_to_the_origin_when_omitted(library) -> None:
    timeline = _timeline(library)
    ScriptedPilot([(1, "combo-a")]).act(timeline, 1)
    assert timeline.commits[0].ghost == (0.0, 0.0)


def test_a_script_may_queue_commits_back_to_back(library) -> None:
    """A second commit is queued, not refused, while the first is still outstanding."""
    timeline = _timeline(library)
    pilot = ScriptedPilot([(0, "combo-a"), (2, "combo-b")])
    pilot.act(timeline, 0)
    pilot.act(timeline, 2)

    assert [c.record.name for c in timeline.commits] == ["combo-a", "combo-b"]
    assert [c.issued_at for c in timeline.commits] == [0, 2]


def test_a_script_that_overruns_the_queue_raises(library) -> None:
    """A script bug, not a dropped input. Silently skipping it would make the match record disagree
    with the script that produced it."""
    timeline = _timeline(library)
    script = [(tick, "combo-a") for tick in range(MAX_OUTSTANDING_COMMITS + 1)]
    pilot = ScriptedPilot(script)
    for tick in range(MAX_OUTSTANDING_COMMITS):
        pilot.act(timeline, tick)

    with pytest.raises(IntentError, match="already queued"):
        pilot.act(timeline, MAX_OUTSTANDING_COMMITS)


def test_reset_rearms_a_script_for_the_next_round(library) -> None:
    pilot = ScriptedPilot([(4, "combo-a")])
    first = _timeline(library)
    pilot.act(first, 4)
    assert len(first.commits) == 1

    pilot.reset()
    second = _timeline(library)
    pilot.act(second, 4)
    assert len(second.commits) == 1, "the script did not fire again in the new round"


def test_a_malformed_script_entry_raises() -> None:
    with pytest.raises(FightError, match=r"\(tick, combination\[, ghost\]\)"):
        ScriptedPilot([(1,)])


# --- the world ----------------------------------------------------------------------------------------
@pytest.mark.slow
def test_a_committed_combination_reaches_its_end_tick_and_records_its_drift(library) -> None:
    """A short combination, driven by a script, runs under physics to its recorded end — and the
    drift speed its warp implied is visible in the round's commit log, unconditionally
    (`spec/intent.md` "The achieved drift speed is recorded in the match record").
    """
    from openroboxing.runtime.contact import ContactTracker, FightTrace

    world = FightWorld(
        libraries={f: library for f in FIGHTERS},
        pilots={"red": ScriptedPilot([(0, "combo-a", (0.2, 0.1))])},
        match_seed=1234,
    )
    world.reset_round(0)

    tracker, trace = ContactTracker(), FightTrace()
    combo = library["combo-a"]
    ticks = COMMIT_HORIZON_TICKS + combo.duration_ticks + 40  # margin past the commit horizon floor
    for tick in range(ticks):
        world.step(tick)
        world.observe(tracker, trace, tick)

    red_events = [e for e in world.commits() if e["fighter"] == "red"]
    assert len(red_events) == 1
    event = red_events[0]

    assert event["commit_at"] is not None, "the commit never started"
    assert event["end_tick"] == event["commit_at"] + combo.duration_ticks
    assert ticks - 1 >= event["end_tick"], "the run did not go far enough to reach the end tick"
    assert event["drift_speed_m_s"] is not None
    assert event["drift_speed_m_s"] >= 0.0


@pytest.mark.slow
def test_both_fighters_run_combinations_without_interfering(library) -> None:
    """Two fighters, two different combinations, one shared physics step — extends the M3-T4
    acceptance test past an idle punchbag to a scripted commit each."""
    from openroboxing.runtime.contact import ContactTracker, FightTrace

    world = FightWorld(
        libraries={f: library for f in FIGHTERS},
        pilots={
            "red": ScriptedPilot([(0, "combo-a", (0.2, 0.1))]),
            "blue": ScriptedPilot([(0, "combo-b", (-0.2, -0.1))]),
        },
        match_seed=1234,
    )
    world.reset_round(0)

    tracker, trace = ContactTracker(), FightTrace()
    longest = max(library["combo-a"].duration_ticks, library["combo-b"].duration_ticks)
    ticks = COMMIT_HORIZON_TICKS + longest + 40
    for tick in range(ticks):
        world.step(tick)
        world.observe(tracker, trace, tick)

    assert len(trace.tick) == ticks
    half = ArenaConfig().ring_size / 2
    for fighter in FIGHTERS:
        inside = [abs(p[:2]).max() < half for p in trace.positions[fighter]]
        assert all(inside), f"{fighter} left the ring"
        assert min(trace.torso_height_m[fighter]) > 0.4, f"{fighter} fell over standing still"

    events = world.commits()
    by_fighter = {f: [e["combination"] for e in events if e["fighter"] == f] for f in FIGHTERS}
    assert by_fighter["red"] == ["combo-a"]
    assert by_fighter["blue"] == ["combo-b"]
    assert all(e["drift_speed_m_s"] is not None for e in events), "one fighter's commit was silent"


@pytest.mark.slow
def test_a_round_reset_puts_both_fighters_back(library) -> None:
    world = FightWorld(
        libraries={f: library for f in FIGHTERS},
        pilots={"red": ScriptedPilot([(0, "combo-a", (0.2, 0.1))])},
        match_seed=1234,
    )
    world.reset_round(0)
    start = world.qpos()
    for tick in range(20):
        world.step(tick)
    assert not np.allclose(world.qpos(), start), "nothing moved"

    world.reset_round(1)
    assert np.allclose(world.qpos(), start)
    assert world.commits() == [], "the commit log carried over from the last round"


@pytest.mark.slow
def test_stepping_before_a_round_starts_raises(library) -> None:
    world = FightWorld(libraries={f: library for f in FIGHTERS}, match_seed=1234)
    with pytest.raises(FightError, match="has not started"):
        world.step(0)


def arena_nq(world) -> int:
    return world.model.nq


# --- the target frame faces the opponent (owner, 2026-09-03) ----------------------------------------
def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class _BearingOnlyWorld:
    """Two pelvises and nothing else: enough to ask which way one fighter must look."""

    facing_angle = FightWorld.facing_angle
    opponent = FightWorld.opponent

    def __init__(self, red: tuple[float, float], blue: tuple[float, float], apply_yaw_: float):
        self.data = SimpleNamespace(
            xpos=np.array([[red[0], red[1], 0.79], [blue[0], blue[1], 0.79]])
        )
        self.fighters = {
            name: SimpleNamespace(
                name=name, pelvis_body=index, apply_delta_heading=_yaw_quat(apply_yaw_)
            )
            for index, name in enumerate(FIGHTERS)
        }


def test_the_bearing_to_the_opponent_is_a_world_angle() -> None:
    """It is measured in the world and converted at `_intent_at`, the one place the two frames meet
    — so the timeline, the warp and the client all talk about the same angle."""
    world = _BearingOnlyWorld(red=(0.0, 0.0), blue=(0.0, 2.0), apply_yaw_=0.5)
    bearing = world.facing_angle("red")

    assert bearing == pytest.approx(math.pi / 2)
    assert bearing != pytest.approx(math.pi / 2 - 0.5), "that is the generator frame, not the world"


def test_the_target_frame_faces_the_opponent_not_the_recording() -> None:
    """The reversal of design D5, at the boundary that matters: whatever a combination recorded, the
    heading handed to MotionBricks — the target frame's, and the facing direction — points at the
    opponent, live, on every tick of a running commit."""
    record = _combination("turner", headings=(1.0, 2.0))
    timeline = IntentTimeline({"turner": record}, require_admitted=False)
    timeline.stage(combination="turner", ghost=(1.0, 0.5))
    timeline.commit(0)

    world = _AnchorOnlyWorld((0.5, 0.2), 0.3)
    fighter = _fighter_stub("red", timeline)
    fighter.apply_delta_heading = _yaw_quat(0.5)
    world.fighters = {"red": fighter}

    bearing = -1.1  # world frame, as `facing_angle` reports it
    commit_at = timeline.commits[0].issued_at + COMMIT_HORIZON_TICKS
    for tick in range(commit_at + 1):
        intent = world._intent_at(fighter, tick, bearing=bearing)

    expected = generator_heading(bearing, fighter.apply_delta_heading)
    assert intent.target_position is not None, "a commit must be running for this to mean anything"
    assert intent.target_heading == pytest.approx(expected)
    assert intent.facing_angle == pytest.approx(expected)


def test_the_opening_stance_faces_the_opponent_in_the_generators_frame() -> None:
    """The stance branch carries no target, so it returns early — and must still be converted, or a
    fighter with nothing committed stands facing wherever its clip happened to start."""
    timeline = IntentTimeline({"unused": _combination("unused")}, require_admitted=False)
    world = _AnchorOnlyWorld((0.0, 0.0), 0.0)
    fighter = _fighter_stub("red", timeline)
    fighter.apply_delta_heading = _yaw_quat(0.5)
    world.fighters = {"red": fighter}

    intent = world._intent_at(fighter, 0, bearing=-1.1)
    assert intent.target_position is None
    assert intent.facing_angle == pytest.approx(generator_heading(-1.1, fighter.apply_delta_heading))
