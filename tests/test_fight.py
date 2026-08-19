"""Two fighters in the ring under physics (M3-T4's world).

The fast tests here cover what does not need a GPU and is where the bugs actually are: the
name-derived indices into a shared ``MjData`` (`CLAUDE.md` invariant 4), the world-to-generator
heading conversion, and the pilots. The slow ones run the thing.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_fight.py -v
    .venv_mb/bin/python -m pytest tests/test_fight.py -v -m slow   # needs a GPU
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.paths import LOADOUT_DIR
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
from openroboxing.runtime.intents import IntentError, IntentTimeline, Loadout, Placement
from openroboxing.spec.constants import (
    APPROACH_LEG_M,
    ARRIVAL_RADIUS_M,
    NUM_JOINTS,
    QPOS_DIM,
)


@pytest.fixture(scope="module")
def arena():
    return build_arena(ArenaConfig())


@pytest.fixture(scope="module")
def loadout() -> Loadout:
    return Loadout.load(LOADOUT_DIR / "orthodox.json")


class _StubGenerator:
    """Never asked for anything by these tests — the reference stream is lazy."""


def _runtime(name: str, arena, loadout) -> FighterRuntime:
    return FighterRuntime(name, arena, _StubGenerator(), loadout, IdlePilot())


# --- indices, derived by name ------------------------------------------------------------------------
def test_a_fighter_finds_all_twenty_nine_joints(arena, loadout) -> None:
    red = _runtime("red", arena, loadout)

    assert red.joint_qpos.shape == (NUM_JOINTS,)
    assert red.joint_dof.shape == (NUM_JOINTS,)
    assert red.actuators.shape == (NUM_JOINTS,)
    assert red.root_qpos.shape == (7,) and red.root_dof.shape == (6,)


def test_the_two_fighters_share_no_index(arena, loadout) -> None:
    """The failure this guards against is silent: red would be driving blue's legs."""
    red = _runtime("red", arena, loadout)
    blue = _runtime("blue", arena, loadout)

    for attribute in ("joint_qpos", "joint_dof", "actuators", "root_qpos", "root_dof"):
        overlap = set(getattr(red, attribute).tolist()) & set(getattr(blue, attribute).tolist())
        assert not overlap, f"{attribute} overlaps between fighters: {sorted(overlap)}"

    assert red.pelvis_body != blue.pelvis_body


def test_the_indices_cover_the_whole_model(arena, loadout) -> None:
    red = _runtime("red", arena, loadout)
    blue = _runtime("blue", arena, loadout)

    qpos = np.concatenate([f.root_qpos for f in (red, blue)] + [f.joint_qpos for f in (red, blue)])
    assert sorted(qpos.tolist()) == list(range(arena.nq)) == list(range(len(FIGHTERS) * QPOS_DIM))

    actuators = np.concatenate([red.actuators, blue.actuators])
    assert sorted(actuators.tolist()) == list(range(arena.nu))


def test_joints_are_indexed_in_mujoco_order(arena, loadout) -> None:
    """Not merely 29 joints, but *these* 29 in the order every gain array assumes."""
    import mujoco

    red = _runtime("red", arena, loadout)
    for position, name in enumerate(G1.mujoco_joint_names):
        joint = mujoco.mj_name2id(arena, mujoco.mjtObj.mjOBJ_JOINT, f"red_{name}")
        assert red.joint_qpos[position] == arena.jnt_qposadr[joint]
        assert red.actuators[position] == int(
            np.flatnonzero(arena.actuator_trnid[:, 0] == joint)[0]
        )


def test_an_unknown_fighter_has_no_joints(arena, loadout) -> None:
    with pytest.raises(FightError, match="is not in the arena"):
        _runtime("green", arena, loadout)


def test_a_fighter_reads_its_own_state_out_of_the_shared_data(arena, loadout) -> None:
    import mujoco

    data = mujoco.MjData(arena)
    reset_to_stance(arena, data, ArenaConfig())

    red = _runtime("red", arena, loadout)
    blue = _runtime("blue", arena, loadout)
    data.qpos[red.joint_qpos] = np.arange(NUM_JOINTS, dtype=float)
    data.qpos[blue.joint_qpos] = -np.arange(NUM_JOINTS, dtype=float)

    assert np.array_equal(red.robot_state(data).joint_pos, np.arange(NUM_JOINTS))
    assert np.array_equal(blue.robot_state(data).joint_pos, -np.arange(NUM_JOINTS))
    assert red.robot_state(data).base_quat.shape == (4,)
    assert red.robot_state(data).base_ang_vel.shape == (3,)


def test_the_fighters_start_apart_and_facing_each_other(arena, loadout) -> None:
    import mujoco

    data = mujoco.MjData(arena)
    reset_to_stance(arena, data, ArenaConfig())
    mujoco.mj_forward(arena, data)

    red = _runtime("red", arena, loadout)
    blue = _runtime("blue", arena, loadout)
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


# --- aiming an approach ------------------------------------------------------------------------------
class _GeometryOnlyWorld:
    """Just enough of a :class:`FightWorld` for the two methods that are pure geometry.

    Both read the pelvis out of the shared ``MjData`` and the generator's buffer tail and nothing
    else, so they can be exercised without a GPU, a policy or a checkpoint.
    """

    leg_target = FightWorld.leg_target
    travel_angle = FightWorld.travel_angle

    def __init__(self, pelvis_xy, context_xy=(0.0, 0.0), leg: float = APPROACH_LEG_M) -> None:
        from types import SimpleNamespace

        self.approach_leg_m = leg
        self.data = SimpleNamespace(xpos=np.array([[pelvis_xy[0], pelvis_xy[1], 0.79]]))
        context = np.zeros((1, QPOS_DIM))
        context[0, 0:2] = context_xy
        self.fighters = {
            "red": SimpleNamespace(
                pelvis_body=0,
                generator=SimpleNamespace(context_qpos=lambda: context),
            )
        }


def test_a_far_placement_is_aimed_one_leg_at_a_time() -> None:
    """The plan is aimed somewhere it can reach; the commit still ends at the placement."""
    world = _GeometryOnlyWorld(pelvis_xy=(0.0, 0.0), leg=1.0)
    leg = world.leg_target("red", Placement(position=(3.0, 4.0), heading=0.9))
    assert np.hypot(*leg.position) == pytest.approx(1.0)          # one leg from the fighter
    assert leg.position == pytest.approx((0.6, 0.8))              # on the line to the placement
    assert leg.heading == pytest.approx(0.9)                      # the player's heading, untouched


def test_the_last_leg_is_the_placement_itself() -> None:
    world = _GeometryOnlyWorld(pelvis_xy=(1.0, 1.0), leg=1.0)
    placement = Placement(position=(1.5, 1.0), heading=0.2)
    assert world.leg_target("red", placement) is placement


def test_a_zero_leg_aims_at_the_whole_distance() -> None:
    """The knob that restores the pre-2026-08-17 aim, for A/B on the bench."""
    world = _GeometryOnlyWorld(pelvis_xy=(0.0, 0.0), leg=0.0)
    placement = Placement(position=(3.0, 4.0), heading=0.0)
    assert world.leg_target("red", placement) is placement


def test_travel_points_where_the_fighter_is_going() -> None:
    """Not where it faces: the difference is what selects a sideways gait upstream."""
    world = _GeometryOnlyWorld(pelvis_xy=(0.0, 0.0), context_xy=(5.0, 5.0))
    # A generator-frame target one metre "left" of the buffer tail, while facing straight ahead.
    angle = world.travel_angle("red", Placement(position=(5.0, 6.0), heading=0.0))
    assert angle == pytest.approx(np.pi / 2)


def test_travel_collapses_onto_the_heading_once_there() -> None:
    """Inside the arrival radius the direction to the target is noise, and noise flips the gait."""
    world = _GeometryOnlyWorld(pelvis_xy=(0.0, 0.0), context_xy=(5.0, 5.0))
    near = Placement(position=(5.0 + ARRIVAL_RADIUS_M / 2, 5.0), heading=1.1)
    assert world.travel_angle("red", near) == pytest.approx(1.1)


# --- pilots ------------------------------------------------------------------------------------------
def _timeline(loadout: Loadout) -> IntentTimeline:
    return IntentTimeline(loadout)


def test_an_idle_pilot_never_commits(loadout) -> None:
    timeline = _timeline(loadout)
    pilot = IdlePilot()
    for tick in range(50):
        pilot.act(timeline, tick)
    assert timeline.commits == ()


def test_a_scripted_pilot_commits_on_its_tick(loadout) -> None:
    timeline = _timeline(loadout)
    pilot = ScriptedPilot([(7, "1")])

    for tick in range(7):
        pilot.act(timeline, tick)
    assert timeline.commits == ()

    pilot.act(timeline, 7)
    assert len(timeline.commits) == 1
    assert timeline.commits[0].slot == "1"
    assert timeline.commits[0].issued_at == 7


def test_a_scripted_pilot_fires_each_entry_once(loadout) -> None:
    timeline = _timeline(loadout)
    pilot = ScriptedPilot([(3, "1")])
    for _ in range(4):
        pilot.act(timeline, 3)
    assert len(timeline.commits) == 1


def test_a_scripted_placement_is_carried_into_the_commit(loadout) -> None:
    timeline = _timeline(loadout)
    placement = Placement(position=(0.4, 0.0), heading=0.2)
    ScriptedPilot([(1, "2", placement)]).act(timeline, 1)

    assert timeline.commits[0].placement == placement


def test_a_script_may_queue_commits_back_to_back(loadout) -> None:
    """`spec/intent.md` 1.0: a second commit is queued, not refused, and runs when the first ends."""
    timeline = _timeline(loadout)
    pilot = ScriptedPilot([(0, "1"), (2, "2")])
    pilot.act(timeline, 0)
    pilot.act(timeline, 2)

    assert len(timeline.commits) == 2
    assert timeline.commits[1].commit_at == timeline.commits[0].end_tick


def test_a_script_that_overruns_the_queue_raises(loadout) -> None:
    """A script bug, not a dropped input. Silently skipping it would make the match record disagree
    with the script that produced it."""
    from openroboxing.spec.constants import MAX_OUTSTANDING_COMMITS

    timeline = _timeline(loadout)
    script = [(tick, "1") for tick in range(MAX_OUTSTANDING_COMMITS + 1)]
    pilot = ScriptedPilot(script)
    for tick in range(MAX_OUTSTANDING_COMMITS):
        pilot.act(timeline, tick)

    with pytest.raises(IntentError, match="already queued"):
        pilot.act(timeline, MAX_OUTSTANDING_COMMITS)


def test_reset_rearms_a_script_for_the_next_round(loadout) -> None:
    pilot = ScriptedPilot([(4, "1")])
    first = _timeline(loadout)
    pilot.act(first, 4)
    assert len(first.commits) == 1

    pilot.reset()
    second = _timeline(loadout)
    pilot.act(second, 4)
    assert len(second.commits) == 1, "the script did not fire again in the new round"


def test_a_malformed_script_entry_raises() -> None:
    with pytest.raises(FightError, match=r"\(tick, slot\[, placement\]\)"):
        ScriptedPilot([(1,)])


# --- the commit log ----------------------------------------------------------------------------------
def test_a_commit_keeps_the_adjustment_the_player_made(loadout) -> None:
    """`spec/match_record.md`'s CommitEvent carries it: "jab, nudged 4 degrees" is not recoverable
    from the resulting angles."""
    timeline = _timeline(loadout)
    pose = loadout.resolve("1")
    joint, bound = next(iter(pose.adjustment_envelope.items()))

    timeline.stage(pose_slot="1", adjustment={joint: bound / 2})
    commit = timeline.commit(0)

    assert commit.adjustment == {joint: bound / 2}
    assert commit.pose.joint_angles[joint] == pytest.approx(
        pose.joint_angles[joint] + bound / 2
    ), "the pose still carries it baked in"


def test_a_commit_with_no_adjustment_records_an_empty_one(loadout) -> None:
    timeline = _timeline(loadout)
    timeline.stage(pose_slot="1")
    assert timeline.commit(0).adjustment == {}


# --- the world ----------------------------------------------------------------------------------------
@pytest.mark.slow
def test_both_fighters_step_and_stay_in_the_ring(loadout) -> None:
    """A short run of the real thing: physics, two generators, two policies, one shared step."""
    from openroboxing.runtime.contact import ContactTracker, FightTrace
    from openroboxing.runtime.fight import FightWorld

    world = FightWorld(loadouts={f: loadout for f in FIGHTERS}, match_seed=1234)
    world.reset_round(0)

    tracker, trace = ContactTracker(), FightTrace()
    for tick in range(100):
        world.step(tick)
        world.observe(tracker, trace, tick)

    assert len(trace.tick) == 100
    assert world.qpos().shape == (arena_nq(world),)

    half = ArenaConfig().ring_size / 2
    for fighter in FIGHTERS:
        inside = [abs(p[:2]).max() < half for p in trace.positions[fighter]]
        assert all(inside), f"{fighter} left the ring"
        assert min(trace.torso_height_m[fighter]) > 0.4, f"{fighter} fell over standing still"


@pytest.mark.slow
def test_a_round_reset_puts_both_fighters_back(loadout) -> None:
    from openroboxing.runtime.fight import FightWorld

    world = FightWorld(loadouts={f: loadout for f in FIGHTERS}, match_seed=1234)
    world.reset_round(0)
    start = world.qpos()
    for tick in range(20):
        world.step(tick)
    assert not np.allclose(world.qpos(), start), "nothing moved"

    world.reset_round(1)
    assert np.allclose(world.qpos(), start)
    assert world.commits() == [], "the commit log carried over from the last round"


@pytest.mark.slow
def test_stepping_before_a_round_starts_raises(loadout) -> None:
    from openroboxing.runtime.fight import FightWorld

    world = FightWorld(loadouts={f: loadout for f in FIGHTERS}, match_seed=1234)
    with pytest.raises(FightError, match="has not started"):
        world.step(0)


def arena_nq(world) -> int:
    return world.model.nq
