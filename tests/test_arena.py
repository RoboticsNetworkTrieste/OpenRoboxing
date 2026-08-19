"""M3-T1 acceptance: the arena scene.

Acceptance criterion from WORKPLAN.md M3-T1:
  both fighters spawn in stance facing each other; `bench_world.py` on this scene meets the
  real-time target.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_arena.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.runtime.arena import (
    FIGHTERS,
    GLOVE_BODIES,
    ArenaConfig,
    ArenaError,
    build_arena,
    fighter_forward,
    fighter_qpos_slice,
    reset_to_stance,
    start_pose,
)
from openroboxing.spec.constants import NUM_JOINTS

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def arena():
    import mujoco

    model = build_arena()
    return mujoco, model, mujoco.MjData(model)


def _geom_name(mujoco, model, index: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or f"geom{index}"


# --- the acceptance criterion ---------------------------------------------------------------------
def test_both_fighters_spawn_in_stance_facing_each_other(arena) -> None:
    mujoco, model, data = arena
    reset_to_stance(model, data)

    positions, forwards = [], []
    for index, fighter in enumerate(FIGHTERS):
        qpos = data.qpos[fighter_qpos_slice(index)]
        positions.append(qpos[:3].copy())
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{fighter}_pelvis")
        forwards.append(data.xmat[body].reshape(3, 3)[:, 0].copy())

    # They stand apart along x, at the same height, on opposite sides of centre.
    assert positions[0][0] < 0 < positions[1][0]
    assert positions[0][2] == pytest.approx(positions[1][2])
    assert positions[0][1] == pytest.approx(0.0) and positions[1][1] == pytest.approx(0.0)

    # Facing each other: each fighter's forward points at the other.
    between = positions[1][:2] - positions[0][:2]
    assert np.dot(forwards[0][:2], between) > 0, "red does not face blue"
    assert np.dot(forwards[1][:2], between) < 0, "blue does not face red"
    assert np.dot(forwards[0][:2], forwards[1][:2]) == pytest.approx(-1.0, abs=1e-6)


def test_the_stance_is_settled_not_interpenetrating(arena) -> None:
    """A scene that starts in contact starts by exploding. This is the check that catches it."""
    mujoco, model, data = arena
    reset_to_stance(model, data)
    mujoco.mj_forward(model, data)

    offenders = [
        (_geom_name(mujoco, model, c.geom1), _geom_name(mujoco, model, c.geom2), float(c.dist))
        for c in data.contact[: data.ncon]
        if c.dist < -1e-4
    ]
    assert not offenders, f"the fighters start interpenetrating: {offenders[:5]}"


# --- geometry ---------------------------------------------------------------------------------------
def test_the_arena_holds_exactly_two_g1s(arena) -> None:
    _, model, _ = arena
    assert model.nq == len(FIGHTERS) * (7 + NUM_JOINTS)
    assert len(FIGHTERS) == 2


def test_each_fighter_has_its_own_qpos_block(arena) -> None:
    mujoco, model, data = arena
    reset_to_stance(model, data)

    first, second = (data.qpos[fighter_qpos_slice(i)] for i in range(2))
    assert len(first) == len(second) == 7 + NUM_JOINTS
    assert not np.allclose(first[:3], second[:3]), "both fighters are in the same place"

    with pytest.raises(ArenaError, match="outside"):
        fighter_qpos_slice(len(FIGHTERS))


def test_the_ring_has_ropes_posts_and_one_canvas(arena) -> None:
    mujoco, model, _ = arena
    names = [_geom_name(mujoco, model, g) for g in range(model.ngeom)]

    assert names.count("canvas") == 1, "the ring must have exactly one canvas"
    assert not [n for n in names if n.endswith("_floor")], "a robot brought its own floor along"
    assert len([n for n in names if n.startswith("rope_")]) == 4 * len(
        ArenaConfig().rope_heights
    ), "each rope height needs four sides"
    assert len([n for n in names if n.startswith("post_")]) == 4


def test_both_fighters_are_gloved(arena) -> None:
    mujoco, model, _ = arena
    gloves = [
        _geom_name(mujoco, model, g)
        for g in range(model.ngeom)
        if "glove" in _geom_name(mujoco, model, g)
    ]
    assert len(gloves) == len(FIGHTERS) * len(GLOVE_BODIES)
    for fighter in FIGHTERS:
        assert len([g for g in gloves if g.startswith(fighter)]) == len(GLOVE_BODIES)


def test_there_are_cameras_to_watch_from(arena) -> None:
    mujoco, model, _ = arena
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in range(model.ncam)}
    assert {"broadcast", "overhead"} <= names


# --- gloves hit the opponent, not their owner -----------------------------------------------------------
def test_a_glove_cannot_hit_its_own_fighter(arena) -> None:
    """It sits on the forearm it would otherwise push against forever — 27 mm of it, measured."""
    mujoco, model, data = arena
    reset_to_stance(model, data)
    mujoco.mj_forward(model, data)

    for contact in data.contact[: data.ncon]:
        first = _geom_name(mujoco, model, contact.geom1)
        second = _geom_name(mujoco, model, contact.geom2)
        if "glove" not in first + second:
            continue
        owner = (first if "glove" in first else second).split("_")[0]
        other = second if "glove" in first else first
        assert not other.startswith(owner), f"{first} is in contact with its own fighter: {second}"


def test_a_glove_can_reach_the_opponent(arena) -> None:
    """The exclusion must not have switched gloves off entirely."""
    mujoco, model, data = arena
    reset_to_stance(model, data)

    # Overlap the fighters so the arms interpenetrate, then look for a cross-fighter glove contact.
    data.qpos[fighter_qpos_slice(0)][0] = -0.15
    data.qpos[fighter_qpos_slice(1)][0] = 0.15
    mujoco.mj_forward(model, data)

    crossed = [
        (_geom_name(mujoco, model, c.geom1), _geom_name(mujoco, model, c.geom2))
        for c in data.contact[: data.ncon]
    ]
    glove_hits = [
        (a, b)
        for a, b in crossed
        if ("glove" in a or "glove" in b) and a.split("_")[0] != b.split("_")[0]
    ]
    assert glove_hits, f"no glove reached the opponent; contacts were {crossed[:6]}"


# --- config ----------------------------------------------------------------------------------------------
def test_a_ring_too_small_for_its_fighters_raises() -> None:
    with pytest.raises(ArenaError, match="cannot hold fighters"):
        ArenaConfig(ring_size=1.0, start_separation=1.2)


def test_a_ring_needs_ropes() -> None:
    with pytest.raises(ArenaError, match="at least one rope"):
        ArenaConfig(rope_heights=())


def test_the_timestep_is_the_one_perf_recommended(arena) -> None:
    """M1-T7 chose 0.001 s for contact fidelity; a scene that quietly used another would invalidate
    the real-time verdict this milestone depends on."""
    _, model, _ = arena
    assert model.opt.timestep == pytest.approx(ArenaConfig().timestep)
    assert ArenaConfig().timestep == 0.001


def test_forward_and_start_pose_agree() -> None:
    for fighter in FIGHTERS:
        position, quat = start_pose(fighter)
        forward = fighter_forward(fighter)
        # The fighter starts on the opposite side from where it looks.
        assert np.dot(position[:2], forward[:2]) < 0
        assert np.linalg.norm(quat) == pytest.approx(1.0)

    with pytest.raises(ArenaError, match="unknown fighter"):
        fighter_forward("green")


def test_fighters_fall_over_without_a_policy(arena) -> None:
    """Sanity: the ring is not holding them up, and it is not launching them either."""
    mujoco, model, data = arena
    reset_to_stance(model, data)
    for _ in range(1000):
        mujoco.mj_step(model, data)

    heights = [float(data.qpos[fighter_qpos_slice(i)][2]) for i in range(len(FIGHTERS))]
    assert all(0.0 < h < 0.8 for h in heights), f"unactuated fighters ended at {heights}"
