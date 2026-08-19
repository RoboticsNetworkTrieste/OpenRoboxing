"""The browser's view of the ring: scene description, mesh blob, binary frames (`spec/protocol.md` 0.4).

This module is a **wire format**, so what matters is that the three pieces agree with each other: the
frame's body count matches the description's body list, a drawable's mesh index points at a real
mesh, and the blob's slices are exactly the running sum of the counts the description declares. A
client that trusted any one of those and was wrong would draw a G1 with its arm on backwards.

Reproduce:
    .venv_mb/bin/python -m pytest tests/test_scene.py -v
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from openroboxing.runtime.arena import FIGHTERS, ArenaConfig, build_arena, reset_to_stance
from openroboxing.server.scene import (
    FLOATS_PER_BODY,
    FRAME_HEADER,
    FRAME_MAGIC,
    Scene,
    SceneError,
)

#: How far the browser's ghost may sit from where MuJoCo would put it, metres. Not a fudge: the
#: kinematic tree ships rounded to 6 decimals so `scene.json` stays ~40 kB, and that rounding
#: compounds along a limb. Measured at 8.5e-07 m worst case.
EXPORT_ROUNDING_M = 5e-06


@pytest.fixture(scope="module")
def model():
    return build_arena(ArenaConfig())


@pytest.fixture(scope="module")
def scene(model):
    return Scene(model, {"ring_size": 4.90})


@pytest.fixture(scope="module")
def data(model):
    import mujoco

    d = mujoco.MjData(model)
    reset_to_stance(model, d, ArenaConfig())
    return d


# --- the description ------------------------------------------------------------------------------
def test_every_streamed_body_is_named_and_the_world_is_not_one(scene, model) -> None:
    """Body 0 never moves, so it is placed once from the description and never streamed."""
    assert len(scene.bodies) == model.nbody - 1
    assert "world" not in scene.bodies
    assert all(isinstance(name, str) and name for name in scene.bodies)


def test_both_fighters_are_present_and_prefixed(scene) -> None:
    for fighter in FIGHTERS:
        assert any(name.startswith(f"{fighter}_") for name in scene.bodies)


def test_geometry_is_deduplicated_across_the_two_fighters(scene, model) -> None:
    """Red and blue are the same model attached twice, so shipping both copies would double a 10 MB
    download for nothing."""
    assert len(scene.meshes) < model.nmesh
    assert len(scene.meshes) == model.nmesh // 2
    names = [mesh["name"] for mesh in scene.meshes]
    assert len(set(names)) == len(names)
    assert not any(n.startswith(("red_", "blue_")) for n in names)


def test_every_drawable_points_at_a_body_that_exists(scene) -> None:
    for drawable in scene.drawables:
        assert -1 <= drawable["body"] < len(scene.bodies)


def test_every_mesh_drawable_points_at_a_shipped_mesh(scene) -> None:
    for drawable in scene.drawables:
        if drawable["type"] == "mesh":
            assert 0 <= drawable["mesh"] < len(scene.meshes)


def test_the_ring_is_primitives_and_the_fighters_are_meshes(scene) -> None:
    world = [d for d in scene.drawables if d["body"] == -1]
    assert world, "the canvas, posts and ropes hang off the world body"
    assert all(d["type"] != "mesh" for d in world)
    assert any(d["type"] == "mesh" for d in scene.drawables if d["body"] >= 0)


def test_an_undrawable_geom_type_is_refused_rather_than_skipped(model) -> None:
    """A silently skipped geom is an invisible obstacle players collide with."""
    import mujoco

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body()
    body.name = "odd"
    geom = body.add_geom()
    geom.name = "ellipsoid"
    geom.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    geom.size = [0.1, 0.2, 0.3]

    with pytest.raises(SceneError, match="cannot draw"):
        Scene(spec.compile())


def test_the_description_is_json_safe(scene) -> None:
    import json

    encoded = json.dumps(scene.description())
    assert "_id" not in encoded, "the internal mesh id must not reach a client"


# --- the shadow's kinematics ----------------------------------------------------------------------
def test_the_shadow_carries_one_fighter_not_two(scene) -> None:
    assert len(scene.shadow_bodies) == len(scene.bodies) // 2
    assert not any(name.startswith(("red_", "blue_")) for name in scene.shadow_bodies)


def test_the_kinematic_tree_is_rooted_and_acyclic(scene) -> None:
    """A client walks parents to compose transforms; a body whose parent came later would be posed
    against a stale one."""
    tree = scene.description()["shadow_kinematics"]["bodies"]
    assert tree[0]["parent"] == -1, "the pelvis hangs off the placement, not off another body"
    for index, body in enumerate(tree):
        assert body["parent"] < index, f"body {index} refers forwards to {body['parent']}"


def test_the_tree_names_every_joint_once_in_mujoco_order(scene) -> None:
    from openroboxing.runtime.conventions import G1

    joints = scene.description()["shadow_kinematics"]["joints"]
    assert joints == list(G1.mujoco_joint_names)


def test_every_joint_in_the_tree_has_an_axis_and_an_anchor(scene) -> None:
    for body in scene.description()["shadow_kinematics"]["bodies"]:
        for joint in body["joints"]:
            assert len(joint["axis"]) == 3 and any(joint["axis"])
            assert len(joint["pos"]) == 3


def test_the_root_free_joint_is_not_in_the_tree(scene) -> None:
    """The client sets the root straight from the placement; a free joint has no axis to speak of."""
    joints = scene.description()["shadow_kinematics"]["joints"]
    assert not any("floating_base" in name for name in joints)


# --- the mesh blob --------------------------------------------------------------------------------
def test_the_blob_is_exactly_the_size_the_description_promises(scene) -> None:
    """The client slices by the running sum of the counts. One byte out and every mesh after the
    first is garbage."""
    expected = sum(
        mesh["verts"] * 3 * 4 * 2 + mesh["faces"] * 3 * 4 for mesh in scene.meshes
    )
    assert len(scene.mesh_blob()) == expected


def test_the_first_mesh_decodes_to_plausible_geometry(scene) -> None:
    mesh = scene.meshes[0]
    blob = scene.mesh_blob()

    verts = np.frombuffer(blob, dtype="<f4", count=mesh["verts"] * 3).reshape(-1, 3)
    faces = np.frombuffer(
        blob,
        dtype="<u4",
        count=mesh["faces"] * 3,
        offset=mesh["verts"] * 3 * 4 * 2,
    )

    assert np.isfinite(verts).all()
    assert np.abs(verts).max() < 2.0, "a G1 link is smaller than the ring"
    assert faces.max() < mesh["verts"], "an index points outside its own mesh"


# --- binary frames --------------------------------------------------------------------------------
def test_a_frame_is_the_header_plus_seven_floats_a_body(scene, data) -> None:
    frame = scene.pack(tick=41, data=data)
    assert len(frame) == FRAME_HEADER.size + len(scene.bodies) * FLOATS_PER_BODY * 4


def test_a_frame_announces_itself(scene, data) -> None:
    magic, tick, bodies, reserved = FRAME_HEADER.unpack_from(scene.pack(41, data))
    assert magic == FRAME_MAGIC
    assert tick == 41
    assert bodies == len(scene.bodies)
    assert reserved == 0


def test_the_magic_spells_orbo(scene, data) -> None:
    """Little-endian, so the bytes read in order. A client checks this before trusting a buffer."""
    assert struct.pack("<I", FRAME_MAGIC) == b"ORBO"


def test_a_frame_carries_the_positions_the_simulator_has(scene, data) -> None:
    frame = scene.pack(7, data)
    values = np.frombuffer(frame, dtype="<f4", offset=FRAME_HEADER.size).reshape(-1, 7)

    assert values.shape == (len(scene.bodies), 7)
    np.testing.assert_allclose(values[:, 0:3], data.xpos[1:], rtol=0, atol=1e-6)
    np.testing.assert_allclose(values[:, 3:7], data.xquat[1:], rtol=0, atol=1e-6)


def test_quaternions_stay_in_mujoco_order(scene, data) -> None:
    """`wxyz` on the wire. three.js wants `xyzw`, and the client converts — one place, named."""
    values = np.frombuffer(scene.pack(0, data), dtype="<f4", offset=FRAME_HEADER.size).reshape(-1, 7)
    norms = np.linalg.norm(values[:, 3:7], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    pelvis = scene.bodies.index("red_pelvis")
    assert values[pelvis, 3] == pytest.approx(1.0, abs=1e-5), "red starts unrotated, w=1"


def test_a_frame_is_the_same_for_everybody(scene, data) -> None:
    """Nothing private is in a frame, which is what lets one pack serve the whole room."""
    assert scene.pack(3, data) == scene.pack(3, data)


def test_the_frame_is_far_smaller_than_the_jpeg_it_replaced(scene, data) -> None:
    """The trade 0.4 is built on: ~25 kB a JPEG became ~1.7 kB of transforms."""
    assert len(scene.pack(0, data)) < 4_000


# --- the client's FK, checked against MuJoCo's -----------------------------------------------------
def _client_fk(kinematics, placement, angles):
    """The algorithm in ``client/ring.js::ShadowSkeleton.solve``, in numpy.

    Transcribed rather than imported, because the thing under test is whether the **exported tree** is
    enough to pose a G1 correctly. If the browser's ghost is wrong, a player aims with a preview of a
    move that will not happen — and nothing else in the system would notice.
    """

    def quat_mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def rotate(q, v):
        w, x, y, z = q
        u = np.array([x, y, z])
        return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)

    def axis_angle(axis, angle):
        half = angle / 2.0
        return np.array([np.cos(half), *(np.asarray(axis) * np.sin(half))])

    x, y, z, heading = placement
    positions, quaternions = [], []
    for body in kinematics["bodies"]:
        parent = body["parent"]
        if parent < 0:
            pos = np.array([x, y, z])
            quat = axis_angle([0, 0, 1], heading)
        else:
            quat = quaternions[parent].copy()
            pos = positions[parent] + rotate(quat, np.asarray(body["pos"]))
            quat = quat_mul(quat, np.asarray(body["quat"]))

        for joint in body["joints"]:
            angle = angles.get(joint["name"])
            if angle is None:
                continue
            anchor = pos + rotate(quat, np.asarray(joint["pos"]))
            quat = quat_mul(quat, axis_angle(joint["axis"], angle))
            pos = anchor - rotate(quat, np.asarray(joint["pos"]))

        positions.append(pos)
        quaternions.append(quat)
    return np.array(positions), np.array(quaternions)


def test_the_shipped_tree_poses_a_g1_exactly_as_mujoco_does(scene) -> None:
    """The browser draws the ghost from `shadow_kinematics` alone. If that tree is short of anything
    MuJoCo uses, the ghost bends wrong and a player aims with a preview of a different move."""
    import mujoco

    from openroboxing.paths import G1_29DOF_SIM_XML
    from openroboxing.runtime.conventions import G1
    from openroboxing.spec.constants import NUM_JOINTS, QPOS_DIM

    reference = mujoco.MjModel.from_xml_path(str(G1_29DOF_SIM_XML))
    data = mujoco.MjData(reference)

    rng = np.random.default_rng(7)
    angles = {name: float(v) for name, v in zip(G1.mujoco_joint_names, rng.uniform(-0.6, 0.6, NUM_JOINTS))}
    placement = (1.25, -0.4, 0.793, 0.9)

    qpos = np.zeros(QPOS_DIM)
    qpos[0], qpos[1], qpos[2] = placement[:3]
    qpos[3], qpos[6] = np.cos(placement[3] / 2), np.sin(placement[3] / 2)
    qpos[7:] = [angles[name] for name in G1.mujoco_joint_names]
    data.qpos[:] = qpos
    mujoco.mj_kinematics(reference, data)

    kinematics = scene.description()["shadow_kinematics"]
    positions, quaternions = _client_fk(kinematics, placement, angles)

    ids = [
        mujoco.mj_name2id(reference, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in scene.shadow_bodies
    ]
    assert all(i >= 0 for i in ids), "a streamed body is missing from the standalone model"

    # The tree is exported rounded to 6 decimals to keep scene.json small, and that rounding
    # compounds down a limb. Measured worst case here: 8.5e-07 m at the fingertips — a micron on a
    # ghost, and the bound is asserted rather than the rounding being assumed harmless.
    np.testing.assert_allclose(positions, data.xpos[ids], atol=EXPORT_ROUNDING_M)

    # Quaternions are sign-ambiguous, so compare what an orientation actually does to a vector.
    probe = np.array([0.13, -0.27, 0.41])
    for index, body in enumerate(ids):
        want = data.xmat[body].reshape(3, 3) @ probe
        w, x, y, z = quaternions[index]
        u = np.array([x, y, z])
        got = probe + 2.0 * np.cross(u, np.cross(u, probe) + w * probe)
        np.testing.assert_allclose(got, want, atol=EXPORT_ROUNDING_M)


def test_the_tree_covers_every_joint_the_pose_library_sets(scene) -> None:
    """A joint an authored pose sets but the tree omits would be silently ignored by the ghost."""
    from openroboxing.paths import LOADOUT_DIR
    from openroboxing.runtime.intents import Loadout

    loadout = Loadout.load(LOADOUT_DIR / "orthodox.json")
    exported = set(scene.description()["shadow_kinematics"]["joints"])
    for slot, pose in loadout.slots.items():
        missing = set(pose.joint_angles) - exported
        assert not missing, f"slot {slot} ({pose.name}) sets {missing}, which the ghost cannot bend"
