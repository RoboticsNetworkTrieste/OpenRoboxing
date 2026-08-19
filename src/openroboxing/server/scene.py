"""What the browser needs to draw the ring, and the frames that move it (`spec/protocol.md` 0.4).

Until 0.4 the host rendered a JPEG and the client was a ``<canvas>``. The project owner overruled
that (`docs/ASSUMPTIONS.md` §A7): the game needs a **shadow you can drive around the ring**, and you
cannot place a ghost in space by looking at a flat video of it. So the client runs three.js and this
module gives it three things:

- a **scene description**, once — every drawable, its shape, and which body it hangs off;
- the **mesh geometry**, once — ~10 MB, cached by the browser;
- a **binary frame**, 30 times a second — one world transform per body, 1708 bytes.

That is ~55 kB/s against the JPEG stream's ~750 kB/s.

The fighters are the host's; the shadow is the client's
-------------------------------------------------------
Frames carry **world** transforms for the two real fighters, which MuJoCo had already computed in
order to step physics. The client never holds *their* kinematic tree and so cannot drift from the
simulation.

The **shadow is different, and deliberately so**: it is drawn entirely in the browser, from the pose
angles in ``welcome`` and a kinematic tree in the description. A ghost that round-tripped to the
server before it moved would be unusable to aim with, and the host has no business knowing where a
player is *thinking* of standing. The host only ever learns a placement when it is committed.

That is the one exception to "the client is a view and a keyboard", and it is bounded: the client
decides where the ghost is *drawn*, the host decides what a commit *means*.

Geometry comes from the compiled model, not the STL files
---------------------------------------------------------
``mjModel`` carries the meshes it actually simulates, already triangulated. Reading them here instead
of serving ``gear_sonic_deploy/g1/meshes/*.STL`` avoids inventing a mesh-name-to-filename mapping
that could silently go stale, needs no STL parser in the client, and guarantees the browser draws the
same geometry the physics used. Red's and blue's copies are identical, so they are **deduplicated by
name suffix** and instanced client-side: 72 compiled meshes become 36 shipped ones.

Conventions
-----------
- **Quaternions are MuJoCo ``wxyz``**, on the wire as well as here. The client converts to three.js's
  ``xyzw`` on the way in, and that is the only place the difference exists.
- **Body index 0 is the world**, which never moves; it is placed once from the scene description and
  never streamed. Streamed bodies are ``1..nbody-1`` and the description's ``bodies`` list is in that
  order, so a frame is a flat array with no keys.
"""

from __future__ import annotations

import struct
from typing import Any, Mapping

import numpy as np

#: Magic at the head of every binary frame: the bytes ``ORBO``, little-endian.
FRAME_MAGIC = 0x4F42524F

#: ``<`` little-endian, ``I`` magic, ``I`` tick, ``H`` body count, ``H`` spare.
FRAME_HEADER = struct.Struct("<IIHH")

#: Floats per body: 3 position + 4 quaternion.
FLOATS_PER_BODY = 7

#: Prefixes the arena gives each fighter's bodies and meshes. Stripped to dedupe geometry.
FIGHTER_PREFIXES = ("red_", "blue_")


class SceneError(RuntimeError):
    """The scene could not be described or packed. Never recovered from silently."""


def _name(mujoco, model, objtype, index: int) -> str:
    name = mujoco.mj_id2name(model, objtype, index)
    if name is None:
        raise SceneError(f"object {index} of type {objtype} has no name; the client indexes by name")
    return name


def _strip_prefix(name: str) -> str:
    for prefix in FIGHTER_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


class Scene:
    """A compiled arena, described for a browser and packed for the wire.

    Built once per match from the ``mjModel`` the fight is running on, so the description can never
    disagree with what is being simulated.
    """

    def __init__(self, model, arena: Mapping[str, Any] | None = None) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.arena = dict(arena or {})

        self.bodies: list[str] = [
            _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)
        ]
        self.meshes = self._collect_meshes()
        self.drawables = self._collect_drawables()

        #: One fighter's bodies, prefix stripped — the frame the client's shadow FK works in. Taken
        #: from red because the two fighters are the same model attached twice.
        self._shadow_ids: list[int] = [
            i
            for i in range(1, model.nbody)
            if _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, i).startswith(FIGHTER_PREFIXES[0])
        ]
        if not self._shadow_ids:
            raise SceneError(
                f"no bodies prefixed {FIGHTER_PREFIXES[0]!r} in the arena; FIGHTER_PREFIXES is stale"
            )
        self.shadow_bodies: list[str] = [
            _strip_prefix(_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, i))
            for i in self._shadow_ids
        ]

    # -- geometry ------------------------------------------------------------------------------------
    def _collect_meshes(self) -> list[dict]:
        """Unique meshes, in the order :meth:`mesh_blob` writes them."""
        mujoco, model = self._mujoco, self.model
        seen: dict[str, dict] = {}
        for index in range(model.nmesh):
            short = _strip_prefix(_name(mujoco, model, mujoco.mjtObj.mjOBJ_MESH, index))
            if short in seen:
                continue
            seen[short] = {
                "name": short,
                "verts": int(model.mesh_vertnum[index]),
                "faces": int(model.mesh_facenum[index]),
                "_id": index,
            }
        return list(seen.values())

    def _collect_drawables(self) -> list[dict]:
        """Every geom, as the client should draw it.

        ``body`` is an index into :attr:`bodies`, or ``-1`` for the world body — which never moves,
        so the client places those once and never touches them again.
        """
        mujoco, model = self._mujoco, self.model
        by_name = {mesh["name"]: position for position, mesh in enumerate(self.meshes)}

        kinds = {
            int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
            int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
        }

        drawables: list[dict] = []
        for index in range(model.ngeom):
            kind = kinds.get(int(model.geom_type[index]))
            if kind is None:
                # Not a silent skip: an arena that grows a geom type the client cannot draw should
                # say so rather than render an invisible obstacle players collide with.
                raise SceneError(
                    f"geom {index} has type {int(model.geom_type[index])}, which the client cannot "
                    "draw; add it to `kinds` and to client/scene.js together"
                )

            entry: dict[str, Any] = {
                "body": int(model.geom_bodyid[index]) - 1,
                "type": kind,
                "pos": [round(float(v), 6) for v in model.geom_pos[index]],
                "quat": [round(float(v), 6) for v in model.geom_quat[index]],
                "size": [round(float(v), 6) for v in model.geom_size[index]],
                "rgba": [round(float(v), 4) for v in model.geom_rgba[index]],
            }
            if kind == "mesh":
                mesh_name = _strip_prefix(
                    _name(mujoco, model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[index]))
                )
                entry["mesh"] = by_name[mesh_name]
            drawables.append(entry)
        return drawables

    # -- the shadow's kinematics ------------------------------------------------------------------------
    def _shadow_kinematics(self) -> dict:
        """One fighter's kinematic tree, so the client can pose a ghost from joint angles alone.

        Exported rather than solved here because the shadow belongs to the client: it must move with
        the mouse, not with a round trip. This is the *only* forward kinematics the browser runs, and
        it never touches the two real fighters, whose transforms arrive already computed.

        Every joint on the G1 is a hinge, so a client needs an axis and an anchor per joint and
        nothing else. A model that grew a slide or ball joint would silently pose wrong, so the
        export refuses one.
        """
        mujoco, model = self._mujoco, self.model
        index_of = {body_id: position for position, body_id in enumerate(self._shadow_ids)}

        bodies = []
        joint_names: list[str] = []
        for body_id in self._shadow_ids:
            joints = []
            start = int(model.body_jntadr[body_id])
            for j in range(start, start + int(model.body_jntnum[body_id])):
                if j < 0:
                    continue
                kind = int(model.jnt_type[j])
                if kind == int(mujoco.mjtJoint.mjJNT_FREE):
                    continue  # the root: the client sets it straight from the placement
                if kind != int(mujoco.mjtJoint.mjJNT_HINGE):
                    raise SceneError(
                        f"joint {_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, j)!r} is type "
                        f"{kind}, but the client's shadow FK only implements hinges"
                    )
                name = _strip_prefix(_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, j))
                joints.append(
                    {
                        "name": name,
                        "axis": [round(float(v), 6) for v in model.jnt_axis[j]],
                        "pos": [round(float(v), 6) for v in model.jnt_pos[j]],
                    }
                )
                joint_names.append(name)

            parent = int(model.body_parentid[body_id])
            bodies.append(
                {
                    "parent": index_of.get(parent, -1),
                    "pos": [round(float(v), 6) for v in model.body_pos[body_id]],
                    "quat": [round(float(v), 6) for v in model.body_quat[body_id]],
                    "joints": joints,
                }
            )

        return {"bodies": bodies, "joints": joint_names}

    def description(self) -> dict:
        """The JSON a client fetches once, at ``/scene.json``."""
        return {
            "spec_version": "0.4",
            "bodies": self.bodies,
            "shadow_bodies": self.shadow_bodies,
            "shadow_kinematics": self._shadow_kinematics(),
            "meshes": [
                {"name": m["name"], "verts": m["verts"], "faces": m["faces"]} for m in self.meshes
            ],
            "drawables": self.drawables,
            "meshes_url": "/meshes.bin",
            "arena": self.arena,
        }

    def mesh_blob(self) -> bytes:
        """Every unique mesh, concatenated, in :attr:`meshes` order.

        Per mesh: ``float32[verts*3]`` positions, ``float32[verts*3]`` normals, ``uint32[faces*3]``
        indices. The client slices it using the counts in the description — there are no offsets on
        the wire because they are exactly the running sum of those counts, and a second copy of a
        derivable number is a second thing that can be wrong.

        Normals are shipped rather than recomputed in the browser because MuJoCo's carry the model's
        hard edges; ``computeVertexNormals`` would smooth them into a soap bar.
        """
        model = self.model
        chunks: list[bytes] = []
        for mesh in self.meshes:
            index = mesh["_id"]
            v0 = int(model.mesh_vertadr[index])
            f0 = int(model.mesh_faceadr[index])
            nv, nf = mesh["verts"], mesh["faces"]

            chunks.append(np.asarray(model.mesh_vert[v0 : v0 + nv], dtype="<f4").tobytes())
            chunks.append(np.asarray(model.mesh_normal[v0 : v0 + nv], dtype="<f4").tobytes())
            chunks.append(np.asarray(model.mesh_face[f0 : f0 + nf], dtype="<u4").tobytes())
        return b"".join(chunks)

    # -- frames --------------------------------------------------------------------------------------
    def pack(self, tick: int, data) -> bytes:
        """One binary frame: the header, then every body's world transform.

        Identical for every viewer — the shadow lives in the browser, so there is nothing private in
        a frame and one pack serves the whole room.
        """
        world = np.empty((len(self.bodies), FLOATS_PER_BODY), dtype="<f4")
        world[:, 0:3] = data.xpos[1 : self.model.nbody]
        world[:, 3:7] = data.xquat[1 : self.model.nbody]
        return FRAME_HEADER.pack(FRAME_MAGIC, int(tick), len(self.bodies), 0) + world.tobytes()
