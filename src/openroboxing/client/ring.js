/* The 3-D ring: scene from /scene.json, geometry from /meshes.bin, poses from binary frames.
 *
 * Implements spec/protocol.md 0.6 (3-D rendering since 0.4). Until 0.4 the host rendered a JPEG and
 * this file did not exist; the game needs a shadow you can drive around the ring, and you cannot
 * place a ghost in space by looking at a flat video of it.
 *
 * Who owns what
 * -------------
 * The two real fighters are the host's: their body transforms arrive already computed, so this file
 * only copies numbers into Object3Ds and can never disagree with the simulation.
 *
 * The shadow is ours. It is posed here, from the selected combination's final-keyframe joint angles
 * (`welcome`'s `pose`, since 0.6) and the kinematic tree in the scene description, because a ghost
 * that round-tripped to the server before it moved would be unusable to aim with. That is the one
 * piece of forward kinematics in the client, and it touches nothing the simulation owns.
 *
 * `fighterHeading` and `fighterPosition` read two more things off those same streamed transforms:
 * the pelvis's own yaw, and where it stands. Both serve the ghost's *derived* heading, which since
 * the owner's 2026-09-03 rule is the bearing from the ghost to the **opponent's** pelvis. Not new
 * kinematics — the same world transforms every other body already gets, read rather than only
 * copied.
 *
 * Quaternions arrive MuJoCo `wxyz`; three.js wants `xyzw`. `setQuat` below is the only place in the
 * client where that difference exists.
 */

'use strict';

import * as THREE from './vendor/three.module.min.js';

/* Header of a binary frame: magic, tick, body count, reserved. See spec/protocol.md. */
const FRAME_MAGIC = 0x4f42524f;   // the bytes "ORBO", little-endian
const FRAME_HEADER_BYTES = 12;
const FLOATS_PER_BODY = 7;

/* MuJoCo builds a capsule/cylinder along its own +z; three.js builds one along +y. */
const Z_TO_Y = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);

const SHADOW_COLOUR = { red: 0xff6b6b, blue: 0x6bb6ff };

/* A ghost placed beyond its combination's own `reach_m` (`spec/protocol.md` 0.6 §"Feasibility"):
 * the host will reject the commit, and the shadow says so before the player pays for it. Distinct
 * from both seat colours so it reads as "rejected" rather than "the other seat's ghost". */
const REJECTED_COLOUR = 0xffa53d;

/* Scratch for `project`, so the render loop allocates nothing. */
const PROJECTED = new THREE.Vector3();

function setQuat(object, w, x, y, z) {
  object.quaternion.set(x, y, z, w);
}

/* ---- shadow kinematics ------------------------------------------------------------------------ */
/* MuJoCo's own composition rule, transcribed: a body sits at its parent's frame plus a local offset,
 * and each hinge rotates the body about an axis through an anchor expressed in that body's frame.
 *
 *   world = parent * local
 *   anchor_world = world * jnt.pos
 *   world.rotation *= rotation(jnt.axis, angle)
 *   world.position = anchor_world - world.rotation * jnt.pos
 *
 * The last line is the part that is easy to get wrong: rotating a joint must leave its *anchor*
 * where it was, not the body origin, or every limb slides as it bends. */
class ShadowSkeleton {
  constructor(kinematics) {
    this.bodies = kinematics.bodies;
    this.jointOrder = kinematics.joints;

    this.local = this.bodies.map((body) => ({
      pos: new THREE.Vector3().fromArray(body.pos),
      quat: new THREE.Quaternion(body.quat[1], body.quat[2], body.quat[3], body.quat[0]),
      joints: body.joints.map((joint) => ({
        name: joint.name,
        axis: new THREE.Vector3().fromArray(joint.axis),
        pos: new THREE.Vector3().fromArray(joint.pos),
      })),
    }));

    this.position = this.bodies.map(() => new THREE.Vector3());
    this.quaternion = this.bodies.map(() => new THREE.Quaternion());
    this._scratch = new THREE.Vector3();
    this._turn = new THREE.Quaternion();
  }

  /* Pose every body from a root placement and a joint-name -> angle map. */
  solve(x, y, z, heading, angles) {
    const root = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), heading);

    for (let i = 0; i < this.local.length; i += 1) {
      const body = this.local[i];
      const parent = this.bodies[i].parent;
      const outPos = this.position[i];
      const outQuat = this.quaternion[i];

      if (parent < 0) {
        outPos.set(x, y, z);
        outQuat.copy(root);
      } else {
        outQuat.copy(this.quaternion[parent]);
        outPos.copy(body.pos).applyQuaternion(outQuat).add(this.position[parent]);
        outQuat.multiply(body.quat);
      }

      for (const joint of body.joints) {
        const angle = angles[joint.name];
        if (angle === undefined) continue;
        /* anchor stays put; the body swings around it */
        this._scratch.copy(joint.pos).applyQuaternion(outQuat).add(outPos);
        this._turn.setFromAxisAngle(joint.axis, angle);
        outQuat.multiply(this._turn);
        outPos.copy(joint.pos).applyQuaternion(outQuat).negate().add(this._scratch);
      }
    }
  }
}

/* ---- the ring ---------------------------------------------------------------------------------- */
export class Ring {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0d12);

    /* MuJoCo is z-up. Telling three.js so once beats rotating every object. */
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    this.camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.05, 200);
    this.camera.up.set(0, 0, 1);

    this.bodies = [];          // Object3D per streamed body, in scene.bodies order
    this.bodyNames = [];       // scene.bodies itself, kept for fighterHeading()
    this.meshes = [];          // BufferGeometry per shipped mesh
    this.skeleton = null;      // ShadowSkeleton — shared solver, one seat at a time
    this.shadows = {};         // seat -> { root, parts }
    this._shadowDrawables = null;

    this.tick = 0;
    this.expectedBodies = 0;
    this._addLights();
    window.addEventListener('resize', () => this.resize());
  }

  _addLights() {
    this.scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x1a1d26, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(-3.5, -3.5, 6.0);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    const span = 5;
    Object.assign(key.shadow.camera, { left: -span, right: span, top: span, bottom: -span, far: 25 });
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0x9db4ff, 0.5);
    fill.position.set(4.0, 4.0, 3.0);
    this.scene.add(fill);
  }

  /* ---- building ------------------------------------------------------------------------------- */
  async load(base = '') {
    const description = await (await fetch(`${base}/scene.json`)).json();
    const blob = await (await fetch(`${base}${description.meshes_url}`)).arrayBuffer();

    this._buildMeshes(description.meshes, blob);
    this._buildBodies(description.bodies);
    this._buildDrawables(description.drawables);
    this._buildShadow(description);
    this.frameRing(description.arena);
    return description;
  }

  _buildMeshes(meshes, blob) {
    /* Offsets are the running sum of the counts already in the description. Sending them again
       would be a second copy of a derivable number, and a second thing that can disagree. */
    let offset = 0;
    this.meshes = meshes.map((mesh) => {
      const vertexFloats = mesh.verts * 3;
      const positions = new Float32Array(blob, offset, vertexFloats);
      offset += vertexFloats * 4;
      const normals = new Float32Array(blob, offset, vertexFloats);
      offset += vertexFloats * 4;
      const indices = new Uint32Array(blob, offset, mesh.faces * 3);
      offset += mesh.faces * 3 * 4;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
      geometry.setIndex(new THREE.BufferAttribute(indices, 1));
      geometry.computeBoundingSphere();
      return geometry;
    });

    if (offset !== blob.byteLength) {
      throw new Error(
        `meshes.bin is ${blob.byteLength} bytes, the description accounts for ${offset}`,
      );
    }
  }

  _buildBodies(names) {
    this.expectedBodies = names.length;
    this.bodyNames = names;         // kept for fighterHeading(); order matches the streamed frame
    this.bodies = names.map(() => {
      const group = new THREE.Group();
      this.scene.add(group);
      return group;
    });
  }

  /* A drawable is a mesh or a MuJoCo primitive. Ours are the ring (primitives) and the two
     fighters (meshes). */
  _geometryFor(drawable) {
    const [a, b] = drawable.size;
    switch (drawable.type) {
      case 'mesh':
        return this.meshes[drawable.mesh];
      case 'sphere':
        return new THREE.SphereGeometry(a, 24, 16);
      case 'capsule':
        return new THREE.CapsuleGeometry(a, b * 2, 6, 12);
      case 'cylinder':
        return new THREE.CylinderGeometry(a, a, b * 2, 20);
      case 'box':
        return new THREE.BoxGeometry(a * 2, b * 2, drawable.size[2] * 2);
      case 'plane':
        return new THREE.PlaneGeometry(Math.max(a, 1) * 2, Math.max(b, 1) * 2);
      default:
        throw new Error(`scene.json asked for a '${drawable.type}', which this client cannot draw`);
    }
  }

  _buildDrawables(drawables) {
    for (const drawable of drawables) {
      const geometry = this._geometryFor(drawable);
      const [r, g, b, alpha] = drawable.rgba;
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, g, b),
        transparent: alpha < 1,
        opacity: alpha,
        roughness: drawable.type === 'mesh' ? 0.55 : 0.8,
        metalness: drawable.type === 'mesh' ? 0.25 : 0.0,
      });

      const object = new THREE.Mesh(geometry, material);
      object.position.fromArray(drawable.pos);
      setQuat(object, ...drawable.quat);
      /* MuJoCo lays capsules and cylinders along +z; three.js along +y. */
      if (drawable.type === 'capsule' || drawable.type === 'cylinder') {
        object.quaternion.multiply(Z_TO_Y);
      }
      object.castShadow = drawable.type === 'mesh';
      object.receiveShadow = true;

      if (drawable.body < 0) {
        this.scene.add(object);          // world geoms never move
      } else {
        this.bodies[drawable.body].add(object);
      }
    }
  }

  _buildShadow(description) {
    this.skeleton = new ShadowSkeleton(description.shadow_kinematics);
    this.shadowBodies = description.shadow_bodies;

    /* Which drawables make up one fighter, keyed by body. Taken from red because the two are the
       same model attached twice, so the ghost is the same robot as the thing it previews. */
    const byBody = new Map();
    for (const drawable of description.drawables) {
      if (drawable.body < 0 || drawable.type !== 'mesh') continue;
      const name = description.bodies[drawable.body];
      if (!name.startsWith('red_')) continue;
      const short = name.slice('red_'.length);
      if (!byBody.has(short)) byBody.set(short, []);
      byBody.get(short).push(drawable);
    }
    this._shadowDrawables = byBody;
  }

  /* A ghost per seat, built on demand. Hotseat puts two players at one screen, so both need one —
     and the geometry is shared, only the material and the transforms differ. */
  shadowFor(seat) {
    if (this.shadows[seat]) return this.shadows[seat];

    /* Depth-write off and additive blending, so a ghost reads as a projection rather than a third
       fighter somebody might mistake for a real one. */
    const material = new THREE.MeshBasicMaterial({
      color: SHADOW_COLOUR[seat] ?? 0xffffff,
      transparent: true,
      opacity: 0.3,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    const root = new THREE.Group();
    root.visible = false;
    const parts = this.shadowBodies.map((name) => {
      const group = new THREE.Group();
      for (const drawable of this._shadowDrawables.get(name) || []) {
        const object = new THREE.Mesh(this.meshes[drawable.mesh], material);
        object.position.fromArray(drawable.pos);
        setQuat(object, ...drawable.quat);
        group.add(object);
      }
      root.add(group);
      return group;
    });

    this.scene.add(root);
    this.shadows[seat] = { root, parts, material };
    return this.shadows[seat];
  }

  /* ---- the camera ------------------------------------------------------------------------------ */
  frameRing(arena) {
    const half = (arena?.ring_size ?? 4.9) / 2;
    this.camera.position.set(0, -(half + 3.4), 2.6);
    this.camera.lookAt(0, 0, 0.9);
    this.resize();
  }

  resize() {
    const width = this.canvas.clientWidth || 960;
    const height = this.canvas.clientHeight || 540;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /* A world point in the canvas's own pixel box.
   *
   * The annotations the design asks for on top of the arena — the distance path, the anchor marker,
   * the contact ring — are drawn in SVG over the canvas, and they have to land on the pixels the
   * renderer drew to. So they go through *this* camera rather than a second guess at where the
   * camera is; if `frameRing` ever moves it, the overlay moves with it and cannot drift.
   */
  project(x, y, z = 0) {
    PROJECTED.set(x, y, z).project(this.camera);
    const width = this.canvas.clientWidth || 1;
    const height = this.canvas.clientHeight || 1;
    return {
      x: (PROJECTED.x * 0.5 + 0.5) * width,
      y: (-PROJECTED.y * 0.5 + 0.5) * height,
      behind: PROJECTED.z > 1,
    };
  }

  /* ---- streaming ------------------------------------------------------------------------------- */
  /* One binary frame: check it is ours and the right shape, then copy transforms straight in. A
     frame that disagrees with the scene description is refused rather than drawn partially — half a
     robot in the right place looks like a physics bug, and would be debugged as one. */
  applyFrame(buffer) {
    const header = new DataView(buffer);
    if (header.getUint32(0, true) !== FRAME_MAGIC) {
      throw new Error('binary frame is not an OpenRoboxing frame');
    }
    const count = header.getUint16(8, true);
    if (count !== this.expectedBodies) {
      throw new Error(`frame carries ${count} bodies, the scene has ${this.expectedBodies}`);
    }
    this.tick = header.getUint32(4, true);

    const values = new Float32Array(buffer, FRAME_HEADER_BYTES, count * FLOATS_PER_BODY);
    for (let i = 0; i < count; i += 1) {
      const at = i * FLOATS_PER_BODY;
      const body = this.bodies[i];
      body.position.set(values[at], values[at + 1], values[at + 2]);
      setQuat(body, values[at + 3], values[at + 4], values[at + 5], values[at + 6]);
    }
  }

  /* ---- the fighter's own heading ---------------------------------------------------------------- */
  /* `spec/protocol.md` 0.6: a ghost's heading is *derived* — it faces the opponent (owner,
   * 2026-09-03) — and is never a field the JSON carries (`position` is a bare `{x, y}`). This is the
   * fallback for the frames before the opponent's body exists, and the fighter's own heading exists
   * in exactly one place: the pelvis's own streamed transform. So that is where this reads it from —
   * the yaw of `${seat}_pelvis`'s world quaternion,
   * by the *same* formula the host uses to read a live fighter's heading and a recorded take's
   * heading off a quaternion (`runtime/conventions.py::quat_wxyz_to_yaw`) — this is that formula
   * with the `wxyz` arguments relabelled to three.js's own `xyzw` quaternion fields, not a second
   * derivation of the convention.
   *
   * A preview only (`spec/protocol.md` "The shadow (0.4, ghost-only since 0.6)"): the client sends
   * nothing about heading, so an approximation that is off by the fighter's own pitch/roll costs
   * nothing but a slightly untrue-looking ghost, never a wrong commit.
   */
  fighterHeading(seat) {
    const index = this.bodyNames.indexOf(`${seat}_pelvis`);
    if (index < 0) return 0;
    const q = this.bodies[index].quaternion;   // three.js order: (x, y, z, w)
    return Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
  }

  /* Where a fighter stands, world `(x, y)` — the same pelvis `fighterHeading` reads, and the same
   * MuJoCo world frame the streamed floats are already in (`applyFrame` copies them straight in).
   * The ghost aims at this: a fighter always faces its opponent (owner, 2026-09-03), so the shadow
   * has to know where the opponent *is*, not merely which way this fighter is turned. `null` until
   * the scene description has arrived and a first frame has been applied.
   */
  fighterPosition(seat) {
    const index = this.bodyNames.indexOf(`${seat}_pelvis`);
    if (index < 0) return null;
    const at = this.bodies[index].position;
    return { x: at.x, y: at.y };
  }

  /* ---- the shadow ------------------------------------------------------------------------------ */
  showShadow(seat, x, y, heading, angles, standHeight, rejected = false) {
    if (!this.skeleton || !angles) { this.hideShadow(seat); return; }
    const shadow = this.shadowFor(seat);
    this.skeleton.solve(x, y, standHeight, heading, angles);
    for (let i = 0; i < shadow.parts.length; i += 1) {
      shadow.parts[i].position.copy(this.skeleton.position[i]);
      shadow.parts[i].quaternion.copy(this.skeleton.quaternion[i]);
    }
    /* A ghost beyond its combination's reach is drawn as rejected — the host will refuse it
       (`spec/protocol.md` §"Feasibility") — so the colour says so before the player commits. */
    shadow.material.color.setHex(rejected ? REJECTED_COLOUR : (SHADOW_COLOUR[seat] ?? 0xffffff));
    shadow.root.visible = true;
  }

  hideShadow(seat) {
    if (this.shadows[seat]) this.shadows[seat].root.visible = false;
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}
