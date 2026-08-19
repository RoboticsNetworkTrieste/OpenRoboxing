"""The ring: two G1s, ropes, padded gloves, cameras (M3-T1).

Composed at run time with :class:`mujoco.MjSpec` rather than written as a static MJCF. The robot
model is upstream and read-only (`CLAUDE.md`), so a checked-in arena file would either duplicate it
or include it by a relative path that breaks when upstream moves. Building the scene means the ring
is parameterised — size, rope heights, glove padding are all things a playtest will want to change,
and `M4-T4` is where they get changed.

Conventions
-----------
- **The ring is centred on the origin**, its axes aligned with the world: ``x`` is the line between
  the fighters, ``y`` is across it. Corner posts sit at ``(±half, ±half)``.
- **Fighters face each other along x.** Red is at ``-x`` facing ``+x``; blue is at ``+x`` facing
  ``-x``, which is a 180° yaw. A fighter's own forward is always its ``+x``, so blue's world motion
  is mirrored — anything reading positions in world frame must account for that, and
  :func:`fighter_forward` exists so nothing has to hard-code it.
- **Body names are prefixed** ``red_`` / ``blue_``. That prefix is the only thing distinguishing two
  otherwise identical models, so hit attribution reads it rather than guessing by position.
- **The robot is** :data:`~openroboxing.paths.G1_29DOF_SIM_XML`, not its similarly-named sibling.
  Only that file carries rotor armature and joint damping, and a stiff PD controller without them
  collapses a fighter in half a second while appearing to be driven correctly. The trap and the
  bisect that found it are in `spec/upstream_notes.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openroboxing.paths import G1_29DOF_SIM_XML
from openroboxing.spec.constants import RING_SIZE_M

#: The two fighters, in order. Index 0 is red.
FIGHTERS: tuple[str, ...] = ("red", "blue")

#: Bodies that get a padded glove geom. The policy was trained penalising contact outside feet,
#: hands and elbows, so these are also the only bodies it expects to strike with.
GLOVE_BODIES: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")


class ArenaError(RuntimeError):
    """The arena could not be built. Never recovered from silently."""


@dataclass(frozen=True)
class ArenaConfig:
    """Ring geometry and the physics it is compiled with.

    Defaults are competition dimensions, not invented ones: a World Boxing / AIBA elite ring is
    **4.90 m square inside the ropes** with **four ropes** at 40, 70, 100 and 130 cm. The G1 stands
    about 1.3 m, so a full-size ring plays *larger* for these fighters than for people — whether to
    scale it down is a feel question for `M4-T4`, which is why it is a parameter and not a constant.
    """

    #: Side length inside the ropes, metres. `spec/constants.RING_SIZE_M`, shared so that anything
    #: deriving a duration from the ring's size cannot drift away from its geometry.
    ring_size: float = RING_SIZE_M
    #: Rope heights above the canvas, metres.
    rope_heights: tuple[float, ...] = (0.40, 0.70, 1.00, 1.30)
    rope_radius: float = 0.02
    post_radius: float = 0.06
    #: How far from the centre each fighter starts, metres. Two arm's reach apart (0.38 m each,
    #: measured in `studio/pose_ik.py`), so neither can land without closing.
    start_separation: float = 1.20
    #: Standing root height, matching the pose library's render height.
    start_height: float = 0.793
    #: Glove radius, metres. A 10 oz competition glove is about 0.09 m across the knuckles; the G1's
    #: hand is smaller, so this pads it out to roughly a glove's contact profile.
    glove_radius: float = 0.055
    #: Contact softness for gloves: `solref` time constant and damping ratio. A longer time constant
    #: than the default 0.02 spreads an impact over more substeps, which is what padding does.
    glove_solref: tuple[float, float] = (0.04, 1.0)
    #: Physics timestep. 0.001 is M1-T7's recommendation — the finest that keeps a 2x real-time
    #: margin, chosen for contact fidelity (`docs/perf/m1_mujoco.md`).
    timestep: float = 0.001
    #: Sliding friction between a fighter and everything else. Not invented: this is the value
    #: ``scene_29dof.xml`` sets, and therefore the grip every M1 measurement — including the one that
    #: showed a fighter walking for 30 s without falling — was made at.
    friction: float = 0.5

    def __post_init__(self) -> None:
        if self.ring_size <= 2 * self.start_separation:
            raise ArenaError(
                f"ring_size {self.ring_size} m cannot hold fighters {self.start_separation} m "
                "either side of centre"
            )
        if not self.rope_heights:
            raise ArenaError("a ring needs at least one rope")


def fighter_forward(fighter: str) -> np.ndarray:
    """The world-frame direction a fighter faces at the start. Red looks ``+x``, blue ``-x``."""
    if fighter not in FIGHTERS:
        raise ArenaError(f"unknown fighter {fighter!r}; expected one of {FIGHTERS}")
    return np.array([1.0, 0.0, 0.0]) if fighter == "red" else np.array([-1.0, 0.0, 0.0])


def start_pose(fighter: str, config: ArenaConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Where a fighter starts: ``(position, quaternion wxyz)`` in world frame."""
    config = config or ArenaConfig()
    sign = 1.0 if fighter == "red" else -1.0
    position = np.array([-sign * config.start_separation, 0.0, config.start_height])
    # Blue is yawed 180 deg so the pair face each other; quaternion (0,0,0,1) in wxyz.
    quat = np.array([1.0, 0.0, 0.0, 0.0]) if fighter == "red" else np.array([0.0, 0.0, 0.0, 1.0])
    return position, quat


def _add_ring(spec, config: ArenaConfig) -> None:
    """Canvas, four corner posts and the ropes. Ropes collide; posts collide; the canvas is a plane."""
    import mujoco

    half = config.ring_size / 2.0
    world = spec.worldbody

    canvas = world.add_geom()
    canvas.name = "canvas"
    canvas.type = mujoco.mjtGeom.mjGEOM_PLANE
    canvas.size = [half + 0.5, half + 0.5, 0.1]
    canvas.rgba = [0.32, 0.36, 0.45, 1.0]

    for xi, x in ((0, -half), (1, half)):
        for yi, y in ((0, -half), (1, half)):
            post = world.add_geom()
            post.name = f"post_{xi}{yi}"
            post.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            post.fromto = [x, y, 0.0, x, y, max(config.rope_heights) + 0.15]
            post.size = [config.post_radius, 0, 0]
            post.rgba = [0.55, 0.12, 0.12, 1.0]

    # Ropes as capsules between posts: four sides per height.
    for index, height in enumerate(config.rope_heights):
        for side, (a, b) in enumerate(
            (
                ((-half, -half), (half, -half)),
                ((half, -half), (half, half)),
                ((half, half), (-half, half)),
                ((-half, half), (-half, -half)),
            )
        ):
            rope = world.add_geom()
            rope.name = f"rope_{index}_{side}"
            rope.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            rope.fromto = [a[0], a[1], height, b[0], b[1], height]
            rope.size = [config.rope_radius, 0, 0]
            rope.rgba = [0.9, 0.9, 0.85, 1.0]


def _add_lights(spec, config: ArenaConfig) -> None:
    """Light the ring.

    A composed :class:`mujoco.MjSpec` starts with **no lights at all**, and MuJoCo falls back to a
    headlight — which renders a fight as two grey figures in a black void, with no cast shadows to
    say where a fighter is standing. Only visible once something rendered a replay (`M3-T5`).

    Three directional lights: a key above and in front, matching the broadcast camera's side, a
    fill from the opposite corner so the far fighter is not a silhouette, and one straight down so
    the canvas reads as a floor.
    """
    import mujoco

    half = config.ring_size / 2.0

    for name, pos, direction, diffuse, castshadow in (
        ("key", [-half, -half, 3.2], [0.4, 0.4, -1.0], 0.55, True),
        ("fill", [half, half, 3.0], [-0.4, -0.4, -1.0], 0.35, False),
        ("top", [0.0, 0.0, 4.0], [0.0, 0.0, -1.0], 0.30, False),
    ):
        light = spec.worldbody.add_light()
        light.name = name
        light.pos = pos
        light.dir = direction
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.diffuse = [diffuse] * 3
        light.ambient = [0.20, 0.20, 0.22]
        light.castshadow = castshadow


def _add_cameras(spec, config: ArenaConfig) -> None:
    """A broadcast camera side-on and one overhead.

    Both aim with ``mode=targetbody`` at a marker body at ring centre rather than with a hand-written
    orientation: a camera that tracks a target keeps framing the fight if the ring is resized, and
    ``MjsCamera`` has no ``xyaxes`` to write anyway.
    """
    import mujoco

    half = config.ring_size / 2.0

    centre = spec.worldbody.add_body()
    centre.name = "ring_centre"
    centre.pos = [0.0, 0.0, 1.0]

    broadcast = spec.worldbody.add_camera()
    broadcast.name = "broadcast"
    broadcast.pos = [0.0, -(half + 2.2), 1.9]
    broadcast.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    broadcast.targetbody = "ring_centre"

    overhead = spec.worldbody.add_camera()
    overhead.name = "overhead"
    overhead.pos = [0.0, -0.01, half + 2.5]
    overhead.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    overhead.targetbody = "ring_centre"


def _pad_gloves(spec, prefix: str, config: ArenaConfig) -> None:
    """Add a soft sphere to each hand.

    Additive: the hand's own geoms stay, so the arm still collides normally. The glove is what makes
    a punch a punch rather than a knuckle strike, and its softness is what keeps the impulse from
    spiking through a 1 ms timestep.
    """
    import mujoco

    for body_name in GLOVE_BODIES:
        body = spec.body(f"{prefix}{body_name}")
        if body is None:
            raise ArenaError(
                f"body {prefix}{body_name!r} is not in the composed model; GLOVE_BODIES is stale"
            )
        glove = body.add_geom()
        glove.name = f"{prefix}glove_{body_name.split('_')[0]}"
        glove.type = mujoco.mjtGeom.mjGEOM_SPHERE
        glove.size = [config.glove_radius, 0, 0]
        glove.rgba = [0.75, 0.10, 0.10, 0.85]
        glove.solref = list(config.glove_solref)
        # Gloves are padding, not ballast: keep the arm's inertia as the robot model defines it.
        glove.mass = 0.0


def _drop_duplicate_floor(spec, prefix: str) -> int:
    """Remove any ground plane the robot model brings with it. Returns how many were removed.

    A robot model may carry its own ground plane so it can be loaded standalone. Attaching two of
    them puts three coincident planes in the ring, which at best wastes contacts and at worst — if
    the attachment frame is ever lifted — buries a fighter inside its own floor. The arena owns the
    canvas; the robots do not get one each.

    Found by geom **type**, not by name: the sim model leaves its geoms unnamed, so a name lookup
    silently finds nothing and reports success.
    """
    import mujoco

    planes = [
        geom
        for geom in spec.geoms
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE and (geom.name or "").startswith(prefix)
    ]
    for plane in planes:
        spec.delete(plane)
    return len(planes)


def _set_friction(spec, friction: float) -> None:
    """Give **every** geom in the ring the same sliding friction.

    Set explicitly because it is a *scene* property the robot file does not carry:
    ``scene_29dof.xml`` supplies it through a top-level default that covers the floor as well as the
    robot, and the arena composes the robot without that scene.

    It has to be every geom, not just the fighters'. MuJoCo combines a contact pair's friction by
    taking the **maximum**, so leaving the canvas at its default would run every footstep at that
    default however carefully the fighters were set — a silent divergence from the configuration all
    of M1 was measured at.
    """
    for geom in spec.geoms:
        current = list(geom.friction)
        geom.friction = [friction, current[1], current[2]]


def _exclude_self_glove_contacts(spec, prefix: str) -> None:
    """A fighter's glove cannot hit that fighter.

    The glove is a sphere large enough to model a real one, so it overlaps the forearm it sits on and
    would push against its own arm forever — measured at 27 mm of standing penetration before this.
    Excluding the pair at body level says the thing we actually mean: a glove is for hitting the
    opponent. The arm's own geoms are untouched, so the *limb* still collides normally.
    """
    hands = [f"{prefix}{body}" for body in GLOVE_BODIES]
    own = [b.name for b in spec.bodies if b.name and b.name.startswith(prefix)]
    for hand in hands:
        for body in own:
            if body == hand:
                continue  # geoms on one body never collide with each other anyway
            exclude = spec.add_exclude()
            exclude.bodyname1 = hand
            exclude.bodyname2 = body


def build_arena(config: ArenaConfig | None = None):
    """Compile the ring with both fighters in it. Returns a ``mujoco.MjModel``."""
    import mujoco

    config = config or ArenaConfig()
    if not G1_29DOF_SIM_XML.exists():
        raise ArenaError(f"robot model not found: {G1_29DOF_SIM_XML}")

    spec = mujoco.MjSpec()
    spec.option.timestep = config.timestep
    _add_ring(spec, config)
    _add_lights(spec, config)
    _add_cameras(spec, config)

    for fighter in FIGHTERS:
        robot = mujoco.MjSpec.from_file(str(G1_29DOF_SIM_XML))
        frame = spec.worldbody.add_frame()
        _, quat = start_pose(fighter, config)
        # Attached at the origin, deliberately. A free joint's qpos is absolute, so where the fighter
        # *starts* is `reset_to_stance`'s business; lifting the attachment frame instead would carry
        # the robot model's own floor plane up with it and bury the fighter inside it.
        frame.pos = [0.0, 0.0, 0.0]
        frame.quat = list(quat)
        spec.attach(robot, prefix=f"{fighter}_", frame=frame)

    for fighter in FIGHTERS:
        _drop_duplicate_floor(spec, f"{fighter}_")
        _pad_gloves(spec, f"{fighter}_", config)
        _exclude_self_glove_contacts(spec, f"{fighter}_")

    _set_friction(spec, config.friction)

    try:
        model = spec.compile()
    except Exception as exc:
        raise ArenaError(f"the arena did not compile: {exc}") from exc

    expected = len(FIGHTERS) * (7 + 29)
    if model.nq != expected:
        raise ArenaError(f"arena has nq={model.nq}, expected {expected} for {len(FIGHTERS)} G1s")
    return model


def reset_to_stance(model, data, config: ArenaConfig | None = None) -> None:
    """Put both fighters in their starting stance, at rest.

    The attachment frame positions the *model*, but ``qpos`` for a free joint is absolute, so a plain
    ``mj_resetData`` drops both fighters at the origin on top of each other. This writes the stance
    in explicitly.
    """
    import mujoco

    from openroboxing.runtime.conventions import G1
    from openroboxing.runtime.obs import default_angles

    config = config or ArenaConfig()
    mujoco.mj_resetData(model, data)
    defaults = default_angles(G1, "mujoco")

    for index, fighter in enumerate(FIGHTERS):
        base = index * (7 + 29)
        position, quat = start_pose(fighter, config)
        data.qpos[base : base + 3] = position
        data.qpos[base + 3 : base + 7] = quat
        data.qpos[base + 7 : base + 36] = defaults

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def fighter_qpos_slice(index: int) -> slice:
    """Where a fighter's 36 qpos values live in the arena's qpos."""
    if not 0 <= index < len(FIGHTERS):
        raise ArenaError(f"fighter index {index} outside 0..{len(FIGHTERS) - 1}")
    base = index * (7 + 29)
    return slice(base, base + 36)
