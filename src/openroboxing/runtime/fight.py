"""Two fighters in the ring, under physics — the world a real match runs on (M3-T4; rewritten M6-T7).

This is the :class:`~openroboxing.runtime.match.MatchWorld` implementation that makes
``Match.run()`` an actual fight: the arena from `M3-T1`, a generator per fighter from `M3-T2`, the
GEAR-SONIC policy from `M1-T4`, and an :class:`~openroboxing.runtime.intents.IntentTimeline` per
fighter carrying what the player did.

Rewritten for `spec/intent.md` 3.0 — the approach is gone
-----------------------------------------------------------
1.0 through 2.2 built a commit out of *a placement and a final pose*, and roughly a third of this
module existed to run the walk that got a fighter there: an approach aimed a leg at a time
(``leg_target``), a travel direction re-derived from the generator's own buffer tail
(``travel_angle``), an arrival test (``has_arrived``) and a settle test (``has_settled``) that
decided when a struck pose was "over". **3.0 deletes all of it.** A commit now carries a recorded
:class:`~openroboxing.studio.combination_record.CombinationRecord` that already contains its own
footwork and a fixed duration (``record.duration_ticks``); ``runtime/warp.py`` and
``runtime/sequence.py`` turn that into a schedule of legs before this module ever sees it. What is
left for `fight.py` to do is exactly what an approach never needed: supply the timeline with
**where the fighter actually is** (:meth:`FightWorld._anchor_now`, passed as ``anchor=`` — see
`spec/intent.md` "Off-target execution"), and convert the combination's world-frame targets into the
frame the generator itself plans in (:meth:`FightWorld.to_generator_frame`, unchanged in kind from
1.0-2.2 even though what feeds it changed completely).

Where the rules are not
-----------------------
Nothing here knows about rounds, counts or knockouts. This world steps physics and records;
``runtime/match.py`` decides what any of it means. The split is what lets a recorded trace be replayed
through the same rules without a GPU.

Both fighters, then one step
----------------------------
Torques for **both** fighters are computed before any physics runs, because ``mj_step`` advances the
whole model. Computing red's torques, stepping, then computing blue's would give red a half-tick head
start on every exchange — small, invisible, and decisive over three rounds.

The PD loop itself runs at the *physics* rate, re-reading ``q`` and ``dq`` each substep against a
target held for the whole control tick. That is what the deploy stack does and what M1 measured
against.

One policy, two fighters
------------------------
:class:`~openroboxing.runtime.policy.GearSonicPolicy` holds only its two ONNX sessions and no
per-fighter state — the history lives in each fighter's ``ObservationBuilder`` — so one instance
serves both. The generators are emphatically *not* shared; see `runtime/pool.py`.

Conventions
-----------
- **Every index is derived by name** (`CLAUDE.md` invariant 4): joint qpos addresses, dof addresses
  and actuators are read out of the compiled model through ``red_``/``blue_``-prefixed names, never
  by assuming the arena lays two 36-value blocks out back to back.
- **A fighter always faces its opponent** (owner, 2026-09-03, reversing design D5). The bearing
  measured here is what the fighter looks at in every state — the opening stance, a running
  combination's legs, and a held final pose alike — and it is the *target frame's* heading too, so
  MotionBricks aims the move at the opponent rather than at wherever the recording happened to turn.
  Re-measured every tick, because the opponent moves while a combination runs. See
  :meth:`FightWorld.facing_angle`.
- **Ticks are 50 Hz** and match every other tick in the project.
- **World frame is MuJoCo's** — ``(x, y)`` on the ground plane. The anchor, the ghost and everything
  a client draws live here; the generator's own frame stops at ``runtime/generator.py``.
- **A fighter's heading is the full yaw-about-Z extraction**
  (:func:`~openroboxing.runtime.conventions.quat_wxyz_to_yaw`), not the pure-yaw shortcut
  :func:`apply_yaw` takes: a fighter mid-move is pitched and rolled, and reading ``(w, z)`` alone
  would report a heading that leans. The same formula reads a recorded take's heading
  (`studio/combination_record.py`'s ``_heading``) — one derivation, shared, per `CLAUDE.md`'s warning
  that most bugs here are convention bugs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

import numpy as np

from openroboxing.runtime.arena import (
    FIGHTERS,
    ArenaConfig,
    build_arena,
    reset_to_stance,
)
from openroboxing.runtime.bridge import compute_apply_delta_heading, encoder_input
from openroboxing.runtime.conventions import G1, quat_wxyz_to_yaw
from openroboxing.runtime.generator import GeneratorIntent
from openroboxing.runtime.intents import OPENING_STANCE_CONTEXT, IntentTimeline
from openroboxing.runtime.obs import ObservationBuilder, RobotState
from openroboxing.runtime.policy import (
    GearSonicPolicy,
    action_to_joint_target,
    effort_limits,
    pd_kd,
    pd_kp,
)
from openroboxing.runtime.pool import GeneratorPool
from openroboxing.runtime.reference import ReferenceStream
from openroboxing.spec.constants import (
    COMMIT_HORIZON_TICKS,
    HISTORY_LEN,
    MAX_OUTSTANDING_COMMITS,
    NUM_JOINTS,
    TICK_DT,
    TICK_HZ,
)

if TYPE_CHECKING:  # `runtime` does not import `studio` at module level - see generator.py's note.
    from openroboxing.studio.combination_record import CombinationRecord

#: The free joint upstream gives the G1, before the arena prefixes it.
ROOT_JOINT_NAME = "floating_base_joint"


class FightError(RuntimeError):
    """The fight world could not be built or stepped. Never recovered from silently."""


def apply_yaw(apply_delta_heading: np.ndarray) -> float:
    """The yaw of a heading quaternion.

    ``apply_delta_heading`` is a product of two yaw-only quaternions and so is yaw-only itself, which
    is why this can read the angle straight off ``(w, z)`` instead of decomposing.
    """
    q = np.asarray(apply_delta_heading, dtype=np.float64)
    if q.shape != (4,):
        raise FightError(f"expected a wxyz quaternion, got shape {q.shape}")
    return float(2.0 * np.arctan2(q[3], q[0]))


def generator_heading(world_angle: float, apply_delta_heading: np.ndarray) -> float:
    """A world-frame heading expressed in the generator's frame.

    The generator plans in its own frame and the bridge rotates the result into the world by
    ``apply_delta_heading`` (see :func:`~openroboxing.runtime.bridge.compute_apply_delta_heading`).
    Asking for a world heading therefore means asking for ``world - yaw(apply)``; passing the world
    angle straight through aims a fighter off by however far its clip happened to start.
    """
    return float(world_angle) - apply_yaw(apply_delta_heading)


# -- pilots ----------------------------------------------------------------------------------------
class Pilot(Protocol):
    """Whoever decides when a fighter commits.

    In `M4-T1` this is a human at a client and in `M5` an AI. Here it is a script, because M3-T4 is
    the match *loop* — what drives it belongs to the milestones that own it.
    """

    def reset(self) -> None:
        """Start of a round. Forget anything carried from the last one."""

    def act(self, timeline: IntentTimeline, tick: int) -> None:
        """Stage and commit, or do nothing. Called once per tick, before the fighter generates."""


class IdlePilot:
    """Commits nothing. The fighter stands in :data:`~openroboxing.runtime.intents.
    OPENING_STANCE_CONTEXT` for the whole round and never throws.

    Useful as a punchbag, and as the control case for anything that claims a commit caused something.
    """

    def reset(self) -> None:
        return

    def act(self, timeline: IntentTimeline, tick: int) -> None:
        return


class ScriptedPilot:
    """Commits from a fixed list of ``(tick, combination)`` or ``(tick, combination, ghost)``.

    ``ghost`` defaults to the world origin when omitted — a script that does not care where a
    combination lands, not a silent substitute for one that does (every entry that cares supplies its
    own three-tuple).

    A script that fires into a full queue raises. That is a bug in the script, not a dropped input:
    silently skipping it would make a match's commit log disagree with the script that produced it
    (`CLAUDE.md` invariant 5).
    """

    def __init__(
        self,
        script: Sequence[tuple[int, str]] | Sequence[tuple[int, str, tuple[float, float]]],
    ):
        self.script = [tuple(entry) for entry in script]
        for entry in self.script:
            if len(entry) not in (2, 3):
                raise FightError(f"script entry {entry!r} is not (tick, combination[, ghost])")
        self._fired: set[int] = set()

    def reset(self) -> None:
        self._fired = set()

    def act(self, timeline: IntentTimeline, tick: int) -> None:
        for index, entry in enumerate(self.script):
            if index in self._fired or entry[0] != tick:
                continue
            ghost = entry[2] if len(entry) == 3 else (0.0, 0.0)
            timeline.stage(combination=entry[1], ghost=ghost)
            timeline.commit(tick)
            self._fired.add(index)


# -- one fighter's control chain --------------------------------------------------------------------
class FighterRuntime:
    """One fighter inside the arena: its indices, its generator, its history, its timeline.

    Holds no simulator state of its own — ``qpos``/``qvel`` live in the shared ``MjData``, and this
    knows only *where* in them to look.
    """

    def __init__(
        self,
        name: str,
        model,
        generator,
        library: Mapping[str, CombinationRecord],
        pilot: Pilot,
        *,
        horizon_ticks: int = COMMIT_HORIZON_TICKS,
        max_outstanding: int = MAX_OUTSTANDING_COMMITS,
        require_admitted: bool = True,
    ) -> None:
        import mujoco

        self.name = name
        self.prefix = f"{name}_"
        self.generator = generator
        self.library = library
        self.pilot = pilot
        self.horizon_ticks = horizon_ticks
        self.max_outstanding = max_outstanding
        self.require_admitted = require_admitted

        self.stream = ReferenceStream(generator)
        self.builder = ObservationBuilder()
        self.timeline = self._new_timeline()
        self.last_action = np.zeros(NUM_JOINTS)
        self.apply_delta_heading = np.array([1.0, 0.0, 0.0, 0.0])

        self.root_qpos, self.root_dof = self._locate_root(mujoco, model)
        self.joint_ids, self.joint_qpos, self.joint_dof = self._locate_joints(mujoco, model)
        self.actuators = self._locate_actuators(mujoco, model)
        self.pelvis_body = self._body_id(mujoco, model, "pelvis")
        self.effort_limit = effort_limits(model, self.actuators, self.joint_ids)

    # -- name-derived indices ------------------------------------------------------------------------
    def _joint_id(self, mujoco, model, name: str) -> int:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}{name}")
        if joint < 0:
            raise FightError(f"joint {self.prefix}{name!r} is not in the arena")
        return joint

    def _body_id(self, mujoco, model, name: str) -> int:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}{name}")
        if body < 0:
            raise FightError(f"body {self.prefix}{name!r} is not in the arena")
        return body

    def _locate_root(self, mujoco, model) -> tuple[np.ndarray, np.ndarray]:
        """``(qpos indices, dof indices)`` of the free joint: 7 and 6 values."""
        joint = self._joint_id(mujoco, model, ROOT_JOINT_NAME)
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise FightError(f"{self.prefix}{ROOT_JOINT_NAME} is not a free joint")
        qpos = int(model.jnt_qposadr[joint])
        dof = int(model.jnt_dofadr[joint])
        return np.arange(qpos, qpos + 7), np.arange(dof, dof + 6)

    def _locate_joints(self, mujoco, model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(joint ids, qpos indices, dof indices)`` for the 29 hinges, in MuJoCo joint order."""
        ids = np.array([self._joint_id(mujoco, model, name) for name in G1.mujoco_joint_names])
        qpos = np.array([int(model.jnt_qposadr[j]) for j in ids])
        dof = np.array([int(model.jnt_dofadr[j]) for j in ids])
        if len(set(qpos.tolist())) != NUM_JOINTS:
            raise FightError(f"{self.name}: two joints share a qpos address")
        return ids, qpos, dof

    def _locate_actuators(self, mujoco, model) -> np.ndarray:
        """Actuator index for each of this fighter's joints, in MuJoCo joint order."""
        by_joint: dict[str, int] = {}
        for a in range(model.nu):
            joint_id = int(model.actuator_trnid[a, 0])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is None:
                raise FightError(f"actuator {a} drives an unnamed joint")
            if name.startswith(self.prefix):
                by_joint[name[len(self.prefix) :]] = a

        missing = [n for n in G1.mujoco_joint_names if n not in by_joint]
        if missing:
            raise FightError(f"{self.name}: joints with no actuator: {missing}")
        return np.array([by_joint[n] for n in G1.mujoco_joint_names], dtype=int)

    # -- per round ---------------------------------------------------------------------------------
    def _new_timeline(self) -> IntentTimeline:
        return IntentTimeline(
            self.library,
            horizon_ticks=self.horizon_ticks,
            max_outstanding=self.max_outstanding,
            require_admitted=self.require_admitted,
        )

    def reset_round(self) -> None:
        """Fresh history, fresh reference, fresh commit log. The generator is reseeded by the pool."""
        self.stream.reset()
        self.builder.reset()
        self.timeline = self._new_timeline()
        self.pilot.reset()
        self.last_action = np.zeros(NUM_JOINTS)

    # -- reading the simulator ----------------------------------------------------------------------
    def robot_state(self, data) -> RobotState:
        return RobotState(
            joint_pos=data.qpos[self.joint_qpos].copy(),
            joint_vel=data.qvel[self.joint_dof].copy(),
            base_quat=data.qpos[self.root_qpos[3:7]].copy(),
            base_ang_vel=data.qvel[self.root_dof[3:6]].copy(),
            last_action=self.last_action,
        )

    def base_quat(self, data) -> np.ndarray:
        return data.qpos[self.root_qpos[3:7]].copy()

    @property
    def apply_yaw(self) -> float:
        """Yaw of ``apply_delta_heading``, which is pure yaw by construction."""
        return apply_yaw(self.apply_delta_heading)


# -- the world -------------------------------------------------------------------------------------
class FightWorld:
    """Both fighters in the arena under physics. Satisfies ``match.MatchWorld``.

    Args:
        libraries: the combinations each fighter may commit, one library per fighter. `D6`
            (`spec/intent.md`) is a shared, un-loadout'd library the whole client pages through; that
            client/protocol phase has not landed, so a caller passes one library per fighter today —
            often the same object for both.
        pilots: what commits for each fighter. Defaults to :class:`IdlePilot` all round.
        match_seed: the one number a whole match reproduces from.
        config: ring geometry and timestep.
        policy: shared GEAR-SONIC. Built if not supplied.
        pool: the generators. Built if not supplied, seeded from ``match_seed``.
        horizon_ticks: the commit horizon floor, in ticks. Currently inert — see `spec/intent.md`
            "What actually sets the floor".
        max_outstanding: how many commits a fighter may have unfinished at once.
        require_admitted: whether library combinations must be admitted. A match always requires it;
            a smoke run of a draft combination may not.
    """

    def __init__(
        self,
        libraries: dict[str, Mapping[str, CombinationRecord]],
        *,
        pilots: dict[str, Pilot] | None = None,
        match_seed: int = 1234,
        config: ArenaConfig | None = None,
        policy: GearSonicPolicy | None = None,
        pool: GeneratorPool | None = None,
        horizon_ticks: int = COMMIT_HORIZON_TICKS,
        max_outstanding: int = MAX_OUTSTANDING_COMMITS,
        require_admitted: bool = True,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        missing = [f for f in FIGHTERS if f not in libraries]
        if missing:
            raise FightError(f"no combination library for {missing}; every fighter brings one")

        self.config = config or ArenaConfig()
        self.match_seed = match_seed
        self.model = build_arena(self.config)
        self.data = mujoco.MjData(self.model)
        self.substeps = max(1, round(TICK_DT / self.model.opt.timestep))
        #: Achieved drift speed (m/s) per commit, keyed by ``id(commit)`` — see :meth:`_record_drift`.
        #: A side table rather than a field on ``Commit`` because ``runtime/intents.py`` is upstream
        #: of this module and untouched by this rewrite (`spec/intent.md` 3.0's "Off-target
        #: execution": the number this module wants is not exposed by the timeline at all, since it
        #: is derived from the *same* anchor/ghost/record inputs the timeline already re-warped from,
        #: recomputed here rather than duplicated into ``intents.py``).
        self._drift_speed_m_s: dict[int, float] = {}

        self.policy = policy or GearSonicPolicy()
        self.pool = pool or GeneratorPool(match_seed=match_seed)
        if not self.pool.independence_holds():
            raise FightError(
                "the generator pool shares state between fighters; one fighter's commits could "
                "steer the other"
            )

        pilots = pilots or {}
        self.fighters: dict[str, FighterRuntime] = {
            name: FighterRuntime(
                name,
                self.model,
                self.pool[name],
                libraries[name],
                pilots.get(name) or IdlePilot(),
                horizon_ticks=horizon_ticks,
                max_outstanding=max_outstanding,
                require_admitted=require_admitted,
            )
            for name in FIGHTERS
        }

        self._kp = pd_kp(G1, "mujoco")
        self._kd = pd_kd(G1, "mujoco")
        self._round = -1

    # -- MatchWorld ---------------------------------------------------------------------------------
    def reset_round(self, index: int) -> None:
        """Both fighters back in stance, reseeded for this round, history primed for tick 0."""
        if index < 0:
            raise FightError(f"round index must not be negative, got {index}")

        reset_to_stance(self.model, self.data, self.config)
        self.pool.reset(round_index=index)
        self._round = index
        # Last round's drift readings are keyed by commit objects that no longer exist once
        # `FighterRuntime.reset_round` rebuilds each timeline below — carrying them forward would
        # either leak memory over a long match or, worse, collide if Python reuses an `id()`.
        self._drift_speed_m_s.clear()

        for fighter in self.fighters.values():
            fighter.reset_round()
            # The ambient intent, not the timeline's: no commit can be executing at tick 0, and a
            # fighter that has committed nothing yet stands in the opening stance
            # (`intents.OPENING_STANCE_CONTEXT`) rather than any travelling style — there is no
            # ambient "walking" context left once the approach is gone.
            ambient = GeneratorIntent(style=OPENING_STANCE_CONTEXT)
            fighter.stream.ensure(lambda _tick, _i=ambient: _i, tick=0)
            fighter.apply_delta_heading = compute_apply_delta_heading(
                init_base_quat_wxyz=fighter.base_quat(self.data),
                init_ref_root_quat_wxyz=fighter.stream.motion[0, 3:7],
            )
            for _ in range(HISTORY_LEN):
                fighter.builder.push(fighter.robot_state(self.data))

    def step(self, tick: int) -> None:
        """One 50 Hz control tick for both fighters, then the physics they share."""
        if self._round < 0:
            raise FightError("step() before reset_round(); the round has not started")

        targets: dict[str, np.ndarray] = {}
        for fighter in self.fighters.values():
            fighter.pilot.act(fighter.timeline, tick)
            # The bearing is captured now and reused for every frame this fill produces, which is
            # the same staleness a leg's target position already carries: both are read at the tick
            # the generator plans on, and a fill covers at most one replan interval. Since the
            # owner's 2026-09-03 rule it applies to *every* state, not just the hold — a running
            # combination faces the opponent too — so it is re-read on the next fill rather than
            # remembered anywhere.
            bearing = self.facing_angle(fighter.name)
            fighter.stream.ensure(
                lambda play_tick, _f=fighter, _b=bearing: self._intent_at(_f, play_tick, _b),
                tick,
            )
            fighter.stream.require(tick)

            fighter.builder.push(fighter.robot_state(self.data))
            enc = encoder_input(
                tick=tick,
                motion_50hz=fighter.stream.motion,
                base_quat_wxyz=fighter.base_quat(self.data),
                motion_joint_vel_50hz=fighter.stream.velocities,
                apply_delta_heading=fighter.apply_delta_heading,
            )
            action, _ = self.policy.step(enc, fighter.builder)
            fighter.last_action = action
            targets[fighter.name] = action_to_joint_target(action)

        self._step_physics(targets)

    def observe(self, tracker, trace, tick: int) -> None:
        tracker.observe(self.model, self.data, tick)
        trace.observe(self.model, self.data, tick)

    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def commits(self) -> list[dict]:
        """Every commit issued this round, as `spec/match_record.md`'s ``CommitEvent``.

        The commit log is per round because :meth:`reset_round` builds a fresh timeline: a round's
        record must hold that round's commits and no others.

        ``drift_speed_m_s`` is ``None`` for a commit that has not started yet (nothing to measure)
        and a finite number from the instant it does — see :meth:`_record_drift`. It is present for
        every *started* commit unconditionally, not only the ones that drifted hard, because a move
        that tracked its ghost cleanly at a low drift speed is exactly the baseline a high one is
        read against (`spec/intent.md` "The achieved drift speed is recorded in the match record").
        """
        events: list[dict] = []
        for name, fighter in self.fighters.items():
            for commit in fighter.timeline.commits:
                events.append(
                    {
                        "fighter": name,
                        "combination": commit.record.name,
                        "ghost": list(commit.ghost),
                        "issued_at": commit.issued_at,
                        "commit_at": commit.commit_at,
                        "end_tick": commit.end_tick,
                        "drift_speed_m_s": self._drift_speed_m_s.get(id(commit)),
                    }
                )
        return sorted(events, key=lambda e: (e["issued_at"], e["fighter"]))

    # -- where things are ----------------------------------------------------------------------------
    def separation_m(self) -> float:
        """Distance between the two pelvises on the ground plane.

        Sent to clients (`spec/protocol.md`) because range is not secret — both fighters can see how
        far apart they are by looking, and a player who cannot judge distance cannot manage it.
        """
        positions = [self.data.xpos[f.pelvis_body][:2] for f in self.fighters.values()]
        return float(np.linalg.norm(positions[0] - positions[1]))

    def opponent(self, fighter: str) -> str:
        others = [f for f in FIGHTERS if f != fighter]
        if len(others) != 1:
            raise FightError(f"{fighter!r} has {len(others)} opponents; a bout is two fighters")
        return others[0]

    def _anchor_now(self, fighter: FighterRuntime) -> tuple[tuple[float, float], float]:
        """Where ``fighter`` actually is right now: world ``(x, y)`` and yaw, read off live physics.

        Passed as :meth:`~openroboxing.runtime.intents.IntentTimeline.generator_intent`'s ``anchor``
        argument. The timeline calls it **exactly once per commit**, at the tick that commit starts,
        to warp the combination into place from wherever the fighter genuinely stands rather than
        wherever it was aimed to be (`spec/intent.md` "Off-target execution") — this method is the
        one piece of geometry that makes that possible, since the timeline itself is deliberately
        ignorant of where anyone is (`runtime/intents.py`'s module docstring).

        Position is the pelvis, not the free joint's own translation — the same body every other
        world-frame reading in this module (:meth:`separation_m`, :meth:`facing_angle`) uses, so a
        combination starts from the same point the client already renders. Heading is the full
        yaw-about-Z extraction (:func:`~openroboxing.runtime.conventions.quat_wxyz_to_yaw`), not the
        yaw-only shortcut :func:`apply_yaw` takes, because a fighter is not guaranteed to be upright
        the instant a queued commit becomes current.
        """
        position = self.data.xpos[fighter.pelvis_body][:2]
        heading = quat_wxyz_to_yaw(self.data.qpos[fighter.root_qpos[3:7]])
        return (float(position[0]), float(position[1])), heading

    def to_generator_frame(
        self, fighter: str, position: tuple[float, float], heading: float
    ) -> tuple[tuple[float, float], float]:
        """A world position and heading, expressed in the frame the generator plans in::

            target_gen = context_gen + R(-yaw) . (target_world - robot_world)

        In words: *from where the generator is, travel the vector from the robot to the target.*
        Unchanged in kind since 1.0-2.2, even though what feeds it changed completely: a warped
        combination's leg targets are world-frame positions exactly like a 1.0-2.2 placement was, and
        the generator still does not plan in that frame (`spec/intent.md` "Coordinates, unchanged").

        **The conversion is not optional and omitting it is not obviously wrong.** The generator
        plans in its own frame — its context is its own previous output, starting near the origin —
        while a leg target is a point on a ring the player can see. Red starts at ``x = -1.2`` yawed
        one way and blue at ``+1.2`` yawed the other, so a raw world target lands somewhere else
        entirely for each of them and the fighter is driven a plausible distance in the wrong
        direction. :meth:`facing_angle` has always done the same conversion for angles via
        :func:`generator_heading`; this is its counterpart for positions, and both use the same yaw.

        Two anchors are defensible and only one arrives
        -----------------------------------------------
        The vector is measured **from the robot**, and it is applied **from the generator's buffer
        tail** — the position it will have planned up to, a lookahead in front of the robot. Those
        two choices are what made a 1.0-2.2 approach converge, and the same reasoning still holds for
        a combination's own drift: MotionBricks is kinematic and arrives at a target every time, so
        anchoring the vector on the generator's own belief about itself (rather than on the live
        robot) would have a fighter conclude it has already arrived and stand still, exactly the 1.0
        defect `spec/intent.md`'s changelog records. Re-deriving the vector from the robot every tick
        keeps a leg's target honest for its whole duration, the same integration that made an
        approach close.
        """
        me = self.fighters[fighter]
        here = self.data.xpos[me.pelvis_body][:2]
        context = me.generator.context_qpos()[-1][:2]

        delta = np.asarray(position, dtype=np.float64) - here
        cos, sin = np.cos(-me.apply_yaw), np.sin(-me.apply_yaw)
        rotated = np.array([cos * delta[0] - sin * delta[1], sin * delta[0] + cos * delta[1]])
        moved = context + rotated
        return (float(moved[0]), float(moved[1])), generator_heading(heading, me.apply_delta_heading)

    def _record_drift(self, commit, anchor: tuple[tuple[float, float], float]) -> None:
        """Compute and store the drift speed a commit's warp implied, the tick it started.

        Owner decision, 2026-08-28 (`spec/intent.md` "Off-target execution" / "The achieved drift
        recorded"): a queued combination that starts off-target still reaches its ghost, running
        whatever drift that needs, and the drift speed that implies must be visible in the match
        record rather than silent. ``runtime/warp.py::warp()`` computes exactly this number
        internally but only when checking it against a ceiling (``speed_ceiling is not None``); the
        execution-time call the timeline makes passes ``speed_ceiling=None`` (nothing clamps, nothing
        raises), which skips that branch and returns only the legs. Rather than teach ``warp()`` or
        ``intents.py`` to expose it — both out of scope for this rewrite — this recomputes the same
        formula from the same inputs, all of which are already public on ``commit``:

            drift_speed = |ghost - anchor_position - R(anchor_heading) . recorded_displacement|
                          / (duration_ticks / TICK_HZ)

        Called from :meth:`_intent_at` only on the tick a commit actually starts (guarded by the
        caller having seen its own ``anchor`` callable invoked), so this runs at most once per
        commit — matching the one time the timeline itself samples the anchor.
        """
        (ax, ay), heading = anchor
        dx, dy = commit.record.recorded_displacement
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        rotated = (cos_h * dx - sin_h * dy, sin_h * dx + cos_h * dy)
        residual = (
            commit.ghost[0] - ax - rotated[0],
            commit.ghost[1] - ay - rotated[1],
        )
        duration_s = commit.record.duration_ticks / TICK_HZ
        self._drift_speed_m_s[id(commit)] = math.hypot(*residual) / duration_s

    def _intent_at(self, fighter: FighterRuntime, play_tick: int, bearing: float) -> GeneratorIntent:
        """This fighter's control signals for the tick a generated frame will be played at.

        Two frames meet here and nowhere else. The timeline deals only in **world** coordinates — it
        is what the player and the client see — and the generator plans in its own, so the conversion
        happens at this boundary.

        The ``anchor`` passed to the timeline is wrapped so this method can tell, after the call,
        whether a commit just started: :meth:`~openroboxing.runtime.intents.IntentTimeline.
        generator_intent` calls ``anchor`` exactly once per commit, at the tick it starts, so an
        empty ``seen`` list after the call means no commit started this tick, and a non-empty one
        means exactly one did — the drift speed that starting implied is recorded right there.
        """
        seen: list[tuple[tuple[float, float], float]] = []

        def _anchor(_f=fighter) -> tuple[tuple[float, float], float]:
            value = self._anchor_now(_f)
            seen.append(value)
            return value

        intent = fighter.timeline.generator_intent(play_tick, facing_angle=bearing, anchor=_anchor)

        if seen:
            starting = next(
                (c for c in fighter.timeline.commits if c.commit_at == play_tick), None
            )
            if starting is not None:
                self._record_drift(starting, seen[0])

        if intent.target_position is None:
            # The opening stance carries no target, but its facing is still a world angle: convert
            # it here or a fighter with nothing committed stands facing wherever its clip started.
            return replace(
                intent,
                facing_angle=generator_heading(intent.facing_angle, fighter.apply_delta_heading),
            )

        position, heading = self.to_generator_frame(
            fighter.name, intent.target_position, intent.target_heading
        )
        return replace(
            intent,
            target_position=position,
            target_heading=heading,
            facing_angle=generator_heading(intent.facing_angle, fighter.apply_delta_heading),
            movement_angle=generator_heading(intent.movement_angle, fighter.apply_delta_heading),
        )

    def facing_angle(self, fighter: str) -> float:
        """The **world-frame** heading that points a fighter at its opponent.

        World frame, not the generator's, even though the generator is the only consumer: everything
        the timeline and the warp deal in is world-frame, and :meth:`_intent_at` is the one place the
        two frames meet (its docstring). Converting here as well would convert twice — a leg's
        ``target_heading`` is *this* angle since the owner's 2026-09-03 rule, and it goes through
        :meth:`to_generator_frame` like every other world quantity.

        The conversion is not optional at that boundary, only relocated: ``apply_delta_heading`` is
        the fixed yaw between the two frames, and passing a world angle straight to MotionBricks aims
        a fighter off by however far its clip happened to start.
        """
        me = self.fighters[fighter]
        them = self.fighters[self.opponent(fighter)]
        here = self.data.xpos[me.pelvis_body]
        there = self.data.xpos[them.pelvis_body]
        return float(np.arctan2(there[1] - here[1], there[0] - here[0]))

    # -- physics -------------------------------------------------------------------------------------
    def _step_physics(self, targets: dict[str, np.ndarray]) -> None:
        """PD both fighters to their targets at the physics rate, stepping the shared model once."""
        for _ in range(self.substeps):
            for fighter in self.fighters.values():
                q = self.data.qpos[fighter.joint_qpos]
                dq = self.data.qvel[fighter.joint_dof]
                tau = self._kp * (targets[fighter.name] - q) - self._kd * dq
                np.clip(tau, -fighter.effort_limit, fighter.effort_limit, out=tau)
                self.data.ctrl[fighter.actuators] = tau
            self._mujoco.mj_step(self.model, self.data)
