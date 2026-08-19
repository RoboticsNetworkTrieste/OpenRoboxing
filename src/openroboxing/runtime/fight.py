"""Two fighters in the ring, under physics — the world a real match runs on (M3-T4).

This is the :class:`~openroboxing.runtime.match.MatchWorld` implementation that makes
``Match.run()`` an actual fight: the arena from `M3-T1`, a generator per fighter from `M3-T2`, the
GEAR-SONIC policy from `M1-T4`, and an :class:`~openroboxing.runtime.intents.IntentTimeline` per
fighter carrying what the player did.

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
- **A fighter faces its opponent while holding.** A committed move carries its own heading in its
  placement; the bearing only decides which way a fighter looks when its queue has run dry. See
  :meth:`FightWorld.facing_angle`.
- **Ticks are 50 Hz** and match every other tick in the project.
- **World frame is MuJoCo's** — ``(x, y)`` on the ground plane. Placements, the shadow's anchor and
  everything a client draws live here; the generator's own frame stops at ``runtime/generator.py``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Protocol, Sequence

import numpy as np

from openroboxing.runtime.arena import (
    FIGHTERS,
    ArenaConfig,
    build_arena,
    reset_to_stance,
)
from openroboxing.runtime.bridge import compute_apply_delta_heading, encoder_input
from openroboxing.runtime.conventions import G1
from openroboxing.runtime.generator import GeneratorIntent
from openroboxing.runtime.intents import (
    TRAVEL_CONTEXT,
    IntentTimeline,
    Loadout,
    Placement,
    approach_timeout_ticks,
)
from openroboxing.runtime.obs import ObservationBuilder, RobotState
from openroboxing.runtime.policy import (
    GearSonicPolicy,
    action_to_joint_target,
    effort_limits,
    pd_kd,
    pd_kp,
)
from openroboxing.runtime.pool import GeneratorPool
from openroboxing.runtime.reference import REPLAN_DT, ReferenceStream
from openroboxing.spec.constants import (
    APPROACH_LEG_M,
    ARRIVAL_RADIUS_M,
    COMMIT_HORIZON_TICKS,
    HISTORY_LEN,
    MAX_OUTSTANDING_COMMITS,
    NUM_JOINTS,
    POSE_SETTLE_IMPROVEMENT_RAD,
    TICK_DT,
    TICK_HZ,
)

#: The free joint upstream gives the G1, before the arena prefixes it.
ROOT_JOINT_NAME = "floating_base_joint"

#: The window a settle test compares against itself, in control ticks: one replan interval. The
#: reference is rebuilt at that cadence, so anything shorter measures the inside of a plan rather
#: than the move.
SETTLE_WINDOW_TICKS = int(round(REPLAN_DT * TICK_HZ))


@dataclass
class _Settling:
    """One commit's pose-error trace, and the target it is measured against."""

    commit: object
    target: np.ndarray
    trace: deque


#: Default fighting context: the clip a fighter travels in. One name, declared beside the opening
#: stance it is paired with — see :data:`~openroboxing.runtime.intents.TRAVEL_CONTEXT` for why it is
#: the release's ``walk`` and not the boxing loop it used to be.
DEFAULT_CONTEXT = TRAVEL_CONTEXT


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
    """Commits nothing. The fighter moves in its context and never throws.

    Useful as a punchbag, and as the control case for anything that claims a commit caused something.
    """

    def reset(self) -> None:
        return

    def act(self, timeline: IntentTimeline, tick: int) -> None:
        return


class ScriptedPilot:
    """Commits from a fixed list of ``(tick, slot)``, optionally with a placement.

    A script that fires into a full queue raises. That is a bug in the script, not a dropped input:
    silently skipping it would make a match's commit log disagree with the script that produced it
    (`CLAUDE.md` invariant 5).
    """

    def __init__(self, script: Sequence[tuple[int, str]] | Sequence[tuple[int, str, Placement]]):
        self.script = [tuple(entry) for entry in script]
        for entry in self.script:
            if len(entry) not in (2, 3):
                raise FightError(f"script entry {entry!r} is not (tick, slot[, placement])")
        self._fired: set[int] = set()

    def reset(self) -> None:
        self._fired = set()

    def act(self, timeline: IntentTimeline, tick: int) -> None:
        for index, entry in enumerate(self.script):
            if index in self._fired or entry[0] != tick:
                continue
            placement = entry[2] if len(entry) == 3 else None
            timeline.stage(pose_slot=entry[1], placement=placement)
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
        loadout: Loadout,
        pilot: Pilot,
        *,
        context: str = DEFAULT_CONTEXT,
        horizon_ticks: int = COMMIT_HORIZON_TICKS,
        max_outstanding: int = MAX_OUTSTANDING_COMMITS,
        approach_timeout: int | None = None,
        require_admitted: bool = True,
    ) -> None:
        import mujoco

        self.name = name
        self.prefix = f"{name}_"
        self.generator = generator
        self.loadout = loadout
        self.pilot = pilot
        self.context = context
        self.horizon_ticks = horizon_ticks
        self.max_outstanding = max_outstanding
        self.approach_timeout = approach_timeout
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
            self.loadout,
            context=self.context,
            horizon_ticks=self.horizon_ticks,
            max_outstanding=self.max_outstanding,
            approach_timeout_ticks=self.approach_timeout,
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
        loadouts: the poses each fighter brought. One per fighter, keyed by name.
        pilots: what commits for each fighter. Defaults to :class:`IdlePilot` all round.
        match_seed: the one number a whole match reproduces from.
        config: ring geometry and timestep.
        policy: shared GEAR-SONIC. Built if not supplied.
        pool: the generators. Built if not supplied, seeded from ``match_seed``.
        max_outstanding: how many commits a fighter may have unfinished at once. A tuning knob
            (`M4-T4`) as much as a rule: since `spec/intent.md` 1.1 a move is its walk plus its pose,
            so five is a far deeper commitment than it was.
        require_admitted: whether loadout poses must be admitted. A match always requires it; a
            smoke run of a draft pose may not.
    """

    def __init__(
        self,
        loadouts: dict[str, Loadout],
        *,
        pilots: dict[str, Pilot] | None = None,
        match_seed: int = 1234,
        config: ArenaConfig | None = None,
        policy: GearSonicPolicy | None = None,
        pool: GeneratorPool | None = None,
        context: str = DEFAULT_CONTEXT,
        horizon_ticks: int = COMMIT_HORIZON_TICKS,
        max_outstanding: int = MAX_OUTSTANDING_COMMITS,
        require_admitted: bool = True,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        missing = [f for f in FIGHTERS if f not in loadouts]
        if missing:
            raise FightError(f"no loadout for {missing}; every fighter brings one")

        self.config = config or ArenaConfig()
        self.match_seed = match_seed
        self.model = build_arena(self.config)
        self.data = mujoco.MjData(self.model)
        self.substeps = max(1, int(round(TICK_DT / self.model.opt.timestep)))
        #: How far ahead an approach aims the generator (:data:`APPROACH_LEG_M`). An attribute, not
        #: the constant itself, so a bench can turn it — including to zero, which aims at the whole
        #: placement the way the runtime did before 2026-08-17.
        self.approach_leg_m = float(APPROACH_LEG_M)
        #: Per-fighter settle traces, rebuilt whenever a different commit starts holding a pose.
        self._settling: dict[str, _Settling] = {}

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
                loadouts[name],
                pilots.get(name) or IdlePilot(),
                context=context,
                horizon_ticks=horizon_ticks,
                max_outstanding=max_outstanding,
                # Derived from *this* ring, not the competition default: a match that shrinks the
                # ring must shrink the patience a stalled approach is given with it.
                approach_timeout=approach_timeout_ticks(self.config.ring_size),
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

        for fighter in self.fighters.values():
            fighter.reset_round()
            # The ambient intent, not the timeline's: no commit can be executing at tick 0, and the
            # timeline needs a facing angle that needs a heading that this call is what produces.
            ambient = GeneratorIntent(style=fighter.context)
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
            # The bearing is captured now and reused for every frame this fill produces. It is only
            # the *hold* facing — a committed move carries its own heading in its placement — and a
            # fighter that is holding is by definition not moving, so a bearing that is up to a
            # lookahead stale is a bearing that has not changed.
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
        for fighter in self.fighters.values():
            self._sample_pose_error(fighter)

    # -- settling ------------------------------------------------------------------------------------
    def _sample_pose_error(self, fighter: FighterRuntime) -> None:
        """Record how far this fighter's body is from the pose it is currently holding.

        One sample per tick, after physics, so the trace is at the control rate whatever rate the
        generator is asking questions at. It belongs to **one** commit: a new move resets it, because
        "the error stopped shrinking" must be a statement about the move being judged.
        """
        commit = None
        for candidate in reversed(fighter.timeline.commits):
            if candidate.strike_at is not None and candidate.ended_at is None:
                commit = candidate
                break

        held = self._settling.get(fighter.name)
        if commit is None:
            self._settling.pop(fighter.name, None)
            return
        if held is None or held.commit is not commit:
            target = np.array(
                [commit.pose.joint_angles[name] for name in G1.mujoco_joint_names],
                dtype=np.float64,
            )
            held = _Settling(commit=commit, target=target, trace=deque(maxlen=2 * SETTLE_WINDOW_TICKS))
            self._settling[fighter.name] = held

        measured = np.asarray(fighter.robot_state(self.data).joint_pos, dtype=np.float64)
        held.trace.append(float(np.abs(measured - held.target).mean()))

    def has_settled(self, fighter: str, commit) -> bool:
        """Whether ``commit``'s move is over: the body has stopped closing on its pose.

        The rule the project owner asked for — *the next target goes in when the plan is finished and
        the robot is in position* — needs a causal test, and "the error has converged" as
        ``tools/measure_dwell.py`` defines it is not one: that definition reads the whole run to find
        the asymptote first. The live analogue compares two consecutive replan windows and asks
        whether the second beat the first: if the best error over the last window is no better than
        the window before it by more than
        :data:`~openroboxing.spec.constants.POSE_SETTLE_IMPROVEMENT_RAD`, the fighter is as near the
        pose as it is going to get and the queue may move on.

        Two windows, not one, because a single window cannot tell "converged" from "still falling" —
        and one **replan** window rather than any other length because the reference is rebuilt at
        that cadence, so a shorter window measures the inside of a plan rather than the move.

        Nothing is settled before there are two full windows to compare, so a move always lasts at
        least ``2 * SETTLE_WINDOW_TICKS`` (1.0 s) — the visible strike the dwell exists to protect.
        """
        held = self._settling.get(fighter)
        if held is None or held.commit is not commit or len(held.trace) < 2 * SETTLE_WINDOW_TICKS:
            return False
        trace = list(held.trace)
        recent = min(trace[SETTLE_WINDOW_TICKS:])
        before = min(trace[:SETTLE_WINDOW_TICKS])
        return recent >= before - POSE_SETTLE_IMPROVEMENT_RAD

    def observe(self, tracker, trace, tick: int) -> None:
        tracker.observe(self.model, self.data, tick)
        trace.observe(self.model, self.data, tick)

    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def commits(self) -> list[dict]:
        """Every commit issued this round, as `spec/match_record.md`'s ``CommitEvent``.

        The commit log is per round because :meth:`reset_round` builds a fresh timeline: a round's
        record must hold that round's commits and no others.
        """
        events: list[dict] = []
        for name, fighter in self.fighters.items():
            for commit in fighter.timeline.commits:
                events.append(
                    {
                        "fighter": name,
                        "slot": commit.slot,
                        "pose_name": commit.pose.name,
                        "issued_at": commit.issued_at,
                        "commit_at": commit.commit_at,
                        "strike_at": commit.strike_at,
                        "end_tick": commit.end_tick,
                        "arrived": commit.arrived,
                        "placement": (
                            None
                            if commit.placement is None
                            else {
                                "position": list(commit.placement.position),
                                "heading": commit.placement.heading,
                            }
                        ),
                        "adjustment": dict(commit.adjustment),
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

    def root_pose(self, fighter: str) -> Placement:
        """Where a fighter is standing right now, in **world** terms: ``(x, y)`` and a yaw.

        This is the shadow's fallback anchor — where the next commit starts from when the queue is
        drained (`spec/protocol.md` §The shadow). World frame, not the generator's: a placement is
        something the player points at on a ring they can see, and the conversion into the
        generator's frame is `runtime/generator.py`'s business.
        """
        me = self.fighters[fighter]
        position = self.data.xpos[me.pelvis_body]
        quat = self.data.qpos[me.root_qpos[3:7]]
        # Yaw of a full orientation quaternion, not the yaw-only shortcut apply_yaw takes: a fighter
        # mid-move is pitched and rolled, and reading (w, z) alone would report a heading that leans.
        yaw = float(
            np.arctan2(
                2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
            )
        )
        return Placement(position=(float(position[0]), float(position[1])), heading=yaw)

    def anchor(self, fighter: str, tick: int) -> Placement:
        """Where ``fighter`` will be standing once its queue has run out.

        The last queued commit's placement, or its current root pose when nothing is outstanding.
        The shadow hangs off this rather than off the live position, which moves under the player's
        cursor for the whole duration of every move.
        """
        queued = self.fighters[fighter].timeline.anchor_placement(tick)
        return queued if queued is not None else self.root_pose(fighter)

    def opponent(self, fighter: str) -> str:
        others = [f for f in FIGHTERS if f != fighter]
        if len(others) != 1:
            raise FightError(f"{fighter!r} has {len(others)} opponents; a bout is two fighters")
        return others[0]

    def to_generator_frame(self, fighter: str, placement: Placement) -> Placement:
        """A world placement, expressed in the frame the generator plans in::

            target_gen = context_gen + R(-yaw) . (target_world - robot_world)

        In words: *from where the generator is, travel the vector from the robot to the target.*

        **The conversion is not optional and omitting it is not obviously wrong.** The generator
        plans in its own frame — its context is its own previous output, starting near the origin —
        while a placement is a point on a ring the player can see. Red starts at ``x = -1.2`` yawed
        one way and blue at ``+1.2`` yawed the other, so a raw world target lands somewhere else
        entirely for each of them and the fighter walks a plausible distance in the wrong direction.
        :meth:`facing_angle` has always done the same conversion for angles via
        :func:`generator_heading`; this is its counterpart for positions, and both use the same yaw.

        Two anchors are defensible and only one arrives
        -----------------------------------------------
        The vector is measured **from the robot**, and it is applied **from the generator's buffer
        tail** — the position it will have planned up to, a lookahead in front of the robot. Those
        two choices are what make an approach converge, and neither is interchangeable.

        Measuring from the robot is the feedback. MotionBricks is kinematic: its plan arrives at the
        target while the policy tracking it under physics lands short. Anchored instead on the
        generator's belief about itself, a fighter concludes it has already arrived and plans to
        stand still — which is why, before `spec/intent.md` 1.1, five queued moves at one target got
        no further than one did.

        Applying it from the tail is what makes the feedback *integrate*. Because the tail is already
        ahead, re-deriving the remaining distance from the robot each frame keeps pushing the plan
        forward until the body catches up; measured, that settles to about 0.1 m. Anchoring on the
        frame the robot is playing right now looks more principled — it cancels the lookahead — but
        it turns the loop proportional, and its steady-state error is the tracking shortfall itself:
        **0.22 m at 1 m, 0.49 m at 2 m, 0.81 m at 3 m**, i.e. 27 % of whatever was asked for
        (2026-08-08, ``scratchpad/probe_approach.py``). An integrator that looks wrong beat a
        proportional controller that looks right.

        Within the **pose** phase there is no correction, because a committed plan must not be
        replanned over (`spec/intent.md`). The approach has no such restriction and re-derives every
        frame, which is where all the convergence happens.
        """
        me = self.fighters[fighter]
        here = self.data.xpos[me.pelvis_body][:2]
        context = me.generator.context_qpos()[-1][:2]

        delta = np.asarray(placement.position, dtype=np.float64) - here
        cos, sin = np.cos(-me.apply_yaw), np.sin(-me.apply_yaw)
        rotated = np.array([cos * delta[0] - sin * delta[1], sin * delta[0] + cos * delta[1]])
        moved = context + rotated
        return Placement(
            position=(float(moved[0]), float(moved[1])),
            heading=generator_heading(placement.heading, me.apply_delta_heading),
        )

    def has_arrived(self, fighter: str, commit) -> bool:
        """Whether ``fighter`` is near enough to ``commit``'s placement to stop walking and throw.

        Pelvis to point on the ground plane, against :data:`ARRIVAL_RADIUS_M` — a **measured**
        radius: over ten placements around a fighter the worst closest approach was 0.30 m, and all
        ten reached 0.40 m (`spec/constants.py`). Tighter than that and an approach can fail to
        close, which does not merely miss — it holds the whole queue behind it until the timeout.

        Measured on the **fighter under physics**, not on the generator's plan. The plan is kinematic
        and arrives every time; the body tracking it is the thing that has to get there, and the gap
        between the two is exactly what an open-ended approach exists to close.

        Read a lookahead early, because the frame this decides is played about a second later. That
        is survivable in the one direction it errs: an approach decelerates into its target rather
        than running through it, so arming the pose slightly early lands the strike at the placement
        rather than past it.
        """
        me = self.fighters[fighter]
        if commit.placement is None:
            raise FightError(
                f"{fighter}: asked whether a commit with no placement has arrived; a commit without "
                "one has nowhere to walk to and never approaches"
            )
        here = self.data.xpos[me.pelvis_body][:2]
        target = np.asarray(commit.placement.position, dtype=np.float64)
        return bool(np.linalg.norm(target - here) <= ARRIVAL_RADIUS_M)

    def _intent_at(self, fighter: FighterRuntime, play_tick: int, bearing: float) -> GeneratorIntent:
        """This fighter's control signals for the tick a generated frame will be played at.

        Two frames meet here and nowhere else. The timeline deals only in **world** coordinates — it
        is what the player and the client see — and the generator plans in its own, so the conversion
        happens at this boundary. The arrival test likewise: the timeline owns *when* a commit stops
        walking, this owns *where the fighter is*.
        """
        intent = fighter.timeline.generator_intent(
            play_tick,
            facing_angle=bearing,
            has_arrived=lambda commit, _n=fighter.name: self.has_arrived(_n, commit),
            has_settled=lambda commit, _n=fighter.name: self.has_settled(_n, commit),
        )
        if intent.target_position is None:
            return intent

        leg = self.leg_target(
            fighter.name, Placement(intent.target_position, intent.target_heading)
        )
        placement = self.to_generator_frame(fighter.name, leg)
        return replace(
            intent,
            target_position=placement.position,
            target_heading=placement.heading,
            facing_angle=placement.heading,
            movement_angle=self.travel_angle(fighter.name, placement),
        )

    def leg_target(self, fighter: str, placement: Placement) -> Placement:
        """The next point of the approach: at most :data:`APPROACH_LEG_M` towards ``placement``.

        The commit still ends at its placement — :meth:`has_arrived` is unchanged and still measures
        the body against the point the player chose. What moves is where the *plan* is aimed, and it
        moves for one reason: MotionBricks in-betweens toward the target as its plan's last frame, so
        a target further than one plan can carry produces a plan that arrives while the body is still
        walking. After that the reference is standing at the goal with nothing left to pull the body
        forward, which is the stall measured at 2.6 m off-axis (`spec/constants.py`).

        Re-derived from the fighter's true position on every frame, so a leg needs no bookkeeping to
        be consumed: it slides forward as the body moves and collapses onto the placement itself once
        the fighter is within one leg of it. Setting :attr:`approach_leg_m` to zero aims the
        generator at the whole distance again, which is what the bench's knob is for.
        """
        if self.approach_leg_m <= 0.0:
            return placement
        me = self.fighters[fighter]
        here = np.asarray(self.data.xpos[me.pelvis_body][:2], dtype=np.float64)
        delta = np.asarray(placement.position, dtype=np.float64) - here
        distance = float(np.linalg.norm(delta))
        if distance <= self.approach_leg_m:
            return placement
        step = here + delta * (self.approach_leg_m / distance)
        return Placement(position=(float(step[0]), float(step[1])), heading=placement.heading)

    def travel_angle(self, fighter: str, target_gen: Placement) -> float:
        """Which way the fighter is **going**, in the generator's frame. Not where it is facing.

        Upstream reads two directions and they are not the same signal. ``facing_direction`` is where
        the fighter looks; ``movement_direction`` is where it travels, and their difference is what
        selects a gait: ``blendspace_modes_remap_from_velocity`` swaps ``walk`` for ``walk_left`` or
        ``walk_right`` when the two differ by more than 45° (``demo/clips.py``). Left unset it
        defaults to 0.0 — "straight ahead, always" — which is how a fighter came to have no sideways
        gait no matter where its placement was, and to walk an off-axis approach as one long turn.

        Measured from the generator's own buffer tail, the same anchor
        :meth:`to_generator_frame` applies the target from, so the two agree about what "remaining"
        means. Inside :data:`ARRIVAL_RADIUS_M` the fighter is *there*: the direction to a target you
        are standing on is noise, and feeding it would flip the gait between left and right for the
        whole dwell, so travel collapses onto the heading and the remap stops firing — which is the
        correct thing for a fighter that is no longer going anywhere.
        """
        me = self.fighters[fighter]
        context = me.generator.context_qpos()[-1][:2]
        delta = np.asarray(target_gen.position, dtype=np.float64) - context
        if float(np.linalg.norm(delta)) <= ARRIVAL_RADIUS_M:
            return float(target_gen.heading)
        return float(np.arctan2(delta[1], delta[0]))

    def facing_angle(self, fighter: str) -> float:
        """The **generator-frame** heading that points a fighter at its opponent.

        Converted out of world frame deliberately. The generator plans in its own frame and
        ``apply_delta_heading`` is the fixed yaw between the two, so subtracting its yaw is what makes
        "face the opponent" mean the same thing on both sides of the bridge. Passing a world angle
        straight through would aim each fighter off by however far its clip happened to start.
        """
        me = self.fighters[fighter]
        them = self.fighters[self.opponent(fighter)]
        here = self.data.xpos[me.pelvis_body]
        there = self.data.xpos[them.pelvis_body]
        world = float(np.arctan2(there[1] - here[1], there[0] - here[0]))
        return generator_heading(world, me.apply_delta_heading)

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
