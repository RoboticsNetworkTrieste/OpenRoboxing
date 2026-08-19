"""The sparring bench: one player fighter, a passive sacco, and every debug tap the match hides.

Implements `spec/sparring_protocol.md` 0.1 over the same aiohttp + websocket shape as
`server/app.py`. The core is imported and never edited: `SparringWorld` subclasses
:class:`~openroboxing.runtime.fight.FightWorld`, the pilot is the match's own ``QueuedPilot``, and
the binary frames come from the same :class:`~openroboxing.server.scene.Scene`.

What is deliberately different from the match host
--------------------------------------------------
- **No rounds, no scoring, no knockouts.** The session runs until reset. A sparring session writes
  no match record — it is a bench, not a bout.
- **Pause, reset, teleport and knobs exist.** A match must never have them; here they are the
  point. Every knob reports its canonical value so a deviation is always visible.
- **Everything is recorded** into a :class:`~openroboxing.server.sparring_tap.DebugTap`, and any
  recorded tick can be re-packed into a binary frame for scrubbing.

Conventions
-----------
- Ticks are 50 Hz, absolute since the last reset. The red seat is the player; blue is the sacco.
- All world coordinates are MuJoCo world frame, as everywhere else in the project.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from openroboxing.paths import OPENROBOXING_ROOT
from openroboxing.runtime.conventions import G1
from openroboxing.runtime.fight import FightError, FightWorld
import openroboxing.runtime.intents as intents_module
from openroboxing.runtime.intents import DEFAULT_APPROACH_TIMEOUT_TICKS
from openroboxing.runtime.match import MatchFormat
from openroboxing.runtime.reference import (
    GENERATOR_MARGIN_FRAMES,
    LOOKAHEAD_TICKS,
    REPLAN_DT,
)
from openroboxing.server import protocol
from openroboxing.server.host import LATE_TICK_S, QueuedPilot
from openroboxing.server.scene import Scene
from openroboxing.server.sparring_tap import (
    MACHINE_STATES,
    DebugTap,
    TapError,
    derive_machine_state,
    viz_ghost,
    viz_world_path,
)
from openroboxing.spec.constants import (
    APPROACH_LEG_M,
    ARRIVAL_RADIUS_M,
    COMMIT_HORIZON_TICKS,
    GENERATOR_HZ,
    POSE_DWELL_TICKS,
    TICK_DT,
    TICK_HZ,
)

CLIENT_DIR = OPENROBOXING_ROOT / "client"

#: The bench's queue bound. Deliberately deeper than the match's 5 (`MAX_OUTSTANDING_COMMITS`):
#: the owner asked to be able to stack "anche 10", and the knob block always reports the deviation.
SPARRING_MAX_OUTSTANDING = 10

#: Client-facing rates. Frames as the match streams them; debug at a rate a panel can read.
STREAM_FPS = 30
DEBUG_HZ = 10

#: Root height below which a fighter is considered fallen, for the optional auto-pause. Same
#: threshold `runtime/world.py` uses to declare a fall.
FALL_HEIGHT_M = 0.4

#: Trail decimation: one point per this many ticks. The plan tail is hundreds of ticks; a drawn
#: polyline does not need them all.
TRAIL_STRIDE = 5

#: How far in front of a tick the reference is a **plan** rather than history: the encoder's
#: lookahead plus the generator margin the stream keeps, converted to ticks. This is exactly how far
#: `ReferenceStream.ensure` fills, so live and scrub draw the same horizon — a scrub that reads to
#: the end of the recording draws the rest of the session instead of the plan (0.1's "strange
#: trail", 2026-08-17).
PLAN_HORIZON_TICKS = LOOKAHEAD_TICKS + int(
    math.ceil(GENERATOR_MARGIN_FRAMES * TICK_HZ / GENERATOR_HZ)
)


def strict_dumps(payload: Any) -> str:
    """``json.dumps`` that refuses non-finite numbers.

    Python writes ``NaN``/``Infinity`` as bare tokens; ``JSON.parse`` rejects them and throws away
    the whole message. On a debug bench that is the worst possible failure — the panel goes blank
    and nothing says why (2026-08-17). Failing here instead puts the offending payload in the
    server's log, where it can be fixed.
    """
    import json

    return json.dumps(payload, allow_nan=False)


def json_response(payload: Any, status: int = 200):
    """Every HTTP body the bench sends, through :func:`strict_dumps`."""
    from aiohttp import web

    return web.json_response(payload, status=status, dumps=strict_dumps)


async def send_json(socket: Any, payload: Any) -> None:
    """Every websocket message the bench sends, through :func:`strict_dumps`."""
    await socket.send_json(payload, dumps=strict_dumps)


def _rounded(value: Any, digits: int = 3) -> float | None:
    """A recorded scalar for the wire: ``None`` where the recording has no value."""
    number = float(value)
    return None if not math.isfinite(number) else round(number, digits)


def bench_queue_entry(commit: Any, tick: int) -> dict:
    """The match's queue entry plus what only a bench needs to see.

    ``arrived`` is the difference between a move that *landed* and one whose approach ran out of
    time and threw the pose where it stood — the same row in a match client, two very different
    events on a bench. The match protocol is untouched; this is the bench's own addition.
    """
    entry = protocol.queue_entry(commit, tick)
    entry["arrived"] = commit.arrived
    entry["completed_by"] = commit.completed_by
    return entry


class SparringError(RuntimeError):
    """The bench was driven wrongly. Never recovered from silently."""


# -- knobs ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Knob:
    """One live-tunable runtime parameter, with its canonical value for deviation marking."""

    canonical: float
    get: Callable[[Any], float]
    set: Callable[[Any, float], None]
    #: Values at or below this are refused. Zero would silently disable most of these mechanisms.
    minimum: float = 0.0
    integer: bool = False


def _set_dwell(world: Any, value: float) -> None:
    """The documented process-local override of the dwell (`spec/sparring_protocol.md` §Knobs)."""
    intents_module.POSE_DWELL_TICKS = int(value)


KNOBS: dict[str, Knob] = {
    "replan_dt": Knob(
        canonical=REPLAN_DT,
        get=lambda w: float(w.fighters["red"].stream.replan_dt),
        set=lambda w, v: setattr(w.fighters["red"].stream, "replan_dt", float(v)),
    ),
    "horizon_ticks": Knob(
        canonical=float(COMMIT_HORIZON_TICKS),
        get=lambda w: float(w.fighters["red"].timeline.horizon_ticks),
        set=lambda w, v: setattr(w.fighters["red"].timeline, "horizon_ticks", int(v)),
        minimum=-1.0,  # zero is legal: a horizonless bench run is a real experiment
        integer=True,
    ),
    "max_outstanding": Knob(
        canonical=float(SPARRING_MAX_OUTSTANDING),
        get=lambda w: float(w.fighters["red"].timeline.max_outstanding),
        set=lambda w, v: setattr(w.fighters["red"].timeline, "max_outstanding", int(v)),
        integer=True,
    ),
    "arrival_radius_m": Knob(
        canonical=ARRIVAL_RADIUS_M,
        get=lambda w: float(w.arrival_radius_m),
        set=lambda w, v: setattr(w, "arrival_radius_m", float(v)),
    ),
    "approach_leg_m": Knob(
        canonical=APPROACH_LEG_M,
        get=lambda w: float(w.approach_leg_m),
        set=lambda w, v: setattr(w, "approach_leg_m", float(v)),
        # Zero is legal and is the A/B: aim the generator at the whole placement, the way the
        # runtime did before the leg existed.
        minimum=-1.0,
    ),
    "approach_timeout_ticks": Knob(
        canonical=float(DEFAULT_APPROACH_TIMEOUT_TICKS),
        get=lambda w: float(w.fighters["red"].timeline.approach_timeout_ticks),
        set=lambda w, v: setattr(
            w.fighters["red"].timeline, "approach_timeout_ticks", int(v)
        ),
        integer=True,
    ),
    "pose_dwell_ticks": Knob(
        canonical=float(POSE_DWELL_TICKS),
        get=lambda w: float(intents_module.POSE_DWELL_TICKS),
        set=_set_dwell,
        integer=True,
    ),
}


def knob_values(world: Any) -> dict[str, dict[str, float]]:
    """Every knob as ``{"current", "canonical"}`` — the block the UI marks deviations from."""
    return {
        name: {"current": knob.get(world), "canonical": knob.canonical}
        for name, knob in KNOBS.items()
    }


def set_knobs(world: Any, updates: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Apply a batch of knob changes, validating before touching anything.

    All-or-nothing: a batch with one bad entry changes no knob, so a UI retry cannot leave the
    bench half-configured.
    """
    staged: list[tuple[Knob, float]] = []
    for name, raw in updates.items():
        knob = KNOBS.get(name)
        if knob is None:
            raise SparringError(f"unknown knob {name!r}; the bench has {sorted(KNOBS)}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise SparringError(f"{name}: {raw!r} is not a number") from exc
        if not math.isfinite(value):
            raise SparringError(f"{name}: value must be finite, got {raw!r}")
        if value <= knob.minimum:
            raise SparringError(
                f"{name}: value must be positive (> {knob.minimum:g}), got {value:g}"
            )
        staged.append((knob, value))

    for knob, value in staged:
        knob.set(world, value)
    return knob_values(world)


# -- the world --------------------------------------------------------------------------------------
class SparringWorld(FightWorld):
    """A :class:`FightWorld` with a bench's affordances: a movable arrival radius and a teleport.

    Everything else — both fighters, the shared policy, the per-fighter generators — is exactly the
    match's world, which is the point: the behaviours debugged here are the behaviours a match runs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: The arrival test's radius. The knob writes this; the canonical value is the measured
        #: `ARRIVAL_RADIUS_M` the core uses.
        self.arrival_radius_m = float(ARRIVAL_RADIUS_M)

    def has_arrived(self, fighter: str, commit) -> bool:
        """The core's arrival test, reading the bench's radius instead of the constant."""
        me = self.fighters[fighter]
        if commit.placement is None:
            raise FightError(
                f"{fighter}: asked whether a commit with no placement has arrived; a commit "
                "without one has nowhere to walk to and never approaches"
            )
        here = self.data.xpos[me.pelvis_body][:2]
        target = np.asarray(commit.placement.position, dtype=np.float64)
        return bool(np.linalg.norm(target - here) <= self.arrival_radius_m)

    def teleport_sacco(self, x: float, y: float, heading: float) -> None:
        """Drop the sacco at a chosen spot, standing, at rest. A bench action; matches never move
        a fighter by decree."""
        if not all(math.isfinite(v) for v in (x, y, heading)):
            raise SparringError(f"teleport needs finite numbers, got {(x, y, heading)!r}")
        blue = self.fighters["blue"]
        qpos = self.data.qpos
        qpos[blue.root_qpos[0]] = x
        qpos[blue.root_qpos[1]] = y
        qpos[blue.root_qpos[2]] = 0.793  # the default standing pelvis height
        half = heading / 2.0
        qpos[blue.root_qpos[3:7]] = [math.cos(half), 0.0, 0.0, math.sin(half)]
        self.data.qvel[blue.root_dof] = 0.0
        self._mujoco.mj_forward(self.model, self.data)


# -- scrubbing --------------------------------------------------------------------------------------
def repack_frame(scene: Scene, model, scratch, tick: int, qpos: np.ndarray) -> bytes:
    """A recorded tick, re-packed into the same binary frame the live stream sends.

    ``mj_forward`` recomputes body transforms from the recorded ``qpos``; the pack is then the
    live code path, so a scrubbed frame cannot drift from what the client saw live.
    """
    import mujoco

    scratch.qpos[:] = np.asarray(qpos, dtype=np.float64)
    mujoco.mj_forward(model, scratch)
    return scene.pack(tick, scratch)


# -- the host ---------------------------------------------------------------------------------------
class SparringHost:
    """One sparring session, driven live. Owns the world, the tap, and every socket.

    Args:
        world: a :class:`SparringWorld`, or any world-shaped object (tests pass a fake). The red
            fighter's pilot must be a ``QueuedPilot`` for the staging messages to land.
        render: whether binary frames are produced. Off in tests, on in the tool.
        tap: the recorder; built with defaults when not supplied.
    """

    def __init__(self, world: Any, *, render: bool = True, tap: DebugTap | None = None) -> None:
        self.world = world
        self.tap = tap or DebugTap()
        self.tick = 0
        self.paused = False
        self.pause_on_fall = False
        self.dropped = 0
        self._render = render
        self._scene: Scene | None = None
        self._scrub_data = None
        self._subscribers: set[Any] = set()
        self._stopped = False
        #: The tick `step_once` is currently filling — what a replan event is stamped with.
        self._filling_tick = 0
        #: The red pilot's most recent refusal, surfaced in the debug message. ``QueuedPilot.act``
        #: records errors instead of raising (a socket handler must not crash the match), which on
        #: a bench would be a silent sink — and a silent refusal is indistinguishable from a
        #: broken bench, which is exactly how the controller-gate bug hid.
        self._pilot_error: dict[str, Any] | None = None
        self._pilot_error_seen: str | None = None
        self._watch_replans()

    # -- wiring ------------------------------------------------------------------------------------
    def _watch_replans(self) -> None:
        """Count the red generator's real replans, without touching the core.

        A real replan **reassigns** ``agent.frames['mujoco_qpos']`` (`full_agent.py` builds a new
        tensor); the cadence no-op returns early without assigning. So an identity change across
        one ``generate`` call is exactly one replan. The wrapper lives on this one instance.

        The comparison holds a **reference** to the old tensor, not its ``id()``. A bare id lets
        the old tensor be freed and the new one allocated at the same address, which silently
        swallows the event — measured: a 35 s session recorded 1 replan instead of ~70.
        """
        generator = self.world.fighters["red"].generator
        original = generator.generate

        def watched(intent, context_qpos, dt, *, force: bool = False):
            before = generator.agent.frames["mujoco_qpos"]
            result = original(intent, context_qpos, dt, force=force)
            frames = generator.agent.frames["mujoco_qpos"]
            if frames is not before:
                self.tap.replans.append(
                    (self._filling_tick, bool(force), int(frames.shape[1]))
                )
            return result

        generator.generate = watched

    @property
    def scene(self) -> Scene:
        if self._scene is None:
            self._scene = Scene(self.world.model, self.world.config.__dict__.copy())
        return self._scene

    @property
    def _red(self):
        return self.world.fighters["red"]

    # -- the loop ----------------------------------------------------------------------------------
    def step_once(self) -> None:
        """One 50 Hz tick: pilot anchor, world step, tap row. A no-op while paused."""
        if self.paused:
            return
        start = time.perf_counter()
        self._filling_tick = self.tick

        red = self._red
        pilot = red.pilot
        if isinstance(pilot, QueuedPilot):
            pilot.anchor = self.world.anchor("red", self.tick)

        self.world.step(self.tick)

        # Surface what the pilot refused this tick. `last_error` is a latch, not a stream, so a
        # repeat of the same sentence is one event until it changes.
        refused = getattr(pilot, "last_error", None)
        if refused and refused != self._pilot_error_seen:
            self._pilot_error_seen = refused
            self._pilot_error = {"tick": self.tick, "message": refused}

        qpos = np.asarray(self.world.data.qpos, dtype=np.float64).copy()
        reference = red.stream.motion[self.tick]
        measured = red.robot_state(self.world.data).joint_pos
        error = np.asarray(measured, dtype=np.float64) - reference[7:]

        commits = red.timeline.commits
        machine = derive_machine_state(commits, self.tick)
        ordinal, distance, plan_distance = -1, float("nan"), float("nan")
        for index, commit in enumerate(commits):
            if commit.is_executing(self.tick):
                ordinal = index
                if commit.placement is not None:
                    target = np.asarray(commit.placement.position)
                    here = np.asarray(self.world.root_pose("red").position)
                    distance = float(np.linalg.norm(target - here))
                    plan_distance = self._plan_distance(target, here)
                break

        blue = self.world.fighters["blue"]
        root_h_red = float(self.world.data.qpos[red.root_qpos[2]])
        root_h_blue = float(self.world.data.qpos[blue.root_qpos[2]])
        blue_reference = blue.stream.motion[self.tick] if len(blue.stream.motion) > self.tick else reference

        self.tap.append(
            self.tick,
            qpos=qpos,
            ref_red=reference,
            ref_blue=blue_reference,
            err_red=error,
            action_red=np.asarray(red.last_action, dtype=np.float32),
            root_h_red=root_h_red,
            root_h_blue=root_h_blue,
            separation=self.world.separation_m(),
            dist_target=distance,
            dist_plan=plan_distance,
            step_ms=(time.perf_counter() - start) * 1e3,
            machine=machine,
            commit_ordinal=ordinal,
        )

        if self.pause_on_fall and (root_h_red < FALL_HEIGHT_M or root_h_blue < FALL_HEIGHT_M):
            self.paused = True

        self.tick += 1

    def reset(self, seed: int | None = None) -> None:
        """Back to the opening stance, optionally reseeded, with a fresh recording."""
        if seed is not None:
            self.world.pool.match_seed = int(seed)
        self.world.reset_round(0)
        self.tap.clear()
        self.tick = 0
        self.paused = False
        self._pilot_error = None
        self._pilot_error_seen = None
        pilot = self._red.pilot
        if hasattr(pilot, "reset"):
            pilot.reset()

    # -- messages ----------------------------------------------------------------------------------
    def frame(self) -> bytes | None:
        if not self._render:
            return None
        return self.scene.pack(self.tick, self.world.data)

    def welcome_message(self) -> dict:
        """The red seat's welcome, so the sparring client reuses the match client's staging."""
        return protocol.welcome(
            seat="red",
            loadout=self._red.loadout,
            match_format=MatchFormat(),
            arena=self.world.config.__dict__.copy(),
            match_id="sparring",
        )

    def state_message(self) -> dict:
        """The red seat's own view. There is no opponent to hide anything from."""
        red = self._red
        timeline = red.timeline
        scheduled = timeline.scheduled(self.tick)
        pilot = red.pilot
        seat = protocol.seat_state(
            handle="red",
            staged=getattr(pilot, "staged", None),
            placement=timeline.staged.placement,
            anchor=self.world.anchor("red", self.tick),
            position=self.world.root_pose("red"),
            queue=[protocol.queue_entry(c, self.tick) for c in scheduled],
            can_commit=len(scheduled) < timeline.max_outstanding,
            hits_landed=0,
            torso_height_m=float(self.world.data.qpos[red.root_qpos[2]]),
            down=False,
        )
        return {
            "type": "state",
            "tick": self.tick,
            "paused": self.paused,
            "seats": {"red": seat},
            "separation_m": round(self.world.separation_m(), 3),
        }

    def _plan_distance(self, target: np.ndarray, robot_xy: np.ndarray) -> float:
        """How far the frame the encoder is chasing is from ``target``. NaN when there is no plan.

        The counterpart of the body's own distance. MotionBricks is kinematic and its plan arrives
        every time; the body tracking it is what actually has to get there, and an approach that
        stalls shows up here as *plan in, body out* — measured 2026-08-17 over seven bearings: the
        plan closed to 0.02-0.19 m in all of them, the body to 0.007 m straight ahead and only
        0.38-0.54 m off-axis, so four of seven approaches ended on the timeout instead of arriving.
        """
        red = self._red
        motion = np.asarray(red.stream.motion)
        if len(motion) <= self.tick + LOOKAHEAD_TICKS:
            return float("nan")
        path = viz_world_path(motion, self.tick, robot_xy, float(red.apply_yaw))
        return float(np.linalg.norm(target - path[LOOKAHEAD_TICKS]))

    def _ghost_and_trail(self, tick: int, motion: np.ndarray) -> tuple[dict | None, list]:
        """The plan ghost and trail for ``tick``, or ``(None, [])`` when the motion is short.

        Bounded to :data:`PLAN_HORIZON_TICKS` explicitly, not to however much the stream happens to
        hold: the buffer's length is an implementation detail of the fill, and the scrub draws the
        same horizon from a recording that has no tail at all.
        """
        if len(motion) <= tick:
            return None, []
        red = self._red
        robot_xy = self.world.root_pose("red").position
        apply_yaw = float(red.apply_yaw)
        plan = motion[: tick + PLAN_HORIZON_TICKS + 1]
        ghost = viz_ghost(plan, tick, LOOKAHEAD_TICKS, robot_xy, apply_yaw, G1.mujoco_joint_names)
        path = viz_world_path(plan, tick, robot_xy, apply_yaw)[::TRAIL_STRIDE]
        trail = [[round(float(x), 3), round(float(y), 3)] for x, y in path]
        return ghost, trail

    @staticmethod
    def _series_head(row: dict) -> dict:
        """One recorded tick, as the panel reads it. Live and scrub share this, so they agree.

        Every number is finite or ``None`` (`spec/sparring_protocol.md` §Host → client): a bare
        ``NaN`` on the wire is a payload the browser cannot parse at all.
        """
        err = np.abs(np.asarray(row["err_red"], dtype=np.float64))
        return {
            "err_mean": round(float(err.mean()), 5),
            "err_max": round(float(err.max()), 5),
            # Keyed by joint name (invariant 4): a bare list would make the client guess the
            # order, and a heatmap painted in the wrong order looks like a physics bug.
            "err_by_joint": {
                name: round(float(v), 4) for name, v in zip(G1.mujoco_joint_names, err)
            },
            "dist": _rounded(row["dist_target"]),
            "dist_plan": _rounded(row["dist_plan"]),
            "root_h_red": round(float(row["root_h_red"]), 3),
            "root_h_blue": round(float(row["root_h_blue"]), 3),
            "step_ms": round(float(row["step_ms"]), 2),
        }

    def debug_message(self) -> dict:
        """The 10 Hz panel feed — `spec/sparring_protocol.md` §Host → client."""
        red = self._red
        tick = max(0, self.tick - 1) if self.tick else 0
        ghost, trail = self._ghost_and_trail(tick, np.asarray(red.stream.motion))

        if len(self.tap):
            first, last = self.tap.window()
            row = self.tap.at(last)
            head = self._series_head(row)
            recording = {"start_tick": first, "end_tick": last}
            machine = MACHINE_STATES[int(row["machine"])]
            ordinal = int(row["commit_ordinal"])
        else:
            head, recording = None, {"start_tick": 0, "end_tick": 0}
            machine = MACHINE_STATES[derive_machine_state(red.timeline.commits, tick)]
            ordinal = -1

        return {
            "type": "debug",
            "tick": tick,
            "paused": self.paused,
            "machine": machine,
            "commit_ordinal": ordinal,
            "queue": [
                bench_queue_entry(c, tick) for c in red.timeline.scheduled(tick)
            ],
            "ghost": ghost,
            "trail": trail,
            "series_head": head,
            "replans": [list(r) for r in self.tap.replans[-20:]],
            "knobs": knob_values(self.world),
            "recording": recording,
            "pilot_error": self._pilot_error,
        }

    def scrub_payload(self, tick: int) -> dict:
        """Everything the client needs to show one recorded tick."""
        row = self.tap.at(tick)  # raises TapError outside the window

        first, last = self.tap.window()
        # Only the plan horizon, not the rest of the recording: past `tick + PLAN_HORIZON_TICKS`
        # the recorded reference is the session's later history, and drawing it puts a minute of
        # future robot path on the ring as if it were the plan.
        horizon = min(last, tick + PLAN_HORIZON_TICKS)
        reference = np.asarray(
            [
                np.asarray(self.tap.at(t)["ref_red"], dtype=np.float64)
                for t in range(tick, horizon + 1)
            ]
        )
        red = self._red
        qpos = np.asarray(row["qpos"], dtype=np.float64)
        robot_xy = (float(qpos[red.root_qpos[0]]), float(qpos[red.root_qpos[1]]))
        apply_yaw = float(red.apply_yaw)
        ghost = viz_ghost(reference, 0, LOOKAHEAD_TICKS, robot_xy, apply_yaw, G1.mujoco_joint_names)
        path = viz_world_path(reference, 0, robot_xy, apply_yaw)[::TRAIL_STRIDE]

        payload: dict[str, Any] = {
            "tick": tick,
            "machine": MACHINE_STATES[int(row["machine"])],
            "commit_ordinal": int(row["commit_ordinal"]),
            # The queue as it stood at `tick`. Commit spans are absolute and only ever filled in,
            # so asking a past tick replays the record. Without it the panel emptied out the moment
            # you scrubbed, which reads as "nothing was queued" rather than "not shown".
            "queue": [
                bench_queue_entry(c, tick) for c in self._red.timeline.scheduled(tick)
            ],
            "ghost": ghost,
            "trail": [[round(float(x), 3), round(float(y), 3)] for x, y in path],
            "series_head": self._series_head(row),
            "recording": {"start_tick": first, "end_tick": last},
        }

        if self._render:
            import mujoco

            if self._scrub_data is None:
                self._scrub_data = mujoco.MjData(self.world.model)
            frame = repack_frame(self.scene, self.world.model, self._scrub_data, tick, row["qpos"])
            payload["frame"] = base64.b64encode(frame).decode()
        return payload

    # -- client messages ---------------------------------------------------------------------------
    def handle(self, message: dict) -> dict | None:
        """Apply one client message: sparring controls first, then the match staging subset."""
        kind = message.get("type")
        if kind == "pause":
            self.paused = True
            return None
        if kind == "resume":
            self.paused = False
            return None
        if kind == "pause_on_fall":
            self.pause_on_fall = bool(message.get("on"))
            return None
        if kind == "reset":
            seed = message.get("seed")
            try:
                self.reset(seed=None if seed is None else int(seed))
            except (TypeError, ValueError):
                return protocol.error(f"reset seed must be an integer, got {seed!r}", rejected=kind)
            return None
        if kind == "teleport_sacco":
            try:
                x = float(message.get("x", math.nan))
                y = float(message.get("y", math.nan))
                heading = float(message.get("heading", 0.0))
            except (TypeError, ValueError) as exc:
                return protocol.error(str(exc), rejected=kind)
            # Validated here, before the world is touched: a fake or paused world must never see
            # a NaN, and the client deserves the message either way.
            if not all(math.isfinite(v) for v in (x, y, heading)):
                return protocol.error(
                    f"teleport needs finite numbers, got {(x, y, heading)!r}", rejected=kind
                )
            try:
                self.world.teleport_sacco(x, y, heading)
            except (SparringError, FightError) as exc:
                return protocol.error(str(exc), rejected=kind)
            return None

        try:
            parsed = protocol.parse(message)
        except protocol.ProtocolError as exc:
            return protocol.error(str(exc), rejected=str(kind))

        if parsed["type"] == "ping":
            return protocol.pong(parsed.get("t"))
        if parsed["type"] == "join":
            return None

        red = self._red
        if parsed["type"] == "stage" and parsed["slot"] not in red.loadout.slots:
            return protocol.error(f"slot {parsed['slot']!r} is not in this loadout", rejected="stage")
        if parsed["type"] == "commit":
            timeline = red.timeline
            if len(timeline.scheduled(self.tick)) >= timeline.max_outstanding:
                return protocol.error(
                    f"{timeline.max_outstanding} moves are already queued; no cancellation",
                    rejected="commit",
                )
        red.pilot.queue(parsed)
        return None

    # -- sockets -----------------------------------------------------------------------------------
    def subscribe(self, socket: Any) -> None:
        self._subscribers.add(socket)

    def unsubscribe(self, socket: Any) -> None:
        self._subscribers.discard(socket)

    async def _broadcast_json(self, message: dict) -> None:
        for socket in list(self._subscribers):
            try:
                await send_json(socket, message)
            except Exception:
                self._subscribers.discard(socket)

    async def _broadcast_frame(self) -> None:
        frame = self.frame()
        if frame is None:
            return
        for socket in list(self._subscribers):
            try:
                await socket.send_bytes(frame)
            except Exception:
                self._subscribers.discard(socket)

    async def run(self) -> None:
        """The session loop: paced like the match host, endless, pause-aware."""
        stream_every = max(1, round(TICK_HZ / STREAM_FPS))
        debug_every = max(1, round(TICK_HZ / DEBUG_HZ))
        deadline = time.perf_counter()
        last_paused_send = 0.0

        while not self._stopped:
            if self.paused:
                now = time.perf_counter()
                if now - last_paused_send > 0.5:
                    await self._broadcast_json(self.debug_message())
                    await self._broadcast_json(self.state_message())
                    last_paused_send = now
                await asyncio.sleep(0.05)
                deadline = time.perf_counter()
                continue

            deadline += TICK_DT
            self.step_once()

            if self.tick % stream_every == 0:
                await self._broadcast_json(self.state_message())
                await self._broadcast_frame()
            if self.tick % debug_every == 0:
                await self._broadcast_json(self.debug_message())

            now = time.perf_counter()
            if now < deadline:
                await asyncio.sleep(deadline - now)
            elif now - deadline > LATE_TICK_S:
                # Behind: drop the deficit rather than run two ticks (the match host's rule).
                self.dropped += 1
                deadline = now

    def stop(self) -> None:
        self._stopped = True


# -- the aiohttp surface ----------------------------------------------------------------------------
def build_sparring_app(host: SparringHost, client_dir: Path = CLIENT_DIR):
    """The bench's HTTP + websocket application. **Every socket drives.**

    0.1 gave control to the first socket and made everyone after it a viewer. That gate defended
    nothing — the bench is one human on localhost — and it failed on the most common action: on a
    page refresh the browser can open the new socket before the server processes the old one's
    close, nothing promoted a viewer, and the live page was locked out of its own bench forever
    (reproduced 2026-08-17: stage/commit rejected, the robot standing in OPENING while the local
    aim ghost still moved). A match needs seat protection; a bench needs a refresh to work.
    """
    from aiohttp import WSMsgType, web

    key_host = web.AppKey("sparring_host", SparringHost)
    app = web.Application()
    app[key_host] = host
    mesh_blob: bytes | None = None

    async def index(request):
        return web.FileResponse(client_dir / "sparring.html")

    async def scene_json(request):
        return json_response(host.scene.description())

    async def meshes_bin(request):
        nonlocal mesh_blob
        if mesh_blob is None:
            mesh_blob = host.scene.mesh_blob()
        return web.Response(
            body=mesh_blob,
            content_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    async def websocket(request):
        socket = web.WebSocketResponse(heartbeat=20.0)
        await socket.prepare(request)

        host.subscribe(socket)
        await send_json(socket, host.welcome_message())
        await send_json(socket, host.state_message())
        await send_json(socket, host.debug_message())

        try:
            async for message in socket:
                if message.type is not WSMsgType.TEXT:
                    continue
                try:
                    import json

                    decoded = json.loads(message.data)
                except ValueError as exc:
                    await send_json(socket, protocol.error(str(exc)))
                    continue
                reply = host.handle(decoded)
                if reply is not None:
                    await send_json(socket, reply)
        finally:
            host.unsubscribe(socket)
        return socket

    async def api_frame(request):
        try:
            tick = int(request.match_info["tick"])
        except ValueError:
            return json_response({"error": "tick must be an integer"}, status=400)
        try:
            return json_response(host.scrub_payload(tick))
        except TapError as exc:
            return json_response({"error": str(exc)}, status=404)

    async def api_series(request):
        try:
            first, last = host.tap.window()
        except TapError as exc:
            return json_response({"error": str(exc)}, status=404)
        query = request.rel_url.query
        try:
            from_tick = int(query.get("from", first))
            to_tick = int(query.get("to", last))
            stride = int(query.get("stride", 1))
            return json_response(host.tap.series(from_tick, to_tick, stride))
        except (TapError, ValueError) as exc:
            return json_response({"error": str(exc)}, status=400)

    async def api_session(request):
        try:
            body = host.tap.to_npz_bytes()
        except TapError as exc:
            return json_response({"error": str(exc)}, status=404)
        return web.Response(
            body=body,
            content_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=sparring_session.npz"},
        )

    async def api_knobs_get(request):
        return json_response(knob_values(host.world))

    async def api_knobs_post(request):
        try:
            updates = await request.json()
            return json_response(set_knobs(host.world, dict(updates)))
        except (SparringError, TypeError, ValueError) as exc:
            return json_response({"error": str(exc)}, status=400)

    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/scene.json", scene_json)
    app.router.add_get("/meshes.bin", meshes_bin)
    app.router.add_get("/api/frame/{tick}", api_frame)
    app.router.add_get("/api/series", api_series)
    app.router.add_get("/api/session.npz", api_session)
    app.router.add_get("/api/knobs", api_knobs_get)
    app.router.add_post("/api/knobs", api_knobs_post)
    app.router.add_static("/static/", path=client_dir, name="static")
    return app


async def serve_sparring(host: SparringHost, port: int = 8081) -> None:
    """Serve the bench until interrupted. No waiting gate — sparring starts alone."""
    from aiohttp import web

    app = build_sparring_app(host)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"  sparring  http://localhost:{port}/")

    try:
        await host.run()
    finally:
        await runner.cleanup()
