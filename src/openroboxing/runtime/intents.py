"""The intent timeline: staging, committing, and the commit queue (M2-T4, remodelled).

Implements ``spec/intent.md`` v2.2 (:data:`SPEC_VERSION`).

The player is always steering a :class:`StagedIntent` — channels edited continuously while the fight
runs, with no pause and no edit mode. Committing freezes whatever is staged at that instant into a
:class:`Commit` and appends it to the queue, and from then on it is out of the player's hands:
**no cancellation, of anything.** That rule is the game, and it is enforced here rather than in the
UI, because a client cannot be trusted to enforce it.

A commit is a plan, not a punch
-------------------------------
Since 1.0 a commit carries **a placement and a final pose**: *go here, arrive like this*. The
generator in-betweens from wherever the fighter actually is to that target, so walking is the first
half of every move rather than a separate control. Up to
:data:`~openroboxing.spec.constants.MAX_OUTSTANDING_COMMITS` may be unfinished at once and they run
back to back.

One continuous intent, held after it arrives
--------------------------------------------
Since 2.0 a commit is **one intent for its whole life** — *be at this placement, in this pose* — with
the pose armed on every replan and the plan's length left to MotionBricks. There is no poseless
approach and no separate pose phase; one motion converges on the placement and the pose together.
A commit still **runs until it arrives** — and since 2.2 it also runs until the *move is over*,
which is asked of whoever owns a body rather than counted out. So its span cannot be computed when
it is issued: ``commit_at``, ``strike_at`` and ``end_tick`` are stamped as it runs and are ``None``
before that, and *the queue is not a schedule*.

When the queue drains, the last completed commit's intent **stays armed**. That is the whole
implementation of holding a pose: the generator keeps in-betweening toward a target it has already
reached. 1.1 switched to an idle clip there instead, and MotionBricks obligingly in-betweened *out
of* the pose the player had just paid for. Before the first commit of a round there is no such intent
to hold, and the fighter stands in :data:`OPENING_STANCE_CONTEXT` — the one place an idle clip
survives 2.0.

This module is therefore a **state machine, not a record**. :meth:`IntentTimeline.generator_intent`
is its clock: each call may advance the current commit from waiting to arrived, so it must be called
once per generated frame with non-decreasing ticks, and never as a casual query.
:meth:`IntentTimeline.executing` and friends are the queries.

Two clocks, deliberately not the same
-------------------------------------
Staging is unbounded and happens during play. :data:`COMMIT_HORIZON_TICKS` applies only to
commit → execution, and only as a **floor**: it is the earliest a move may begin, which is what makes
it readable, not a pause inserted between queued moves. Adding one of those would stutter every
combination. See ``spec/intent.md`` §"A commit's span".

Conventions
-----------
- **All ticks are 50 Hz** (:data:`~openroboxing.spec.constants.TICK_HZ`), matching every other
  ``tick`` / ``commit_at`` field in the project.
- **Placement is MuJoCo world ``(x, y)`` on the ground plane** plus a heading in radians — the same
  frame the arena, the shadow and the client use. The axis swap upstream wants is owned by
  ``runtime/generator.py`` and is not visible here.
- A commit is **scheduled** from ``issued_at`` and **executing** from ``commit_at``. The queue is
  bounded on the first; the fighter's motion follows the second.
- **Nothing here knows where a fighter is.** Whether an approach has arrived is geometry, and it
  arrives as a callable from whoever owns the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Mapping

from openroboxing.runtime.generator import GeneratorIntent
from openroboxing.spec.constants import (
    APPROACH_SPEED_M_S,
    COMMIT_HORIZON_TICKS,
    MAX_DWELL_TICKS,
    MAX_OUTSTANDING_COMMITS,
    POSE_DWELL_TICKS,
    RING_SIZE_M,
    TICK_HZ,
)
from openroboxing.studio.pose_record import PoseRecord, PoseRecordError, validate

#: The `spec/intent.md` version this module implements. The test that pairs them is what caught
#: 2.0 shipping without a changelog entry, so the two move together or not at all.
SPEC_VERSION = "2.2"


class IntentError(RuntimeError):
    """An intent was malformed, or the commit rule was broken. Never recovered from silently."""


@dataclass(frozen=True)
class Placement:
    """Where the move ends: MuJoCo world ``(x, y)`` on the ground plane, heading in radians.

    Since ``spec/intent.md`` 1.0 this is the game's primary control — it is how a player moves at
    all, not an optional extra on a punch.
    """

    position: tuple[float, float]
    heading: float

    def __post_init__(self) -> None:
        if len(self.position) != 2:
            raise IntentError(f"placement position must be (x, y), got {self.position!r}")
        if not all(math.isfinite(v) for v in (*self.position, self.heading)):
            raise IntentError(
                f"placement must be finite, got position={self.position!r} "
                f"heading={self.heading!r}"
            )


@dataclass(frozen=True)
class Loadout:
    """The poses a fighter brought to the match, keyed by the slot the player presses.

    Fixed for the duration of a match: swapping a loadout mid-fight would let a player carry more
    than six moves, which is the constraint the format is built on.
    """

    name: str
    version: str
    slots: Mapping[str, PoseRecord]

    def resolve(self, slot: str) -> PoseRecord:
        if slot not in self.slots:
            raise IntentError(
                f"slot {slot!r} is not in loadout {self.name!r}; it has "
                f"{sorted(self.slots)}"
            )
        return self.slots[slot]

    @classmethod
    def load(cls, path, library_dir=None) -> Loadout:
        """Load a loadout: a name, a library version, and slot → pose *name*.

        Slots hold names rather than inline angles so a loadout is a few lines a human can read and
        edit, and so two loadouts naming the same pose cannot disagree about what it is.

        Loadouts live in ``poses/loadouts/`` and the poses they name in ``poses/<version>/``, so
        ``library_dir`` defaults to ``<loadout's parent's parent>/<version>``. Keeping them in
        separate directories is not tidiness: :func:`load_library` reads *every* JSON file it finds,
        so a loadout sitting among the poses is parsed as a malformed pose.
        """
        import json
        from pathlib import Path

        from openroboxing.studio.pose_record import load_library

        path = Path(path)
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise IntentError(f"{path}: cannot read loadout ({exc})") from exc

        for key in ("name", "version", "slots"):
            if key not in data:
                raise IntentError(f"{path}: loadout is missing {key!r}")

        resolved = (
            Path(library_dir) if library_dir else path.parent.parent / str(data["version"])
        )
        if not resolved.is_dir():
            raise IntentError(
                f"{path}: no pose library at {resolved}; a loadout for version "
                f"{data['version']!r} expects one beside its own directory"
            )
        library = load_library(resolved)
        slots: dict[str, PoseRecord] = {}
        for slot, pose_name in data["slots"].items():
            if pose_name not in library:
                raise IntentError(
                    f"{path}: slot {slot!r} names pose {pose_name!r}, which is not in the library; "
                    f"it has {sorted(library)}"
                )
            slots[str(slot)] = library[pose_name]

        return cls(name=data["name"], version=data["version"], slots=slots)

    def validate(self, require_admitted: bool = True) -> None:
        """Raise unless every pose is loadable, and admitted when the match requires it."""
        if not self.slots:
            raise IntentError(f"loadout {self.name!r} has no slots")
        for slot, pose in self.slots.items():
            try:
                validate(pose)
            except PoseRecordError as exc:
                raise IntentError(f"loadout {self.name!r} slot {slot!r}: {exc}") from exc
            if require_admitted and not pose.is_admitted():
                raise IntentError(
                    f"loadout {self.name!r} slot {slot!r} holds pose {pose.name!r}, which is "
                    f"{pose.admission!r}; a match may only use admitted poses"
                )


#: The clip a fighter stands in at the opening bell, **before the first commit of a round has become
#: current**. Nothing else uses it: once a commit has completed, the fighter holds *that commit's*
#: intent (:meth:`IntentTimeline._hold_intent`), which is what "no idle clip after a commit" means in
#: `spec/intent.md` 2.0. Before there has been a commit there is nothing to hold, and this is what a
#: fighter does instead — a state of its own, not a fallback.
#:
#: ``idle`` because it is the one clip with ``avg_root_vel = 0.0``. Standing still is a *clip*, not a
#: zero vector: upstream's ``movement_direction`` is always a unit vector, so "do not move" can only
#: be said by choosing a clip that does not travel. It must not be the ambient context clip —
#: ``walk_boxing`` carries 2.0 m/s in a fixed direction, which is `docs/ASSUMPTIONS.md` §A18, the bug
#: where fighters walked until they hit the ropes and ended a round at opposite ends of the ring.
OPENING_STANCE_CONTEXT = "idle"

#: The clip a fighter **travels** in: the one every commit is generated in unless a player stages
#: another. ``walk`` — the release's default locomotion, straight out of upstream's own registry
#: (``demo/clips.py``: ``neutral_idle_loop_001__A076`` driven at ``avg_root_vel`` 2.0, actual ≈1 m/s).
#:
#: It was ``walk_boxing`` until 2026-08-17, and that was the reason a fighter could not reach a
#: placement that was not straight ahead. ``walk_boxing`` is ``shadow_boxing_R_003__A360_M``
#: frames 25-35 — a shadow-boxing loop — and upstream's lateral blendspace is **gated on the mode**:
#: ``blendspace_modes_remap_from_velocity`` swaps in ``walk_left``/``walk_right`` only when the mode
#: is ``slow_walk`` or ``walk`` (``demo/clips.py``). In the boxing style the remap can never fire, so
#: the fighter has no sideways gait at all and an off-axis approach can only be walked as a turn.
#: Measured before the change (`tools/measure_approach.py`, 1.5 m, seven bearings): the plan closed
#: to 0.02-0.19 m every time while the body closed to 0.007 m straight ahead and 0.38-0.54 m
#: off-axis, and four of seven commits threw their pose on the approach timeout.
#:
#: The poses in ``poses/v0.1`` were harvested and admitted in the boxing style, so their
#: ``generator_error_rad`` is a number measured in a context that is no longer the default; it wants
#: re-measuring (`studio/rehearsal.py`, `tools/measure_dwell.py`).
TRAVEL_CONTEXT = "walk"


@dataclass(frozen=True)
class StagedIntent:
    """What the player is currently steering. Every field is editable until the commit fires."""

    context: str
    pose_slot: str | None = None
    adjustment: Mapping[str, float] = field(default_factory=dict)
    #: Where the move should end. The player drives this with a shadow of their own fighter; see
    #: `spec/protocol.md` §The shadow.
    placement: Placement | None = None

    def is_committable(self) -> bool:
        return self.pose_slot is not None


@dataclass
class Commit:
    """A staged intent, frozen: *go to this placement and arrive in this pose*.

    The **intent** half — pose, context, placement, ``issued_at`` — is fixed the instant the player
    commits and is never written again. The **span** half is not, because a commit runs until it
    arrives: ``commit_at``, ``strike_at`` and ``arrived`` are ``None`` until the move reaches each
    stage, and ``end_tick`` follows from ``strike_at``.

    Since ``spec/intent.md`` 2.0 the intent is the *same* for every tick of the commit's life — one
    continuous "be at this placement, in this pose". So the stages below are not different motions,
    they are only how far along one motion is:

    ==================  =================================================================
    ``commit_at`` None  issued, still inside the readable window or queued behind another
    ``strike_at`` None  **executing, not there yet** — converging on placement *and* pose
    both set            **arrived** — standing in the pose until ``end_tick``
    ==================  =================================================================

    A commit therefore has no length of its own. ``end_tick`` is arrival plus
    :data:`~openroboxing.spec.constants.POSE_DWELL_TICKS`, the measured time the slowest pose in the
    library takes to settle — nothing here is derived from the pose's ``horizon_tokens`` any more,
    because MotionBricks chooses how long the motion runs.
    """

    pose: PoseRecord  # the adjustment is already applied
    context: str
    placement: Placement | None
    issued_at: int
    slot: str
    #: The live adjustment as the player set it. Kept even though :attr:`pose` already carries it
    #: baked in, because a match record's ``CommitEvent`` is the record of what the *player did* —
    #: and "jab, nudged 4 degrees left" is not recoverable from the resulting angles alone.
    adjustment: Mapping[str, float] = field(default_factory=dict)

    #: Tick the approach began. ``None`` until it does.
    commit_at: int | None = None
    #: Tick the fighter arrived and the pose was armed. ``None`` until it does.
    strike_at: int | None = None
    #: Whether :attr:`strike_at` came from arriving or from the approach timing out. Recorded rather
    #: than inferred, because "threw it short" is a thing a replay should be able to show.
    arrived: bool | None = None
    #: Tick the move finished and the next commit could become current. ``None`` until it does.
    ended_at: int | None = None
    #: What ended it: ``"settled"`` (the body stopped closing on the pose), ``"dwell"`` (the counted
    #: rule, for a caller with no body to measure) or ``"timeout"`` (:data:`MAX_DWELL_TICKS`, the
    #: guard). Recorded rather than inferred, for the same reason as :attr:`arrived`.
    completed_by: str | None = None

    @property
    def end_tick(self) -> int | None:
        """The tick this commit finished, or ``None`` while it is still running.

        ``None`` means *later than any tick you can name*, not zero and not "already over". Every
        predicate below reads it that way and so must every caller.

        **Stamped, not computed** since 2.2. It used to be ``strike_at + POSE_DWELL_TICKS`` — a
        counter, so the queue advanced on a clock: every move waited out the settle time of the
        *slowest pose in the library* whether or not its own had settled, and the measured
        distribution is `[0, 0, 0, 0, 0, 2, 5, 12, 12, 74]` ticks, so nine moves in ten spent the
        wait on nothing. Now :meth:`IntentTimeline.generator_intent` asks whoever owns a body
        whether the move is over, and stamps the tick it says yes — the project owner's rule:
        *when the plan is finished and the robot is in position, pass the next target*.

        The counted dwell survives in two places, both explicit: as the rule for a caller that has
        no body to measure (the Studio's rehearsals), and as :data:`MAX_DWELL_TICKS`, the guard that
        stops a pose the body never settles into from holding the queue forever.
        """
        return self.ended_at

    def is_scheduled(self, tick: int) -> bool:
        """True while this commit exists and has not finished — issued, walking, or waiting."""
        if tick < self.issued_at:
            return False
        end = self.end_tick
        return end is None or tick < end

    def is_executing(self, tick: int) -> bool:
        """True once the move itself is under way. **Walking counts** — walking is the move."""
        if self.commit_at is None or tick < self.commit_at:
            return False
        end = self.end_tick
        return end is None or tick < end

    def is_approaching(self, tick: int) -> bool:
        """True while this commit is under way and has not reached its placement yet.

        Still exactly "executing but not yet arrived" — 2.0 removed the poseless approach, not the
        approach. ``server/protocol.py`` sends this so a client can show a move as travelling rather
        than landed.
        """
        return self.is_executing(tick) and (self.strike_at is None or tick < self.strike_at)


def approach_timeout_ticks(ring_size: float, speed: float = APPROACH_SPEED_M_S) -> int:
    """How long an approach may walk before the fighter throws the pose where it stands.

    Derived, not chosen: **the ring's diagonal at the measured sustained approach speed.** An
    approach that has taken longer than crossing the whole ring corner to corner is not walking
    anywhere — it is wedged in a corner, leaning on its opponent, or knocked down — and every commit
    queued behind it is waiting on something that will not happen.

    At competition dimensions (4.90 m) that is 8.4 s, against a slowest *observed* arrival of 4.3 s
    (``scratchpad/probe_arrival.py``), so roughly a factor of two in hand. The factor is the
    consequence of the derivation, not a safety margin someone picked.

    It is a function of ``ring_size`` because ``ArenaConfig.ring_size`` is a match parameter that
    `M4-T4` is expected to change; a constant would silently stop matching the ring.
    """
    if not math.isfinite(ring_size) or ring_size <= 0.0:
        raise IntentError(f"ring_size must be a positive length, got {ring_size!r}")
    if not math.isfinite(speed) or speed <= 0.0:
        raise IntentError(f"approach speed must be positive, got {speed!r}")
    return int(math.ceil(ring_size * math.sqrt(2.0) / speed * TICK_HZ))


#: The timeout for a competition-sized ring. A world with its own ``ArenaConfig`` passes its own.
DEFAULT_APPROACH_TIMEOUT_TICKS = approach_timeout_ticks(RING_SIZE_M)


def apply_adjustment(pose: PoseRecord, adjustment: Mapping[str, float]) -> PoseRecord:
    """Apply a bounded live adjustment to an admitted pose.

    The base pose is admitted offline; the adjustment is not individually admitted, so it is legal
    only inside the envelope whose corners *were* admitted (``spec/intent.md`` §Feasibility). An
    adjustment outside the envelope, or on a joint the envelope does not cover, raises — clamping it
    would silently produce a pose nobody has ever measured.
    """
    if not adjustment:
        return pose

    angles = dict(pose.joint_angles)
    for joint, delta in adjustment.items():
        bound = pose.adjustment_envelope.get(joint)
        if bound is None:
            raise IntentError(
                f"pose {pose.name!r} has no adjustment envelope for {joint!r}; it allows "
                f"{sorted(pose.adjustment_envelope) or 'no adjustment at all'}"
            )
        if abs(delta) > bound:
            raise IntentError(
                f"adjustment {delta:+.4f} on {joint!r} leaves pose {pose.name!r}'s envelope of "
                f"+-{bound}"
            )
        angles[joint] = angles[joint] + delta

    return replace(pose, joint_angles=angles)


class IntentTimeline:
    """The commit queue for one fighter.

    Args:
        loadout: the poses this fighter may commit.
        horizon_ticks: **minimum** lead from commit to execution. Defaults to the canonical
            :data:`COMMIT_HORIZON_TICKS`; a match may parameterise it.
        max_outstanding: how many commits may be unfinished at once. Defaults to
            :data:`MAX_OUTSTANDING_COMMITS`.
        approach_timeout_ticks: how long an approach may walk before the fighter throws where it
            stands. Defaults to the competition ring's diagonal at :data:`APPROACH_SPEED_M_S`; a
            world with a different ring must pass its own, or a fighter in a small ring waits far
            longer than that ring can justify.
        require_admitted: whether the loadout's poses must be admitted. A match always requires it;
            the Studio sets it False so a draft pose can be rehearsed before it is measured.
    """

    def __init__(
        self,
        loadout: Loadout,
        *,
        context: str = TRAVEL_CONTEXT,
        horizon_ticks: int = COMMIT_HORIZON_TICKS,
        max_outstanding: int = MAX_OUTSTANDING_COMMITS,
        approach_timeout_ticks: int | None = None,
        max_dwell_ticks: int = MAX_DWELL_TICKS,
        require_admitted: bool = True,
    ) -> None:
        if horizon_ticks < 0:
            raise IntentError(f"horizon_ticks must not be negative, got {horizon_ticks}")
        if max_outstanding < 1:
            raise IntentError(f"max_outstanding must be at least 1, got {max_outstanding}")
        if approach_timeout_ticks is not None and approach_timeout_ticks < 1:
            raise IntentError(
                f"approach_timeout_ticks must be at least 1, got {approach_timeout_ticks}"
            )
        if max_dwell_ticks < 1:
            raise IntentError(f"max_dwell_ticks must be at least 1, got {max_dwell_ticks}")
        loadout.validate(require_admitted=require_admitted)

        self.loadout = loadout
        self.horizon_ticks = horizon_ticks
        self.max_outstanding = max_outstanding
        self.approach_timeout_ticks = (
            DEFAULT_APPROACH_TIMEOUT_TICKS
            if approach_timeout_ticks is None
            else approach_timeout_ticks
        )
        #: The guard on a dwell: a move ends here whatever the body is doing.
        self.max_dwell_ticks = max_dwell_ticks
        self._staged = StagedIntent(context=context)
        self._commits: list[Commit] = []
        #: Highest tick :meth:`generator_intent` has been driven to. The queue advances there, so it
        #: must never be asked about a tick it has already passed.
        self._driven_to = -1

    # -- staging ------------------------------------------------------------------------------------
    @property
    def staged(self) -> StagedIntent:
        return self._staged

    @property
    def commits(self) -> tuple[Commit, ...]:
        """Every commit issued, in order. This is the log a match record is built from."""
        return tuple(self._commits)

    def stage(
        self,
        *,
        pose_slot: str | None = None,
        adjustment: Mapping[str, float] | None = None,
        placement: Placement | None = None,
        context: str | None = None,
    ) -> StagedIntent:
        """Edit the staged intent. Always allowed, including while a commit is executing.

        Only the channels passed are changed; the rest keep their current value. Staging never
        affects the fighter — it is the player's private aim, and it fires only on :meth:`commit`.
        """
        if pose_slot is not None:
            self.loadout.resolve(pose_slot)  # fail at staging time, not at commit time
        self._staged = StagedIntent(
            context=context if context is not None else self._staged.context,
            pose_slot=pose_slot if pose_slot is not None else self._staged.pose_slot,
            adjustment=dict(adjustment) if adjustment is not None else self._staged.adjustment,
            placement=placement if placement is not None else self._staged.placement,
        )
        return self._staged

    def clear_pose(self) -> StagedIntent:
        """Unstage the pose, leaving the other channels. Not a cancellation — nothing has fired."""
        self._staged = StagedIntent(
            context=self._staged.context,
            pose_slot=None,
            adjustment=self._staged.adjustment,
            placement=self._staged.placement,
        )
        return self._staged

    # -- the queue ----------------------------------------------------------------------------------
    def scheduled(self, tick: int) -> tuple[Commit, ...]:
        """Every commit that exists and has not finished at ``tick``, executing first.

        This is what the queue bound counts and what a client is shown. A commit leaves it by
        finishing, never by being cancelled.
        """
        return tuple(c for c in self._commits if c.is_scheduled(tick))

    def executing(self, tick: int) -> Commit | None:
        """The commit whose move is actually under way at ``tick``, if any.

        At most one: a commit only begins once the one in front of it has finished, so the executing
        spans cannot overlap. A **query** — unlike :meth:`generator_intent` it advances nothing, so
        asking never changes what a fighter does.
        """
        for commit in self._commits:
            if commit.is_executing(tick):
                return commit
        return None

    def current(self, tick: int) -> Commit | None:
        """The commit that should be driving the fighter at ``tick``, started or not.

        The **oldest unfinished** one, once its readable window has elapsed. ``None`` means hold:
        either the queue is drained, or the only thing in it was committed less than
        ``horizon_ticks`` ago.

        A commit still inside its window blocks the queue rather than being skipped over. Letting a
        later commit start first would reorder them, and order is the one thing a player controls
        about a queue they cannot cancel.
        """
        for commit in self._commits:
            if not commit.is_scheduled(tick):
                continue  # finished; look further down the queue
            if commit.commit_at is None and tick < commit.issued_at + self.horizon_ticks:
                return None  # its window has not elapsed, and nothing may jump ahead of it
            return commit
        return None

    def anchor_placement(self, now: int) -> Placement | None:
        """Where the queue leaves this fighter standing, or ``None`` if it is drained.

        The shadow hangs off this rather than off the live root pose: the next move starts from the
        end of the queue, and an anchor that moves under the player's cursor for the duration of
        every move cannot be aimed (`spec/protocol.md` §The shadow).

        The **last unfinished commit's** placement, by issue order. Since 1.1 that cannot be found by
        latest ``end_tick`` — an unfinished commit has none.
        """
        outstanding = [c for c in self._commits if c.is_scheduled(now)]
        if not outstanding:
            return None
        return outstanding[-1].placement

    def commit(self, now: int) -> Commit:
        """Queue the staged intent.

        The move begins no earlier than ``now + horizon_ticks`` — the floor that makes an isolated
        commit readable — and no earlier than the commit in front of it finishes. Which of those
        binds is not known here and is not decided here: since 1.1 an approach ends by arriving, so
        the queue's timing is settled as it runs, by :meth:`generator_intent`.

        Raises:
            IntentError: if nothing is staged, or the queue is full. A refused commit is refused
                outright — there is no cancellation and no replacing a queued move, so the only way
                a slot frees up is a move finishing.
        """
        if now < 0:
            raise IntentError(f"tick must not be negative, got {now}")
        if self._commits and now < self._commits[-1].issued_at:
            raise IntentError(
                f"tick {now} is before the last commit at {self._commits[-1].issued_at}; "
                "the timeline does not run backwards"
            )

        outstanding = self.scheduled(now)
        if len(outstanding) >= self.max_outstanding:
            # No "frees at tick N": since 1.1 the move in front ends when the fighter gets there,
            # and a number invented here would be a promise the ring has not agreed to.
            raise IntentError(
                f"cannot commit at tick {now}: {len(outstanding)} moves are already queued, the "
                f"limit is {self.max_outstanding}. A slot frees when the move in front arrives and "
                "throws. No cancellation."
            )
        if not self._staged.is_committable():
            raise IntentError("cannot commit: no pose is staged")

        pose = apply_adjustment(
            self.loadout.resolve(self._staged.pose_slot), self._staged.adjustment
        )
        commit = Commit(
            pose=pose,
            context=self._staged.context,
            placement=self._staged.placement,
            issued_at=now,
            slot=self._staged.pose_slot,
            adjustment=dict(self._staged.adjustment),
        )
        self._commits.append(commit)
        return commit

    # -- driving the generator -----------------------------------------------------------------------
    def generator_intent(
        self, tick: int, *, facing_angle: float = 0.0, has_arrived=None, has_settled=None
    ) -> GeneratorIntent:
        """The control signals for the tick **this frame will be played at**, advancing the queue.

        Ask for the tick the frame is *for*, not the tick you happen to be generating on. Frames are
        produced ahead of the tick that consumes them, so asking about "now" slides every move a
        fixed lookahead late. ``runtime/reference.py`` owns that mapping.

        **This is the timeline's clock, not a query.** Calling it is what starts an approach, ends
        one, and so decides when the next commit in the queue begins. Call it exactly once per
        generated frame, with non-decreasing ticks — which is enforced below, because a caller that
        drifts backwards would rewrite a move's history rather than fail.

        What comes back, by stage:

        - **a commit is current** — its own continuous intent, placement and pose together, from the
          tick it starts to the tick it ends. See :meth:`_commit_intent`.
        - **nothing current, but a commit has completed** — that commit's intent, unchanged: the
          fighter holds the pose it was commanded into. See :meth:`_hold_intent`.
        - **nothing current and none has completed** — the round's opening stance,
          :data:`OPENING_STANCE_CONTEXT`, which is the only place an idle clip is still used.

        Args:
            facing_angle: generator-frame heading to face when nothing supplies one — the opening
                stance, and commits issued without a placement. A commit with a placement faces its
                own heading, held included. The caller owns the conversion out of world frame;
                nothing here knows about the generator's frame.
            has_arrived: ``(commit) -> bool``, asked once per approach frame. **Required for any
                commit that carries a placement** — there is nothing else that could end its
                approach, and defaulting to "arrived" would quietly restore the 1.0 bug where a move
                never reached where it was pointed.
            has_settled: ``(commit) -> bool``, asked once per frame after a commit has struck: has
                the body stopped closing on the pose? Omitting it is legal and means *the counted
                dwell* — the only rule available to a caller with no body to measure, such as a
                Studio rehearsal. A world that has one passes it, and then the queue advances on the
                move being over rather than on a clock (`spec/intent.md` 2.2).
        """
        if tick < self._driven_to:
            raise IntentError(
                f"generator_intent went backwards: tick {tick} after {self._driven_to}. It advances "
                "the commit queue, so a repeated or out-of-order tick would rewrite a move's span."
            )
        self._driven_to = tick

        commit = self.current(tick)
        # Completion is decided *before* the current move is chosen, so a commit that ends on this
        # tick hands over on this tick. Deciding it afterwards leaves a one-tick hole between queued
        # moves — small, and exactly the stutter `spec/intent.md` §"A commit's span" forbids.
        if commit is not None and commit.strike_at is not None:
            self._resolve_completion(commit, tick, has_settled)
            if commit.ended_at is not None:
                commit = self.current(tick)

        if commit is None:
            return self._hold_intent(tick, facing_angle)

        if commit.commit_at is None:
            commit.commit_at = tick

        if commit.strike_at is None:
            self._resolve_approach(commit, tick, has_arrived)

        return self._commit_intent(commit, facing_angle)

    def _commit_intent(self, commit: Commit, facing_angle: float) -> GeneratorIntent:
        """The one intent a commit has for its whole life: *be there, in that pose*.

        **The pose is armed on every replan, not only once it has arrived.** That is the change 2.0
        is: the generator in-betweens toward a target it is given, so a walk that carries the pose
        converges on the pose *while it travels* instead of snapping into it at the end. Measured
        (2026-08-13, ``tools/measure_dwell.py`` over ``studio/rehearsal.rehearse_approach``) against
        a placement 2.5 m away: hook-right closes 2.500 m → 0.028 m and 17.0° → 6.0°, and then holds.
        The same six seconds with the pose withheld until arrival never converges at all — 17.0° →
        18.5°.

        ``horizon_tokens`` is deliberately ``None``: one plan covers only ~54 % of the distance asked
        for at *any* length, so a forced length buys nothing and costs the model its own choice.
        Left free it picks 11 tokens at 0.5 m and 16 at 6.0 m, which lands plan speed near the
        0.83 m/s the robot actually sustains. The pose's own ``horizon_tokens`` stays in the record
        as the author's statement of how long the move is meant to take, and as the Studio's
        rehearsal parameter; the runtime simply stops forwarding it.
        """
        placement = commit.placement
        return GeneratorIntent(
            style=commit.context,
            facing_angle=placement.heading if placement else facing_angle,
            target_position=placement.position if placement else None,
            target_heading=placement.heading if placement else None,
            pose=commit.pose,
            horizon_tokens=None,
        )

    def _hold_intent(self, tick: int, facing_angle: float) -> GeneratorIntent:
        """What a fighter with nothing current does. **Two states, not one:**

        - **a commit has completed** — hold *that commit's* intent, unchanged.
        - **none has yet** — the round's opening stance, :data:`OPENING_STANCE_CONTEXT`.

        They are different because the fighter is in a different situation, not because one is a
        degenerate form of the other. After a commit there is a placement and a pose the player paid
        for and the fighter is standing in; before one there is neither, and a fighter at the opening
        bell must stand rather than travel (§A18 — see :data:`OPENING_STANCE_CONTEXT`).

        **Re-issuing the last commit's intent is the entire implementation of holding a pose.** There
        is no idle clip, no freeze branch and no repeated-frame counter — the generator simply keeps
        in-betweening toward a placement and a pose it has already reached, which is a fixed point.
        1.1 switched to an idle clip here, so MotionBricks in-betweened *out of* the pose the instant
        the move ended and the fighter drifted back to a neutral stance; the pose the player paid for
        existed for one 30 Hz frame. That is what "no idle clip **after** a commit" means, and it is
        the whole of what 2.0 removed: the opening stance below is untouched by it.

        The held intent carries its own ``target_heading``, so a held fighter **does not turn to
        track its opponent**. Re-orienting is paid for by the next commit. Both that and "the
        committed pose is the resting state" are the project owner's calls, 2026-08-13.
        """
        for commit in reversed(self._commits):
            end = commit.end_tick
            if end is not None and tick >= end:
                return self._commit_intent(commit, facing_angle)
        return GeneratorIntent(style=OPENING_STANCE_CONTEXT, facing_angle=facing_angle)

    def _resolve_approach(self, commit: Commit, tick: int, has_arrived) -> None:
        """Decide whether ``commit`` is still walking, or has arrived and should throw."""
        if commit.placement is None:
            # Nothing to walk to: "do this where you stand". Not a fallback - a commit without a
            # placement is a complete instruction, and it is what the Studio and the tools issue.
            commit.strike_at, commit.arrived = tick, True
            return

        if tick - commit.commit_at >= self.approach_timeout_ticks:
            commit.strike_at, commit.arrived = tick, False
            return

        if has_arrived is None:
            raise IntentError(
                f"commit {commit.slot!r} carries a placement but no arrival test was passed to "
                "generator_intent, so its approach could never end and everything queued behind it "
                "would wait out approach_timeout_ticks. Pass has_arrived= from whoever knows where "
                "the fighter is."
            )
        if has_arrived(commit):
            commit.strike_at, commit.arrived = tick, True

    def _resolve_completion(self, commit: Commit, tick: int, has_settled) -> None:
        """Decide whether the move is over, and stamp :attr:`Commit.ended_at` if it is.

        Three ways a commit can end, and they are recorded apart because they mean different things
        to a replay:

        - **settled** — ``has_settled`` says the body has stopped closing on the pose. This is the
          normal case and the whole point of 2.2: the next target goes in when the move is done, not
          when a counter says the slowest pose in the library would have been done.
        - **dwell** — no ``has_settled`` was passed, so the counted
          :data:`~openroboxing.spec.constants.POSE_DWELL_TICKS` applies. Read as a module global so
          the bench's knob still reaches it.
        - **timeout** — :attr:`max_dwell_ticks`, the guard. Without it a pose the body never settles
          into holds every commit behind it forever, which is the one failure the counted dwell did
          not have.

        The strike frame itself is never a settle test: a commit that struck this tick has not yet
        had a tick in which to settle, and a body already standing in the pose would end the move on
        the same frame it began it.
        """
        held = tick - commit.strike_at
        if held <= 0:
            return
        if held >= self.max_dwell_ticks:
            commit.ended_at, commit.completed_by = tick, "timeout"
            return
        if has_settled is None:
            if held >= POSE_DWELL_TICKS:
                commit.ended_at, commit.completed_by = tick, "dwell"
            return
        if has_settled(commit):
            commit.ended_at, commit.completed_by = tick, "settled"
