"""The intent timeline: staging, committing, and the commit queue (M2-T4; rewritten M6-T5).

Implements ``spec/intent.md`` v3.1 (:data:`SPEC_VERSION`).

The player is always steering a :class:`StagedIntent` — channels edited continuously while the fight
runs, with no pause and no edit mode. Committing freezes whatever is staged at that instant into a
:class:`Commit` and appends it to the queue, and from then on it is out of the player's hands:
**no cancellation, of anything.** That rule is the game, and it is enforced here rather than in the
UI, because a client cannot be trusted to enforce it. None of that changed at 3.0.

A commit is a combination, not a plan to walk somewhere
---------------------------------------------------------
1.0 through 2.2 built a commit out of *a placement and a final pose*: the generator walked the
fighter to the placement and arrived in the pose, and roughly a third of this module existed to run
that walk — an approach that timed out, an arrival test, a dwell that decided when the pose had
"settled". **3.0 deletes all of it.** A commit now carries a :class:`~openroboxing.studio.
combination_record.CombinationRecord` — 3-6 recorded key poses with recorded timing — plus a
**ghost**: the world ``(x, y)`` its final keyframe should land on. It starts **in place**, wherever
the fighter already stands, and runs the combination's own recorded footwork while the leftover
travel to the ghost is added as an even drift (``runtime/warp.py``). There is no walk left to be a
separate phase of, because the combination already contains the footwork.

The consequence that reverses 2.2: a commit's length is known the instant it starts
--------------------------------------------------------------------------------------
2.2's whole reason for stamping ``commit_at`` / ``strike_at`` / ``ended_at`` as a move ran, rather
than computing them at issue time, was that an *approach's* length depended on distance under physics
and could not be known until the fighter got there. A combination has no such phase: its length is
``record.duration_ticks``, fixed the day it was captured. So the instant :attr:`Commit.commit_at` is
known, ``end_tick = commit_at + record.duration_ticks`` is exact arithmetic — nothing is watched for
and nothing can be "not yet". See :attr:`Commit.end_tick`.

One consequence of that arithmetic is worth naming because it looks like it should need more
machinery than it does: **the hand-over between queued commits needs no explicit resolution step.**
:meth:`Commit.is_scheduled` and :meth:`Commit.is_executing` both test ``tick < end_tick`` — a strict
inequality — so a commit that ends at tick *T* is no longer scheduled *at* T, and whichever commit is
next in the queue is simply the one :meth:`IntentTimeline._current` finds there. 2.2 needed a
separate ``_resolve_completion`` pass, run *before* choosing the current commit, purely because its
``end_tick`` was unknowable until a callback said so; 3.0's ``end_tick`` is knowable the moment
``commit_at`` is, so there is nothing left to resolve.

Off-target execution: re-warped for real, from wherever the fighter actually is
-----------------------------------------------------------------------------------
A commit's *timing* is fixed the instant it starts, but not its *placement* — physics does not track
a plan exactly, and a fighter can be pushed or knocked down while a queued commit waits its turn. So
building a commit's :class:`~openroboxing.runtime.sequence.CombinationRunner` needs to know the
fighter's **true** position and heading at the tick the commit starts, not the position that was
assumed when the ghost was placed. That is what the ``anchor`` callable passed to
:meth:`IntentTimeline.generator_intent` is for: it is called **exactly once per commit**, at the tick
that commit starts, and :func:`~openroboxing.runtime.warp.warp` is called with ``speed_ceiling=None``
— nothing is clamped and nothing raises; a combination that starts off-target still reaches its
ghost, running whatever drift that needs (``spec/intent.md`` "Off-target execution").

The two hold states, unchanged in kind since 2.0
---------------------------------------------------
When the queue is drained, :meth:`IntentTimeline.generator_intent` re-issues an intent rather than
falling back to an idle clip — and there are still **two different situations**, not one degenerate
case of the other:

- **a commit has completed** — hold *that commit's* intent: its runner's **final leg**, which
  :class:`~openroboxing.runtime.sequence.CombinationRunner` already returns for any tick past its
  end. This is the whole implementation of "holding a pose": the runner keeps being asked for the leg
  live at ``tick``, and past its own end that answer is a fixed point.
- **none has completed yet** — :data:`OPENING_STANCE_CONTEXT`, the one place an idle clip survives.

They are different because the fighter is in a different situation, not because one is a fallback
for the other: after a commit there is a specific recorded motion the player paid for and the fighter
is standing in its last pose; before one there is neither, and a fighter at the opening bell must
stand rather than travel (`docs/ASSUMPTIONS.md` §A18 — see :data:`OPENING_STANCE_CONTEXT`).

This module is therefore still a **state machine, not a record**. :meth:`IntentTimeline.
generator_intent` is its clock: each call may start the next commit and build its runner, so it must
be called once per generated frame with non-decreasing ticks, and never as a casual query.

What is gone, and why it is safe to delete rather than merely retire
--------------------------------------------------------------------
The approach existed to fill a gap 1.0's only control left open: a placement with no idea what motion
should get the fighter there. A combination fills that gap by construction — it already contains the
footwork — so there is nothing left for a generic walk, an arrival test, a settle test or a counted
dwell to do. Removed in this rewrite: the approach itself, ``TRAVEL_CONTEXT``,
``approach_timeout_ticks`` / ``DEFAULT_APPROACH_TIMEOUT_TICKS``, ``has_arrived`` / ``has_settled`` and
the parameters that carried them, ``apply_adjustment`` and the bounded live adjustment it applied
(a combination carries no adjustment envelope — ``spec/combination.md``), ``Placement`` (the ghost is
a bare ``(x, y)``; its heading is derived, never chosen — it faces the opponent, see
``runtime/warp.py::ghost_heading``), and
every ``Commit`` field that only made sense once a move had an approach phase and a settle phase to
distinguish (``strike_at``, ``arrived``, ``completed_by``, ``is_approaching``, ``slot``,
``adjustment``, ``pose``, ``context``, ``placement``). Full accounting: ``spec/intent.md`` "Removed at
3.0".

Conventions
-----------
- **All ticks are 50 Hz** (:data:`~openroboxing.spec.constants.TICK_HZ`), matching every other
  ``tick`` / ``commit_at`` field in the project.
- **The ghost is MuJoCo world ``(x, y)`` on the ground plane** — the same frame the arena, the shadow
  and the client use. Unlike 1.0-2.2's ``Placement`` it carries no heading: a fighter always faces
  its opponent (owner, 2026-09-03), so the ghost's heading is derived by
  ``runtime/warp.py::ghost_heading`` as the bearing from the ghost to the opponent, and is never
  player-set (``spec/intent.md`` "The ghost").
- **Where a fighter looks is the world's business, not the timeline's.** ``generator_intent`` is
  handed the live bearing each tick and hands it straight to the runner; nothing here remembers a
  heading between ticks, because the opponent moves while a combination runs.
- A commit is **scheduled** from ``issued_at`` and **executing** from ``commit_at``. The queue is
  bounded on the first; the fighter's motion follows the second.
- **Nothing here knows where a fighter is**, except at the one instant a commit starts, and even then
  only through the ``anchor`` callable passed in — geometry belongs to whoever owns the world.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openroboxing.runtime.generator import GeneratorIntent
from openroboxing.runtime.sequence import CombinationRunner
from openroboxing.runtime.warp import warp
from openroboxing.spec.constants import COMMIT_HORIZON_TICKS, MAX_OUTSTANDING_COMMITS

if TYPE_CHECKING:  # `runtime` does not import `studio` at module level - see generator.py's note.
    from openroboxing.studio.combination_record import CombinationRecord

#: The `spec/intent.md` version this module implements. The test that pairs them is what caught
#: 2.0 shipping without a changelog entry, so the two move together or not at all.
SPEC_VERSION = "3.1"


class IntentError(RuntimeError):
    """An intent was malformed, or the commit rule was broken. Never recovered from silently."""


#: The clip a fighter stands in at the opening bell, **before the first commit of a round has become
#: current**. Nothing else uses it: once a commit has completed, the fighter holds *that commit's*
#: intent — its runner's final leg — which is what "no idle clip after a commit" means. Before there
#: has been a commit there is nothing to hold, and this is what a fighter does instead — a state of
#: its own, not a fallback.
#:
#: ``idle`` because it is the one clip with ``avg_root_vel = 0.0``. Standing still is a *clip*, not a
#: zero vector: upstream's ``movement_direction`` is always a unit vector, so "do not move" can only
#: be said by choosing a clip that does not travel. It must not be the ambient context clip —
#: ``walk_boxing`` carries 2.0 m/s in a fixed direction, which is `docs/ASSUMPTIONS.md` §A18, the bug
#: where fighters walked until they hit the ropes and ended a round at opposite ends of the ring.
OPENING_STANCE_CONTEXT = "idle"


@dataclass(frozen=True)
class StagedIntent:
    """What the player is currently steering: a combination and a ghost, or neither yet.

    Every field is editable until the commit fires — staging never touches a fired commit
    (``spec/intent.md`` "Staging never touches a fired commit"). 3.0 leaves exactly two channels
    where 1.0-2.2 had four: the adjustment envelope is gone because a combination carries no
    adjustment (``spec/combination.md``), and the style preset is gone because every leg runs
    :data:`~openroboxing.runtime.sequence.COMBINATION_CONTEXT` — there is nothing left to choose.
    """

    #: A name in the combination library, or ``None`` if nothing is selected yet.
    combination: str | None = None
    #: Where the combination's **last keyframe** must land: world ``(x, y)`` only. The heading is
    #: derived — it faces the opponent (``runtime/warp.py::ghost_heading``) — and is never part of
    #: what is staged, see `spec/intent.md` "Ghost heading is derived, not staged".
    ghost: tuple[float, float] | None = None

    def is_committable(self) -> bool:
        return self.combination is not None and self.ghost is not None


def _validate_ghost(ghost: tuple[float, float]) -> None:
    """A client sends this. A malformed value must fail here, not silently reach the generator."""
    if len(ghost) != 2:
        raise IntentError(f"ghost must be (x, y), got {ghost!r}")
    if not all(math.isfinite(v) for v in ghost):
        raise IntentError(f"ghost must be finite, got {ghost!r}")


@dataclass
class Commit:
    """A staged intent, frozen: play ``record``, starting wherever the fighter is, landing on
    ``ghost``.

    Unlike 1.0-2.2, a commit's span is arithmetic rather than watched-for. ``record.duration_ticks``
    is fixed the day the combination was captured, so the instant :attr:`commit_at` is known,
    :attr:`end_tick` is too (``spec/intent.md`` "A commit's span"). There is no approach to time out
    and no dwell to count, so the two-stage ``commit_at`` / ``strike_at`` distinction 1.0-2.2 needed
    is gone with them — a commit is either not started, running, or finished, and that is one field
    (``commit_at``) plus one piece of arithmetic (``end_tick``), not two clocks.

    :attr:`runner` is built exactly once, the first time :meth:`IntentTimeline.generator_intent`
    finds this commit current. It re-warps from the fighter's *true* position at that tick rather
    than trusting wherever the ghost was aimed when the player committed (``spec/intent.md``
    "Off-target execution") — which is why building it needs a live ``anchor`` callback rather than a
    value stashed at issue time.
    """

    #: The combination this commit plays, in full — footwork, poses and timing all come from it.
    record: CombinationRecord
    #: Where the combination's last keyframe must land: world ``(x, y)``.
    ghost: tuple[float, float]
    #: The tick the player pressed commit.
    issued_at: int
    #: The tick this commit became current and started running. ``None`` until it does.
    commit_at: int | None = None
    #: ``commit_at + record.duration_ticks``, stamped in the same step as ``commit_at`` because it
    #: is exact arithmetic the instant that is known — never watched for, unlike 2.2's ``ended_at``.
    #: Kept as a plain field (not only derived by :attr:`end_tick`) so a match record or a replay can
    #: read a commit's history off its attributes the same way it reads ``commit_at``.
    ended_at: int | None = None
    #: The warped, sequenced motion this commit runs once it has started. ``None`` until then.
    runner: CombinationRunner | None = None

    def __post_init__(self) -> None:
        _validate_ghost(self.ghost)

    @property
    def end_tick(self) -> int | None:
        """The tick this commit finishes, or ``None`` before it has started.

        Computed the instant :attr:`commit_at` is, not stamped when a body settles — the reversal
        `spec/intent.md` 3.0 makes over 2.2's "the queue is not a schedule". ``None`` here means
        "has not started yet", not "unknown": once set, this value is exact and is never revised.
        """
        return self.ended_at

    def is_scheduled(self, tick: int) -> bool:
        """True while this commit exists and has not finished — issued, queued, or running."""
        if tick < self.issued_at:
            return False
        end = self.end_tick
        return end is None or tick < end

    def is_executing(self, tick: int) -> bool:
        """True once this commit's motion is actually under way."""
        if self.commit_at is None or tick < self.commit_at:
            return False
        end = self.end_tick
        return end is None or tick < end


class IntentTimeline:
    """The commit queue for one fighter.

    Args:
        library: every combination this fighter may commit, keyed by the name a player selects —
            the whole shared library, both fighters identical, per D6 (``spec/intent.md``).
        horizon_ticks: **minimum** lead from commit to execution. Defaults to the canonical
            :data:`~openroboxing.spec.constants.COMMIT_HORIZON_TICKS`; a match may parameterise it.
        max_outstanding: how many commits may be unfinished at once. Defaults to
            :data:`~openroboxing.spec.constants.MAX_OUTSTANDING_COMMITS`.
        require_admitted: whether every combination in ``library`` must be ``"admitted"``. A match
            always requires it; the Studio sets it False so a draft combination can be rehearsed
            before it has been measured (``spec/intent.md`` "Admission is enforced at construction").
    """

    def __init__(
        self,
        library: Mapping[str, CombinationRecord],
        *,
        horizon_ticks: int = COMMIT_HORIZON_TICKS,
        max_outstanding: int = MAX_OUTSTANDING_COMMITS,
        require_admitted: bool = True,
    ) -> None:
        if horizon_ticks < 0:
            raise IntentError(f"horizon_ticks must not be negative, got {horizon_ticks}")
        if max_outstanding < 1:
            raise IntentError(f"max_outstanding must be at least 1, got {max_outstanding}")
        if not library:
            raise IntentError("the combination library is empty")
        if require_admitted:
            for name, record in library.items():
                if record.admission != "admitted":
                    raise IntentError(
                        f"combination {name!r} is {record.admission!r}; a match may only use "
                        "admitted combinations"
                    )

        self.library: dict[str, CombinationRecord] = dict(library)
        self.horizon_ticks = horizon_ticks
        self.max_outstanding = max_outstanding
        self._staged = StagedIntent()
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

    def _resolve(self, name: str) -> CombinationRecord:
        if name not in self.library:
            raise IntentError(
                f"combination {name!r} is not in the library; it has {sorted(self.library)}"
            )
        return self.library[name]

    def stage(
        self,
        *,
        combination: str | None = None,
        ghost: tuple[float, float] | None = None,
    ) -> StagedIntent:
        """Edit the staged intent. Always allowed, including while a commit is executing.

        Only the channels passed are changed; the rest keep their current value. Staging never
        affects the fighter — it is the player's private aim, and it fires only on :meth:`commit`.
        """
        if combination is not None:
            self._resolve(combination)  # fail at staging time, not at commit time
        if ghost is not None:
            _validate_ghost(ghost)
        self._staged = StagedIntent(
            combination=combination if combination is not None else self._staged.combination,
            ghost=ghost if ghost is not None else self._staged.ghost,
        )
        return self._staged

    def clear_combination(self) -> StagedIntent:
        """Unstage the combination, leaving the ghost. Not a cancellation — nothing has fired."""
        self._staged = StagedIntent(combination=None, ghost=self._staged.ghost)
        return self._staged

    # -- the queue ----------------------------------------------------------------------------------
    def scheduled(self, tick: int) -> tuple[Commit, ...]:
        """Every commit that exists and has not finished at ``tick``, in queue order.

        This is what the queue bound counts and what a client is shown. A commit leaves it by
        finishing, never by being cancelled.
        """
        return tuple(c for c in self._commits if c.is_scheduled(tick))

    def _current(self, tick: int) -> Commit | None:
        """The commit that should be driving the fighter at ``tick``, started or not.

        The **oldest unfinished** one, once its readable window has elapsed. ``None`` means hold:
        either the queue is drained, or the only thing in it was committed less than
        ``horizon_ticks`` ago.

        A commit still inside its window blocks the queue rather than being skipped over. Letting a
        later commit start first would reorder them, and order is the one thing a player controls
        about a queue they cannot cancel.

        No separate "has this finished" pass runs first, unlike 2.2: :meth:`Commit.is_scheduled`
        already excludes the boundary tick itself, so a commit that ends at tick *T* is simply absent
        from consideration at *T* and the next one in the queue is found here directly.
        """
        for commit in self._commits:
            if not commit.is_scheduled(tick):
                continue  # finished; look further down the queue
            if commit.commit_at is None and tick < commit.issued_at + self.horizon_ticks:
                return None  # its window has not elapsed, and nothing may jump ahead of it
            return commit
        return None

    def commit(self, now: int) -> Commit:
        """Queue the staged intent.

        The move begins no earlier than ``now + horizon_ticks`` — the floor that makes an isolated
        commit readable — and no earlier than the commit in front of it finishes.

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
            raise IntentError(
                f"cannot commit at tick {now}: {len(outstanding)} moves are already queued, the "
                f"limit is {self.max_outstanding}. A slot frees when the move in front finishes. "
                "No cancellation."
            )
        if self._staged.combination is None:
            raise IntentError("cannot commit: no combination is staged")
        if self._staged.ghost is None:
            raise IntentError("cannot commit: no ghost is staged")

        record = self._resolve(self._staged.combination)
        commit = Commit(record=record, ghost=self._staged.ghost, issued_at=now)
        self._commits.append(commit)
        return commit

    # -- driving the generator -----------------------------------------------------------------------
    def generator_intent(
        self,
        tick: int,
        *,
        facing_angle: float | None = None,
        anchor: Callable[[], tuple[tuple[float, float], float]] | None = None,
    ) -> GeneratorIntent:
        """The control signals for the tick **this frame will be played at**, advancing the queue.

        Ask for the tick the frame is *for*, not the tick you happen to be generating on. Frames are
        produced ahead of the tick that consumes them, so asking about "now" slides every move a
        fixed lookahead late. ``runtime/reference.py`` owns that mapping.

        **This is the timeline's clock, not a query.** Calling it is what starts a commit and builds
        its runner, so it decides when the next commit in the queue begins. Call it exactly once per
        generated frame, with non-decreasing ticks — which is enforced below, because a caller that
        drifts backwards would rewrite a move's history rather than fail.

        What comes back, by state:

        - **a commit is current** — the live leg of its :class:`~openroboxing.runtime.sequence.
          CombinationRunner`, built the first time this method finds the commit current.
        - **nothing current, but a commit has completed** — that commit's runner's **final leg**,
          held forever: the fighter stands in the combination's last pose.
        - **nothing current and none has completed** — the round's opening stance,
          :data:`OPENING_STANCE_CONTEXT`, the only place an idle clip is still used.

        Args:
            facing_angle: **world-frame** bearing to the opponent, measured this tick by whoever
                owns the world. It is what the fighter faces in every state — the opening stance, a
                running commit's live leg, and a held final leg alike (owner, 2026-09-03: a fighter
                is always turned towards the fighter it is boxing, reversing design D5's recorded
                heading). ``None`` means there is no opponent, as on a lone-fighter bench: a
                combination then keeps its own recorded heading and the stance faces ``0.0``.
            anchor: ``() -> ((x, y), heading)``, the fighter's *true* position and heading right now.
                Called **exactly once per commit**, the tick that commit starts, to build its runner
                — never once per tick, and never for a commit that is already running. Required for
                any commit that is starting: a combination that starts in place cannot be built
                without knowing where "in place" is, and defaulting to the origin would silently
                teleport the fighter.
        """
        if tick < self._driven_to:
            raise IntentError(
                f"generator_intent went backwards: tick {tick} after {self._driven_to}. It advances "
                "the commit queue, so a repeated or out-of-order tick would rewrite a move's span."
            )
        self._driven_to = tick

        commit = self._current(tick)
        if commit is None:
            return self._hold_intent(tick, facing_angle)


        if commit.commit_at is None:
            if anchor is None:
                raise IntentError(
                    f"commit {commit.record.name!r} starts at tick {tick} but no anchor was passed "
                    "to generator_intent, so 'in place' has no place to start from and everything "
                    "queued behind it would wait forever. Pass anchor= from whoever knows where the "
                    "fighter is."
                )
            position, heading = anchor()
            commit.commit_at = tick
            commit.ended_at = tick + commit.record.duration_ticks
            legs = warp(commit.record, position, heading, commit.ghost, speed_ceiling=None)
            commit.runner = CombinationRunner(commit.record, legs, commit_at=tick)

        return commit.runner.intent_for(tick, facing_angle)

    def _hold_intent(self, tick: int, facing_angle: float | None) -> GeneratorIntent:
        """What a fighter with nothing current does. **Two states, not one:**

        - **a commit has completed** — hold *that commit's* runner at its final leg.
        - **none has yet** — the round's opening stance, :data:`OPENING_STANCE_CONTEXT`.

        They are different because the fighter is in a different situation, not because one is a
        degenerate form of the other — see the module docstring.

        **Re-issuing the last commit's final leg is the entire implementation of holding a pose.**
        There is no idle clip, no freeze branch and no repeated-frame counter — the runner simply
        keeps being asked for the leg live at ``tick``, which past its own end is a fixed point
        (:meth:`~openroboxing.runtime.sequence.CombinationRunner.leg_index`).

        The last commit in queue order with a runner is exactly the last *completed* one whenever
        this is reached: if :meth:`_current` returned ``None``, every commit that has started must
        already be finished — a still-running one would have been current instead, since commits
        execute strictly in order with no gaps (the hand-over rule above).
        """
        for commit in reversed(self._commits):
            if commit.runner is not None:
                return commit.runner.intent_for(tick, facing_angle)
        return GeneratorIntent(
            style=OPENING_STANCE_CONTEXT,
            facing_angle=0.0 if facing_angle is None else facing_angle,
        )
