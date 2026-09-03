"""CombinationRunner: which leg of a warped combination is live at a tick (M6-T3).

Pure arithmetic — no generator, no physics, no MuJoCo. Turns the ``list[Leg]`` that
``runtime/warp.py`` produces into the one thing the timeline needs each tick: the live leg's
:class:`~openroboxing.runtime.generator.GeneratorIntent`. Wiring this into the timeline (arming a
runner from a commit, advancing it tick by tick) is Task 5, not here.

Conventions
-----------
- **Legs are measured in tokens; the timeline runs in ticks.** One token is
  ``SECONDS_PER_TOKEN * TICK_HZ`` = (4/30) x 50 = 6.667 ticks. Leg ``i`` (0-indexed) ends at::

      commit_at + round(cumulative_tokens_through_leg_i * SECONDS_PER_TOKEN * TICK_HZ)

  the same expression ``CombinationRecord.duration_ticks`` already uses, rounded once against the
  *cumulative* token count rather than accumulated from per-leg roundings — so the final boundary
  equals ``commit_at + record.duration_ticks`` exactly, by construction, and no leg's rounding can
  drift the next leg's start. Boundaries are derived once at construction, not counted from calls:
  :meth:`CombinationRunner.leg_index` is queried with the tick a frame is *for*, and
  ``generator_intent`` is called once per generated frame at 30 Hz with the 50 Hz tick that frame
  targets — a per-call counter would drift the moment a frame is skipped.
- **A fighter always faces its opponent** (owner, 2026-09-03, reversing design D5). The heading is
  not warped and not recorded: the world measures the bearing to the opponent every tick and passes
  it to :meth:`CombinationRunner.intent_for`, which uses it for the target frame's heading and for
  the facing signal alike. A recorded turn still moves the *body* - it is in the keyframe joint
  angles and in the footwork - it just no longer aims the fighter at the ropes.
- **A finished runner holds its last leg.** Past the final boundary, :meth:`leg_index` keeps
  returning the last leg forever — the existing runtime's "holding a pose is the same target
  re-armed" behaviour, and this is not a special case of it.
- **The style is always ``walk_boxing``.** Measured 2026-08-28: ``walk`` permits only 6-11 tokens
  (``narrow_allowed_tokens`` raises on 12) while legs run to 16, so ``walk_boxing`` is the only clip
  that can express a forced leg length. This does not reintroduce ``CLAUDE.md``'s warning that
  ``walk_boxing`` leaves a fighter with no sideways gait: here travel comes from ``target_position``
  rather than from the gait remap, and a ghost 1 m to the side is reached as well as one 1 m ahead
  (0.79 m either way, measured).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openroboxing.runtime.generator import GeneratorIntent
from openroboxing.spec.constants import SECONDS_PER_TOKEN, TICK_HZ

if TYPE_CHECKING:  # `runtime` does not import `studio` at module level - see generator.py's note.
    from openroboxing.runtime.warp import Leg
    from openroboxing.studio.combination_record import CombinationRecord

#: Forced-length legs need `walk_boxing`; `walk` cannot express a 12-16 token leg. See module
#: docstring for the measurement.
COMBINATION_CONTEXT = "walk_boxing"


class SequenceError(RuntimeError):
    """A runner was asked about a tick it does not own. Never recovered from silently."""


def _tokens_to_ticks(tokens: float) -> int:
    """Tokens to ticks, the same rounding ``CombinationRecord.duration_ticks`` uses."""
    return round(tokens * SECONDS_PER_TOKEN * TICK_HZ)


class CombinationRunner:
    """Which leg of a warped combination is live at a tick, and the intent it produces.

    Built from a :class:`~openroboxing.studio.combination_record.CombinationRecord` (for its
    keyframe count and token durations) and the matching ``list[Leg]`` that
    ``runtime.warp.warp(record, ...)`` produced from it.
    """

    def __init__(self, record: CombinationRecord, legs: list[Leg], *, commit_at: int) -> None:
        if len(legs) != len(record.keyframes) - 1:
            raise SequenceError(
                f"{record.name}: {len(legs)} legs for {len(record.keyframes)} keyframes, "
                f"expected {len(record.keyframes) - 1} - legs and record are mismatched"
            )
        self._record = record
        self._legs = tuple(legs)
        self._commit_at = commit_at

        cumulative_tokens = 0
        boundaries: list[int] = []
        for leg in self._legs:
            cumulative_tokens += leg.horizon_tokens
            boundaries.append(commit_at + _tokens_to_ticks(cumulative_tokens))
        self._boundaries = tuple(boundaries)

    @property
    def legs(self) -> tuple[Leg, ...]:
        return self._legs

    @property
    def end_tick(self) -> int:
        """The tick at which the last leg's boundary is crossed and the runner is finished."""
        return self._boundaries[-1]

    def leg_index(self, tick: int) -> int:
        """Which leg (0-indexed) is live at ``tick``.

        Past :attr:`end_tick` this keeps returning the last leg's index - see the module docstring.

        Raises:
            SequenceError: if ``tick`` is before ``commit_at``, which this runner never owned.
        """
        if tick < self._commit_at:
            raise SequenceError(
                f"{self._record.name}: tick {tick} is before commit_at {self._commit_at}"
            )
        for index, boundary in enumerate(self._boundaries):
            if tick < boundary:
                return index
        return len(self._legs) - 1

    def is_finished(self, tick: int) -> bool:
        """Whether ``tick`` is at or past the final boundary."""
        return tick >= self.end_tick

    def intent_for(self, tick: int, facing_angle: float | None = None) -> GeneratorIntent:
        """The live leg's :class:`GeneratorIntent` at ``tick``.

        Carries ``movement_angle`` and ``facing_angle`` through separately - `CLAUDE.md`'s named
        trap is leaving the former at its default, which silently means "straight ahead, always".

        Args:
            facing_angle: the bearing to the opponent, world frame, measured this tick. It replaces
                the recorded heading in **both** places a heading reaches the generator - the target
                frame's ``target_heading`` and the ``facing_angle`` control signal - and a leg that
                does not travel (``Leg.is_still``) takes it as its ``movement_angle`` too, since a
                still leg has no direction of its own to travel in. ``None`` means *there is no
                opponent*: the Studio's rehearsal and the warp tools drive a lone fighter, and there
                the recording is the only heading there is.
        """
        index = self.leg_index(tick)
        leg = self._legs[index]
        facing = leg.facing_angle if facing_angle is None else facing_angle
        heading = leg.target_heading if facing_angle is None else facing_angle
        return GeneratorIntent(
            style=COMBINATION_CONTEXT,
            movement_angle=facing if leg.is_still else leg.movement_angle,
            facing_angle=facing,
            target_position=leg.target_position,
            target_heading=heading,
            pose=self._pose_for(leg, index),
            horizon_tokens=leg.horizon_tokens,
        )

    def _pose_for(self, leg: Leg, index: int) -> object:
        """The leg's target as a real ``PoseRecord`` - local import keeps `studio` out of `runtime`'s
        module-level imports (`CLAUDE.md`; see ``generator.py``'s note on ``GeneratorIntent.pose``).
        """
        from openroboxing.studio.pose_record import PoseRecord

        return PoseRecord(
            name=f"{self._record.name}-leg{index}",
            joint_angles=dict(leg.joint_angles),
            horizon_tokens=leg.horizon_tokens,
            library_version=self._record.library_version,
        )
