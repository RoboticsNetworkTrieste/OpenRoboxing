"""A fighter's reference motion: generator frames pulled, resampled, kept ahead of the tick.

Extracted from :mod:`openroboxing.runtime.world` when the two-fighter match loop needed the same
thing per fighter. There is now **one path**: pull a frame, and replan at the ambient
:data:`REPLAN_DT` cadence whatever the intent says. The stream needs no idea what a commit is.

Why there is no forced-plan rule any more
-----------------------------------------
MotionBricks in-betweens from the current context to a target pose, and **the target is the plan's
last frame**. Under ``spec/intent.md`` 1.1 that made a strike a plan you had to consume exactly:
replanning part-way through discarded the tail the pose lived in, so the module forced one plan per
commit, bound it to that commit (``_plan_key``), counted its frames
(``_committed_plan_length``, which asserted ``plan_frames == horizon_tokens * 4``) and tolerated a
single frame of overrun (``MAX_HELD_STRIKE_FRAMES``, which raised on the second).

Under ``spec/intent.md`` 2.0 the pose is armed on **every** replan for a commit's whole life, with
the length chosen by the model (``horizon_tokens=None``). "Consume one plan whole" is replaced by
"keep re-aiming at the same target", and it converges without any of that machinery. Measured
(:func:`openroboxing.studio.rehearsal.rehearse_approach`, 2026-08-13), armed against a placement
2.5 m away, ``hook-right`` closes 2.500 m -> 0.028 m and 17.0 deg -> 6.0 deg and then *holds*;
unarmed over the same 6 s it never converges at all (17.0 deg -> 18.5 deg). So a plan outliving its
move is not an error to bound — it is what holding a pose looks like.

Three defects went with that code: an 8-token pose losing its final frame (the authored pose) on
20 % of tick alignments; that lost frame then playing into the *next* commit's approach; and the
end-of-strike replan bypassing the ambient cadence.

Frames are for a tick in the future
-----------------------------------
A fighter runs a **queue** of committed moves, and which one is running at a given tick is decided by
the timeline as it is asked. The stream produces frames :data:`LOOKAHEAD_TICKS` ahead of the tick
that plays them, so it must ask **about the tick each frame is for**, not the tick it happens to be
filling on. :meth:`ReferenceStream.ensure` therefore takes a *callable*, not an intent.

Getting this wrong does not crash. It slides every move a fixed lookahead late — and since the same
call is what advances the queue, it would also start and end every approach a lookahead out of step
with the fighter it is steering.

Conventions
-----------
- Generator frames come out at :data:`GENERATOR_HZ`; :attr:`ReferenceStream.motion` is at
  :data:`TICK_HZ`, MuJoCo order, ``(N, 36)``.
- The stream is always kept :data:`LOOKAHEAD_TICKS` **plus a margin** ahead of the consumed tick,
  because the encoder samples the reference 45 ticks in front of now. Falling behind raises rather
  than padding (`CLAUDE.md` invariant 5).
"""

from __future__ import annotations

import numpy as np

from openroboxing.runtime.bridge import (
    ENCODER_FRAME_STRIDE,
    joint_velocities,
    resample_qpos,
)
from openroboxing.spec.constants import (
    GENERATOR_HZ,
    HISTORY_LEN,
    NUM_JOINTS,
    QPOS_DIM,
    TICK_HZ,
)

#: Reference ticks the encoder looks ahead: ``(HISTORY_LEN - 1) * stride`` = 45.
LOOKAHEAD_TICKS = (HISTORY_LEN - 1) * ENCODER_FRAME_STRIDE

#: Generator frames kept in front of the consumed tick, over and above the lookahead.
GENERATOR_MARGIN_FRAMES = 12

#: Ambient replan interval, in seconds. Shared with `studio.rehearsal` so a physics trial and a
#: rehearsal see the same cadence. It applies to every intent, armed or not.
REPLAN_DT = 0.5

#: Generator calls one :meth:`ReferenceStream.ensure` may make before it is declared stuck. A plan is
#: tens of frames, so a fill needing thousands means the generator is returning nothing.
_FILL_GUARD = 10_000


class ReferenceError(RuntimeError):
    """The reference motion could not be produced. Never recovered from silently."""


class ReferenceStream:
    """The reference motion for one fighter, kept ahead of the tick that consumes it.

    Args:
        generator: a :class:`~openroboxing.runtime.generator.MotionBricksGenerator`, already built.
        replan_dt: ambient replan interval in seconds. It applies to every intent.
    """

    def __init__(self, generator, replan_dt: float = REPLAN_DT) -> None:
        if replan_dt <= 0.0:
            raise ReferenceError(f"replan_dt must be positive, got {replan_dt}")
        self.generator = generator
        self.replan_dt = replan_dt
        self.motion = np.zeros((0, QPOS_DIM))
        self.velocities = np.zeros((0, NUM_JOINTS))
        self._frames: list[np.ndarray] = []

    def reset(self) -> None:
        """Drop the buffered motion. Does **not** reseed the generator — that is its owner's call."""
        self.motion = np.zeros((0, QPOS_DIM))
        self.velocities = np.zeros((0, NUM_JOINTS))
        self._frames = []

    def ensure(self, intent_at, tick: int, ticks_ahead: int = LOOKAHEAD_TICKS) -> None:
        """Pull frames until the motion covers ``ticks_ahead`` beyond ``tick``.

        Args:
            intent_at: ``tick -> GeneratorIntent``, asked once per frame **for the tick that frame
                will be played at**. Usually
                :meth:`~openroboxing.runtime.intents.IntentTimeline.generator_intent`. A commit arms
                ``intent.pose``; that changes what is generated, not how it is pulled.
            tick: the control tick about to be consumed.
            ticks_ahead: how far in front of ``tick`` the motion must reach.
        """
        frames_ahead = int(np.ceil(ticks_ahead * GENERATOR_HZ / TICK_HZ)) + GENERATOR_MARGIN_FRAMES
        needed = self.frame_cursor(tick) + frames_ahead
        if len(self._frames) >= needed:
            return  # already covered; resampling again would be identical work

        guard = 0
        while len(self._frames) < needed:
            guard += 1
            if guard > _FILL_GUARD:
                raise ReferenceError(
                    f"the generator produced {len(self._frames)} frames in {_FILL_GUARD} calls, "
                    f"short of the {needed} this tick needs"
                )

            intent = intent_at(self.tick_of_frame(len(self._frames)))
            self._frames.append(self.generator.next_frame())
            self._plan(intent, force=False)

        self.motion = resample_qpos(
            np.asarray(self._frames), source_hz=GENERATOR_HZ, target_hz=TICK_HZ
        )
        self.velocities = joint_velocities(self.motion)

    def _plan(self, intent, *, force: bool) -> None:
        self.generator.generate(
            intent, self.generator.context_qpos(), dt=self.replan_dt, force=force
        )

    @staticmethod
    def frame_cursor(tick: int) -> int:
        """Where a control tick sits, expressed in generator frames."""
        return int(np.ceil(tick * GENERATOR_HZ / TICK_HZ))

    @staticmethod
    def tick_of_frame(index: int) -> int:
        """Which control tick a generator frame is produced for — the inverse of :meth:`frame_cursor`.

        30 Hz and 50 Hz do not divide, so the round trip is exact only where the grids coincide;
        elsewhere this rounds up and a commit's intent can take effect **at most one frame (33 ms)
        early**. That error does not accumulate: each commit's start is computed from its own
        ``commit_at``, so a move is at worst a frame early or a frame late and the next one
        re-derives from the schedule.
        """
        return int(np.ceil(index * TICK_HZ / GENERATOR_HZ))

    def require(self, tick: int, ticks_ahead: int = LOOKAHEAD_TICKS) -> None:
        """Raise unless the motion reaches ``ticks_ahead`` past ``tick``.

        The encoder reads that far in front; a short buffer would be silently clamped to the last
        frame by :func:`~openroboxing.runtime.bridge.lookahead_indices`, which looks like a fighter
        freezing mid-move for no reason.
        """
        if tick + ticks_ahead >= self.motion.shape[0]:
            raise ReferenceError(
                f"reference motion has {self.motion.shape[0]} ticks; tick {tick} needs "
                f"{tick + ticks_ahead + 1}"
            )
