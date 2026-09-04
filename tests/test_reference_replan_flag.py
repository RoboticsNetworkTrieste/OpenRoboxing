"""The reference stream must not plan on a frame whose intent says there is no hole to fill.

`spec/intent.md` 3.2: time in MotionBricks is a continuous array filled where there are holes, and a
keyframe placed in it stays put while the array moves forward. Once the hole in front of a keyframe
is shorter than ``MIN_TOKENS`` — the shortest plan the model can produce — there is nothing left to
fill, and replanning would only re-aim the keyframe past its own boundary. The runner decides that;
the stream only honours it, because the stream deliberately knows nothing about legs
(``runtime/reference.py``: "The stream needs no idea what a commit is").
"""

from __future__ import annotations

import numpy as np

from openroboxing.runtime.generator import GeneratorIntent
from openroboxing.runtime.reference import ReferenceStream
from openroboxing.spec.constants import QPOS_DIM


class _CountingGenerator:
    """Counts generate() calls; frames are inert but valid input to `bridge.resample_qpos`."""

    def __init__(self) -> None:
        self.calls = 0

    def next_frame(self) -> np.ndarray:
        qpos = np.zeros(QPOS_DIM)
        qpos[3] = 1.0  # identity quaternion, so the resampler's slerp sees valid input
        return qpos

    def generate(self, intent, context_qpos, dt, *, force: bool = False) -> None:
        self.calls += 1

    def context_qpos(self) -> np.ndarray:
        return np.zeros((1, QPOS_DIM))


def test_replan_false_suppresses_every_generate_call() -> None:
    generator = _CountingGenerator()
    stream = ReferenceStream(generator)
    stream.ensure(lambda _tick: GeneratorIntent(replan=False), tick=100, ticks_ahead=10)
    assert generator.calls == 0


def test_replan_true_still_plans() -> None:
    generator = _CountingGenerator()
    stream = ReferenceStream(generator)
    stream.ensure(lambda _tick: GeneratorIntent(replan=True), tick=100, ticks_ahead=10)
    assert generator.calls > 0


class _CursorGenerator:
    """Models upstream's play cursor and its clamp, which is where a frozen context comes from.

    ``full_agent.get_next_frame`` clamps the cursor to the plan's last index, and
    ``get_context_mujoco_qpos`` reads 4 frames from the cursor with the same clamp
    (``full_agent.py:503-521``). So a plan played to its end hands the *next* plan four copies of one
    frame — a zero-velocity context telling the model the fighter is standing still while it is in
    fact mid-combination. This stub reproduces that behaviour exactly, so the runner's ``ceil``
    rounding can be shown to avoid it rather than assumed to.
    """

    def __init__(self) -> None:
        self._plan = self._make(6, 0.0)
        self._cursor = 0
        self._generation = 0.0

    @staticmethod
    def _make(tokens: int, start: float) -> np.ndarray:
        frames = np.zeros((tokens * 4, QPOS_DIM))
        frames[:, 3] = 1.0
        # Strictly increasing, so "four distinct values" means real motion and "one value" means
        # the cursor was clamped.
        frames[:, 0] = start + np.arange(tokens * 4)
        return frames

    def next_frame(self) -> np.ndarray:
        frame = self._plan[self._cursor]
        self._cursor = max(0, min(self._cursor + 1, self._plan.shape[0] - 1))
        return frame

    def generate(self, intent, context_qpos, dt, *, force: bool = False) -> None:
        self._generation += 1.0
        self._plan = self._make(intent.horizon_tokens or 6, self._generation * 1000.0)
        self._cursor = 0

    def context_qpos(self) -> np.ndarray:
        last = self._plan.shape[0] - 1
        return np.asarray([self._plan[min(self._cursor + i, last)] for i in range(4)])


def _two_leg_runner(tokens):
    from openroboxing.runtime import sequence, warp
    from openroboxing.runtime.conventions import G1
    from openroboxing.studio import combination_record as cr

    angles = {name: 0.0 for name in G1.mujoco_joint_names}
    keyframes = [cr.Keyframe(dict(angles), None, (0.0, 0.0), 0.0)]
    for i, token in enumerate(tokens, start=1):
        keyframes.append(cr.Keyframe(dict(angles), token, (0.1 * i, 0.0), 0.0))
    rec = cr.CombinationRecord(
        name="c",
        library_version="v0.2",
        source=cr.CombinationSource("t", 0, 100, False),
        keyframes=keyframes,
    )
    legs = warp.warp(rec, (0.0, 0.0), 0.0, rec.recorded_displacement)
    return sequence.CombinationRunner(rec, legs, commit_at=0)


def test_the_context_is_never_one_frame_repeated() -> None:
    """A frozen context is silent: the fighter keeps moving, but the model plans as if it were
    standing still. Asserted across a whole combination, including a long leg and the hold."""
    run = _two_leg_runner([6, 24])
    generator = _CursorGenerator()
    stream = ReferenceStream(generator)
    frozen: list[int] = []

    plan = generator.generate

    def watched(intent, context_qpos, dt, *, force: bool = False):
        if len(np.unique(np.asarray(context_qpos)[:, 0])) == 1:
            frozen.append(len(stream._frames))
        plan(intent, context_qpos, dt, force=force)

    generator.generate = watched
    stream.ensure(run.intent_for, tick=run.end_tick + 100, ticks_ahead=10)

    assert not frozen, f"the context collapsed to a single repeated frame at frames {frozen[:10]}"

