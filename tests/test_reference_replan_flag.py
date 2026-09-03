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
