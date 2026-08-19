"""M2-T3 acceptance: telegraph measurement.

Acceptance criterion from WORKPLAN.md M2-T3:
  reports a window in ms and a pass/fail against a configurable floor; two obviously different poses
  (a slow hook and a snap jab) produce clearly different windows.

Motions here are synthesised analytically rather than generated, so the test is fast and
deterministic and exercises the *measurement*, which is what M2-T3 is about.
"""

from __future__ import annotations

import numpy as np
import pytest

from openroboxing.runtime.conventions import G1
from openroboxing.spec.constants import QPOS_DIM, TICK_HZ
from openroboxing.studio.telegraph import (
    TelegraphError,
    contact_frame,
    divergence_frame,
    measure,
)

pytest.importorskip("mujoco")

ELBOW = "left_elbow_joint"
SHOULDER = "left_shoulder_pitch_joint"


def _guard(n: int = 60, jitter: float = 0.002, seed: int = 0) -> np.ndarray:
    """A guard baseline: standing, with a little natural wander."""
    from openroboxing.runtime.obs import default_angles

    rng = np.random.default_rng(seed)
    frames = np.zeros((n, QPOS_DIM))
    frames[:, 2] = 0.793
    frames[:, 3] = 1.0
    frames[:, 7:] = default_angles(G1, "mujoco")[None, :]
    frames[:, 7:] += rng.normal(0.0, jitter, size=(n, 29))
    return frames


def _punch(n: int = 60, windup_frames: int = 10, reach: float = 1.2) -> np.ndarray:
    """A strike: hold guard, then extend the arm to full reach at the final frame.

    `windup_frames` controls how early the motion starts, i.e. how long the telegraph is.
    """
    frames = _guard(n, jitter=0.0)
    elbow = G1.mujoco_joint_names.index(ELBOW)
    shoulder = G1.mujoco_joint_names.index(SHOULDER)

    start = n - 1 - windup_frames
    ramp = np.zeros(n)
    ramp[start:] = np.linspace(0.0, 1.0, n - start)
    frames[:, 7 + elbow] += ramp * reach
    frames[:, 7 + shoulder] += ramp * reach * 0.5
    return frames


# --- the acceptance criterion ---------------------------------------------------------------------
def test_reports_a_window_in_milliseconds() -> None:
    result = measure(_punch(), _guard())
    assert result.window_ms > 0
    assert result.contact_frame > result.divergence_frame
    # a 10-frame windup at 50 Hz is 200 ms; allow the threshold crossing to lag a frame or two
    assert 100.0 <= result.window_ms <= 220.0, result


def test_pass_fail_against_a_configurable_floor() -> None:
    result = measure(_punch(windup_frames=10), _guard())
    assert result.passes(floor_ms=100.0)
    assert not result.passes(floor_ms=500.0)


def test_slow_hook_and_snap_jab_differ_clearly() -> None:
    """The headline check: a slow, telegraphed move must measure far longer than a snap."""
    slow = measure(_punch(windup_frames=25), _guard())
    snap = measure(_punch(windup_frames=4), _guard())

    assert slow.window_ms > snap.window_ms
    assert (
        slow.window_ms - snap.window_ms > 100.0
    ), f"slow {slow.window_ms:.0f} ms vs snap {snap.window_ms:.0f} ms — not clearly different"


# --- the pieces -------------------------------------------------------------------------------------
def test_threshold_is_derived_from_the_baseline_not_invented() -> None:
    """A noisier guard must raise the bar for what counts as distinguishable."""
    _, quiet = divergence_frame(_punch(), _guard(jitter=0.001))
    _, noisy = divergence_frame(_punch(), _guard(jitter=0.02))
    assert noisy > quiet


def test_contact_is_the_peak_reach() -> None:
    frame, peak = contact_frame(_punch(n=60))
    assert frame >= 55, "the strike extends to the final frame, so contact should be near the end"
    assert peak > 0.1


def test_root_translation_is_not_a_telegraph() -> None:
    """Walking toward the opponent must not register as a windup."""
    walking = _guard(jitter=0.0)
    walking[:, 0] = np.linspace(0.0, 3.0, walking.shape[0])  # stride forward 3 m
    with pytest.raises(TelegraphError, match="never becomes distinguishable"):
        measure(walking, _guard())


def test_a_move_identical_to_guard_has_no_telegraph() -> None:
    with pytest.raises(TelegraphError, match="never becomes distinguishable"):
        measure(_guard(jitter=0.0), _guard())


def test_window_scales_with_the_rate() -> None:
    fast = measure(_punch(), _guard(), rate_hz=TICK_HZ)
    slow = measure(_punch(), _guard(), rate_hz=TICK_HZ / 2)
    assert slow.window_ms == pytest.approx(fast.window_ms * 2)


def test_bad_qpos_shape_raises() -> None:
    with pytest.raises(TelegraphError, match="expected"):
        contact_frame(np.zeros((10, 20)))
