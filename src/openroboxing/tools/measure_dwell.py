"""Measure the two numbers `spec/intent.md` 2.0 needs (M4-T4).

1. ``POSE_DWELL_TICKS`` - how long after arriving a fighter needs before the pose has settled,
   so the next queued commit does not cut the strike short.
2. The admission tolerance for ``generator_error_rad`` under continuous arming, which is looser
   than the forced-plan number every pose in ``poses/v0.1`` was admitted against.

Both are reported per pose and as a distribution. Nothing is written; the numbers go into
``spec/constants.py`` and ``spec/rates.md`` by hand, with this tool's output as their citation.

Conventions
-----------
- Errors come from :func:`~openroboxing.studio.rehearsal.rehearse_approach`, so they are **mean
  absolute joint error in radians** against the commanded pose, sampled per generator frame at
  :data:`~openroboxing.spec.constants.GENERATOR_HZ`, and distances are metres in the **generator's
  own frame** (not the MuJoCo world frame - see ``rehearse_approach``).
- The dwell is reported in **control ticks** at :data:`~openroboxing.spec.constants.TICK_HZ`,
  because that is the rate the intent queue counts in, and as a **distribution** rather than a single
  number: whether the rule is "the max, so the slowest pose completes" or something tighter is a
  design decision, and it needs the spread in front of it.
- The tolerance is read over the **final replan cycle**, never at a single frame. Each plan converges
  over its own length, so the error oscillates within every cycle and any single frame - the settle
  index above all, which is one frame at the top of the band by construction, misreads the pose.

Usage
-----
    .venv_mb/bin/python -m openroboxing.tools.measure_dwell
    .venv_mb/bin/python -m openroboxing.tools.measure_dwell --travel 2.5 --seconds 8
"""

from __future__ import annotations

import argparse

import numpy as np

from openroboxing.paths import POSE_DIR
from openroboxing.spec.constants import ARRIVAL_RADIUS_M, GENERATOR_HZ, TICK_HZ
from openroboxing.studio import pose_record
from openroboxing.studio.rehearsal import REPLAN_DT, rehearse_approach

#: The window that defines steady state, in generator frames: one replan interval, derived rather
#: than written down. Anything shorter cannot distinguish settling from the gap between two plans,
#: and each plan converges over its own length, so the error oscillates within every cycle.
SETTLE_WINDOW_FRAMES = int(round(REPLAN_DT * GENERATOR_HZ))

#: Floating-point slack when testing band membership, radians. ``steady`` is a *mean*, so a
#: perfectly flat curve does not compare equal to it (the mean of fifteen 0.1s is 0.1 + 2e-17) and a
#: zero-width band would exclude every sample. 1e-9 rad is 6e-8 degrees: below any real signal.
BAND_EPSILON_RAD = 1e-9


def settle_index(curve: np.ndarray, *, from_index: int) -> int:
    """The first index at or after ``from_index`` where ``curve`` has converged to steady state.

    Steady state is the mean over the final :data:`SETTLE_WINDOW_FRAMES` samples, and the band around
    it is that same window's peak-to-peak spread. "Converged" means the curve has fallen to
    ``steady + band`` and **stays at or below it for the rest of the run**: a single touch is an
    excursion, sustained residence is an asymptote. Returns ``from_index`` when the curve is already
    there and never rises out again.

    The band is deliberately **one-sided**, and this is the subtle part. Only an excursion *above*
    ``steady + band`` resets residence. A frame *below* steady state means the pose is being held
    better than the library average — it is arrival, not instability, and there is nothing left to
    wait for. Scoring dips as instability instead put the library's median dwell at 188 ticks (3.8 s)
    against 1 tick one-sided, because the residual oscillation dips below steady state constantly.

    Requiring residence at all is the other half. Taking the first *touch* of the curve's minimum
    instead reads the bottom of a late transient dip as the asymptote — measured on ``jab-right``,
    which arrives ~10° off, grinds down, touches 4.70° at 5.1 s and drifts back up to ~7.5°: its true
    asymptote is ~7.5°, reached far earlier, and the dip inflated its dwell to 105 ticks.

    The returned index is always a valid index into ``curve``: the final sample cannot exceed its own
    window's mean by more than that window's spread, so it is in-band by construction.
    """
    curve = np.asarray(curve, dtype=float)
    if not 0 <= from_index < curve.shape[0]:
        raise ValueError(f"from_index {from_index} outside a curve of {curve.shape[0]} samples")

    window = curve[-SETTLE_WINDOW_FRAMES:]  # the whole curve, when it is shorter than the window
    steady = float(window.mean())
    band = float(np.ptp(window))

    above = np.flatnonzero(curve[from_index:] - steady > band + BAND_EPSILON_RAD)
    # The last departure, not the first arrival: everything after it is sustained residence.
    return int(from_index + (above[-1] + 1 if above.size else 0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="measure_dwell",
        description="Measure POSE_DWELL_TICKS and the continuous-arming pose tolerance (M4-T4).",
    )
    parser.add_argument("--travel", type=float, default=2.5, help="metres to the placement")
    parser.add_argument("--seconds", type=float, default=8.0, help="how long to rehearse each pose")
    parser.add_argument("--library", default="v0.1", help="pose library under poses/")
    parser.add_argument("--seed", type=int, default=1234, help="generator seed")
    args = parser.parse_args(argv)

    from openroboxing.runtime.generator import GeneratorConfig, MotionBricksGenerator

    records = pose_record.load_library(POSE_DIR / args.library)
    print(f"{len(records)} poses from {args.library}, placement {args.travel} m away\n")

    # One generator for the whole library: building one loads a checkpoint onto the GPU, and
    # `rehearse_approach` calls `reset(seed=...)` itself, so no state carries between poses.
    generator = MotionBricksGenerator(GeneratorConfig(random_seed=args.seed))

    print(
        f"{'pose':<16} {'arrive_s':>9} {'settle_s':>9} {'dwell_ticks':>12} "
        f"{'at_settle':>10} {'cyc_mean':>9} {'cyc_max':>8}"
    )

    dwells: list[int] = []
    errors: list[float] = []  # at the settle index: one frame, kept only to show the working
    cycle_means: list[float] = []
    cycle_maxima: list[float] = []  # steady state, and what the tolerance is actually set from
    for name in sorted(records):
        result = rehearse_approach(
            records[name],
            travel_m=args.travel,
            seconds=args.seconds,
            seed=args.seed,
            generator=generator,
        )

        inside = np.flatnonzero(result.distance_to_goal <= ARRIVAL_RADIUS_M)
        if inside.size == 0:
            print(
                f"{name:<16} {'never':>9} {'-':>9} {'-':>12} "
                f"{np.degrees(result.pose_error_rad[-1]):>9.1f}d {'-':>9} {'-':>8}"
            )
            continue

        arrived = int(inside[0])
        settled = settle_index(result.pose_error_rad, from_index=arrived)
        dwell_ticks = int(np.ceil((settled - arrived) / GENERATOR_HZ * TICK_HZ))
        final = float(result.pose_error_rad[settled])
        # Steady state read over a whole replan cycle. The value at the settle index is the frame the
        # error last came down through the band, so it sits at the band's top by construction - a
        # single frame, and not a reading the tolerance may rest on either way.
        cycle = result.pose_error_rad[-SETTLE_WINDOW_FRAMES:]
        cycle_mean, cycle_max = float(cycle.mean()), float(cycle.max())

        dwells.append(dwell_ticks)
        errors.append(final)
        cycle_means.append(cycle_mean)
        cycle_maxima.append(cycle_max)
        print(
            f"{name:<16} {arrived / GENERATOR_HZ:>9.2f} {settled / GENERATOR_HZ:>9.2f} "
            f"{dwell_ticks:>12} {np.degrees(final):>9.1f}d {np.degrees(cycle_mean):>8.1f}d "
            f"{np.degrees(cycle_max):>7.1f}d"
        )

    if not dwells:
        print("\nno pose arrived; nothing to report")
        return 1

    # Ticks are whole, so every percentile is rounded up: a dwell of 20.4 ticks is not satisfied
    # until tick 21.
    def _ticks(percentile: float) -> int:
        return int(np.ceil(np.percentile(dwells, percentile)))

    print(f"\ndwell over {len(dwells)} poses, ticks at {TICK_HZ} Hz:")
    print(
        f"  min {min(dwells)}   median {_ticks(50)}   p90 {_ticks(90)}   max {max(dwells)}"
        f"   ({min(dwells) / TICK_HZ:.2f}s .. {max(dwells) / TICK_HZ:.2f}s)"
    )
    print(f"  all: {sorted(dwells)}")

    print("\npose error, degrees:")
    print(
        f"  at the settle index   {np.degrees(np.mean(errors)):.1f} mean, "
        f"{np.degrees(max(errors)):.1f} worst   <- one frame, at the band's top; not the tolerance"
    )
    print(
        f"  over the final cycle  {np.degrees(np.mean(cycle_means)):.1f} mean, "
        f"{np.degrees(max(cycle_maxima)):.1f} worst   <- steady state"
    )
    print(
        f"\nGENERATOR_POSE_TOLERANCE_RAD >= {max(cycle_maxima):.4f}  "
        f"({np.degrees(max(cycle_maxima)):.1f}d, the worst pose's worst frame of its final "
        f"{SETTLE_WINDOW_FRAMES}-frame replan cycle)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
