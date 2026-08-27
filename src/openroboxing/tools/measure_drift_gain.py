"""Measure the drift gain: what fraction of the ADDED drift MotionBricks actually covers (M6-T1).

A warped combination (``runtime/warp.py``) keeps the rotated recorded offsets at true size (design
D4) and scales only the leftover travel — the residual. So the quantity `runtime/warp.py`'s
correction needs is how well the generator covers the *residual*, not how well it covers the
combination's total displacement.

**Revision 2026-08-28: the first framing conflated the two.** The original measurement computed
``reached / asked`` where ``asked = hypot(recorded_displacement + drift)``. For a `shadow-boxing`
combination (2 cm of its own recorded travel) that is almost entirely a measurement of the drift, but
for an `ib-combat-turn-jog` combination (up to ~4.3 m of recorded travel) the drift is a small
addition to a much larger number, so the fraction mostly measures how well the recorded path itself
is covered — a different and much better-understood quantity (the phase-1 spike already showed the
recorded path is reached to within a few percent; see ``docs/perf/2026-08-27-warp-tracking-spike.md``
and the parity harness). That is almost certainly why the jog family's ``reached/asked`` sat
systematically low: not a property of the drift, but of averaging in a large denominator the residual
gain never touches.

**The fix — measure the increment.** For each combination, a **baseline run at drift = 0** first:
ghost exactly ``recorded_displacement``, so the residual is zero and every leg aims at the recorded
path alone. Call the reached distance there ``R0``. Then for each drift ``d``, with reached distance
``R(d)``::

    incremental_gain = (R(d) - R0) / d

This isolates coverage of the *added* travel alone, and it cancels any per-combination baseline
offset in the recorded path — which is exactly the quantity Task 2 needs, because `warp()` divides
only the residual by the gain, never the recorded offsets.

Both metrics are computed and printed side by side: the superseded ``reached/asked`` fraction (kept
so the write-up shows why it misled) and ``incremental_gain`` (the one the decision is made on).

This drives the generator directly, exactly as ``tools.spike_warp_tracking`` does — no physics, no
policy.

Method
------
For each sampled combination:

1. **Baseline.** ``warp(record, (0, 0), 0.0, recorded_displacement, speed_ceiling=1e9)``, drive every
   leg, and record ``R0 = hypot(*final_plan_frame_root_xy)``.
2. For each drift distance in 0.25/0.5/1.0/2.0 m: ``ghost = recorded_displacement + drift *
   direction``, where ``direction`` is the unit vector of ``recorded_displacement`` (or ``(1, 0)`` if
   that displacement is under ``1e-6`` m — the same convention ``spike_warp_tracking`` uses). Warp
   and drive exactly as the baseline did, recording ``R(d)``.
3. ``asked = hypot(*ghost)``, ``fraction = R(d) / asked`` (superseded), ``incremental_gain =
   (R(d) - R0) / d`` (the metric this task decides on).

``speed_ceiling=1e9`` disables the ceiling with a value no real placement can exceed; the
``speed_ceiling=None`` escape hatch is Task 2, not yet in ``warp``'s signature. Every leg is driven
with the plan length forced to the recorded leg duration (as the phase-1 spike did), consuming every
generated frame so the next leg's context follows on. The same seed drives the baseline and every
drift for a given combination, so the runs are comparable.

Combinations are sampled deterministically: files are sorted per family and every
``len(files) // per_family``-th one is taken, so the run is reproducible without depending on
``--seed`` (which only reseeds the generator's own RNG for each combination).

Run: ``.venv_mb/bin/python -m openroboxing.tools.measure_drift_gain``
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from openroboxing.paths import COMBINATION_DIR
from openroboxing.runtime.generator import GeneratorIntent, MotionBricksGenerator
from openroboxing.runtime.warp import WarpError, warp
from openroboxing.spec.constants import GENERATOR_DT
from openroboxing.studio import combination_record as cr
from openroboxing.studio.pose_record import PoseRecord

#: The three families in the library, by filename prefix (`CLAUDE.md`).
DEFAULT_PREFIXES = ("shadow-boxing", "ib-dodge", "ib-combat-turn-jog")

#: Drift distances to probe, metres. Matches the preliminary measurement this task follows up on.
DRIFT_DISTANCES_M = (0.25, 0.5, 1.0, 2.0)

#: Below this, `recorded_displacement` has no direction; fall back to +x. Mirrors
#: `spike_warp_tracking`'s `DRIFT_EXTRA_M` handling and `warp.py`'s `STILL_LEG_M` convention.
NO_DIRECTION_M = 1e-6

#: A ceiling no real placement can exceed, standing in for the `speed_ceiling=None` this task's
#: sibling (Task 2) adds to `warp`. Without it every drifted placement above ~0.5 m/s would raise
#: `WarpError` before the fraction could even be measured.
DISABLED_SPEED_CEILING = 1e9


@dataclass(frozen=True)
class Sample:
    """One (combination, drift) measurement, alongside that combination's drift=0 baseline."""

    family: str
    name: str
    drift_m: float
    asked_m: float
    reached_m: float
    r0_m: float
    """Reached distance at drift=0 for this same combination (ghost = recorded_displacement)."""

    @property
    def fraction(self) -> float:
        """Superseded: total-displacement coverage, ``reached / asked``.

        Conflates the combination's own recorded travel with the added drift — see the module
        docstring's 2026-08-28 revision note. Kept only so the write-up can show why it misled.
        """
        return self.reached_m / self.asked_m

    @property
    def incremental_gain(self) -> float:
        """Coverage of the added drift alone: ``(reached - baseline) / drift``. The decision metric."""
        return (self.reached_m - self.r0_m) / self.drift_m


def pick_records(
    prefixes: tuple[str, ...], per_family: int
) -> list[tuple[str, cr.CombinationRecord]]:
    """``per_family`` records per prefix, chosen by even stride so the sample is reproducible."""
    files = sorted(COMBINATION_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"no combinations in {COMBINATION_DIR}; run tools.import_motions first")
    chosen: list[tuple[str, cr.CombinationRecord]] = []
    for prefix in prefixes:
        matches = [p for p in files if p.stem.startswith(prefix)]
        if len(matches) < per_family:
            raise SystemExit(
                f"only {len(matches)} combinations start {prefix!r}, need {per_family}"
            )
        stride = len(matches) // per_family
        for i in range(per_family):
            chosen.append((prefix, cr.load(matches[i * stride])))
    return chosen


def leg_pose(joint_angles: dict, horizon_tokens: int, name: str) -> PoseRecord:
    """A leg's target as the ``PoseRecord`` the override consumes. See ``spike_warp_tracking``."""
    return PoseRecord(
        name=name,
        joint_angles=dict(joint_angles),
        horizon_tokens=horizon_tokens,
        library_version="v0.2",
    )


def drive_to_ghost(
    generator: MotionBricksGenerator,
    record: cr.CombinationRecord,
    ghost: tuple[float, float],
    *,
    seed: int,
) -> tuple[float, float]:
    """Drive one combination leg by leg toward ``ghost``. Returns the final plan frame's root (x, y).

    Raises ``WarpError`` if ``warp`` itself refuses the placement (should not happen with the
    disabled ceiling, other than a zero-duration combination, which the library does not contain).
    """
    generator.reset(seed)
    legs = warp(record, (0.0, 0.0), 0.0, ghost, speed_ceiling=DISABLED_SPEED_CEILING)
    final_xy: tuple[float, float] | None = None
    for index, leg in enumerate(legs):
        intent = GeneratorIntent(
            style="walk_boxing",
            movement_angle=leg.movement_angle,
            facing_angle=leg.facing_angle,
            target_position=leg.target_position,
            target_heading=leg.target_heading,
            pose=leg_pose(leg.joint_angles, leg.horizon_tokens, f"{record.name}-leg{index}"),
            horizon_tokens=leg.horizon_tokens,
        )
        # force=True: this measures the plan, not the replan schedule (spike_warp_tracking).
        generator.generate(intent, generator.context_qpos(), GENERATOR_DT, force=True)
        plan = generator.plan()
        final_xy = (float(plan[-1, 0]), float(plan[-1, 1]))
        # Consume the plan so the next leg's context is where this one left off.
        for _ in range(len(plan)):
            generator.next_frame()
    assert final_xy is not None  # a validated record always has at least one leg
    return final_xy


def _direction(displacement: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*displacement)
    if length < NO_DIRECTION_M:
        return (1.0, 0.0)
    return (displacement[0] / length, displacement[1] / length)


def measure(
    generator: MotionBricksGenerator,
    records: list[tuple[str, cr.CombinationRecord]],
    drifts: tuple[float, ...],
    *,
    seed: int,
) -> tuple[list[Sample], int]:
    """Run every combination's baseline plus every (combination, drift) pair.

    Returns the samples (one per drift, each carrying its combination's baseline) and a count of
    skipped (combination, distance) pairs. A baseline failure skips every drift for that combination.
    """
    samples: list[Sample] = []
    skipped = 0
    for family, record in records:
        try:
            r0_xy = drive_to_ghost(generator, record, record.recorded_displacement, seed=seed)
        except WarpError as exc:
            print(f"  skip {record.name} baseline (drift=0): {exc}")
            skipped += len(drifts)
            continue
        r0 = math.hypot(*r0_xy)

        direction = _direction(record.recorded_displacement)
        dx, dy = record.recorded_displacement
        for drift in drifts:
            ghost = (dx + direction[0] * drift, dy + direction[1] * drift)
            try:
                reached_xy = drive_to_ghost(generator, record, ghost, seed=seed)
            except WarpError as exc:
                print(f"  skip {record.name} @ {drift:.2f} m: {exc}")
                skipped += 1
                continue
            samples.append(
                Sample(
                    family=family,
                    name=record.name,
                    drift_m=drift,
                    asked_m=math.hypot(*ghost),
                    reached_m=math.hypot(*reached_xy),
                    r0_m=r0,
                )
            )
    return samples, skipped


def _stats(values: list[float]) -> tuple[int, float, float, float, float]:
    arr = np.array(values)
    return len(arr), float(np.median(arr)), float(arr.mean()), float(arr.min()), float(arr.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefixes", nargs="*", default=list(DEFAULT_PREFIXES))
    parser.add_argument("--drifts", nargs="*", type=float, default=list(DRIFT_DISTANCES_M))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-family", type=int, default=3)
    args = parser.parse_args()

    records = pick_records(tuple(args.prefixes), args.per_family)
    print(
        f"sampling {len(records)} combinations "
        f"({', '.join(f'{p}x{args.per_family}' for p in args.prefixes)})"
    )
    generator = MotionBricksGenerator()

    samples, skipped = measure(generator, records, tuple(args.drifts), seed=args.seed)

    print(
        f"\n{'family':22s} {'combination':30s} {'drift m':>7s} {'r0 m':>7s} {'asked m':>8s} "
        f"{'reached m':>9s} {'fraction':>8s} {'incr gain':>9s}"
    )
    for s in samples:
        print(
            f"{s.family:22s} {s.name[:30]:30s} {s.drift_m:7.2f} {s.r0_m:7.3f} {s.asked_m:8.3f} "
            f"{s.reached_m:9.3f} {s.fraction:8.3f} {s.incremental_gain:9.3f}"
        )

    fractions = [s.fraction for s in samples]
    gains = [s.incremental_gain for s in samples]

    n, median, mean, lo, hi = _stats(fractions)
    print(
        f"\noverall reached/asked (superseded)  n={n:3d}  median {median:.3f}  mean {mean:.3f}  "
        f"min {lo:.3f}  max {hi:.3f}"
    )
    gn, gmedian, gmean, glo, ghi = _stats(gains)
    print(
        f"overall incremental_gain (decision) n={gn:3d}  median {gmedian:.3f}  mean {gmean:.3f}  "
        f"min {glo:.3f}  max {ghi:.3f}"
    )
    if skipped:
        print(f"skipped {skipped} (combination, distance) pairs (WarpError)")

    print("\nper-family median (fraction / incremental_gain):")
    for prefix in args.prefixes:
        family_fractions = [s.fraction for s in samples if s.family == prefix]
        family_gains = [s.incremental_gain for s in samples if s.family == prefix]
        if not family_fractions:
            print(f"  {prefix:22s} (no samples)")
            continue
        fn, fmedian, _fmean, flo, fhi = _stats(family_fractions)
        _gfn, gfmedian, _gfmean, gflo, gfhi = _stats(family_gains)
        print(
            f"  {prefix:22s} n={fn:3d}  fraction: median {fmedian:.3f} min {flo:.3f} max {fhi:.3f}  "
            f"|  incremental_gain: median {gfmedian:.3f} min {gflo:.3f} max {gfhi:.3f}"
        )

    spread_ok = (ghi - gmedian) <= 0.10 and (gmedian - glo) <= 0.10
    print(
        f"\ndecision bar (on incremental_gain): min/max within +/-0.10 of the median -> "
        f"{'PASS, a single DRIFT_GAIN constant is justified' if spread_ok else 'FAIL, do not pick a single constant'}"
    )


if __name__ == "__main__":
    main()
