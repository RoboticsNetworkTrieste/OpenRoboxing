"""Measure a pose's telegraph window (M2-T3).

Measures the window between the first frame a move is distinguishable from a guard baseline and the
frame it lands, and checks it against a configurable floor.

Two input modes:

- ``--pose``: generate the move from a pose record and measure it. Needs a GPU and the MotionBricks
  checkpoint, because it runs the generator.
- ``--motion``/``--baseline``: measure two qpos streams that already exist (npz or npy, ``(N, 36)``).
  No generator, no GPU — this is what the test suite exercises.

The baseline
------------
``--pose`` compares against the *same style with no authored pose*, which is the generator's
unmodified behaviour. That is the right control: it isolates what the authored pose changed, rather
than what the clip was already doing. Pass ``--baseline-pose`` to compare against an authored guard
instead, which is what a library measurement should do once a guard pose exists.

Usage
-----
    python -m openroboxing.tools.measure_telegraph --pose poses/dev/hook_R.json
    python -m openroboxing.tools.measure_telegraph --pose poses/dev/hook_R.json --write
    python -m openroboxing.tools.measure_telegraph --motion hook.npz --baseline guard.npz
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from openroboxing.spec.constants import GENERATOR_HZ, TICK_HZ
from openroboxing.studio.pose_record import load, save
from openroboxing.studio.telegraph import DIVERGENCE_SIGMA, TelegraphError, measure

#: Starting point, not a rule. M2-T5 calibrates a real floor against measured poses and records it in
#: `spec/rates.md`; inventing one here would be exactly what CLAUDE.md forbids.
DEFAULT_FLOOR_MS = 150.0

#: How long a rehearsal runs. Long enough to contain a full commit at the maximum horizon
#: (MAX_TOKENS = 16 tokens ≈ 2.1 s, see spec/rates.md) plus the windup that precedes it.
DEFAULT_REHEARSAL_SECONDS = 3.0


def _load_stream(path: Path) -> np.ndarray:
    """Load a (N, 36) qpos stream from .npy, or from .npz (first array, or key 'qpos')."""
    if path.suffix == ".npy":
        return np.load(path)
    with np.load(path) as data:
        key = "qpos" if "qpos" in data.files else data.files[0]
        return data[key]


def _rehearse_pair(args) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate the move and its baseline. Returns ``(motion, baseline, rate_hz)``."""
    from openroboxing.runtime.generator import GeneratorConfig, MotionBricksGenerator
    from openroboxing.studio.rehearsal import rehearse

    record = load(args.pose)
    baseline_record = load(args.baseline_pose) if args.baseline_pose else None

    print(f"building the generator (style={args.style}, seed={args.seed})...")
    generator = MotionBricksGenerator(GeneratorConfig(random_seed=args.seed))

    common = dict(
        style=args.style, seconds=args.seconds, seed=args.seed, generator=generator
    )
    print(f"rehearsing {record.name!r}...")
    motion = rehearse(record, **common)
    label = baseline_record.name if baseline_record else f"{args.style} with no authored pose"
    print(f"rehearsing the baseline ({label})...")
    baseline = rehearse(baseline_record, **common)

    return motion.qpos, baseline.qpos, motion.rate_hz


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="measure_telegraph",
        description="Measure a move's telegraph window against a guard baseline (M2-T3).",
    )
    parser.add_argument("--pose", type=Path, help="pose record to generate the move from")
    parser.add_argument(
        "--baseline-pose", type=Path, help="pose record for the baseline; default is no pose"
    )
    parser.add_argument("--motion", type=Path, help="a pre-generated move, (N, 36) qpos")
    parser.add_argument("--baseline", type=Path, help="a pre-generated baseline, (N, 36) qpos")
    parser.add_argument("--style", default="walk_boxing", help="clip driving the rehearsal")
    parser.add_argument("--seed", type=int, default=1234, help="generator seed")
    parser.add_argument(
        "--seconds", type=float, default=DEFAULT_REHEARSAL_SECONDS, help="rehearsal length"
    )
    parser.add_argument("--floor-ms", type=float, default=DEFAULT_FLOOR_MS, help="pass/fail floor")
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=None,
        help=f"rate of --motion/--baseline streams (default {TICK_HZ}); "
        f"rehearsals are always {GENERATOR_HZ}",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DIVERGENCE_SIGMA,
        help="divergence threshold, in baseline standard deviations",
    )
    parser.add_argument("--write", action="store_true", help="write telegraph_ms into --pose")
    args = parser.parse_args(argv)

    from_streams = args.motion is not None or args.baseline is not None
    if args.pose is None and not from_streams:
        parser.error("give --pose, or both --motion and --baseline")
    if from_streams and (args.motion is None or args.baseline is None):
        parser.error("--motion and --baseline must be given together")
    if args.write and args.pose is None:
        parser.error("--write needs --pose")

    if from_streams:
        motion = _load_stream(args.motion)
        baseline = _load_stream(args.baseline)
        rate_hz = args.rate_hz if args.rate_hz is not None else TICK_HZ
    else:
        motion, baseline, rate_hz = _rehearse_pair(args)
        if args.rate_hz is not None:
            rate_hz = args.rate_hz

    try:
        result = measure(motion, baseline, rate_hz=rate_hz, sigma=args.sigma)
    except TelegraphError as exc:
        print(f"FAIL: {exc}")
        return 1

    verdict = "PASS" if result.passes(args.floor_ms) else "FAIL"
    print(f"  telegraph window : {result.window_ms:.1f} ms")
    print(f"  distinguishable  : frame {result.divergence_frame}")
    print(f"  contact          : frame {result.contact_frame}")
    print(f"  measured at      : {rate_hz:.0f} Hz")
    print(
        f"  threshold        : {result.threshold_m * 1e3:.1f} mm (from the baseline's own spread)"
    )
    print(f"  peak reach       : {result.peak_displacement_m * 1e3:.0f} mm")
    print(f"  floor            : {args.floor_ms:.0f} ms  -> {verdict}")

    if args.write:
        record = load(args.pose)
        save(dataclasses.replace(record, telegraph_ms=result.window_ms), args.pose)
        print(f"\nwrote telegraph_ms={result.window_ms:.1f} into {args.pose}")

    return 0 if result.passes(args.floor_ms) else 1


if __name__ == "__main__":
    raise SystemExit(main())
