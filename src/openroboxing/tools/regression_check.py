"""The regression gate: record a baseline, or check a candidate against one (S-T3).

Acceptance criterion from WORKPLAN.md S-T3:
  a deliberately over-fitted checkpoint fails the regression gate.

Usage
-----
    python -m openroboxing.tools.regression_check --record            # baseline from current weights
    python -m openroboxing.tools.regression_check                     # check against it
    python -m openroboxing.tools.regression_check --degrade 0.75      # prove the gate can fail

`--degrade` stands in for an over-fitted checkpoint: it scales and biases the policy's actions the
way a network that has forgotten how to walk would. A gate nobody has watched fail is not a gate.

Exit code is 1 on any regression, so this drops straight into a release script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openroboxing.studio.regression import (
    DEFAULT_BATTERY,
    SECONDS,
    TOLERANCE,
    Baseline,
    DegradedPolicy,
    compare,
    default_baseline_path,
    format_report,
    notes_for,
    run_battery,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="regression_check",
        description="Check a weight set against the general-behaviour battery (S-T3).",
    )
    parser.add_argument("--record", action="store_true", help="write a new baseline instead")
    parser.add_argument("--baseline", type=Path, default=None, help="baseline file")
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--seconds", type=float, default=SECONDS)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument(
        "--battery", nargs="*", default=list(DEFAULT_BATTERY), help="motions to run"
    )
    parser.add_argument(
        "--degrade",
        type=float,
        default=None,
        help="scale the policy's actions by this, standing in for an over-fitted checkpoint",
    )
    args = parser.parse_args(argv)

    path = args.baseline or default_baseline_path()

    policy = None
    if args.degrade is not None:
        from openroboxing.runtime.policy import GearSonicPolicy

        policy = DegradedPolicy(GearSonicPolicy(), scale=args.degrade)
        print(f"using a DEGRADED policy (actions x {args.degrade}) — this should fail the gate")

    def report(result) -> None:
        print(
            f"  {result.style:<14} mean {result.mean_joint_error_rad:.4f}  "
            f"max {result.max_joint_error_rad:.4f}  "
            f"{'FELL' if result.fell else 'stood'}  {result.distance_m:.2f} m"
        )

    print(f"running the battery: {', '.join(args.battery)}  ({args.seconds:.0f}s each)")
    candidate = run_battery(
        args.battery,
        label=args.label,
        seconds=args.seconds,
        seed=args.seed,
        policy=policy,
        on_result=report,
    )

    if args.record:
        candidate.label = args.label if args.label != "candidate" else "baseline"
        candidate.notes = notes_for()
        print(f"\nwrote {candidate.save(path)}")
        return 0

    if not path.exists():
        print(f"\nno baseline at {path}. Record one first:")
        print("    python -m openroboxing.tools.regression_check --record")
        return 1

    baseline = Baseline.load(path)
    findings = compare(baseline, candidate, tolerance=args.tolerance)

    print()
    print(format_report(baseline, candidate, findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
