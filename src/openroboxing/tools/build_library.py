"""Build, measure and admit the authored pose library (M2-T5).

Solves `studio/library_v0_1.py` into pose records under `poses/v0.1/`, and with ``--measure`` runs
each pose through the generator to fill in the two numbers admission requires:

- **reachability** — will the generator produce this pose? Measured at the plan endpoint, since that
  is where an in-between puts its target.
- **telegraph window** — how long the move is readable before it lands, against the guard as
  baseline. Both are measured on the *committed plan*, never on a replanning run.

Without ``--measure`` the records are written as ``draft``, which is what they are: a pose nobody has
measured must not reach a match (`spec/pose_record.md`).

Usage
-----
    python -m openroboxing.tools.build_library                    # write drafts
    python -m openroboxing.tools.build_library --measure --admit  # measure, then admit what passes
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from openroboxing.paths import OPENROBOXING_ROOT
from openroboxing.runtime.conventions import G1
from openroboxing.spec.constants import GENERATOR_HZ
from openroboxing.studio.library_v0_1 import ENVELOPES, HORIZONS, LIBRARY_VERSION, build
from openroboxing.studio.pose_record import PoseRecord, PoseSource, save
from openroboxing.studio.render import contact_sheet, render_pose, save_png

#: A pose whose plan endpoint misses by more than this is not admitted. Set from measurement: real
#: boxing poses land at 7-11° worst-joint error, and the one deliberately absurd pose tested reached
#: 22.5°, so 20° separates "the generator produced it" from "the generator declined".
#: `spec/upstream_notes.md`.
REACHABILITY_TOLERANCE_DEG = 20.0

#: Below this a move is unreadable and unfair. A placeholder until a playtest sets it — recorded, not
#: enforced, so a pose is never silently dropped for failing a number nobody has justified yet.
TELEGRAPH_FLOOR_MS = 150.0


def _records(defaults: dict[str, float]) -> dict[str, PoseRecord]:
    poses = build()
    return {
        name: PoseRecord(
            name=name,
            joint_angles={**defaults, **overrides},
            horizon_tokens=HORIZONS[name],
            library_version=LIBRARY_VERSION,
            adjustment_envelope=ENVELOPES.get(name, {}),
            source=PoseSource(clip="walk_boxing", start_frame=323, end_frame=323),
        )
        for name, overrides in poses.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_library", description="Build and measure the pose library (M2-T5)."
    )
    parser.add_argument(
        "--out", type=Path, default=OPENROBOXING_ROOT / "poses" / LIBRARY_VERSION
    )
    parser.add_argument("--measure", action="store_true", help="run each pose through the generator")
    parser.add_argument(
        "--admit", action="store_true", help="mark measured poses that pass as admitted"
    )
    parser.add_argument("--style", default="walk_boxing", help="clip the moves are generated in")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)

    if args.admit and not args.measure:
        parser.error("--admit needs --measure: admission requires the measurements")

    from openroboxing.runtime.obs import default_angles

    defaults = dict(zip(G1.mujoco_joint_names, default_angles(G1, "mujoco")))
    records = _records(defaults)
    print(f"solved {len(records)} poses\n")

    if args.measure:
        from openroboxing.runtime.generator import GeneratorConfig, MotionBricksGenerator
        from openroboxing.studio.rehearsal import measure_reachability, rehearse, rehearse_commit
        from openroboxing.studio.telegraph import TelegraphError, measure

        print(f"building the generator (style={args.style}, seed={args.seed})...")
        generator = MotionBricksGenerator(GeneratorConfig(random_seed=args.seed))
        common = dict(style=args.style, seed=args.seed, generator=generator)

        # The baseline is what the fighter does when it is *not* committing: ambient, replanning
        # motion in the same style. Not a plan that travels to the guard — that is itself a move, and
        # measuring one move against another gave mirrored poses windows differing by 6x.
        baseline = rehearse(None, style=args.style, seconds=6.0, seed=args.seed,
                            generator=generator).qpos
        print(f"ambient baseline: {baseline.shape[0]} frames\n")
        print(f"{'pose':<16}{'reach mean':>11}{'worst':>8}  {'telegraph':>10}  admission")
        print("-" * 62)

        for name, record in sorted(records.items()):
            reach = measure_reachability(record, **common)
            telegraph_ms = None
            note = ""
            if name != "guard":
                try:
                    plan = rehearse_commit(record, **common).qpos
                    telegraph_ms = measure(plan, baseline, rate_hz=GENERATOR_HZ).window_ms
                except TelegraphError as exc:
                    note = f"  ({str(exc)[:38]})"

            # Admission turns on reachability alone. A telegraph window is recorded when the proxy
            # produces one, but is not a gate: see spec/pose_record.md.
            reachable = reach.max_error_deg <= REACHABILITY_TOLERANCE_DEG
            admission = record.admission
            if args.admit:
                admission = "admitted" if reachable else "rejected"

            records[name] = dataclasses.replace(
                record,
                telegraph_ms=telegraph_ms,
                generator_error_rad=reach.max_error_rad,
                admission=admission,
            )
            shown = f"{telegraph_ms:.0f} ms" if telegraph_ms is not None else "n/a"
            print(
                f"{name:<16}{reach.mean_error_deg:9.1f}d{reach.max_error_deg:7.1f}d  "
                f"{shown:>10}  {admission}{note}"
            )

        print(f"\nreachability tolerance {REACHABILITY_TOLERANCE_DEG:.0f}d; "
              f"telegraph floor {TELEGRAPH_FLOOR_MS:.0f} ms is recorded, not enforced")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, record in records.items():
        save(record, args.out / f"{name}.json")
    print(f"\nwrote {len(records)} records to {args.out}")

    if not args.no_render:
        names = sorted(records)
        sheet = contact_sheet(
            [render_pose(records[n], width=380, height=430) for n in names], names, columns=5
        )
        print(f"contact sheet: {save_png(sheet, args.out / 'library.png')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
