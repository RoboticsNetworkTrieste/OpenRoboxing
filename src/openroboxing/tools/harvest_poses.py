"""Propose candidate key poses from generated motion, and render them for a human (M2-T5).

The pose target is a soft constraint, so a pose the generator will not produce cannot become a move
however sensible it looks in joint space (`spec/upstream_notes.md`). This tool therefore proposes
poses the generator *already* reaches: it runs a style, picks out the frames that stand out, measures
how closely each can be commanded back, and renders a contact sheet.

The output is a set of **draft** records and a PNG. Naming them — deciding which candidate is a jab
and which is a guard — is the human's job, and is what `--name` does on a second pass.

Usage
-----
    python -m openroboxing.tools.harvest_poses --style walk_boxing --count 8 --out poses/candidates
    python -m openroboxing.tools.harvest_poses --style walk_boxing --count 8 --measure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openroboxing.paths import POSE_DIR
from openroboxing.spec.constants import GENERATOR_HZ
from openroboxing.studio.harvest import harvest
from openroboxing.studio.pose_record import save
from openroboxing.studio.rehearsal import measure_reachability, rehearse
from openroboxing.studio.render import contact_sheet, render_pose, save_png

#: Nothing hangs on this yet — it is the reporting bar for the table this tool prints, not a rule.
#: The real admission threshold is `M2-T5`'s to set from these measurements (`spec/intent.md`).
REPORT_TOLERANCE_RAD = 0.35


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harvest_poses",
        description="Propose candidate key poses from generated motion (M2-T5).",
    )
    parser.add_argument("--style", default="walk_boxing", help="clip to harvest from")
    parser.add_argument("--seconds", type=float, default=20.0, help="how much motion to generate")
    parser.add_argument("--seed", type=int, default=1234, help="generator seed")
    parser.add_argument("--count", type=int, default=8, help="how many candidates to propose")
    parser.add_argument(
        "--min-separation",
        type=int,
        default=15,
        help="minimum frames between candidates, so they are not the same pose",
    )
    parser.add_argument(
        "--skip-seconds",
        type=float,
        default=3.0,
        help="ignore this much motion at the start, while the generator settles into the style",
    )
    # Resolved through `paths`, not from the working directory. The old cwd-relative default named
    # a directory that stopped existing when the package moved under `src/`, so the tool wrote its
    # candidates wherever it happened to be launched from.
    parser.add_argument(
        "--out", type=Path, default=POSE_DIR / "candidates", help="output directory"
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="rehearse each candidate back and report how closely the generator reaches it",
    )
    parser.add_argument("--no-render", action="store_true", help="skip the contact sheet")
    args = parser.parse_args(argv)

    from openroboxing.runtime.generator import GeneratorConfig, MotionBricksGenerator

    print(f"building the generator (style={args.style}, seed={args.seed})...")
    generator = MotionBricksGenerator(GeneratorConfig(random_seed=args.seed))

    print(f"generating {args.seconds:.0f}s of {args.style}...")
    motion = rehearse(
        None, style=args.style, seconds=args.seconds, seed=args.seed, generator=generator
    )

    candidates = harvest(
        motion.qpos,
        count=args.count,
        min_separation=args.min_separation,
        skip_frames=int(round(args.skip_seconds * GENERATOR_HZ)),
        style=args.style,
        seed=args.seed,
    )
    if len(candidates) < args.count:
        print(
            f"note: {len(candidates)} candidates, not {args.count} — "
            f"{args.seconds:.0f}s at {GENERATOR_HZ} Hz cannot hold that many "
            f"{args.min_separation} frames apart"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for index, candidate in enumerate(candidates):
        record = candidate.to_record(name=f"{args.style}-cand-{index:02d}")
        save(record, args.out / f"{record.name}.json")
        records.append(record)

    print(f"\nwrote {len(records)} draft records to {args.out}")
    print(f"\n{'#':>3}  {'frame':>6}  {'distinct':>9}", end="")
    print(f"  {'reach mean':>11}  {'reach max':>10}  worst joint" if args.measure else "")

    for index, (candidate, record) in enumerate(zip(candidates, records)):
        line = f"{index:>3}  {candidate.frame:>6}  {candidate.distinctiveness:>9.3f}"
        if args.measure:
            reach = measure_reachability(
                record, style=args.style, seed=args.seed, generator=generator
            )
            flag = " " if reach.passes(REPORT_TOLERANCE_RAD) else "*"
            line += (
                f"  {reach.mean_error_deg:>9.1f}d  {reach.max_error_deg:>8.1f}d{flag}"
                f"  {reach.worst_joint}"
            )
        print(line)

    if args.measure:
        print(f"\n* worst-joint error above {np.degrees(REPORT_TOLERANCE_RAD):.0f}d")

    if not args.no_render:
        print("\nrendering...")
        images = [render_pose(record) for record in records]
        labels = [f"#{index:02d} f{c.frame}" for index, c in enumerate(candidates)]
        sheet = save_png(contact_sheet(images, labels), args.out / "candidates.png")
        print(f"contact sheet: {sheet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
