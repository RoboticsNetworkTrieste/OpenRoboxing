"""Build the combination library from the mocap corpus (M5-T9).

Reads every take under ``paths.MOTIONS_DIR``, segments it, and writes one draft
:class:`~openroboxing.studio.combination_record.CombinationRecord` per combination into
``paths.COMBINATION_DIR``.

Records are written **draft**. Nothing here measures, so nothing here may admit.

Run: ``.venv_mb/bin/python -m openroboxing.tools.import_motions --report``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openroboxing.paths import COMBINATION_DIR, MOTIONS_DIR
from openroboxing.spec.constants import GENERATOR_HZ, SECONDS_PER_TOKEN
from openroboxing.studio import combination_record as cr
from openroboxing.studio.motion_import import load_take

LIBRARY_VERSION = "v0.2"


def build_library(corpus: Path, out: Path, *, report: bool = False) -> int:
    """Build every take under ``corpus`` into ``out``. Returns the number of records written."""
    takes = sorted(corpus.glob("*.csv"))
    if not takes:
        raise SystemExit(f"no takes under {corpus}")

    total = 0
    if report:
        print(f"{'take':44s} {'secs':>6s} {'combos':>7s} {'keyframes':>10s} {'planned s':>10s}")
    for path in takes:
        qpos = load_take(path)
        records = cr.build_from_take(path.stem, qpos, library_version=LIBRARY_VERSION)
        for record in records:
            cr.save(record, out / f"{record.name}.json")
        total += len(records)
        if report:
            keyframes = sum(len(r.keyframes) for r in records)
            planned = sum(
                sum(k.leg_tokens or 0 for k in r.keyframes) * SECONDS_PER_TOKEN for r in records
            )
            print(
                f"{path.stem[:44]:44s} {len(qpos) / GENERATOR_HZ:6.1f} "
                f"{len(records):7d} {keyframes:10d} {planned:10.1f}"
            )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=MOTIONS_DIR)
    parser.add_argument("--out", type=Path, default=COMBINATION_DIR)
    parser.add_argument("--report", action="store_true", help="print a per-take summary")
    args = parser.parse_args()
    total = build_library(args.corpus, args.out, report=args.report)
    print(f"\n{total} combinations written to {args.out}")


if __name__ == "__main__":
    main()
