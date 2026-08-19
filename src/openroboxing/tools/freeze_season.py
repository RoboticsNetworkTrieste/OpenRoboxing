"""Freeze a season, verify one, or trace a match record to it (M6-T1).

Acceptance criterion from WORKPLAN.md M6-T1:
  a released season manifest pins every asset by version and hash; a match record can be traced to
  the exact assets that produced it.

Usage
-----
    python -m openroboxing.tools.freeze_season --season season-0 --out seasons/season-0.json
    python -m openroboxing.tools.freeze_season --verify seasons/season-0.json
    python -m openroboxing.tools.freeze_season --verify seasons/season-0.json --trace matches/a.json

**Publishing weights is blocked on a human.** `WORKPLAN` M6-T2 requires a licence review and a
sign-off before any finetuned derivative of an NVIDIA-licensed checkpoint is redistributed. A
manifest is written unreleased unless ``--released`` is passed with the acknowledgement, and this
tool will not do that for you.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openroboxing.league.manifest import (
    LICENCE_ACKNOWLEDGEMENT,
    ManifestError,
    SeasonManifest,
    format_manifest,
    freeze,
    trace_record,
    verify,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="freeze_season", description="Pin every asset a season is fought with (M6-T1)."
    )
    parser.add_argument("--season", default="season-0")
    parser.add_argument("--library", default="v0.1", help="pose library directory under poses/")
    parser.add_argument("--out", type=Path, default=None, help="write the manifest here")
    parser.add_argument("--verify", type=Path, default=None, help="check a manifest against disk")
    parser.add_argument("--trace", type=Path, default=None, help="trace a match record to it")
    parser.add_argument(
        "--released",
        default=None,
        help=f"mark released; must be exactly {LICENCE_ACKNOWLEDGEMENT!r} (WORKPLAN M6-T2)",
    )
    parser.add_argument(
        "--at",
        default=None,
        help="freeze timestamp, ISO 8601. Passed in so a manifest is reproducible.",
    )
    args = parser.parse_args(argv)

    if args.verify is not None:
        manifest = SeasonManifest.load(args.verify)
        discrepancies = verify(manifest)
        print(format_manifest(manifest, discrepancies))

        if args.trace is not None:
            record = json.loads(args.trace.read_text())
            findings = trace_record(record, manifest)
            print(f"\ntracing {findings['match_id']}:")
            for check in findings["checks"]:
                print(
                    f"  {check['asset']:<16} record={str(check.get('record')):<24} "
                    f"manifest={str(check.get('manifest')):<20} {check['result']}"
                )
            print(f"  -> {'traced' if findings['traced'] else 'NOT fully traced'}")
            if not findings["traced"]:
                return 1

        if discrepancies:
            print(f"\n{len(discrepancies)} asset(s) no longer match the freeze.")
            return 1
        print("\nevery asset matches the freeze.")
        return 0

    if args.at is None:
        parser.error("--at is required when freezing: a manifest must be reproducible")

    try:
        manifest = freeze(
            args.season,
            timestamp=args.at,
            pose_library=args.library,
            release_acknowledgement=args.released,
        )
    except ManifestError as exc:
        print(f"cannot freeze: {exc}")
        return 1

    print(format_manifest(manifest))
    if not manifest.released:
        print(
            "\n  This manifest is NOT released. WORKPLAN M6-T2 is blocking: publishing finetuned\n"
            "  derivatives of NVIDIA-licensed checkpoints is redistribution of derived models, and\n"
            "  a human must confirm the terms and produce the attribution text first."
        )

    if args.out is not None:
        print(f"\nwrote {manifest.save(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
