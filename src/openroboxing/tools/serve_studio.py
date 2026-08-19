"""Serve the Pose Studio (S-T1).

Acceptance criterion from WORKPLAN.md S-T1:
  an author with no repo access creates a valid, admitted pose record through the browser.

Usage
-----
    python -m openroboxing.tools.serve_studio
    python -m openroboxing.tools.serve_studio --port 8081 --drafts poses/dev

Then open http://localhost:8081/ and drag sliders.

**Authoring is the half this serves.** Admission needs a measured ``generator_error_rad``, which
means asking MotionBricks to actually reach the pose — seconds per pose and a checkpoint in memory.
Save a draft here, then admit it with `tools/build_library.py`, which is where that measurement
already lives.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openroboxing.server.studio_app import DRAFT_DIR, build_studio_app


def main(argv: list[str] | None = None) -> int:
    from aiohttp import web

    parser = argparse.ArgumentParser(
        prog="serve_studio", description="Author poses in a browser (S-T1)."
    )
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--drafts", type=Path, default=DRAFT_DIR, help="where drafts are saved")
    args = parser.parse_args(argv)

    print(f"Pose Studio on http://localhost:{args.port}/")
    print(f"  drafts -> {args.drafts}")
    print("  admission is separate: save a draft, then run tools/build_library.py to measure it")

    web.run_app(build_studio_app(draft_dir=args.drafts), port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
