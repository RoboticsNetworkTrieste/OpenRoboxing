"""Replay a recorded match: watch it, or re-derive the rules from it (M3-T5).

Acceptance criterion from WORKPLAN.md M3-T5:
  a recorded match replays visually identically from the trace; the intent log alone is under a few
  hundred kilobytes for a full match.

Usage
-----
    python -m openroboxing.tools.replay_match matches/match-1234.json --video replay.mp4
    python -m openroboxing.tools.replay_match matches/match-1234.json --round 2 --stride 2
    python -m openroboxing.tools.replay_match matches/match-1234.json --rescore

``--rescore`` runs the *rules* over the recording — no GPU, no generator, no policy — and prints what
it derives beside what was recorded. Knockdowns must agree exactly. Hit **forces** are expected to
differ, because they depend on velocities the trace does not carry; see `spec/match_record.md`
§"What replays, and what is a reconstruction".

Headless by design: rendering is offscreen through EGL and needs no display.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from openroboxing.runtime.replay import (
    DEFAULT_CAMERA,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    RecordedMatch,
    ReplayWorld,
    encode_video,
    replay_frames,
)
from openroboxing.spec.constants import TICK_HZ


def _describe(recorded: RecordedMatch) -> None:
    record = recorded.record
    fmt = recorded.format()
    print(f"{record.get('match_id')}  (schema {record.get('schema_version')})")
    print(f"  format   : {fmt.rounds} x {fmt.round_seconds:.0f}s, count {fmt.get_up_seconds:.0f}s")
    print(f"  versions : {record.get('versions', {})}")
    print(f"  seeds    : {record.get('seeds', {})}")
    for entry in record.get("rounds", []):
        print(
            f"  round {entry['index'] + 1}: {entry['ticks']:>5} ticks  {entry['ended_by']:<9} "
            f"hits {len(entry['hits']):>3}  knockdowns {len(entry['knockdowns'])}  "
            f"commits {len(entry['commits'])}"
        )


def _rescore(recorded: RecordedMatch) -> int:
    """Re-run the rules over the recording and compare with what was recorded."""
    from openroboxing.runtime.match import Match

    world = ReplayWorld(recorded)
    fmt = recorded.format()
    replayed = Match(world, match_id=f"{recorded.match_id}-replay", match_format=fmt).run()

    print("\nre-derived from the trace:")
    disagreements = 0
    for index in range(recorded.round_count):
        was = recorded.round(index)
        now = replayed.rounds[index]

        recorded_kd = [(k["fighter"], k["start_tick"], k["became_knockout"]) for k in was["knockdowns"]]
        replayed_kd = [(k.fighter, k.start_tick, k.became_knockout) for k in now.knockdowns]
        agree = recorded_kd == replayed_kd and was["knocked_out"] == now.knocked_out
        disagreements += 0 if agree else 1

        print(
            f"  round {index + 1}: knockdowns {len(replayed_kd)} vs {len(recorded_kd)} recorded  "
            f"{'AGREE' if agree else 'DISAGREE'}   "
            f"hits {len(now.hits)} vs {len(was['hits'])} recorded (forces are a reconstruction)"
        )

    if disagreements:
        print(f"\n{disagreements} round(s) disagree on knockdowns — that is a bug, not a tolerance")
    else:
        print("\nevery knockdown re-derives exactly from the trace alone")
    return 1 if disagreements else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="replay_match",
        description="Replay a recorded match from its state trace (M3-T5).",
    )
    parser.add_argument("record", type=Path, help="path to the match record json")
    parser.add_argument("--round", type=int, default=None, help="one round; default is all of them")
    parser.add_argument("--video", type=Path, default=None, help="write an mp4 here")
    parser.add_argument("--frames", type=Path, default=None, help="write PNG frames to this dir")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="arena camera: broadcast|overhead")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--stride", type=int, default=1, help="render every Nth tick; 2 halves the work"
    )
    parser.add_argument(
        "--rescore", action="store_true", help="re-derive the rules from the trace and compare"
    )
    args = parser.parse_args(argv)

    recorded = RecordedMatch.load(args.record)
    _describe(recorded)

    size_kb = args.record.stat().st_size / 1024
    trace_kb = args.record.with_suffix(".trace.npz").stat().st_size / 1024
    print(f"  on disk  : {size_kb:.0f} kB record + {trace_kb:.0f} kB trace")

    status = 0
    if args.rescore:
        status = _rescore(recorded)

    if args.video is None and args.frames is None:
        return status

    rounds = [args.round] if args.round is not None else list(range(recorded.round_count))
    world = ReplayWorld(recorded)
    fps = TICK_HZ / args.stride

    for index in rounds:
        frames = replay_frames(
            world,
            index,
            stride=args.stride,
            width=args.width,
            height=args.height,
            camera=args.camera,
        )
        start = time.perf_counter()

        if args.video is not None:
            path = (
                args.video
                if len(rounds) == 1
                else args.video.with_name(f"{args.video.stem}-round{index + 1}{args.video.suffix}")
            )
            encode_video(frames, path, fps=fps)
            print(
                f"  round {index + 1}: wrote {path} at {fps:.0f} fps "
                f"in {time.perf_counter() - start:.1f}s"
            )
        else:
            from openroboxing.studio.render import save_png

            out = args.frames / f"round{index + 1}"
            out.mkdir(parents=True, exist_ok=True)
            count = 0
            for count, frame in enumerate(frames, start=1):
                save_png(frame, out / f"{count:05d}.png")
            print(
                f"  round {index + 1}: wrote {count} frames to {out} "
                f"in {time.perf_counter() - start:.1f}s"
            )

    return status


if __name__ == "__main__":
    raise SystemExit(main())
