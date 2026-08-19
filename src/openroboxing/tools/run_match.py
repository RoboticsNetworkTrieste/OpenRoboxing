"""Run a full match, headless, and write the record (M3-T4).

Acceptance criterion from WORKPLAN.md M3-T4:
  a full three-round match runs headless and produces a match record containing every field in
  `spec/match_record.md`.

Usage
-----
    python -m openroboxing.tools.run_match
    python -m openroboxing.tools.run_match --rounds 1 --round-seconds 20 --out matches/test.json
    python -m openroboxing.tools.run_match --red orthodox --blue orthodox --seed 7

Both fighters are driven by :class:`~openroboxing.runtime.fight.ScriptedPilot`, cycling their slots
on a fixed cadence. That is a stand-in and is meant to look like one: a human drives this in `M4-T1`
and an AI in `M5`. What is being demonstrated here is the *loop* — that three rounds run, that
contacts are attributed, that knockdowns are counted and that the record comes out whole.

**Each commit carries a placement**, and it has to. Since `spec/intent.md` 1.1 walking *is* the first
half of every move, so a script that committed poses and no placements would run a match in which
neither fighter ever moves — and would leave the approach, which is most of what a commit now does,
out of the one tool that claims to demonstrate the loop.

The two circle the centre from opposite sides, **stepping and then striking**: odd commits move to
the next point on the circle, even ones repeat it, so half of them are thrown from a settled stance
rather than at the end of a walk. Placing every commit somewhere new was tried first and landed
nothing at all — a fighter that is always walking is a fighter that is never quite in range when its
punch fires.

Headless by design: nothing opens a display.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

from openroboxing.paths import LOADOUT_DIR
from openroboxing.runtime.arena import FIGHTERS
from openroboxing.runtime.fight import DEFAULT_CONTEXT, FightWorld, ScriptedPilot
from openroboxing.runtime.intents import Loadout, Placement
from openroboxing.runtime.match import SCHEMA_VERSION, Match, MatchFormat
from openroboxing.runtime.pool import fighter_seed
from openroboxing.spec.constants import TICK_HZ

#: Ticks between one fighter's commits. Wide enough that a move finishes before the next is staged —
#: a jab is 6 tokens, 40 ticks — and it staggers the two fighters so they are not mirror images.
COMMIT_PERIOD_TICKS = 150

#: How far into a round the first commit lands, per fighter. Offset so red leads.
FIRST_COMMIT_TICK = {"red": 100, "blue": 175}

#: How far round the centre each successive commit steps. A fifth of a turn, so a fighter comes back
#: round every five commits and the pattern is visible in a replay rather than looking random.
ORBIT_STEP_RAD = 2.0 * math.pi / 5.0

#: How far apart the script plans to stand the two fighters, metres.
#:
#: **Measured, not chosen** (2026-08-08, one 40 s round at each): planning them a full
#: ``CONTACT_RANGE_M`` (0.80 m) apart lands **nothing at all**, because each arrives within
#: ``ARRIVAL_RADIUS_M`` of its own point and two independent scatters of 0.40 m usually add up to out
#: of reach. Planning 0.40 m lands 17, but a third of those are *leg* contacts — out of distribution
#: for a policy trained penalising anything but feet, hands and elbows, i.e. the fighters are
#: treading on each other. **0.60 m lands 26 and none of them are legs.**
#:
#: The gap between this and ``CONTACT_RANGE_M`` is the real finding: arrival is coarse relative to
#: punching distance, so a placement *at* the edge of reach is a coin flip. See `docs/ASSUMPTIONS.md`
#: §A23.
ORBIT_SEPARATION_M = 0.60


def _git_sha(path: Path) -> str:
    """Short SHA of the tree at ``path``, or ``"unknown"``. Never fabricated."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _versions(pose_library: str) -> dict[str, str]:
    """What produced this match, for `spec/match_record.md`'s ``versions``.

    `WORKPLAN` M6-T1 wants a match traceable to the assets that made it. These are names and SHAs, so
    that stays possible; `CLAUDE.md` invariant 6 is explicit that this does **not** promise
    re-derivation.
    """
    from openroboxing.paths import G1_29DOF_SIM_XML, POLICY_DIR, REPO_ROOT

    return {
        "policy": POLICY_DIR.name,
        "pose_library": pose_library,
        "robot_model": G1_29DOF_SIM_XML.name,
        "rules": SCHEMA_VERSION,
        "openroboxing_sha": _git_sha(REPO_ROOT),
    }


def _script(fighter: str, slots: list[str], round_ticks: int) -> list[tuple[int, str, Placement]]:
    """A fighter's commits for one round: every slot in turn, on the cadence, until the bell.

    Each is placed on a circle about the centre so the two stand :data:`ORBIT_SEPARATION_M` apart
    while circling it from opposite sides — a measured distance, not a picked one, and notably
    *closer* than the scorer's contact range for the reason recorded there.

    The circle advances on every **other** commit. The one in between repeats the same point, which
    since 1.1 means "do this where you will be" and arrives instantly — so the rhythm is step, then
    strike from a stance, which is both what boxing looks like and what actually lands.
    """
    start = FIRST_COMMIT_TICK[fighter]
    ticks = list(range(start, round_ticks, COMMIT_PERIOD_TICKS))
    order = slots if fighter == "red" else list(reversed(slots))
    radius = ORBIT_SEPARATION_M / 2.0
    phase = 0.0 if fighter == "red" else math.pi

    script = []
    for index, tick in enumerate(ticks):
        angle = phase + (index // 2) * ORBIT_STEP_RAD
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        # Facing the centre, which is where the opponent is standing on the far side of it.
        placement = Placement(position=(x, y), heading=math.atan2(-y, -x))
        script.append((tick, order[index % len(order)], placement))
    return script


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_match",
        description="Run a full match between two G1s and write the match record (M3-T4).",
    )
    parser.add_argument("--red", default="orthodox", help="red's loadout name, in poses/loadouts/")
    parser.add_argument("--blue", default="orthodox", help="blue's loadout name")
    parser.add_argument("--seed", type=int, default=1234, help="match seed; a match reproduces from it")
    parser.add_argument("--rounds", type=int, default=None, help="override the format's rounds")
    parser.add_argument(
        "--round-seconds", type=float, default=None, help="override the round length, seconds"
    )
    parser.add_argument(
        "--get-up-seconds", type=float, default=None, help="override the get-up window, seconds"
    )
    parser.add_argument("--context", default=DEFAULT_CONTEXT, help="ambient clip between commits")
    parser.add_argument("--match-id", default=None, help="record id; defaults to the seed")
    parser.add_argument("--out", type=Path, default=None, help="write the record to this json path")
    args = parser.parse_args(argv)

    default = MatchFormat()
    match_format = MatchFormat(
        rounds=args.rounds if args.rounds is not None else default.rounds,
        round_ticks=(
            int(round(args.round_seconds * TICK_HZ))
            if args.round_seconds is not None
            else default.round_ticks
        ),
        get_up_window_ticks=(
            int(round(args.get_up_seconds * TICK_HZ))
            if args.get_up_seconds is not None
            else default.get_up_window_ticks
        ),
    )

    loadouts = {
        "red": Loadout.load(LOADOUT_DIR / f"{args.red}.json"),
        "blue": Loadout.load(LOADOUT_DIR / f"{args.blue}.json"),
    }
    slots = {f: sorted(loadouts[f].slots) for f in FIGHTERS}
    for fighter in FIGHTERS:
        entries = ", ".join(f"{s}={loadouts[fighter].slots[s].name}" for s in slots[fighter])
        print(f"{fighter:<5} {loadouts[fighter].name!r} ({loadouts[fighter].version}): {entries}")

    print(
        f"\nformat: {match_format.rounds} x {match_format.round_seconds:.0f}s, "
        f"count {match_format.get_up_seconds:.0f}s "
        f"({match_format.rounds * match_format.round_ticks} ticks total)"
    )

    print("building the ring...")
    build_start = time.perf_counter()
    world = FightWorld(
        loadouts=loadouts,
        pilots={
            f: ScriptedPilot(_script(f, slots[f], match_format.round_ticks)) for f in FIGHTERS
        },
        match_seed=args.seed,
        context=args.context,
    )
    print(
        f"  ready in {time.perf_counter() - build_start:.1f}s: "
        f"nq={world.model.nq}, {world.substeps} substeps per tick"
    )

    match = Match(
        world,
        match_id=args.match_id or f"match-{args.seed}",
        match_format=match_format,
        fighters={
            f: {
                "handle": f,
                "loadout": {
                    "name": loadouts[f].name,
                    "version": loadouts[f].version,
                    "slots": {s: loadouts[f].slots[s].name for s in slots[f]},
                },
            }
            for f in FIGHTERS
        },
        versions=_versions(loadouts["red"].version),
        seeds={"match_seed": args.seed, **{f: fighter_seed(args.seed, f) for f in FIGHTERS}},
    )

    run_start = time.perf_counter()
    record = match.run()
    wall = time.perf_counter() - run_start
    simulated = sum(r.ticks for r in record.rounds) / TICK_HZ

    print()
    for round_record in record.rounds:
        landed = {f: sum(1 for h in round_record.hits if h.attacker == f) for f in FIGHTERS}
        ending = round_record.ended_by
        if round_record.knocked_out:
            ending = f"knockout ({round_record.knocked_out})"
        print(
            f"round {round_record.index + 1}: {round_record.ticks:>5} ticks "
            f"({round_record.ticks / TICK_HZ:5.1f}s)  {ending:<20} "
            f"hits red {landed['red']:>3} / blue {landed['blue']:>3}  "
            f"knockdowns {len(round_record.knockdowns)}  commits {len(round_record.commits)}"
        )

    print()
    print(
        f"ran {simulated:.1f}s simulated in {wall:.1f}s wall "
        f"(real-time factor {simulated / wall:.2f}x)"
    )
    for fighter in FIGHTERS:
        hits = record.hits_by(fighter)
        peak = max((h.peak_force_n for h in hits), default=0.0)
        by_region: dict[str, int] = {}
        for hit in hits:
            by_region[hit.region] = by_region.get(hit.region, 0) + 1
        regions = ", ".join(f"{r} {n}" for r, n in sorted(by_region.items())) or "none"
        print(f"  {fighter:<5} landed {len(hits):>3}  peak {peak:7.1f} N  ({regions})")

    knockouts = record.knockouts()
    print(f"  knockouts: {knockouts if knockouts else 'none'}")
    down = {
        f: sum(k.duration_ticks for r in record.rounds for k in r.knockdowns if k.fighter == f)
        for f in FIGHTERS
    }
    print("  ticks on the canvas: " + ", ".join(f"{f} {n}" for f, n in down.items()))

    if args.out is not None:
        trace_path = record.save(args.out)
        size_kb = args.out.stat().st_size / 1024
        trace_kb = trace_path.stat().st_size / 1024
        print(f"\nwrote {args.out} ({size_kb:.0f} kB) and {trace_path.name} ({trace_kb:.0f} kB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
