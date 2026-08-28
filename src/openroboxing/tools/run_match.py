"""Run a full match, headless, and write the record (M3-T4; ported for combinations, B2).

Acceptance criterion from WORKPLAN.md M3-T4:
  a full three-round match runs headless and produces a match record containing every field in
  `spec/match_record.md`.

``--rounds``, ``--round-seconds`` and ``--get-up-seconds`` override the format; ``--out`` writes the
record and its state trace. The on-disk combination library is all-draft today, so a real run needs
``--allow-draft`` (with the warning that implies — see below).

Both fighters are driven by :class:`~openroboxing.runtime.fight.ScriptedPilot`, cycling combinations
from the shared library on a fixed cadence. That is a stand-in and is meant to look like one: a human
drives this in `M4-T1` and an AI in `M5`. What is being demonstrated here is the *loop* — that three
rounds run, that contacts are attributed, that knockdowns are counted and that the record comes out
whole.

`spec/intent.md` 3.0: a commit is a combination, not a placement plus a pose
-----------------------------------------------------------------------------
Since 3.0 there is no approach: a combination already carries its own footwork and a fixed
``duration_ticks``, so what this script has to get right is not "does the fighter get there" but
"does the script ever ask for more than :data:`~openroboxing.spec.constants.MAX_OUTSTANDING_COMMITS`
unfinished at once". A commit's length is now known the moment it is scripted (`record.duration_ticks`
per `spec/intent.md` "A commit's span"), so :func:`_script` spaces each fighter's own commits by the
combination's own duration rather than a fixed period — a fixed period risked outrunning the queue
the instant a combination ran longer than it, which the corpus's travelling takes (up to 10.67 s,
`spec/intent.md`'s rates table) do relative to the old 3 s cadence.

The two circle the centre from opposite sides, ghosts advancing one step of the circle per commit —
where the ghost lands, not how the fighter gets there, since a combination's own recorded footwork now
does the moving (D4, `spec/intent.md`). The 1.0-2.2 script alternated "step" and "strike" commits
because those were different things then; a combination is both at once, so that distinction is gone.

Headless by design: nothing opens a display.

Run: ``.venv_mb/bin/python -m openroboxing.tools.run_match --allow-draft --rounds 1 --round-seconds 20``
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from openroboxing.paths import COMBINATION_DIR
from openroboxing.runtime.arena import FIGHTERS
from openroboxing.runtime.fight import FightWorld, ScriptedPilot
from openroboxing.runtime.match import SCHEMA_VERSION, Match, MatchFormat
from openroboxing.runtime.pool import fighter_seed
from openroboxing.spec.constants import TICK_HZ
from openroboxing.studio import combination_record as cr

#: How far into a round the first commit lands, per fighter. Offset so red leads, and both start well
#: past `COMMIT_HORIZON_TICKS` (30 ticks) so the first move is never rejected for being too soon.
FIRST_COMMIT_TICK = {"red": 100, "blue": 175}

#: How far round the centre each successive commit steps. A fifth of a turn, so a fighter comes back
#: round every five commits and the pattern is visible in a replay rather than looking random.
ORBIT_STEP_RAD = 2.0 * math.pi / 5.0

#: How far apart the script plans to stand the two fighters' ghosts, metres.
#:
#: **Measured under 1.0-2.2's approach model** (2026-08-08, one 40 s round at each) and carried over
#: rather than re-measured: `ARRIVAL_RADIUS_M`'s scatter reasoning that produced this number no longer
#: applies verbatim under 3.0's warp/drift placement, but the value itself — comfortably inside
#: `CONTACT_RANGE_M` (0.80 m) without the fighters treading on each other — is still a reasonable
#: standoff to script from, and re-measuring it against combination lengths is `spec/intent.md`'s
#: `M4-T4` question, not this tool's.
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


def _versions(library_version: str) -> dict[str, str]:
    """What produced this match, for `spec/match_record.md`'s ``versions``.

    `WORKPLAN` M6-T1 wants a match traceable to the assets that made it. These are names and SHAs, so
    that stays possible; `CLAUDE.md` invariant 6 is explicit that this does **not** promise
    re-derivation. The schema's field is still named ``pose_library`` (`spec/match_record.md`, not
    touched by this task); its value is now the combination library's own version stamp.
    """
    from openroboxing.paths import G1_29DOF_SIM_XML, POLICY_DIR, REPO_ROOT

    return {
        "policy": POLICY_DIR.name,
        "pose_library": library_version,
        "robot_model": G1_29DOF_SIM_XML.name,
        "rules": SCHEMA_VERSION,
        "openroboxing_sha": _git_sha(REPO_ROOT),
    }


def _script(
    fighter: str, order: list[str], library: dict[str, cr.CombinationRecord], round_ticks: int
) -> list[tuple[int, str, tuple[float, float]]]:
    """A fighter's commits for one round: every combination in turn, spaced so the queue never fills.

    Each is placed on a circle about the centre so the two ghosts stand :data:`ORBIT_SEPARATION_M`
    apart while circling it from opposite sides. The circle advances one step per commit — the
    combination's own recorded footwork does the travelling (D4), so there is no separate "step" commit
    left to alternate with a "strike" one the way 1.0-2.2 needed.

    A commit's length is exact arithmetic the moment it is scripted (`record.duration_ticks`,
    `spec/intent.md` "A commit's span"), so the next one for this fighter is spaced no sooner than
    this one's own duration — never a fixed period, which the corpus's longest combinations (up to
    10.67 s) would outrun.
    """
    radius = ORBIT_SEPARATION_M / 2.0
    phase = 0.0 if fighter == "red" else math.pi
    names = order if fighter == "red" else list(reversed(order))

    script: list[tuple[int, str, tuple[float, float]]] = []
    tick = FIRST_COMMIT_TICK[fighter]
    index = 0
    while tick < round_ticks:
        name = names[index % len(names)]
        angle = phase + index * ORBIT_STEP_RAD
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        script.append((tick, name, (x, y)))
        tick += library[name].duration_ticks
        index += 1
    return script


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_match",
        description="Run a full match between two G1s and write the match record (M3-T4).",
    )
    parser.add_argument(
        "--library", type=Path, default=COMBINATION_DIR, help="combination library directory"
    )
    parser.add_argument("--seed", type=int, default=1234, help="match seed; a match reproduces from it")
    parser.add_argument("--rounds", type=int, default=None, help="override the format's rounds")
    parser.add_argument(
        "--round-seconds", type=float, default=None, help="override the round length, seconds"
    )
    parser.add_argument(
        "--get-up-seconds", type=float, default=None, help="override the get-up window, seconds"
    )
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help=(
            "commit draft (unmeasured) combinations. The on-disk library is entirely draft today — "
            "telegraph and tracking error have not been measured — so results are NOT admissible "
            "and this is for exercising the loop only, never for a real match."
        ),
    )
    parser.add_argument("--match-id", default=None, help="record id; defaults to the seed")
    parser.add_argument("--out", type=Path, default=None, help="write the record to this json path")
    args = parser.parse_args(argv)

    if args.allow_draft:
        print(
            "WARNING: --allow-draft is set. The combination library has not been measured "
            "(telegraph, tracking error) and every result below is not admissible."
        )

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

    library = {p.stem: cr.load(p) for p in sorted(args.library.glob("*.json"))}
    if not library:
        raise SystemExit(f"no combinations in {args.library}")
    order = sorted(library)
    print(f"combination library {args.library} ({library[order[0]].library_version}): {len(library)} moves")

    print(
        f"\nformat: {match_format.rounds} x {match_format.round_seconds:.0f}s, "
        f"count {match_format.get_up_seconds:.0f}s "
        f"({match_format.rounds * match_format.round_ticks} ticks total)"
    )

    print("building the ring...")
    build_start = time.perf_counter()
    world = FightWorld(
        libraries={f: library for f in FIGHTERS},
        pilots={
            f: ScriptedPilot(_script(f, order, library, match_format.round_ticks)) for f in FIGHTERS
        },
        match_seed=args.seed,
        require_admitted=not args.allow_draft,
    )
    print(
        f"  ready in {time.perf_counter() - build_start:.1f}s: "
        f"nq={world.model.nq}, {world.substeps} substeps per tick"
    )

    match = Match(
        world,
        match_id=args.match_id or f"match-{args.seed}",
        match_format=match_format,
        fighters={f: {"handle": f, "combinations": order} for f in FIGHTERS},
        versions=_versions(library[order[0]].library_version),
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

    # Mean achieved drift speed, the signal 3.0 records in place of the old arrival/timeout split
    # (`spec/intent.md` "Off-target execution"): how hard a fighter had to run beyond its recorded
    # footwork to reach a ghost, across every commit that actually started.
    drift_speeds = [
        c["drift_speed_m_s"]
        for r in record.rounds
        for c in r.commits
        if c["drift_speed_m_s"] is not None
    ]
    if drift_speeds:
        print(f"  mean drift speed: {sum(drift_speeds) / len(drift_speeds):.2f} m/s")

    if args.out is not None:
        trace_path = record.save(args.out)
        size_kb = args.out.stat().st_size / 1024
        trace_kb = trace_path.stat().st_size / 1024
        print(f"\nwrote {args.out} ({size_kb:.0f} kB) and {trace_path.name} ({trace_kb:.0f} kB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
