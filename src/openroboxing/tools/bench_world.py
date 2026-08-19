"""Measure whether two fighters hold real time (M1-T7).

Acceptance criterion from WORKPLAN.md M1-T7:
  a short report in docs/perf/m1_mujoco.md with real-time factors and a recommended timestep, and an
  explicit verdict on whether two fighters at 50 Hz control hold real time.

What is measured
----------------
The control loop has three costs and they scale differently, so they are measured separately and then
combined rather than guessed at:

1. **Physics** — scales with the number of fighters and with 1/timestep.
2. **Inference** — two ONNX runs per fighter per tick (the encoder runs inline; see
   ``runtime/policy.py``), so it scales linearly with fighters and not at all with timestep.
3. **Generation** — MotionBricks replans; measured per call and amortised over the ticks between
   replans.

Usage
-----
    .venv_mb/bin/python -m openroboxing.tools.bench_world
    .venv_mb/bin/python -m openroboxing.tools.bench_world --seconds 3 --out docs/perf/m1_mujoco.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import time

import numpy as np

from openroboxing.paths import G1_29DOF_XML, SCENE_EMPTY_XML
from openroboxing.spec.constants import ENCODER_INPUT_DIM, POLICY_INPUT_DIM, TICK_DT, TICK_HZ

SCENE_EMPTY = SCENE_EMPTY_XML

#: Timesteps to sweep, seconds. 0.002 is the shipped scene's value.
TIMESTEPS = (0.001, 0.002, 0.004, 0.005)

#: Lateral separation between fighters, metres. 0.35 forces sustained contact.
SPACING_APART = 1.5
SPACING_CONTACT = 0.35


def build_arena(num_fighters: int, timestep: float, spacing: float):
    """Compose an arena with `num_fighters` G1s, via MjSpec attachment."""
    import mujoco

    if not SCENE_EMPTY.exists():
        raise FileNotFoundError(f"empty scene not found: {SCENE_EMPTY}")

    base = mujoco.MjSpec.from_file(str(SCENE_EMPTY))
    for i in range(num_fighters):
        robot = mujoco.MjSpec.from_file(str(G1_29DOF_XML))
        frame = base.worldbody.add_frame()
        frame.pos = [0.0, i * spacing, 0.0]
        base.attach(robot, prefix=f"f{i}_", frame=frame)

    model = base.compile()
    model.opt.timestep = timestep
    return model


def bench_ring(seconds: float) -> list[dict]:
    """The real M3-T1 arena — ropes, posts, padded gloves — rather than the bare composition.

    Two spacings: the stance both fighters start in, and a clinch, which is the worst case for
    contact count and therefore for the physics budget.
    """
    import mujoco

    from openroboxing.runtime.arena import ArenaConfig, build_arena, reset_to_stance

    rows = []
    for label, separation in (("stance", 1.20), ("clinch", 0.35)):
        config = ArenaConfig(start_separation=separation)
        model = build_arena(config)
        data = mujoco.MjData(model)
        reset_to_stance(model, data, config)
        for _ in range(200):  # let the fighters settle before timing
            mujoco.mj_step(model, data)

        steps = int(seconds / model.opt.timestep)
        start = time.perf_counter()
        for _ in range(steps):
            mujoco.mj_step(model, data)
        wall = time.perf_counter() - start

        rows.append({
            "label": label,
            "timestep": model.opt.timestep,
            "ncon": int(data.ncon),
            "us_per_step": wall / steps * 1e6,
            "rtf": steps * model.opt.timestep / wall,
            "ms_per_tick": wall / steps * (TICK_DT / model.opt.timestep) * 1e3,
        })
    return rows


def bench_physics(num_fighters: int, timestep: float, spacing: float, seconds: float) -> dict:
    """Physics-only real-time factor, with the fighters settled onto the floor."""
    import mujoco

    model = build_arena(num_fighters, timestep, spacing)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    steps = int(seconds / timestep)
    for _ in range(int(0.2 / timestep)):  # settle, so contacts are live during timing
        mujoco.mj_step(model, data)

    start = time.perf_counter()
    for _ in range(steps):
        mujoco.mj_step(model, data)
    wall = time.perf_counter() - start

    return {
        "fighters": num_fighters,
        "timestep": timestep,
        "spacing": spacing,
        "sim_seconds": steps * timestep,
        "wall_seconds": wall,
        "rtf": (steps * timestep) / wall,
        "ncon": int(data.ncon),
        "us_per_step": wall / steps * 1e6,
    }


def bench_inference(iterations: int = 200) -> dict:
    """Cost of one fighter's two ONNX runs per tick."""
    from openroboxing.runtime.policy import GearSonicPolicy

    policy = GearSonicPolicy()
    rng = np.random.default_rng(0)
    enc_in = rng.standard_normal(ENCODER_INPUT_DIM) * 0.1
    pol_in = rng.standard_normal(POLICY_INPUT_DIM) * 0.1

    start = time.perf_counter()
    for _ in range(iterations):
        policy.encode(enc_in)
    encode_ms = (time.perf_counter() - start) / iterations * 1e3

    start = time.perf_counter()
    for _ in range(iterations):
        policy.act(pol_in)
    act_ms = (time.perf_counter() - start) / iterations * 1e3

    return {
        "encode_ms": encode_ms,
        "act_ms": act_ms,
        "per_fighter_tick_ms": encode_ms + act_ms,
    }


def bench_generation(frames: int = 60) -> dict:
    """Cost of pulling one generator frame plus its replan call."""
    from openroboxing.runtime.generator import GeneratorIntent, MotionBricksGenerator

    gen = MotionBricksGenerator()
    gen.reset(seed=1234)
    intent = GeneratorIntent(style="walk_boxing")

    for _ in range(5):  # warm up
        gen.next_frame()
        gen.generate(intent, gen.context_qpos(), dt=0.5)

    start = time.perf_counter()
    for _ in range(frames):
        gen.next_frame()
        gen.generate(intent, gen.context_qpos(), dt=0.5)
    total = time.perf_counter() - start

    per_frame_ms = total / frames * 1e3
    # generator frames are 30 Hz; a 50 Hz tick consumes 0.6 of one
    return {"per_frame_ms": per_frame_ms, "per_tick_ms": per_frame_ms * 30.0 / TICK_HZ}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench_world", description="Real-time factor for two fighters (M1-T7)."
    )
    parser.add_argument(
        "--seconds", type=float, default=2.0, help="simulated seconds per physics run"
    )
    parser.add_argument("--out", type=Path, default=None, help="write a markdown report here")
    parser.add_argument(
        "--skip-generator", action="store_true", help="skip the generator benchmark"
    )
    parser.add_argument(
        "--arena",
        action="store_true",
        help="benchmark the M3-T1 ring only, and report its verdict",
    )
    args = parser.parse_args(argv)

    if args.arena:
        print("M3-T1 ring: two gloved fighters, ropes and posts\n")
        print(f"  {'case':>8} {'contacts':>9} {'us/step':>9} {'ms/tick':>9} {'RTF':>8}")
        ring = bench_ring(args.seconds)
        for row in ring:
            print(
                f"  {row['label']:>8} {row['ncon']:>9} {row['us_per_step']:>9.1f} "
                f"{row['ms_per_tick']:>9.2f} {row['rtf']:>8.2f}"
            )
        inf = bench_inference()
        worst = max(row["ms_per_tick"] for row in ring)
        budget = worst + 2 * inf["per_fighter_tick_ms"]
        print(f"\n  inference   : {2 * inf['per_fighter_tick_ms']:.2f} ms per tick (both fighters)")
        print(f"  physics     : {worst:.2f} ms per tick (clinch, the worst case)")
        print(f"  total       : {budget:.2f} ms of a {TICK_DT * 1e3:.1f} ms tick "
              f"-> {TICK_DT * 1e3 / budget:.2f}x real time")
        print(
            f"\n  VERDICT: the ring {'HOLDS' if budget < TICK_DT * 1e3 else 'MISSES'} real time "
            "(generation excluded; it is amortised across a replan interval)"
        )
        return 0 if budget < TICK_DT * 1e3 else 1

    print("physics sweep (contact enabled)\n")
    print(
        f"  {'fighters':>8} {'timestep':>9} {'spacing':>8} {'contacts':>9} {'us/step':>9} {'RTF':>8}"
    )
    physics = []
    for fighters in (1, 2):
        for timestep in TIMESTEPS:
            for spacing in (SPACING_APART, SPACING_CONTACT):
                if fighters == 1 and spacing == SPACING_CONTACT:
                    continue  # spacing is meaningless with one fighter
                row = bench_physics(fighters, timestep, spacing, args.seconds)
                physics.append(row)
                print(
                    f"  {row['fighters']:>8} {row['timestep']:>9.4f} {row['spacing']:>8.2f} "
                    f"{row['ncon']:>9} {row['us_per_step']:>9.1f} {row['rtf']:>8.2f}"
                )

    print("\ninference (per fighter, per tick)")
    inf = bench_inference()
    print(
        f"  encoder {inf['encode_ms']:.2f} ms + decoder {inf['act_ms']:.2f} ms "
        f"= {inf['per_fighter_tick_ms']:.2f} ms"
    )

    gen = None
    if not args.skip_generator:
        print("\ngeneration")
        gen = bench_generation()
        print(
            f"  {gen['per_frame_ms']:.2f} ms per 30 Hz frame -> {gen['per_tick_ms']:.2f} ms per tick"
        )

    # Projected budget for two fighters at 50 Hz.
    print("\nprojected budget for TWO fighters at 50 Hz (20.0 ms per tick)")
    rows = []
    for timestep in TIMESTEPS:
        phys = next(
            r
            for r in physics
            if r["fighters"] == 2 and r["timestep"] == timestep and r["spacing"] == SPACING_CONTACT
        )
        physics_ms = phys["us_per_step"] / 1e3 * (TICK_DT / timestep)
        inference_ms = 2 * inf["per_fighter_tick_ms"]
        generation_ms = 2 * gen["per_tick_ms"] if gen else 0.0
        total = physics_ms + inference_ms + generation_ms
        rows.append(
            (timestep, physics_ms, inference_ms, generation_ms, total, TICK_DT * 1e3 / total)
        )
        print(
            f"  dt={timestep:.4f}: physics {physics_ms:6.2f} + inference {inference_ms:6.2f} "
            f"+ generation {generation_ms:6.2f} = {total:6.2f} ms  -> RTF {TICK_DT * 1e3 / total:.2f}x"
        )

    # Recommend the FINEST timestep that keeps a 2x margin, not the largest that merely holds.
    # Physics decides the hit in this game, so contact fidelity is worth more than spare headroom,
    # and the margin absorbs what M3 still has to add: ring geometry, gloves, a real clinch.
    MARGIN = 2.0
    comfortable = [r for r in rows if r[5] >= MARGIN]
    if comfortable:
        best = min(comfortable, key=lambda r: r[0])  # finest timestep with margin to spare
        verdict = (
            f"TWO FIGHTERS HOLD REAL TIME. Recommended timestep {best[0]:.4f}s "
            f"({TICK_DT / best[0]:.0f} substeps per tick) at RTF {best[5]:.2f}x — the finest "
            f"timestep that keeps a {MARGIN:.0f}x margin, chosen for contact fidelity rather than "
            f"for maximum headroom."
        )
    elif any(r[5] >= 1.0 for r in rows):
        best = max((r for r in rows if r[5] >= 1.0), key=lambda r: r[5])
        verdict = (
            f"TWO FIGHTERS HOLD REAL TIME BUT WITHOUT MARGIN. Best is {best[0]:.4f}s at "
            f"RTF {best[5]:.2f}x, under the {MARGIN:.0f}x margin M3's added contacts will need. "
            "Report before M3."
        )
    else:
        verdict = "TWO FIGHTERS DO NOT HOLD REAL TIME at any tested timestep. Report before M3."
    print(f"\nVERDICT: {verdict}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(_report(physics, inf, gen, rows, verdict, args.seconds))
        print(f"\nwrote {args.out}")
    return 0


def _report(physics, inf, gen, rows, verdict, seconds) -> str:
    lines = [
        "# m1_mujoco.md — two-fighter performance (M1-T7)",
        "",
        f"Measured on `{platform.node()}`, {platform.platform()}.",
        f"Physics runs are {seconds:.1f} simulated seconds each, after a 0.2 s settle so contacts are live.",
        "Inference is the CPU execution provider; see the note below.",
        "",
        "## Physics only",
        "",
        "| fighters | timestep (s) | spacing (m) | contacts | us/step | RTF |",
        "|---|---|---|---|---|---|",
    ]
    for r in physics:
        lines.append(
            f"| {r['fighters']} | {r['timestep']:.4f} | {r['spacing']:.2f} | {r['ncon']} | "
            f"{r['us_per_step']:.1f} | {r['rtf']:.2f} |"
        )

    lines += [
        "",
        "## Inference",
        "",
        f"- encoder: **{inf['encode_ms']:.2f} ms**",
        f"- decoder: **{inf['act_ms']:.2f} ms**",
        f"- per fighter per tick: **{inf['per_fighter_tick_ms']:.2f} ms** (both run every tick)",
        "",
    ]
    if gen:
        lines += [
            "## Generation",
            "",
            f"- per 30 Hz frame: **{gen['per_frame_ms']:.2f} ms**",
            f"- amortised per 50 Hz tick: **{gen['per_tick_ms']:.2f} ms**",
            "",
        ]

    lines += [
        "## Projected budget — two fighters at 50 Hz",
        "",
        "One control tick is 20.0 ms. Contact spacing is used, i.e. the expensive case.",
        "",
        "| timestep (s) | physics (ms) | inference (ms) | generation (ms) | total (ms) | RTF |",
        "|---|---|---|---|---|---|",
    ]
    for timestep, phys, inference, generation, total, rtf in rows:
        lines.append(
            f"| {timestep:.4f} | {phys:.2f} | {inference:.2f} | {generation:.2f} | "
            f"{total:.2f} | {rtf:.2f} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "## Caveats",
        "",
        "- Inference is measured on the **CPU** execution provider, chosen in M1-T4 for determinism",
        "  during parity work. A CUDA provider should reduce it; if the budget is tight, that is the",
        "  first lever to pull, and it is a configuration change, not a redesign.",
        "- The generator cost is measured with a replan attempted every frame. The real runtime",
        "  replans far less often, so the generation column is an upper bound.",
        "- Contact counts are for two G1s standing close; a real clinch will produce more.",
        "- This is a bare arena. M3-T1 adds ring geometry and padded gloves, which will add contacts.",
        "",
        (
            "Reproduce with `.venv_mb/bin/python -m openroboxing.tools.bench_world"
            " --out docs/perf/m1_mujoco.md`."
        ),
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
