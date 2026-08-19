# m1_mujoco.md — two-fighter performance (M1-T7)

Measured on `hpc-dev`, Linux-6.17.0-1030-oem-x86_64-with-glibc2.39.
Physics runs are 2.0 simulated seconds each, after a 0.2 s settle so contacts are live.
Inference is the CPU execution provider; see the note below.

## Physics only

| fighters | timestep (s) | spacing (m) | contacts | us/step | RTF |
|---|---|---|---|---|---|
| 1 | 0.0010 | 1.50 | 37 | 108.3 | 9.23 |
| 1 | 0.0020 | 1.50 | 36 | 101.1 | 19.79 |
| 1 | 0.0040 | 1.50 | 18 | 110.1 | 36.32 |
| 1 | 0.0050 | 1.50 | 28 | 111.9 | 44.67 |
| 2 | 0.0010 | 1.50 | 75 | 222.1 | 4.50 |
| 2 | 0.0010 | 0.35 | 84 | 247.4 | 4.04 |
| 2 | 0.0020 | 1.50 | 81 | 218.8 | 9.14 |
| 2 | 0.0020 | 0.35 | 82 | 256.6 | 7.79 |
| 2 | 0.0040 | 1.50 | 84 | 221.2 | 18.08 |
| 2 | 0.0040 | 0.35 | 69 | 266.6 | 15.00 |
| 2 | 0.0050 | 1.50 | 72 | 215.3 | 23.22 |
| 2 | 0.0050 | 0.35 | 85 | 304.8 | 16.41 |

## Inference

- encoder: **0.34 ms**
- decoder: **0.24 ms**
- per fighter per tick: **0.58 ms** (both run every tick)

## Generation

- per 30 Hz frame: **2.76 ms**
- amortised per 50 Hz tick: **1.65 ms**

## Projected budget — two fighters at 50 Hz

One control tick is 20.0 ms. Contact spacing is used, i.e. the expensive case.

| timestep (s) | physics (ms) | inference (ms) | generation (ms) | total (ms) | RTF |
|---|---|---|---|---|---|
| 0.0010 | 4.95 | 1.15 | 3.31 | 9.41 | 2.13 |
| 0.0020 | 2.57 | 1.15 | 3.31 | 7.02 | 2.85 |
| 0.0040 | 1.33 | 1.15 | 3.31 | 5.79 | 3.45 |
| 0.0050 | 1.22 | 1.15 | 3.31 | 5.68 | 3.52 |

## Verdict

**TWO FIGHTERS HOLD REAL TIME. Recommended timestep 0.0010s (20 substeps per tick) at RTF 2.13x — the finest timestep that keeps a 2x margin, chosen for contact fidelity rather than for maximum headroom.**

## Caveats

- Inference is measured on the **CPU** execution provider, chosen in M1-T4 for determinism
  during parity work. A CUDA provider should reduce it; if the budget is tight, that is the
  first lever to pull, and it is a configuration change, not a redesign.
- The generator cost is measured with a replan attempted every frame. The real runtime
  replans far less often, so the generation column is an upper bound.
- Contact counts are for two G1s standing close; a real clinch will produce more.
- This is a bare arena. M3-T1 adds ring geometry and padded gloves, which will add contacts.

Reproduce with `.venv_mb/bin/python -m openroboxing.tools.bench_world --out docs/perf/m1_mujoco.md`.
