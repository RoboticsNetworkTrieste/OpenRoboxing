# Drift gain re-measured under the pinned-keyframe schedule

**Date:** 2026-09-03 · `spec/intent.md` 3.2 · supersedes `docs/perf/2026-08-28-drift-gain.md`

```bash
.venv_mb/bin/python -m openroboxing.tools.measure_drift_gain
```

9 combinations (3 per family) × 4 drift distances (0.25, 0.50, 1.00, 2.00 m) = 36 pairs, on an
NVIDIA RTX 6000 Ada.

## Why this had to be re-measured

`runtime/warp.py` divides every residual by `DRIFT_GAIN`, so the constant is only correct for the
schedule it was measured on. The 2026-08-28 figure of **0.803** was measured with `force=True`, one
clean plan per leg — `measure_drift_gain.py` said so in a comment ("this measures the plan, not the
replan schedule"). The runtime has never run that way, and since 3.2 it also stops replanning inside
a leg's tail. A gain measured on a schedule the runtime does not run is a gain for nothing.

The tool now drives the real production path: `CombinationRunner.intent_for` for the intents and
`ReferenceStream` for the pull/replan loop, honouring `intent.replan`.

## Result

| metric | 2026-08-28 (forced plans) | **2026-09-03 (real schedule)** |
|---|---|---|
| incremental gain, median | 0.803 | **0.935** |
| mean | — | 0.912 |
| min / max | 0.723 / 0.839 | **0.645 / 1.054** |
| per-family median | 0.786–0.811 | 0.903 (shadow-boxing) · 0.993 (ib-dodge) · 0.952 (ib-combat-turn-jog) |

**The gain is much closer to 1.0 than under forced plans, and the mechanism is clear.** A single
forced plan lands wherever it lands. Replanning at the ambient cadence re-aims at a target that is
*pinned*, so the fighter gets repeated chances to converge on it and closes most of the shortfall
that a one-shot plan leaves.

At the old 0.803, `warp` asks for `residual / 0.803` = 1.25× the residual while the generator now
covers 0.935 of it — **a ~16 % overshoot past the ghost on every commit**. Leaving the constant alone
was not an option; 0.803 is not merely imprecise now, it is wrong in a measurable direction.

`DRIFT_GAIN` is therefore set to **0.935**.

## The consistency bar fails, and that is a real caveat

The tool's own decision bar — min and max within ±0.10 of the median — **FAILS**: the spread runs
0.645 to 1.054 against a median of 0.935. This is recorded rather than smoothed over, because
`spec/constants.py` holds `DRIFT_GAIN` to that bar and a reader is entitled to know it is not met.

The spread is **structured, not random: it is concentrated at small drifts.**

| drift | observed incremental gains |
|---|---|
| 0.25 m | 0.762, 1.024, 0.838, 1.037, 0.894, 0.780, 0.645, … — wide |
| 2.00 m | 0.983, 0.874, 1.051, 0.884, 1.017, 1.000, 0.970, 0.934 — tight, ≈0.97 |

At 0.25 m the commanded residual is comparable to the combination's own recorded travel (0.09–0.23 m
for these records) and to the measurement's noise floor, so the ratio is dominated by whatever the
recording was already doing. At 1–2 m the residual dominates and the gain settles near 0.97.

This matters less in play than the raw spread suggests, because the gain multiplies the residual: a
0.10 error on a 0.25 m residual is 2.5 cm, well inside `ARRIVAL_RADIUS_M` and inside the ghost's own
rendering. The same error on a 2 m residual is 20 cm, and it is precisely at 2 m that the
measurements are tight.

**What would close it properly, and why it was not done here:** a drift-dependent gain (or simply
weighting the fit toward larger residuals, which gives ≈0.96) would fit the data better than any
single number. That is a change to `warp`'s signature and to `spec/combination.md`'s contract, not a
constant update, and it should be decided deliberately rather than folded into this change. The
single constant is retained, set to the measured median, with this file as the record of its known
weakness.
