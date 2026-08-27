# Drift gain measurement — phase 2, Task 1

**Date** 2026-08-28 · **Task** `M6-T1` · **Verdict** **NO SINGLE CONSTANT** (spread exceeds the bar)

Reproduce with:

```bash
.venv_mb/bin/python -m openroboxing.tools.measure_drift_gain
```

(default `--seed 0 --per-family 3`, sampling 9 combinations: 3 `shadow-boxing-*`, 3 `ib-dodge-*`,
3 `ib-combat-turn-jog-*`, deterministically by even stride over each family's sorted file list.)

---

## The question

A preliminary probe on one combination (`shadow-boxing-r-001-a359-00`) found MotionBricks converges
toward a commanded drift but arrives short, by a fraction that looked constant:

| drift asked | reached | fraction |
|---|---|---|
| 0.25 m | 0.22 m | 0.77 |
| 0.50 m | 0.41 m | 0.77 |
| 1.00 m | 0.81 m | 0.79 |
| 2.00 m | 1.65 m | 0.81 |

One combination is not a sample. This task measures the fraction across 9 combinations spanning all
three families at four drift distances (0.25, 0.5, 1.0, 2.0 m) and decides whether a single
`DRIFT_GAIN` constant is justified, per the bar CLAUDE.md sets: no invented numbers, every constant
cited to where it came from.

## Method

For each sampled combination and each drift:

1. `ghost = recorded_displacement + drift * direction`, `direction` the unit vector of
   `recorded_displacement` (or `(1, 0)` if that displacement is under `1e-6` m).
2. `warp(record, (0, 0), 0.0, ghost, speed_ceiling=1e9)` — the ceiling disabled with a value no real
   placement can exceed, standing in for the `speed_ceiling=None` Task 2 adds.
3. Every leg driven through the generator exactly as the phase-1 spike did — plan length forced to
   the recorded leg duration, style `"walk_boxing"`, every generated frame consumed so the next leg's
   context follows on.
4. `reached = hypot(*final_plan_frame_root_xy)`, `asked = hypot(*ghost)`, both from the anchor at the
   origin. `fraction = reached / asked`.

No `(combination, distance)` pair was skipped: 36 of 36 ran (`WarpError` count: 0), because the
disabled ceiling accepts every placement this sample produced.

## Results

Full table, one row per (combination, drift):

| family | combination | drift m | asked m | reached m | fraction |
|---|---|---:|---:|---:|---:|
| shadow-boxing | shadow-boxing-r-001-a359-00 | 0.25 | 0.284 | 0.217 | 0.766 |
| shadow-boxing | shadow-boxing-r-001-a359-00 | 0.50 | 0.534 | 0.412 | 0.773 |
| shadow-boxing | shadow-boxing-r-001-a359-00 | 1.00 | 1.034 | 0.813 | 0.786 |
| shadow-boxing | shadow-boxing-r-001-a359-00 | 2.00 | 2.034 | 1.648 | 0.810 |
| shadow-boxing | shadow-boxing-r-001-a362-m-00 | 0.25 | 0.350 | 0.304 | 0.870 |
| shadow-boxing | shadow-boxing-r-001-a362-m-00 | 0.50 | 0.600 | 0.498 | 0.830 |
| shadow-boxing | shadow-boxing-r-001-a362-m-00 | 1.00 | 1.100 | 0.899 | 0.817 |
| shadow-boxing | shadow-boxing-r-001-a362-m-00 | 2.00 | 2.100 | 1.735 | 0.826 |
| shadow-boxing | shadow-boxing-r-003-a359-00 | 0.25 | 0.347 | 0.251 | 0.724 |
| shadow-boxing | shadow-boxing-r-003-a359-00 | 0.50 | 0.597 | 0.439 | 0.735 |
| shadow-boxing | shadow-boxing-r-003-a359-00 | 1.00 | 1.097 | 0.838 | 0.764 |
| shadow-boxing | shadow-boxing-r-003-a359-00 | 2.00 | 2.097 | 1.660 | 0.792 |
| ib-dodge | ib-dodge-270-r-001-a437-00 | 0.25 | 0.944 | 0.646 | 0.684 |
| ib-dodge | ib-dodge-270-r-001-a437-00 | 0.50 | 1.194 | 0.861 | 0.721 |
| ib-dodge | ib-dodge-270-r-001-a437-00 | 1.00 | 1.694 | 1.278 | 0.754 |
| ib-dodge | ib-dodge-270-r-001-a437-00 | 2.00 | 2.694 | 2.121 | 0.787 |
| ib-dodge | ib-dodge-back-l-001-a437-00 | 0.25 | 0.457 | 0.378 | 0.828 |
| ib-dodge | ib-dodge-back-l-001-a437-00 | 0.50 | 0.707 | 0.570 | 0.807 |
| ib-dodge | ib-dodge-back-l-001-a437-00 | 1.00 | 1.207 | 0.985 | 0.816 |
| ib-dodge | ib-dodge-back-l-001-a437-00 | 2.00 | 2.207 | 1.819 | 0.824 |
| ib-dodge | ib-dodge-down-l-002-a437-00 | 0.25 | 0.470 | 0.419 | 0.891 |
| ib-dodge | ib-dodge-down-l-002-a437-00 | 0.50 | 0.720 | 0.626 | 0.870 |
| ib-dodge | ib-dodge-down-l-002-a437-00 | 1.00 | 1.220 | 1.034 | 0.847 |
| ib-dodge | ib-dodge-down-l-002-a437-00 | 2.00 | 2.220 | 1.850 | 0.833 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-00 | 0.25 | 1.567 | 1.090 | 0.696 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-00 | 0.50 | 1.817 | 1.292 | 0.711 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-00 | 1.00 | 2.317 | 1.691 | 0.730 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-00 | 2.00 | 3.317 | 2.511 | 0.757 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-m-00 | 0.25 | 1.567 | 1.080 | 0.689 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-m-00 | 0.50 | 1.817 | 1.283 | 0.706 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-m-00 | 1.00 | 2.317 | 1.689 | 0.729 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-m-00 | 2.00 | 3.317 | 2.511 | 0.757 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-360-r-001-a437-00 | 0.25 | 0.966 | 0.648 | 0.670 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-360-r-001-a437-00 | 0.50 | 1.216 | 0.859 | 0.706 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-360-r-001-a437-00 | 1.00 | 1.716 | 1.262 | 0.735 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-360-r-001-a437-00 | 2.00 | 2.716 | 2.060 | 0.758 |

(The two `ib-combat-turn-jog-start-270-r-*` rows have near-identical `asked` values because they are
mirror pairs of the same take — mirroring preserves displacement magnitude — but their `reached`
values differ, so these are two independent runs, not a duplicate.)

Summary statistics over all 36 (combination, distance) pairs:

| | n | median | mean | min | max |
|---|---:|---:|---:|---:|---:|
| overall | 36 | **0.765** | 0.772 | 0.670 | 0.891 |

Per-family median:

| family | n | median | mean | min | max |
|---|---:|---:|---:|---:|---:|
| shadow-boxing | 12 | 0.789 | 0.791 | 0.724 | 0.870 |
| ib-dodge | 12 | 0.820 | 0.805 | 0.684 | 0.891 |
| ib-combat-turn-jog | 12 | 0.720 | 0.720 | 0.670 | 0.758 |

Pairs skipped (`WarpError`): **0**.

## Reading

**1. The preliminary reading replicates almost exactly.** The one combination it was measured on,
`shadow-boxing-r-001-a359-00`, reproduces here as 0.766 / 0.773 / 0.786 / 0.810 against the
preliminary 0.77 / 0.77 / 0.79 / 0.81 — the tool is measuring the same thing the probe measured.

**2. But the fraction is not one number across the library.** The bar this task set was min and max
within ±0.10 of the median. Overall: median 0.765, max 0.891 is **+0.126** above it — outside the
bar on the high side, driven by `ib-dodge-down-l-002-a437-00`'s 0.891 at the smallest drift (0.25 m).
The low side (0.670, **−0.095**) is inside the bar on its own, but the high side alone is enough to
fail it.

**3. The families separate.** Medians run 0.720 (`ib-combat-turn-jog`) → 0.789 (`shadow-boxing`) →
0.820 (`ib-dodge`), a 0.10 span between family medians before even looking at within-family spread.
`ib-combat-turn-jog`'s 12 samples sit tightly (0.670–0.758, a 0.088 band) but centred a full 0.069
below the shadow-boxing median and 0.10 below the ib-dodge median — the travelling family is
systematically harder to fully close on than the corrective one, not just noisier.

**4. The small-drift end is where it breaks down worst.** `ib-dodge-down-l-002-a437-00` at drift
0.25 m has an `asked` of only 0.470 m (its own recorded displacement is already ~0.22 m in the same
direction), and reaches 0.891 of it — well above every other row in the table. Small commanded
displacements riding on top of a combination's own recorded footwork do not dilute cleanly into "0.8
of whatever is asked"; the correction that would fit the four largest drifts overcorrects here.

## Decision

**FAIL.** The spread (max 0.891, min 0.670, median 0.765) is wider than the ±0.10 bar set for this
task. Per the task's instruction, no single constant is picked from this data — a per-family or
per-combination gain is a schema change and a design decision for the project owner, not something
to add quietly by taking a median anyway.

**Not added to `spec/constants.py`.** `DRIFT_GAIN` is deliberately absent. `runtime/warp.py`'s
residual correction (Task 2) needs the owner to choose between: a single constant accepted as an
approximation with known ±0.13 error at the extremes, a per-family constant (medians 0.720 / 0.789 /
0.820 are each tighter within their own family), or a per-combination value carried in
`CombinationRecord` alongside `tracking_error_rad`. This document and its table are the input to that
decision, not a stand-in for it.

## What would sharpen this before deciding

- The failing case is one `ib-dodge` sample at the smallest drift. A larger per-family sample
  (`--per-family` above the default 3) would show whether 0.891 is an outlier or representative of
  a whole sub-class (e.g., combinations whose own recorded displacement already dominates a small
  commanded drift).
- If a per-family constant is chosen, `ib-combat-turn-jog`'s tight in-family spread (0.088) is the
  strongest case that family-scoping alone would clear the bar; `ib-dodge`'s 0.207 in-family spread
  says it would not, on its own.
