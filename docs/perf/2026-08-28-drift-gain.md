# Drift gain measurement — phase 2, Task 1

**Date** 2026-08-28 · **Task** `M6-T1` · **Verdict** **PASS — `DRIFT_GAIN = 0.803`**
(after correcting the experiment design; see "Revision" below)

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

One combination is not a sample. This task measures the gain across 9 combinations spanning all
three families at four drift distances (0.25, 0.5, 1.0, 2.0 m) and decides whether a single
`DRIFT_GAIN` constant is justified, per the bar CLAUDE.md sets: no invented numbers, every constant
cited to where it came from.

## Revision 2026-08-28: the first framing was wrong, and is kept here rather than deleted

The first pass computed `reached / asked` where `asked = hypot(recorded_displacement + drift)` —
i.e. the fraction of the combination's **total** displacement covered, recorded footwork and drift
together. That run measured `reached/asked` medians of 0.789 / 0.820 / 0.720 for shadow-boxing /
ib-dodge / ib-combat-turn-jog respectively, with an overall min/max of 0.670–0.891 — **outside** the
±0.10 bar, so the first pass reported **FAIL** and added no constant.

The flaw: `warp()` keeps the rotated recorded offsets at true size (design D4) and scales only the
leftover travel, the *residual*. So the quantity the correction needs is how well the generator
covers the residual, not the total displacement. A `shadow-boxing` combination carries ~2–10 cm of
its own recorded travel, so its `asked` is almost entirely the drift and `reached/asked` is nearly
the quantity wanted. An `ib-combat-turn-jog` combination carries **0.46–0.89 m** of recorded travel
(see the `r0 m` column below), so at a 0.25 m drift its `asked` is dominated by the recorded part,
and `reached/asked` mostly measures how well *that* is covered — a different, better-behaved quantity
the phase-1 spike had already shown holds well. That is why the jog family sat systematically ~0.07
below the other two: not a property of the drift, but of averaging in a baseline the residual gain
never touches.

**The fix.** For each combination, a baseline run at drift = 0 first — ghost exactly
`recorded_displacement`, so the residual is zero and every leg aims at the recorded path alone.
Call the reached distance there `R0`. Then for each drift `d`, with reached distance `R(d)`:

```
incremental_gain = (R(d) - R0) / d
```

This isolates coverage of the *added* travel and cancels each combination's own baseline. It is the
quantity actually used by Task 2: `warp()` never touches the recorded offsets, only the residual.

## Method

For each sampled combination:

1. **Baseline.** `warp(record, (0, 0), 0.0, recorded_displacement, speed_ceiling=1e9)`, drive every
   leg, record `R0 = hypot(*final_plan_frame_root_xy)`.
2. For each drift: `ghost = recorded_displacement + drift * direction`, `direction` the unit vector
   of `recorded_displacement` (or `(1, 0)` if that displacement is under `1e-6` m). Warp and drive
   exactly as the baseline did, recording `R(d)`.
3. `asked = hypot(*ghost)`, `fraction = R(d) / asked` (superseded, kept for comparison),
   `incremental_gain = (R(d) - R0) / d` (the decision metric).

`speed_ceiling=1e9` disables the ceiling with a value no real placement can exceed — the
`speed_ceiling=None` escape hatch is Task 2, not yet in `warp`'s signature. Every leg is driven with
the plan length forced to the recorded leg duration, style `"walk_boxing"`, every generated frame
consumed so the next leg's context follows on — exactly as the phase-1 spike did. The same seed
drives the baseline and every drift for a given combination.

No `(combination, distance)` pair was skipped: 36 of 36 ran (`WarpError` count: 0).

## Results

Full table, one row per (combination, drift), both metrics:

| family | combination | drift m | r0 m | asked m | reached m | fraction (superseded) | incremental_gain |
|---|---|---:|---:|---:|---:|---:|---:|
| shadow-boxing | shadow-boxing-r-001-a359-00 | 0.25 | 0.034 | 0.284 | 0.217 | 0.766 | 0.734 |
| shadow-boxing | shadow-boxing-r-001-a359-00 | 0.50 | 0.034 | 0.534 | 0.412 | 0.773 | 0.757 |
| shadow-boxing | shadow-boxing-r-001-a359-00 | 1.00 | 0.034 | 1.034 | 0.813 | 0.786 | 0.779 |
| shadow-boxing | shadow-boxing-r-001-a359-00 | 2.00 | 0.034 | 2.034 | 1.648 | 0.810 | 0.807 |
| shadow-boxing | shadow-boxing-r-001-a362-m-00 | 0.25 | 0.101 | 0.350 | 0.304 | 0.870 | 0.814 |
| shadow-boxing | shadow-boxing-r-001-a362-m-00 | 0.50 | 0.101 | 0.600 | 0.498 | 0.830 | 0.794 |
| shadow-boxing | shadow-boxing-r-001-a362-m-00 | 1.00 | 0.101 | 1.100 | 0.899 | 0.817 | 0.798 |
| shadow-boxing | shadow-boxing-r-001-a362-m-00 | 2.00 | 0.101 | 2.100 | 1.735 | 0.826 | 0.817 |
| shadow-boxing | shadow-boxing-r-003-a359-00 | 0.25 | 0.070 | 0.347 | 0.251 | 0.724 | 0.723 |
| shadow-boxing | shadow-boxing-r-003-a359-00 | 0.50 | 0.070 | 0.597 | 0.439 | 0.735 | 0.737 |
| shadow-boxing | shadow-boxing-r-003-a359-00 | 1.00 | 0.070 | 1.097 | 0.838 | 0.764 | 0.768 |
| shadow-boxing | shadow-boxing-r-003-a359-00 | 2.00 | 0.070 | 2.097 | 1.660 | 0.792 | 0.795 |
| ib-dodge | ib-dodge-270-r-001-a437-00 | 0.25 | 0.443 | 0.944 | 0.646 | 0.684 | 0.812 |
| ib-dodge | ib-dodge-270-r-001-a437-00 | 0.50 | 0.443 | 1.194 | 0.861 | 0.721 | 0.836 |
| ib-dodge | ib-dodge-270-r-001-a437-00 | 1.00 | 0.443 | 1.694 | 1.278 | 0.754 | 0.836 |
| ib-dodge | ib-dodge-270-r-001-a437-00 | 2.00 | 0.443 | 2.694 | 2.121 | 0.787 | 0.839 |
| ib-dodge | ib-dodge-back-l-001-a437-00 | 0.25 | 0.188 | 0.457 | 0.378 | 0.828 | 0.761 |
| ib-dodge | ib-dodge-back-l-001-a437-00 | 0.50 | 0.188 | 0.707 | 0.570 | 0.807 | 0.765 |
| ib-dodge | ib-dodge-back-l-001-a437-00 | 1.00 | 0.188 | 1.207 | 0.985 | 0.816 | 0.797 |
| ib-dodge | ib-dodge-back-l-001-a437-00 | 2.00 | 0.188 | 2.207 | 1.819 | 0.824 | 0.816 |
| ib-dodge | ib-dodge-down-l-002-a437-00 | 0.25 | 0.225 | 0.470 | 0.419 | 0.891 | 0.776 |
| ib-dodge | ib-dodge-down-l-002-a437-00 | 0.50 | 0.225 | 0.720 | 0.626 | 0.870 | 0.803 |
| ib-dodge | ib-dodge-down-l-002-a437-00 | 1.00 | 0.225 | 1.220 | 1.034 | 0.847 | 0.809 |
| ib-dodge | ib-dodge-down-l-002-a437-00 | 2.00 | 0.225 | 2.220 | 1.850 | 0.833 | 0.813 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-00 | 0.25 | 0.890 | 1.567 | 1.090 | 0.696 | 0.802 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-00 | 0.50 | 0.890 | 1.817 | 1.292 | 0.711 | 0.804 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-00 | 1.00 | 0.890 | 2.317 | 1.691 | 0.730 | 0.801 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-00 | 2.00 | 0.890 | 3.317 | 2.511 | 0.757 | 0.811 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-m-00 | 0.25 | 0.875 | 1.567 | 1.080 | 0.689 | 0.822 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-m-00 | 0.50 | 0.875 | 1.817 | 1.283 | 0.706 | 0.816 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-m-00 | 1.00 | 0.875 | 2.317 | 1.689 | 0.729 | 0.815 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-270-r-003-a437-m-00 | 2.00 | 0.875 | 3.317 | 2.511 | 0.757 | 0.818 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-360-r-001-a437-00 | 0.25 | 0.457 | 0.966 | 0.648 | 0.670 | 0.762 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-360-r-001-a437-00 | 0.50 | 0.457 | 1.216 | 0.859 | 0.706 | 0.803 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-360-r-001-a437-00 | 1.00 | 0.457 | 1.716 | 1.262 | 0.735 | 0.805 |
| ib-combat-turn-jog | ib-combat-turn-jog-start-360-r-001-a437-00 | 2.00 | 0.457 | 2.716 | 2.060 | 0.758 | 0.801 |

(The two `ib-combat-turn-jog-start-270-r-*` rows have near-identical `asked` values because they are
mirror pairs of the same take — mirroring preserves displacement magnitude — but their `reached` and
`r0` values differ, so these are two independent runs, not a duplicate. Note `r0` itself: 0.03–0.10 m
for shadow-boxing, 0.19–0.44 m for ib-dodge, **0.46–0.89 m** for ib-combat-turn-jog — the recorded
travel the superseded metric was diluting itself with.)

Summary statistics over all 36 (combination, distance) pairs, both metrics:

| metric | n | median | mean | min | max |
|---|---:|---:|---:|---:|---:|
| fraction (superseded, total displacement) | 36 | 0.765 | 0.772 | 0.670 | 0.891 |
| **incremental_gain (decision)** | 36 | **0.803** | 0.796 | 0.723 | 0.839 |

Per-family medians, both metrics:

| family | n | fraction: median (min–max) | incremental_gain: median (min–max) |
|---|---:|---|---|
| shadow-boxing | 12 | 0.789 (0.724–0.870) | 0.786 (0.723–0.817) |
| ib-dodge | 12 | 0.820 (0.684–0.891) | 0.811 (0.761–0.839) |
| ib-combat-turn-jog | 12 | 0.720 (0.670–0.758) | 0.805 (0.762–0.822) |

Pairs skipped (`WarpError`): **0**.

## Reading

**1. The family effect disappears once the baseline is subtracted out.** `reached/asked` medians
span 0.720–0.820 (a 0.10 range) across families; `incremental_gain` medians span 0.786–0.811 (a 0.025
range) — an order of magnitude tighter. `ib-combat-turn-jog`, the family the first pass flagged as
systematically low, has the **highest** minimum (0.762) and a median (0.805) squarely between the
other two once its 0.46–0.89 m of recorded travel is no longer part of the denominator. This is
exactly the mechanism the revision predicted: the first framing was measuring how well the recorded
path is covered for the travelling family, not how well the drift is.

**2. The preliminary single-combination reading was itself measuring a mix.** `shadow-boxing-r-001-
a359-00` has `r0 = 0.034` m, small enough that its `reached/asked` (0.766/0.773/0.786/0.810) and its
`incremental_gain` (0.734/0.757/0.779/0.807) nearly coincide — which is why the preliminary probe
looked clean using the flawed metric. It was a lucky choice of combination, not evidence the metric
was sound.

**3. The corrected spread clears the bar.** Overall `incremental_gain`: median 0.803, min 0.723
(**−0.080**), max 0.839 (**+0.036**) — both within the ±0.10 bar. Per-family medians (0.786 / 0.811 /
0.805) also sit within 0.01–0.02 of the overall median, so family is not doing hidden work here
either.

**4. Where the residual spread does live.** The two lowest incremental_gain values (0.723, 0.734) are
both shadow-boxing takes (`-003-` and `-001-`) at the smallest drift, 0.25 m; the three highest
(0.836, 0.836, 0.839) are all `ib-dodge-270-r-001-a437-00` at drift ≥ 0.5 m. Both sit comfortably
inside the bar, so this is measurement noise at the scale the bar tolerates, not a second family
effect — but it is the reason the constant is a median over 36 points rather than a single clean
number.

## Decision

**PASS.** `incremental_gain` over 36 (combination, distance) pairs across 9 combinations and all
three families: median **0.803**, min 0.723, max 0.839 — both within ±0.10 of the median. Per the
task's bar, a single constant is justified.

**`DRIFT_GAIN = 0.803`** added to `src/openroboxing/spec/constants.py`, cited to this document and
this measurement.

## What would sharpen this further

- The lowest two points are both `shadow-boxing` at the smallest drifts (0.25–0.50 m), where `R(d) -
  R0` is a small difference of two similar numbers and is the most sensitive to any per-frame noise
  in the generator's plan tail. A larger `--per-family` sample weighted toward small drifts would
  show whether that is a real small-drift effect or sampling noise.
- This measurement drives the generator directly, with no physics and no policy in the loop. Whether
  `DRIFT_GAIN` still holds once a fighter is walking under MuJoCo and GEAR-SONIC is a phase-3
  question, not one this spike answers.
