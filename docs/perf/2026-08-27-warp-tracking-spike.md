# Warp tracking spike — the phase-1 checkpoint

**Date** 2026-08-27 · **Task** `M5-T12` · **Verdict** **GO**

Reproduce with:

```bash
.venv_mb/bin/python -m openroboxing.tools.spike_warp_tracking
```

---

## The question

`docs/superpowers/specs/2026-08-27-motion-combinations-design.md` phasing says this is not a
formality:

> Nothing yet demonstrates that MotionBricks will hit a forced 0.8 s leg to an authored pose *while
> its root is being dragged along a drift*. Poses are reached to 2–3° when they are the target of an
> unhurried plan; a short forced leg with a moving root is a different ask, and it is the assumption
> the whole feature rests on.

Three combinations, one of each character, driven leg by leg through the generator with the plan
length **forced** to the recorded leg duration. Each is placed twice: `recorded` puts the ghost
exactly where the take ends (zero drift, isolating "can it reach the pose"), `drifted` pushes it 1.0 m
further along the same direction so every leg carries drift on top of the recorded footwork.

## Results

Worst-joint error between the plan's final frame and the commanded pose, degrees:

| placement | legs | median | mean | worst |
|---|---|---|---|---|
| `recorded` | 15 | **8.7** | 9.3 | 27.3 |
| `drifted` | 15 | **8.6** | 9.4 | 27.4 |

Plan length was **exact on all 30 legs** — every leg returned precisely `tokens × 4` frames.

## Reading

**1. The drift is free.** This was the risk the checkpoint existed for, and it is not real: dragging
the root a further 1.0 m over the combination moves the median by −0.1° and the worst case by +0.1°.
Whatever error there is, the warp is not causing it. **D4 is safe.**

**2. Forcing the plan length works.** 30 of 30 legs honoured `horizon_tokens` exactly. The owner's
"motions have to last the same time" is therefore mechanically achievable, and combined with the
residual-diffusion tokenisation already measured to hold every combination inside one token, the
timing requirement is met end to end.

This does still reverse `spec/intent.md` 2.0's move to `horizon_tokens=None`, so the three defects
`runtime/reference.py` documents deleting remain phase-2 regression tests. Length being honoured is
not the same as the *stream* handling it correctly.

**3. Leg length does not drive the error.** Six-token legs — the shortest the planner allows — span
4.9° to 12.1°; the 12- and 14-token legs sit at 8.9° and 9.4°. There is no visible relationship, so
**`MIN_TOKENS` is not the constraint** and D2 does not need revisiting.

**4. The error is in line with the game's own shipped library.** The design quoted 2–3° for authored
poses, and that number does not describe what is actually admitted today. The ten **admitted** poses
in `poses/v0.1` record:

| | median | worst |
|---|---|---|
| admitted v0.1 poses (`generator_error_rad`) | **12.0°** | 14.1° |
| combination legs, this spike | **8.7°** | 27.3° |

The combinations' median is **better than every pose currently in the game**. `jab-left`, which the
library treats as its best-behaved pose, records 8.2° — the same neighbourhood as our median.

So the honest statement is not "poses are reached to 2–3°" but "this generator places an authored
pose within roughly 8–14° of worst-joint error, and the game is built on that". Combinations sit
inside that envelope.

## The one thing to carry into phase 3

One leg is an outlier: `shadow-boxing-r-001-a359-00` leg 4 (7 tokens) at **27.3°**, roughly double
the worst admitted v0.1 pose, and it reproduces identically at both placements so it is a property of
that pose rather than of the drift. That is exactly the kind of move the admission gate exists to
reject. It is a data point for phase 3's threshold, not a defect to fix now — and it is a reason to
expect real attrition when 120 combinations are measured rather than 3.

## Conclusion

Proceed to phase 2. The assumption the design flagged as load-bearing holds: the drift costs
essentially nothing, forced leg lengths are honoured exactly, and pose error is no worse than the
library already shipping. Nothing measured here argues for changing D2 or D4.
