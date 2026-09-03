# Pinned keyframes and merged legs

**Date:** 2026-09-03
**Status:** implemented 2026-09-03. Figures below corrected against the built result — see "Corrections" at the end.
**Owner decision:** project owner, 2026-09-03
**Supersedes:** the "consumed exactly" sentence in `spec/intent.md` 3.0 (§"Forced plan lengths are
back, and that has a cost"), which specifies a contract the runtime never implemented.

---

## The problem

MotionBricks is an in-betweening model. It fills a hole in a timeline between **context keyframes**
(the first 4 frames, always supplied, describing the recent past) and a **target keyframe** (the
plan's last token). The owner's framing, 2026-09-03:

> Time in MotionBricks is a continuous array that has to be filled where there are holes. The
> keyframes you put stay in place while the array moves forward accordingly with the time. The
> consumed keyframes after the 0.5 seconds have to be discarded.

**A keyframe is pinned in absolute time. The window slides; the keyframe does not.** Today's runtime
violates exactly this, and it is the root cause of the reported "motions broken in pieces".

### How it breaks today

`ReferenceStream.ensure` calls `generate(intent, ctx, dt=REPLAN_DT, force=False)` after every frame
it pulls (`runtime/reference.py:134`). Upstream's cadence gate (`full_agent.py:122-124`) lets a
replan through every `REPLAN_DT * GENERATOR_HZ` = **15 generator frames (0.5 s)**.
`CombinationRunner.intent_for` passes `horizon_tokens=leg.horizon_tokens` — the leg's **full**
length — on every one of those frames (`runtime/sequence.py:144`). So for a 12-token leg
(48 frames, 1.6 s):

| replan at frame | requested tokens | keyframe lands at frame |
|---|---|---|
| 0 | 12 | 48 |
| 15 | 12 | 63 |
| 30 | 12 | 78 |

The keyframe is a **receding horizon**: re-aimed 15 frames further out every 0.5 s, never arriving
at the leg's boundary. The fighter converges toward it — which is why "holding a pose" works at all
(`runtime/reference.py`'s docstring, measured 2026-08-13) — but the recorded rhythm is stretched and
boundary poses are only ever partially attained. A leg's intent also takes effect up to 15 frames
after its boundary, which for a 6-token leg (24 frames) is most of the leg.

`spec/intent.md` 3.0 already states the correct contract — each leg's plan "forced to
`leg.horizon_tokens` **and consumed exactly**" — but only the forcing half was built. The consuming
half does not exist, and `tests/test_reference_forced_length.py::test_defect3_...` explicitly
asserts no replan is ever forced. This spec resolves that contradiction in favour of a third
mechanism that needs no forcing at all.

### The second constraint: the plan cap

`max_tokens: 16` in the checkpoint config (`out/motionbricks_root/version_1/config.yaml:49`) caps a
single plan at **16 tokens = 64 frames = 2.13 s**. Measured over the 130-combination v0.2 library
(623 legs): median leg 9 tokens (1.2 s), max 16. Only **131 of 493 adjacent leg pairs (27%)** fit
inside 16 tokens when merged, and only 281 of 623 legs could double within it. **Pairwise merging
alone cannot make legs longer**; it silently no-ops on three quarters of the library.

### What is not negotiable

**Context is exactly 4 frames and cannot be increased.** Verified, not assumed:
`_generate_inbetween_frames` builds `context_local_root_values` as a fixed
`[batch, NUM_FRAMES_PER_TOKEN, 4]` tensor and assigns `angle[:, 1:] - angle[:, :-1]` into it
(`full_agent.py:406-412`), so a 5-frame context raises a shape error outright. Upstream then slices
`local_pose_emb[:, :num_frames_per_token, :]` into `_proj_start_input`
(`root_backbone.py:154-157`), a **learned projection sized for 4 frames**. More context requires a
retrained backbone and is out of scope. What is in scope is ensuring those 4 frames are real
consecutive motion with true velocities — see "Context integrity" below.

---

## The design

Two halves. Both follow from pinning the keyframe.

### Half 1 — runtime: pin the keyframe, shrink the hole

`CombinationRunner` already fixes every leg's boundary tick at construction
(`runtime/sequence.py:81-86`), and `intent_for(tick)` is already called once per generated frame
with the tick that frame will play at. So the runner computes the hole rather than the leg:

```
remaining_ticks  = leg_boundary_tick - tick
remaining_tokens = ceil(remaining_ticks / (SECONDS_PER_TOKEN * TICK_HZ))
```

**`ceil`, not `round`, and this is load-bearing.** A plan that ends a frame or two *short* of the
boundary leaves the play cursor clamped at the plan's final frame, and
`get_context_mujoco_qpos` then returns four copies of it — the frozen-context failure described
below. `ceil` guarantees the last plan always reaches at least the boundary, overshooting by **less
than one token** (4 frames), which the next leg's replan simply writes over. Asserted directly as
`0 <= landing - boundary < TICKS_PER_TOKEN` in `tests/test_sequence_pinned.py`.

and asks for `remaining_tokens` instead of `leg.horizon_tokens`. The keyframe stays at its absolute
tick; each replan fills only the shrinking hole between the new context and it. Consumed frames are
discarded by the generator's own cursor, exactly as the owner describes.

Three regimes fall out, each with a distinct meaning:

| `remaining_tokens` | behaviour | why |
|---|---|---|
| `> MAX_TOKENS` (16) | request `MAX_TOKENS`, **no pose target** (`pose=None`) | the hole is bigger than any plan; the keyframe is not reachable yet, so nothing is aimed at it. Untargeted `walk_boxing` shaped only by the leg's `target_position`. |
| `MIN_TOKENS..MAX_TOKENS` (6–16) | request `remaining_tokens`, keyframe as target | the real in-between. This is where the recorded pose lands. |
| `< MIN_TOKENS` (6) | **do not replan at all** | no hole remains that the model can fill. Letting the last plan play out is what makes the keyframe land exactly. |

**The `< MIN_TOKENS` rule is the one that fixes the landing.** `MIN_TOKENS` is a hard floor — a
shorter plan cannot be requested — so without this rule the final 0.8 s of every leg would overshoot
its boundary, which is the residual defect the alternative "shrink-horizon everywhere" design could
not remove. The landing is **never early and less than one token late** — never early is the half
that matters, since an early end is what clamps the cursor and freezes the context.

Note the no-replan tail is `MIN_TOKENS - 1` tokens, not `MIN_TOKENS`: `ceil` lifts a 5.85-token hole
back to a 6-token request, so replanning stops strictly below that.

**Landing tick vs intent-switch tick.** The runner's leg boundary remains authoritative for *which*
leg's intent is issued; the pose lands within one token of it. That mismatch is well inside the
5-tick `_SETTLE_TOLERANCE_TICKS` the existing regression tests already tolerate.

**The hold state is unchanged in behaviour.** Past a combination's end there is no future keyframe.
The runner keeps returning the final leg (`leg_index` is a fixed point past `end_tick`) and requests
`MIN_TOKENS`, re-aiming at the final pose every replan — which is today's converge-and-hold
behaviour, now stated explicitly rather than emerging from a full-length request.

**The opening stance is unchanged**: `style=OPENING_STANCE_CONTEXT` (`idle`),
`horizon_tokens=None`, no pose.

#### Mechanism: how "do not replan" reaches the stream

`ReferenceStream` decides whether to call `generate()`; the runner decides whether there is a hole.
`GeneratorIntent` gains one explicit boolean field carrying that decision, and `ReferenceStream._plan`
skips the `generate()` call when it is set. Rejected alternatives:

- **`horizon_tokens=None`** — means "the model picks its own length by argmax", which would overshoot
  the boundary. Wrong semantics.
- **Letting `ReferenceStream` compute it** — the stream would need to know about legs and boundaries,
  which it deliberately does not (`runtime/reference.py`: "The stream needs no idea what a commit
  is").

Nothing is ever forced, so `force=False` and the ambient `replan_dt` remain invariant on every call
that is made. `tests/test_reference_forced_length.py::test_defect3_...` therefore passes unchanged —
skipping a call does not violate an assertion about the calls that happen.

#### Context integrity

A plan must never be played to its final frame. `get_context_mujoco_qpos` reads 4 frames at the play
cursor and clamps to the last index (`full_agent.py:517-521`), so a cursor pinned at the end returns
**four copies of one frame** — zero velocity, and the next plan starts from a fighter the model
believes is frozen. Under this design the `< MIN_TOKENS` rule stops replanning but the stream keeps
pulling, so the cursor can reach the plan's end while holding. A test must assert the context's 4
frames are distinct (non-zero finite-difference velocity) across a leg boundary and into the hold
state.

### Half 2 — library: half the keyframes, double the legs

Combinations keep their current duration (2.4–7.6 s, unchanged — a longer combination is a longer
uncancellable commitment and that is a separate game-feel question, `docs/ASSUMPTIONS.md` §A23) but
carry roughly half the targets. Legs go from a median of 1.2 s to **2.4–3.8 s**.

Thinning happens inside `studio/segment.py::keyframe_indices`, because that is the only place the
provenance survives: it knows which picks came from `reach` (punches) and which from `level`/`shift`
(fill). A `CombinationRecord` stores neither, so thinning after the fact is impossible.

The rule, per the owner's "keep first+last punch" decision:

1. **Always keep** the first keyframe (index 0) and the final keyframe of the run.
2. **Always keep** the first and last `reach` (punch) keyframe.
3. **Drop** fill keyframes and interior punches as needed to enforce a minimum spacing of
   `MIN_TARGET_GAP_FRAMES` between surviving targets.

`MIN_TARGET_GAP_FRAMES = 2 * MIN_KEYFRAME_GAP_FRAMES` = **48 frames = 1.6 s**, a new constant. It is
deliberately derived from the existing spacing floor rather than chosen: `MIN_KEYFRAME_GAP_FRAMES`
(24 frames) is the shortest leg MotionBricks can plan, and doubling it is the smallest change that
delivers the owner's "longer than double". The two constants are **not** the same quantity and must
not be merged: `MIN_KEYFRAME_GAP_FRAMES` still governs *detection* spacing (which turning points are
found at all, where the measured 39/48 punch-capture rate comes from), while `MIN_TARGET_GAP_FRAMES`
governs *selection* spacing (which of those become hard targets). Thinning must run after detection,
so the punch-capture measurement is not invalidated.

#### A leg is no longer a plan, and three bounds must stop saying it is

This is the change with the widest blast radius, and it is easy to miss. Until now
`leg_tokens ≤ MAX_TOKENS` held **because a leg was exactly one plan**. Half 1 breaks that identity: a
long leg is an untargeted phase *plus* a landing plan. Three places still enforce the old identity
and would silently defeat the whole design if left alone:

- `segment.leg_tokens` **raises** when a gap exceeds `MAX_LEG_FRAMES` (`segment.py:312`);
- the same function **clamps** `chosen` to `min(MAX_TOKENS, …)` (`segment.py:319`) — this one is
  worse than the raise, because it would quietly truncate every merged leg back to 16 tokens and the
  library would look rebuilt while being unchanged;
- `combination_record._validate` **rejects** `leg_tokens > MAX_TOKENS`
  (`combination_record.py:179`).

All three move to a new bound, `MAX_TARGET_LEG_TOKENS = 24` (`MAX_TARGET_LEG_FRAMES = 96` frames =
3.2 s), which is what now caps a *leg*. `MAX_TOKENS` keeps its meaning untouched — it is the cap on a
single **plan**, enforced at runtime in Half 1, and nothing else.

`densify` therefore keeps its job but changes its threshold, from `MAX_LEG_FRAMES` (64) to
`MAX_TARGET_LEG_FRAMES` (96). It is not removed: without a cap, measured legs reach 36 tokens (4.8 s)
and combination duration runs past what the no-cancellation rule was sized for.

`COMBINATION_MIN_KEYFRAMES` / `COMBINATION_MAX_KEYFRAMES` move from **3–6 to 2–3**, i.e. **1–2 legs**.
Three rather than four, derived from duration rather than taste: at up to 3.2 s per leg, 2 legs is
6.4 s and 3 legs would be 9.6 s — past the 7.6 s the library already ships and past what
`COMBINATION_MAX_KEYFRAMES`' own docstring says the no-cancellation rule can survive
(`docs/ASSUMPTIONS.md` §A23).

The v0.2 library is rebuilt from the corpus in `motions/`; `MAX_RECORDED_TRAVEL_M`'s exclusion of 6
long-travel combinations is re-derived, not assumed to hold, because leg boundaries move.

#### Measured outcome of the thinning rule

Simulated against the shipped 130-combination library (spacing rule only, no punch constraint, so a
lower bound on what survives):

Predicted by simulation before implementing, and **measured after**. The simulation thinned within
each existing combination without re-densifying, which is why it over-stated the leg length and the
share over `MAX_TOKENS`:

| | before | predicted | **measured after** |
|---|---|---|---|
| combinations | 130 | — | **174** |
| legs | 623 | 313 | **310** |
| median leg | 9 tokens (1.20 s) | 17 tokens (2.27 s) | **15 tokens (2.00 s)** |
| leg range | 6–16 tokens | 12–36 | **6–24 tokens** |
| legs over `MAX_TOKENS` | 0 | 55 % | **39 %** |
| combination duration | 2.40–7.60 s | unchanged | **0.93–6.00 s**, median 3.87 s |

The median leg is **1.67× longer**. The 39 % figure is the number to keep in view: the
untargeted-then-land path is the **majority-adjacent case**, not an edge case, so it is tested as a
first-class path rather than as an exception.

---

## Costs, accepted explicitly

**`DRIFT_GAIN = 0.803` must be re-measured. This is a required task, not a follow-up.** It was
measured with `force=True`, one clean plan per leg
(`tools/measure_drift_gain.py:175`: "this measures the plan, not the replan schedule") — a schedule
this design does not use. `runtime/warp.py` divides the residual by it, so if effective coverage
under the pinned-keyframe schedule differs, every fighter lands off-ghost by the difference. The
re-measurement must drive the real `ReferenceStream` path. The constant's docstring already says
"Re-measure after any submodule bump"; this is a stronger trigger.

**Long legs contain model-improvised motion.** For a 3.8 s leg, roughly the first 1.7 s is
untargeted `walk_boxing` shaped only by `target_position`, and only the final 2.13 s is the
in-between to the recorded pose. This is the direct consequence of the 2.13 s cap and is the trade
the owner accepted in choosing long legs. It is consistent with MotionBricks' design intent — sparse
keyframes, model in-fills — but it is a real reduction in how much of the corpus reaches the ring.

**Interior punches become improvised rather than recorded**, by the "keep first+last punch" rule.

**Aim staleness is bounded by the replan cadence, not the leg.** Because replanning continues every
0.5 s until the final 0.8 s, the opponent bearing and leg target stay fresh — this is the property
that made shrink-horizon preferable to one-plan-per-leg, where aim could be 2.13 s stale.

---

## Testing

New tests:

1. **The keyframe is pinned.** Across every leg length (6–16 tokens) and every tick alignment,
   the requested `horizon_tokens` at successive replans must decrease monotonically and the implied
   landing tick must stay constant within ±2 frames — the direct inverse of the receding-horizon
   table above.
2. **Replanning stops below `MIN_TOKENS`.** No `generate()` call is made in the final
   `MIN_TOKENS * NUM_FRAMES_PER_TOKEN` frames of a leg.
3. **No pose target above `MAX_TOKENS`.** A leg longer than 16 tokens issues `pose=None` until the
   hole is reachable, then issues the pose for the remainder.
4. **Context integrity** (above): the 4 context frames are distinct across a leg boundary and in the
   hold state.
5. **Hold is unchanged**: past `end_tick`, the final leg's pose is re-issued at `MIN_TOKENS`.

Existing tests that must keep passing unchanged: all three in
`tests/test_reference_forced_length.py` (nothing is forced; the ambient `dt` is invariant), and the
`CombinationRunner` boundary arithmetic in `tests/test_sequence.py`.

Tests that must change: any asserting `intent_for` returns `horizon_tokens == leg.horizon_tokens`,
and the `COMBINATION_MIN_KEYFRAMES`/`MAX_KEYFRAMES` bounds in the combination-record tests.

---

## Spec and doc changes

- `spec/intent.md` → **3.2**: replace §"Forced plan lengths are back, and that has a cost" with the
  pinned-keyframe rule. The "consumed exactly" contract is withdrawn: it was never implemented and
  this design achieves the same guarantee (a leg lasts its recorded duration) by a different
  mechanism that needs no forced replan.
- `spec/combination.md` → keyframe-count bounds, the thinning rule, the relaxed leg-length cap.
- `spec/constants.py` → `COMBINATION_MIN_KEYFRAMES` 3→2, `COMBINATION_MAX_KEYFRAMES` 6→3,
  new `MIN_TARGET_GAP_FRAMES` (48), new `MAX_TARGET_LEG_FRAMES` (96) / `MAX_TARGET_LEG_TOKENS` (24),
  `MAX_LEG_FRAMES` docstring rewritten to say it no longer bounds a leg, `DRIFT_GAIN` re-measured
  with a new provenance note.

## Implementation order

Half 1 is independently shippable and verifiable, and Half 2 depends on it (a thinned library
produces legs longer than 16 tokens, which only Half 1 can run). The plan must therefore sequence:

1. **Half 1** — runtime pinning, the `GeneratorIntent` flag, the five new tests. The existing library
   is unchanged and every leg stays inside 16 tokens, so the `> MAX_TOKENS` branch is exercised by
   synthetic tests only.
2. **`DRIFT_GAIN` re-measurement** through the `ReferenceStream` path, on the unchanged library.
   Doing this between the halves isolates the schedule change from the library change; measuring
   after both would confound them.
3. **Half 2** — segmenter thinning, constants, library rebuild, re-derivation of
   `MAX_RECORDED_TRAVEL_M`'s exclusions.
- `CLAUDE.md` → the "Plan length" and "Combination" rows of the canonical-rates table, and the
  `sequence.py` / `reference.py` layout lines.


---

## Corrections made during implementation

Three things this design got wrong, each found by running it rather than by review. They are recorded
here because the reasoning that produced them is the reasoning most likely to produce them again.

**1. `densify` must run *after* thinning, not before.** The design left densification where it was,
in the detection step. But thinning *removes* keyframes and so re-opens gaps a prior densify had
already closed — measured, a 108-frame gap against a 96-frame maximum, which `leg_tokens` then
refused outright. Densification last is what makes the cap a guarantee instead of an aspiration. It
also has to use `MIN_TARGET_GAP_FRAMES` for its own spacing, or the frame it inserts undoes the
thinning it was inserted into.

**2. `combination_runs` had to stop dropping its remainder.** It filled greedily and discarded a
trailing group below the minimum. That was tolerable at `max_len` 6 and not at 3, where remainders
are frequent: it discarded recorded motion outright, and it **broke mirror balance**. A take and its
mirror can differ by one detected target, because mirroring a real capture is not bit-symmetric and a
prominence near threshold falls either side of it; under greedy filling that one target became a
whole missing combination (13 targets → 4 runs, but 14 → 5), so a move existed in one stance and not
the other. Even distribution into `ceil(n / max_len)` runs fixes both — 7 targets become 3+2+2 — and
the rebuilt library is balanced 87/87.

**3. The punch-capture regression test was measuring the wrong thing.** It ran against the final
keyframes, where thinning deliberately drops interior punches, so thinning appeared to break it
(76.9 % → 53.8 %). The 39/48 measurement belongs to *detection* — the segmenter's ability to find a
reversal rather than a mid-swing frame — so the test was repointed there, where it still reads
30/39 = 76.9 % against an unchanged 0.70 floor. What thinning itself guarantees, that the first and
last punch survive, became a separate test.

**Also worth recording:** `MAX_RECORDED_TRAVEL_M` now excludes nothing. Shorter combinations each
carry far less of their own recorded travel — the library tops out at 0.78 m against a 1.2 m
threshold — so the constant is retained as a guard rather than as an active filter, and its
docstring records the re-derivation.
