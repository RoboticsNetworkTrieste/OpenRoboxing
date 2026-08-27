# Motion combinations — design

**Date** 2026-08-27 · **Status** approved by the project owner, not yet implemented ·
**Origin** the owner added a mocap corpus under `motions/` and asked for it to replace the six
selectable poses

Replaces the single authored key pose as the unit of selection. When implemented, `spec/intent.md`
goes to **3.0** (the walk approach is removed), and a new `spec/combination.md` 0.1 is added.
`spec/pose_record.md` is unchanged — a combination *contains* pose records.

---

## Why

The move set today is six single key poses ([`poses/loadouts/orthodox.json`](../../../src/openroboxing/poses/loadouts/orthodox.json)).
MotionBricks in-betweens the fighter to one pose and holds it. The owner's objective is
**"to take many keyframes in order to compose a complicated motion"** — a selected move should be a
timed *sequence* of poses, and the sequence should last as long as the recording it came from.

`motions/` holds 19 mocap takes × 2 (base + `_M` mirror) = 38 CSVs.

## What the corpus is — measured 2026-08-27

| Property | Value | Evidence |
|---|---|---|
| Columns | `Frame`, `root_translate{X,Y,Z}`, `root_rotate{X,Y,Z}`, 29 × `<joint>_dof` | CSV header |
| Joint names | exactly the 29 G1 names with a `_dof` suffix | compared against `G1.mujoco_joint_names` |
| Units | joints and root rotation in **degrees**, root translation in **cm**, Z-up | root Z spans 50–107 cm = pelvis height |
| Rate | **30 fps** = `GENERATOR_HZ` | see below |
| Take length | 411–1485 frames = **13.7–49.5 s** | measured |

**The corpus shares MotionBricks' training-set naming.** Upstream's `walk_boxing` clip is
`clip_id: "shadow_boxing_R_003__A360_M"` (`demo/clips.py:153`), resolved by matching the training
set's `meta['original_path']` (`clips.py:25`), and that name is a file in `motions/`.

**Corrected 2026-08-27 — same name, different data.** Upstream's preprocessed clip cache
(`motionbricks/out/G1-clip.ckpt`) stores 10 frames of `mujoco_qpos` for that clip. Compared against
our take, the **elbows have opposite signs** (ours +48.9°, upstream −34.5°) and the knees differ by
2.7×; the best 10-frame alignment anywhere in the 1254-frame take still has a worst-joint error of
**22.9°**. Upstream's stored clip is a different retarget of the same capture, not the file itself.

Two consequences, replacing what this section originally claimed:

1. **"in-distribution" is plausible but unverified.** The naming says the captures come from the same
   corpus; the retarget differs, so how the generator responds is a question for the Task 12 spike,
   not something the file names settle.
2. **There is no upstream golden fixture.** The conversion is instead validated against the *robot*:
   every joint value in all 38 takes falls inside the G1's own `jnt_range` from
   `paths.G1_29DOF_SIM_XML` — measured, zero violations. That is a strong check, because a flipped
   sign convention would drive knees or elbows out of range somewhere in ~30 000 frames, and none do.
   It is what establishes that the corpus is already in the robot's joint convention.

The Euler convention was pinned separately and by its own measurement — see
`tools/pin_euler_order.py` and `EULER_ORDER` in `studio/motion_import.py`. It affects the root
quaternion only, never the joint angles.

### The takes are not moves

Nine of the 19 are shadow boxing at 1000–1500 frames, containing many punches each. The action peak
sits at 20–72 % through a take and the last 100 frames are near-still in almost every one (motion at
0.2–5 % of peak). One frame per file would harvest a standing pose from all 19.

## Decisions

Six decisions were taken with the owner, in order. Each rejected alternative is recorded because the
rejection is the reasoning.

### D1 — a selectable move is a *combination*: 3–6 key poses with recorded timing

Rejected: one take = one move (49.5 s uninterruptible, and commits **cannot be cancelled** — that
rule is the game); one take truncated to N seconds (arbitrary cut, discards most of each take).

Segmenting on salient-joint speed peaks at least 0.8 s apart yields **300 key poses across the 19
takes**, grouped into **60 combinations**, or **120 with mirrors**.

### D2 — the generator's timing quantum bounds everything

A plan is `MIN_TOKENS=6` to `MAX_TOKENS=16` tokens of `NUM_FRAMES_PER_TOKEN=4` frames at 30 Hz, so a
pose-to-pose leg is **0.8 s minimum, 2.13 s maximum, quantised to 0.133 s**. Keyframes are therefore
selected no closer than 0.8 s. Legs longer than 2.13 s (present in 12 of 19 takes) become the same
pose re-armed, which is the existing hold behaviour.

"Motions last the same time" is met to **within one token over the whole combination**, by diffusing
the rounding residual across legs rather than rounding each independently.

### D3 — a combination starts in place; the ghost anchors its end

No walk approach. The player picks the ghost's **position**; the combination runs from wherever the
fighter stands and its final keyframe lands on the ghost.

### D4 — intermediate keyframes: preserve the footwork, ramp the difference

Rejected: proportional rescaling of the recorded path. Measured, the scale factor needed to reach a
ghost 2 m away is 0.6–2.1× for the jog takes but **30–141×** for shadow boxing, whose combinations
travel 1–7 cm; seven combinations travel under 5 cm, where the factor is undefined. Proportional
scaling multiplies every intermediate weight-shift by the same factor, turning a 2 cm shift into a
2.8 m lurch. Rejected also: proportional with a clamp — a clamped combination silently misses the
ghost, which invariant 5 forbids.

Instead the recorded offsets are kept at **true size** and the leftover travel is added as an even
drift. Every combination stays usable at fighting range, the degenerate case disappears, and the jog
takes still land correctly because their recorded path already points where it is going.

### D5 — the ghost's heading is derived, never chosen, and nothing aims at the target

`h_ghost = h₀ + recorded_heading_delta`. A combination that ends 90° off its start puts the ghost
90° off the fighter's current heading, wherever the ghost sits. Per keyframe,
`heading_i = h₀ + heading_offset_i`.

This is load-bearing: the jog combinations turn up to **267°**, and a ghost that always faced the
target would discard the turn that *is* the motion. It also lands on the trap `CLAUDE.md` names —
`facing_angle` is where the fighter looks, `movement_angle` is where it travels, and here they
genuinely differ, per leg.

### D6 — there is no loadout; every combination is selectable, nine at a time

**Revised 2026-08-27 by the owner**, replacing "six slots chosen from ~120".

A fighter carries the **whole library**. The client shows **nine moves at a time** in a grid, with a
forward and a back button paging through all ~120 — fourteen pages at nine per page. Keys 1–9 select
within the current page.

This deletes `Loadout` rather than repurposing it. What goes with it:

- the six-slot format constraint, and the deckbuilding choice of which six to bring;
- the per-seat `loadout` and `horizons` maps in the `welcome` message (`server/protocol.py:93`) —
  the library is now shared and identical for both fighters, so it is sent once, or served as a
  static asset;
- `server/agent.py`'s slot learning (`:172-183`), which infers a stance slot and strike slots from
  the loadout it was dealt, and now has to choose from the library instead.

Unchanged: **no cancellation**, `MAX_OUTSTANDING_COMMITS`, `OPENING_STANCE_CONTEXT`.

The consequence worth stating plainly: both fighters now have identical and complete access to every
move, so what separates them is what they choose in the moment and where they place it, not what they
brought. The format's asymmetry moves from preparation to play.

---

## Architecture

```
motions/*.csv
  → studio/motion_import.py   CSV → (N,36) MuJoCo qpos
  → studio/segment.py         qpos → keyframe indices → combinations
  → CombinationRecord         poses/v0.2/combinations/*.json   ← the seam
  → studio/rehearsal.py       measure telegraph + tracking → admission
  → runtime/warp.py           record + anchor + ghost → per-leg intents   (pure)
  → runtime/sequence.py       per-leg intents + tick → GeneratorIntent
  → runtime/intents.py        delegates; commit queue unchanged
```

`CombinationRecord` is the boundary. Everything left of it is offline and needs no GPU; everything
right of it needs no CSV.

### `studio/motion_import.py`

`load_take(path) -> (N, 36)` in MuJoCo qpos convention: degrees→radians, cm→m, Euler→quaternion
`wxyz`, joints mapped **by name** (strip `_dof`) with the permutation asserted invertible against
`G1.mujoco_joint_names` (invariant 4). The CSV order happens to match; the permutation is derived
anyway.

**The Euler order is unknown and is not guessed.** Candidate orders are tested against the
`walk_boxing` clip upstream loads from `shadow_boxing_R_003__A360_M`, and the winner is committed as
a named constant citing that measurement. If no candidate matches, that is a stop-and-ask, not a
default.

### `studio/segment.py`

Salient-joint speed reuses `harvest.py`'s `SALIENT_JOINT_SUBSTRINGS` (shoulder, elbow, wrist) — two
guards differ at the hands, and scoring on everything ranks a long stride above a thrown punch.

Keyframes are local maxima at least `MIN_TOKENS * NUM_FRAMES_PER_TOKEN` = 24 frames apart. The
prototype's "20 % of peak" threshold is an invented number and does not survive: the threshold is
derived from the take's own speed distribution, as `telegraph.py` derives its divergence threshold
from the baseline's own variation, with the sigma multiple as the one stated free parameter.

Combinations are runs of 3–6 consecutive keyframes, split at the longest quiet intervals.

### `spec/combination.md` 0.1

```json
{
  "schema_version": "0.1",
  "name": "shadow-boxing-a361-01",
  "library_version": "v0.2",
  "source": {"take": "shadow_boxing_R_001__A361", "start_frame": 504,
             "end_frame": 690, "mirrored": false},
  "keyframes": [
    {"joint_angles": {"...29, radians, MuJoCo names...": 0.0},
     "leg_tokens": 7,
     "root_offset": [0.02, -0.01],
     "heading_offset": 0.34}
  ],
  "duration_ticks": 214,
  "recorded_displacement": [0.05, 0.02],
  "recorded_heading_delta": 4.71,
  "telegraph_ms": null,
  "tracking_error_rad": null,
  "admission": "draft"
}
```

`root_offset` (metres) and `heading_offset` (radians) are **relative to keyframe 0**, because the
motion starts in place (D3). `leg_tokens` is the duration of the leg *ending* at that keyframe;
keyframe 0 has none.

Each keyframe's `joint_angles` is exactly a `PoseRecord`'s 29, so pose validation and the adjustment
envelope are reused rather than reimplemented. `telegraph_ms` and `tracking_error_rad` keep the
pose-record rule: a record claiming `"admitted"` with either null is invalid.

### `runtime/warp.py` — pure, and the core of the feature

Given a record, the fighter's `(p₀, h₀)` at commit, and the ghost position:

```
r_i       = R(h₀) · root_offset_i                     rotate the recording into the world
Δ         = (p_ghost − p₀) − r_{K−1}                  leftover travel
pos_i     = p₀ + r_i + (t_i / t_{K−1}) · Δ            ramp on ELAPSED TIME, not index
heading_i = h₀ + heading_offset_i
movement_angle_i = atan2(pos_i − pos_{i−1})           direction of travel
facing_angle_i   = heading_i                          where the fighter looks
```

The ramp is on cumulative ticks, not keyframe index: legs differ in length, and an index ramp would
make the drift speed jump between them.

Feasibility: `|Δ| / duration_s` must not exceed a measured speed ceiling (seeded from
`APPROACH_SPEED_M_S` = 0.83 m/s, re-measured for the warped case). Exceeded → `IntentError` at
commit time carrying the number. **No clamping** (invariant 5).

Output is `[(pose, target_position, target_heading, horizon_tokens, movement_angle, facing_angle)]`.
No MuJoCo, no torch, no upstream import — unit-testable on its own.

### `runtime/sequence.py`

`CombinationRunner`, built from the warp output at commit time, answers `intent_for(tick)` with the
leg live at that tick. Legs advance on **recorded time**, which is what makes durations hold. After
the final keyframe the last intent stays armed — the existing hold behaviour, unchanged.

`intents.py` delegates to it. A `Commit` carries a `CombinationRecord` plus its anchor instead of a
single `PoseRecord`. The commit queue, the horizon floor and the no-cancellation rule are untouched.

### The `horizon_tokens` reversal — a known regression risk

`spec/intent.md` 2.0 deliberately moved to `horizon_tokens=None` ("the model picks its own length"),
and `runtime/reference.py` documents three defects that came from the forced-plan machinery it
deleted. Timed legs reintroduce plan-length control, so those three become explicit regression tests:

1. an 8-token pose losing its final frame — the authored pose — on 20 % of tick alignments;
2. that lost frame then playing into the *next* commit;
3. the end-of-move replan bypassing the ambient cadence.

### `spec/intent.md` 3.0 — removals

Deleted, not deprecated: `TRAVEL_CONTEXT` and the walk approach; the arrival-geometry callable;
`rehearse_approach`; `Placement.heading` as a player-set field; `POSE_DWELL_TICKS` and
`MAX_DWELL_TICKS` (a move ends when its recorded duration ends); the ghost-heading control in
`client/app.js`.

`APPROACH_SPEED_M_S` survives as the feasibility ceiling, not as a control.

`SPEC_VERSION` in `intents.py` moves to `"3.0"` in the same change — a test pairs them.

### Admission

- `telegraph_ms` — from the take's own frames leading into keyframe 0; `telegraph.py` already takes
  an `(N, 36)` stream, which is what `motion_import` produces.
- `tracking_error_rad` — rehearse the **warped** sequence under physics, per keyframe, worst case.
  A new `rehearse_combination` alongside the existing `rehearse_commit`.
- duration error — within one token over the combination.

Admitted only if every keyframe is reached within tolerance **and** duration holds. Attrition is
expected and is the gate working.

### What happens to the v0.1 pose library

`poses/v0.1/` stays as the fixtures the pose-level tests and the golden library already depend on;
nothing in the runtime reads them once combinations are selectable. `poses/loadouts/orthodox.json`
and the `Loadout` class go with D6 — there is no loadout to migrate, because the library *is* the
move set.

### Client and protocol

The client holds the whole library and shows **nine moves per page** with forward/back paging
(D6). Each entry needs its name, duration in seconds, the final keyframe's pose (for the ghost) and
`recorded_heading_delta` — sent once at `welcome` rather than per seat, since both fighters see the
same library.

The ghost renders at the derived heading and the player drags position only. Placements whose drift
speed exceeds the ceiling render as rejected, because commit will reject them — and because a move's
reach depends on its own recorded duration, that rejection boundary **differs per move** and moves as
the player pages.

---

## Testing

| Test | What it protects |
|---|---|
| CSV → qpos → CSV round trip | the conversion (`CLAUDE.md`: most bugs here are convention bugs) |
| **joint-limit conformance**: every value of all 38 takes inside the G1's `jnt_range` | that the corpus is in the robot's joint convention — replaces the upstream golden, which the data disproved |
| `pin_euler_order` recovers exactly one convention | that the Euler order is measured, not assumed |
| `remap(unremap(x)) == x` on the joint permutation | invariant 4 |
| warp: zero recorded travel, 180° heading delta, ghost at the fighter's own position | D4's degenerate cases |
| warp: legs of unequal length ramp at constant speed | the time-vs-index ramp |
| duration error ≤ 1 token over all 120 combinations | D2 |
| the three `reference.py` defects | the `horizon_tokens` reversal |
| `test_generator_pose_override.py` against a pristine agent | invariant 3, unchanged |

## Phasing

1. **Ingest, segment, library** — offline, verifiable alone: 120 combinations to inspect and render.
2. **Warp, sequence, `intent.md` 3.0** — the runtime.
3. **Admission measurement.**
4. **Client and protocol.**

**Checkpoint after 1, and it is not a formality.** Nothing yet demonstrates that MotionBricks will
hit a forced 0.8 s leg to an authored pose *while its root is being dragged along a drift*. Poses are
reached to 2–3° when they are the target of an unhurried plan; a short forced leg with a moving root
is a different ask, and it is the assumption the whole feature rests on. Measure it on three
combinations — one shadow boxing, one dodge, one jog — before building phases 3 and 4 on top of it.
If it fails, D2's leg length and D4's drift are the parameters to revisit, not the design.
