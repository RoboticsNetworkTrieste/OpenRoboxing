# combination.md — the selectable motion combination

Version **0.2** · created 2026-08-27 · design `docs/superpowers/specs/2026-08-27-motion-combinations-design.md`
· **rebuilt on sparse targets 2026-09-03**, design
`docs/superpowers/specs/2026-09-03-pinned-keyframes-and-merged-legs-design.md`

A combination is one selectable move: an ordered run of 2–3 key poses with the timing they were
recorded at. It replaces the single `pose_record.md` key pose as the unit a player selects — what a
loadout slot used to hold, before `spec/intent.md` 3.0's `D6` retired the loadout in favour of the
whole shared library, paged.

---

## Fields

| Field | Type | Notes |
|---|---|---|
| `schema_version` | str | `"0.1"` |
| `name` | str | unique within a library, kebab-case |
| `library_version` | str | the library release this belongs to |
| `source` | object | `take`, `start_frame`, `end_frame`, `mirrored` |
| `keyframes` | list | 2–3 entries, in order; see below |
| `duration_ticks` | int | total length at `TICK_HZ`, derived from `leg_tokens` |
| `recorded_displacement` | `[dx, dy]` | metres, keyframe 0 → last, in the take's own frame |
| `recorded_heading_delta` | float | radians, keyframe 0 → last |
| `telegraph_ms` | float \| null | **measured**, never authored |
| `tracking_error_rad` | float \| null | **measured** in a runtime trial |
| `admission` | str | `"draft"` \| `"admitted"` \| `"rejected"` |

### A keyframe

| Field | Type | Notes |
|---|---|---|
| `joint_angles` | map name → float | **all 29**, radians, MuJoCo joint names — identical to `pose_record.md` |
| `leg_tokens` | int \| null | length of the leg *ending* here, in tokens, 6–16. `null` on keyframe 0 only. |
| `root_offset` | `[dx, dy]` | metres, **relative to keyframe 0** |
| `heading_offset` | float | radians, **relative to keyframe 0** |

`root_offset` and `heading_offset` are relative to keyframe 0 because a combination **starts in
place**: the fighter does not travel to a start, so keyframe 0 is wherever it already stands. Both
are `[0, 0]` / `0.0` on keyframe 0 by construction.

`leg_tokens` is bounded by `MIN_TOKENS`..`MAX_TARGET_LEG_TOKENS` (6..24). **Not `MAX_TOKENS`** —
that bounds a *plan*, and since `spec/intent.md` 3.2 a leg is no longer one plan: a leg longer than
`MAX_TOKENS` runs an untargeted phase and then a landing in-between (`runtime/sequence.py`). A
recorded gap longer than `MAX_TARGET_LEG_FRAMES` is densified — a keyframe is added at the strongest
turning point inside it (or its midpoint, if the gap has none) — rather than held, so every leg stays
reachable. Corrected 2026-08-28: this used to add a keyframe at the gap's *busiest* frame, which is
the same mid-swing bug `studio/segment.py`'s module docstring documents for the top-level selection.

## Sparse targets — which recorded poses become keyframes

Added 0.2 (owner, 2026-09-03). Segmentation **detects** turning points at `MIN_KEYFRAME_GAP_FRAMES`
spacing, and the measured 39/48 punch-capture rate belongs to that step. A second step then
**selects** which of them become keyframes a plan is aimed at, at the wider `MIN_TARGET_GAP_FRAMES`
(48 frames, 1.6 s), so that each leg carries roughly twice the motion and MotionBricks in-fills
between the survivors. What survives, in priority order:

1. the first and last keyframe of the run;
2. the **first and last punch** — a combination's signature opening and closing strike stay recorded;
   interior punches become model-improvised;
3. anything else still `MIN_TARGET_GAP_FRAMES` clear of what is already kept, punches first.

Measured over the rebuilt library: 174 combinations, median leg **15 tokens (2.00 s)** against 9
(1.20 s) before, maximum 24 (3.20 s), and **39 % of legs longer than one plan**. Combination duration
is 0.93–6.00 s, median 3.87 s.

## Why the measured fields are nullable

Identical to `pose_record.md`: they are outputs of measurement, not inputs of authoring. A record
claiming `"admitted"` with either field null is **invalid**.

## Relationship to pose_record.md

A keyframe's `joint_angles` is exactly a `PoseRecord`'s 29 angles, so a combination reuses pose
validation rather than restating it. A combination is not a superset of a pose record: it carries no
`adjustment_envelope` and no `horizon_tokens`, because length is per-leg and recorded, not authored.
