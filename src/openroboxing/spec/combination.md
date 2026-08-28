# combination.md — the selectable motion combination

Version **0.1** · created 2026-08-27 · design `docs/superpowers/specs/2026-08-27-motion-combinations-design.md`

A combination is one selectable move: an ordered run of 3–6 key poses with the timing they were
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
| `keyframes` | list | 3–6 entries, in order; see below |
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

`leg_tokens` is bounded by `MIN_TOKENS`..`MAX_TOKENS` because that is what MotionBricks can plan
(`spec/rates.md`). A recorded gap longer than `MAX_TOKENS` is densified — a keyframe is added at the
strongest turning point inside it (or its midpoint, if the gap has none) — rather than held, so every
leg is plannable. Corrected 2026-08-28: this used to add a keyframe at the gap's *busiest* frame,
which is the same mid-swing bug `studio/segment.py`'s module docstring documents for the top-level
selection.

## Why the measured fields are nullable

Identical to `pose_record.md`: they are outputs of measurement, not inputs of authoring. A record
claiming `"admitted"` with either field null is **invalid**.

## Relationship to pose_record.md

A keyframe's `joint_angles` is exactly a `PoseRecord`'s 29 angles, so a combination reuses pose
validation rather than restating it. A combination is not a superset of a pose record: it carries no
`adjustment_envelope` and no `horizon_tokens`, because length is per-leg and recorded, not authored.
