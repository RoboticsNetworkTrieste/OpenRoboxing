# pose_record.md — the authored key pose

Version **0.1** · created 2026-08-07 · task `M2-T2`

A pose record is one authored key pose: what the fighter's body should look like at the moment a
committed move lands. The library of these records *is* the game's move set.

---

## Which joint space — decided

**A pose record stores the 29 robot joint angles, in MuJoCo order, keyed by joint name.**

The generator's pose space is the 34-joint `g1skel34` skeleton, not the robot
(`spec/upstream_patches.md` P0). Storing 29 is nonetheless lossless, because the skeleton is a strict
superset: all 29 robot joints appear in it, and the 5 extras are each derivable rather than authored.

| Extra skeleton joint | Where its value comes from |
|---|---|
| `pelvis` | the root — supplied by *placement*, not by the pose |
| `left_toe_base`, `right_toe_base` | dummy joints the skeleton carries for structure |
| `left_hand_roll_joint`, `right_hand_roll_joint` | not actuated on the 29-DOF G1 |

Storing 29 is chosen deliberately over storing 34:

- it is the space an author, a robot and a tracking-error measurement all think in;
- it is the space the policy is scored against, so admission thresholds mean something;
- the 34-joint form is a *derived* artefact, and deriving it on load keeps one source of truth.

**The 29 → 34 conversion belongs to the Studio, not the runtime.** Records are stored as authored;
`studio/` expands them to the generator's skeleton on the way in.

Angles are **radians**, absolute (not relative to the default standing pose), keyed by joint name so
that no index ordering can silently change meaning (`CLAUDE.md` invariant 4).

---

## Fields

| Field | Type | Notes |
|---|---|---|
| `schema_version` | str | `"0.1"`. Bumped on any breaking change. |
| `name` | str | unique within a library, kebab-case, e.g. `"jab-left"` |
| `joint_angles` | map name → float | **all 29**, radians, MuJoCo joint names |
| `horizon_tokens` | int | commit length, 6–16 (`spec/rates.md`) |
| `source` | object \| null | provenance: `clip`, `start_frame`, `end_frame` |
| `adjustment_envelope` | object \| null | bounded live adjustment; see below |
| `telegraph_ms` | float \| null | **measured**, never authored (`M2-T3`) |
| `tracking_error_rad` | float \| null | **measured** in a runtime trial (`M2-T5`) |
| `admission` | str | `"draft"` \| `"admitted"` \| `"rejected"` |
| `library_version` | str | the library release this belongs to |

### Why the measured fields are nullable

`telegraph_ms` and `tracking_error_rad` are outputs of measurement, not inputs of authoring. A record
is created with both null and `admission: "draft"`; measurement fills them; only then can admission
become `"admitted"`. A record claiming `"admitted"` with either field null is **invalid** — that is
the rule that stops an unmeasured pose reaching a match.

### Adjustment envelope

Per `spec/intent.md`, a player picks a preset pose and adjusts it within bounds. The envelope is
per-joint symmetric limits in radians:

```json
"adjustment_envelope": {"left_shoulder_pitch_joint": 0.30, "left_elbow_joint": 0.20}
```

Joints absent from the map are not adjustable. Admission covers the envelope's **corners**, not
every interior point — see `spec/intent.md` for why that is the weak link and what would strengthen
it.

---

## Example

```json
{
  "schema_version": "0.1",
  "name": "guard-high",
  "joint_angles": {"left_shoulder_pitch_joint": 0.35, "...": 0.0},
  "horizon_tokens": 8,
  "source": {"clip": "walk_boxing", "start_frame": 25, "end_frame": 35},
  "adjustment_envelope": {"left_elbow_joint": 0.2},
  "telegraph_ms": null,
  "tracking_error_rad": null,
  "admission": "draft",
  "library_version": "v0.1"
}
```

## Validation

`studio/pose_record.py` validates on load and **raises with the offending field named** — never
coerces, never fills a default for a missing angle. Enforced:

1. every one of the 29 joints present, no extras, no NaN;
2. `horizon_tokens` within `[MIN_TOKENS, MAX_TOKENS]`;
3. `admission == "admitted"` requires both measured fields non-null;
4. envelope keys are real joints and bounds are positive;
5. angles within the robot's joint limits, read from the MJCF — **not** a hard-coded table.

## Changelog

- **0.1** (2026-08-07) — first version. 29-joint decision recorded, measured-field rule, envelope.


---

## Admission requires tracking error, not a telegraph window

**Decided 2026-08-07 (M2-T5).** An admitted pose must carry `tracking_error_rad`. It need **not**
carry `telegraph_ms`.

The original rule required both. Measuring the second turned out not to work: the M2-T3 proxy asks
when a move becomes distinguishable from a baseline, and once the baseline is a fighter who is
already shadow-boxing, that question has no stable answer — mirrored poses whose reachability differs
by 0.1° were given windows of 0 ms and 433 ms (`spec/upstream_notes.md`).

The deeper reason not to fix it: **a telegraph is a player's intuition, not a property of a
trajectory.** Whether a windup is readable depends on what the opponent is attending to, what else
the fighter might have thrown, and how long they have played. That is a playtest question
(`WORKPLAN` M4-T4), which is where the telegraph floor gets set from observed behaviour rather than
from a geometric proxy.

`telegraph_ms` stays in the schema and `studio/telegraph.py` stays as a tool: a rough window is worth
recording when it can be had, and is a useful sanity check on a wildly fast or slow move. It is just
not a gate.

What admission *does* mean, then:

| Field | Question | Where measured | Gates admission? |
|---|---|---|---|
| `generator_error_rad` | will MotionBricks produce this pose? | `rehearsal.measure_reachability`, at the plan endpoint | **yes** |
| `tracking_error_rad` | can the robot execute it under physics? | a runtime trial | not yet — see below |
| `telegraph_ms` | how readable is the windup? | `studio/telegraph.py`, when it can | no |

**Schema 0.2** splits what 0.1 called `tracking_error_rad` in two. They were being conflated, and
they are different questions with different fixes: a pose the *generator* declines needs re-authoring,
while one the *robot* cannot execute needs a finetune. `build_library` was writing the generator's
number into the field the spec defined as the physics one.

Only `generator_error_rad` gates admission today, because it is the one measured per pose.
`tracking_error_rad` is measured per *run* rather than per pose (`tools/run_single.py --loadout`
reports 0.098 rad mean, 0.63 rad max over the whole v0.1 library), so promoting it to a gate needs a
per-pose trial that does not exist yet. Recorded when available; gate to follow.
