# Continuous pose targeting — design

**Date** 2026-08-13 · **Status** approved by the project owner, not yet implemented ·
**Closest workplan task** `M4-T4` (this came out of playing the game, not out of review)

Supersedes the two-phase commit introduced by `spec/intent.md` 1.1. When implemented,
`spec/intent.md` goes to **2.0** and `spec/pose_record.md` takes a version bump.

---

## Why

The project owner played the game and reported three things:

1. the fighter returns to a default position after a commit instead of holding the pose it was
   commanded into;
2. commits do not run as separate moves — they melt into one another;
3. some committed moves never happen.

Investigation (2026-08-13) confirmed all three and located them. The relevant findings:

| Finding | Evidence |
|---|---|
| The pose is the last frame of one plan, held for at most one generator frame | `runtime/reference.py:86` `MAX_HELD_STRIKE_FRAMES = 1` |
| At `end_tick` the intent becomes the `idle` clip and MotionBricks in-betweens *out of* the pose | `runtime/intents.py:579`, `HOLD_CONTEXT = "idle"` at `:188` |
| That departure fires immediately, not at the 0.5 s cadence, because the plan cursor is pinned at its last frame and the upstream gate's second conjunct is false | `full_agent.py:122-124` |
| Once idle, `_should_regenerate` returns `False` for idle→idle and the reference freezes on that plan's final frame permanently | `full_agent.py:644-651`; measured: joint speed decays to **1.1 °/s** and the posture stops changing |
| A 6-token pose phase is 40 ticks, **shorter than the 45-tick encoder lookahead**, so at the tick the pose is the reference, 9 of the 10 samples the policy tracks are already the following idle motion | `runtime/reference.py:69`, `spec/constants.py:36` |
| 8-token poses claim 31 of their plan's 32 frames on 20 % of tick alignments, losing the frame that *is* the authored pose | simulated over all alignments, 2026-08-13 |
| That lost frame is then played into the *next* commit's approach, contradicting the comment two lines above it | `runtime/reference.py:158-162` |

Measured on a real match record: red's jab arm error falls 15.9° → 5.3° exactly at `end_tick` and
then climbs back while the error against the default standing pose falls 17.8° → 11.9°. A
`hook-left` issued straight behind a `jab-left` moved the arm **−1.2°** — it ended further from its
own pose than it started.

## The decision

**A commit is one continuous intent — "be at this placement, in this pose" — armed on every replan
for the commit's whole life, with the plan length chosen by MotionBricks.**

There is no approach phase and no pose phase. One motion converges on the placement and the pose
together.

### Why this is possible

The two-phase split was never an upstream constraint. Established 2026-08-13:

- The P0 pose override is **not single-use**. `full_agent.generate_new_frames` calls
  `_override_target_joint_transforms` unconditionally (`full_agent.py:152-160`); the hook is pure and
  stateless (`:325-378`). `studio/rehearsal.rehearse()` already arms a pose on every replan.
- The forced pose plan **already carries the placement as well as the pose**
  (`runtime/intents.py:587-599` gates `target_position` on `placement`, not on `striking`). The split
  exists only because one plan does not travel far enough.
- With `horizon_tokens = None`, `narrow_allowed_tokens` passes the clip's own mask through
  (`runtime/generator.py:79-80`) and the model picks its own length by argmax
  (`root_backbone.py:204-206`). `walk_boxing` permits all of 6…16 tokens.

### Why it is better, measured

One plan covers **~54 % of the distance asked for**, at every horizon: 0.5 m → 0.22, 1.5 m → 0.81,
3.0 m → 1.63, 6.0 m → 3.21. More tokens do not travel further, they travel slower — 6 tokens covers
2.88 m of a 6 m ask in 0.80 s (3.6 m/s, untrackable), 16 tokens covers 3.21 m in 2.13 s (1.5 m/s).
Left to choose, the model picks 11 tokens at 0.5 m, 14 at 1.5 m, 15 at 3.0 m and 16 at 6.0 m, which
lands plan speed near the 0.83 m/s the robot actually sustains.

The authored pose still lands while travelling: **3.7–5.0° mean error across every distance and
horizon tested**, with no degradation at range.

Arming the pose throughout, against a placement 2.5 m away:

| | arrives | pose error at 3 s | after arrival |
|---|---|---|---|
| today, pose not armed during approach | 0.41 m | 18.2° mean / 71° max | never converges — 17–22° |
| **pose armed, model-chosen length** | **0.19 m** | **6.7° mean / 21° max** | **holds ~6° for 4+ s** |
| pose armed, horizon forced to 8 | 0.30 m | 8.5° mean / 38° max | plateaus 0.13 m short |

It arrives sooner, converges monotonically with no jitter or stall, and **holds the pose after
arrival with no extra mechanism**. Both of the owner's corrections are one change.

---

## The model

### A commit's life

| Field | Meaning | Set when |
|---|---|---|
| `issued_at` | the tick the player committed | at commit |
| `commit_at` | the tick it becomes current — `max(issued_at + COMMIT_HORIZON_TICKS, previous.end_tick)` | when it starts |
| | *(`COMMIT_HORIZON_TICKS` stays in the formula and stays **inert**: 30 < the 65-tick reference lead, so it has never bound anything — `docs/ASSUMPTIONS.md` §A24. This design does not change that and does not pretend to.)* | |
| `strike_at` | the first tick the pelvis is within `ARRIVAL_RADIUS_M` of the placement, **or** the timeout tick | when it arrives |
| `end_tick` | `strike_at + POSE_DWELL_TICKS` | when it arrives |
| `arrived` | `False` when `strike_at` came from the timeout | when it arrives |

`strike_at` keeps its contract — *the moment the punch was thrown* — because
`league/scoring.py:235` reads it to decide whether the fighters were in range when the blow landed.
Changing its meaning silently stops the scorer counting hits.

`POSE_DWELL_TICKS` is the interval during which the fighter stands in the pose before the next
queued commit becomes current. Without it the next commit starts at the instant of arrival and the
strike is cut short again — the melting, reintroduced. **It is measured, not chosen** (see below).

### The intent

For the current commit, on every replan:

```
GeneratorIntent(
    style=commit.context,
    target_position=commit.placement.position,
    target_heading=commit.placement.heading,
    facing_angle=commit.placement.heading,
    pose=commit.pose,
    horizon_tokens=None,      # the model chooses
)
```

`plan_key` is removed: it existed only to bind a forced plan to the commit that forced it.

### The hold

**When the queue is empty, the intent of the most recently completed commit stays armed** — same
placement, same pose, same heading. That is the entire implementation of "hold the pose". No idle
clip, no freeze branch, no repeated-frame counter.

Because the intent carries its own `target_heading`, a held fighter does **not** turn to track its
opponent — re-orienting is paid for by the next commit. Both decisions were taken by the project
owner on 2026-08-13.

At round start, before any commit has run, the ambient style intent is used as it is today.

### Replanning

One cadence, `REPLAN_DT = 0.5 s`, `force=False`, for the whole commit. The mode is the commit's
style throughout, so `_should_regenerate` never suppresses a replan the way it does for idle→idle.

---

## What this deletes

`CLAUDE.md` prefers deleting to disabling. All of the following go:

- the approach/pose phase distinction in `runtime/intents.py`;
- forced plans: `_committed_frame`, `_committed_plan_length` and its equality assertion,
  `_plan_key`, `_held_strike_frames`, `MAX_HELD_STRIKE_FRAMES` (`runtime/reference.py`);
- `HOLD_CONTEXT = "idle"` and the drained-queue idle branch (`runtime/intents.py:188`, `:577-579`);
- `duration_ticks` derived from `horizon_tokens` (`runtime/intents.py:282`);
- the rule in `spec/intent.md` that asking for a length the clip does not permit raises.

Four of the seven suspect items reported to the owner disappear with that code rather than being
fixed individually: the lost strike frame, the leaked frame, the cadence bypass and the idle freeze.
The remaining one that matters — commits rejected silently, because `QueuedPilot.act` writes
`self.last_error` (`server/host.py:161`) and nothing ever reads it — is **independent of this design
and is fixed separately.**

## What stays

`ARRIVAL_RADIUS_M` (0.40 m, measured), `MAX_OUTSTANDING_COMMITS` (5), no cancellation of anything,
and an approach timeout. The timeout still earns its place: a fighter wedged in a corner, walking
into its opponent or knocked down never satisfies arrival, and without a cap it holds the whole queue
forever. On timeout the commit completes with `arrived = False`, exactly as today.

`horizon_tokens` **stays in the pose record**. The runtime stops sending it to the generator, but it
remains the Studio's rehearsal parameter and the author's statement of how long the move is meant to
take. That is a real use, so it is not dead code; `spec/pose_record.md` documents the changed
meaning.

---

## Numbers to measure

`CLAUDE.md` standing rule 3: never invent a number. Three are needed, and none may be chosen.

1. **`POSE_DWELL_TICKS`** — run the armed approach for every library pose and, for each, take the
   interval from the arrival tick until the plan's joint error against the pose first reaches its
   asymptote (its minimum over the following 2 s). The dwell is the **upper end** of that
   distribution, so the slowest pose in the library still completes inside it. Record it in
   `spec/rates.md` with its derivation.
2. **The admission tolerance for `generator_error_rad`** — re-measure all ten `poses/v0.1` records
   under the new scheme and set the threshold from the measured distribution. Any pose that fails is
   re-authored or dropped, not waved through.

   Note the comparison this replaces, because the obvious one is wrong. The library's recorded 2–3°
   was measured on a single forced plan with **no placement** (`spec/upstream_notes.md:279-293`). The
   runtime always has a placement, and a single forced plan *with* one measures **3.7–5.0° mean**
   (2026-08-13). So the honest regression is roughly **4–5° → 6°**, not 2–3° → 6°.
3. **Telegraph windows** — `studio/telegraph.py` measures a window on generated motion, and this
   changes that motion. Re-run `tools/measure_telegraph.py` for every pose and rewrite the recorded
   values.

Until 1 and 2 exist the design cannot be implemented, so measuring them is the first implementation
step, not a follow-up.

---

## Testing

**Unaffected, and must stay green untouched:** everything in the observation path — `test_obs_parity`,
`test_policy_parity`, `test_bridge`, `test_conventions`, `test_golden_fixture`. This design changes
what the reference motion *is*, never how it is turned into observations. If any of these move, the
change has reached somewhere it should not have.

**Rewritten:** `tests/test_intents.py` encodes the two-phase rules extensively (commit spans,
`strike_at` semantics, back-to-back queueing) and `tests/test_fight.py` asserts commit spans.
27 tests were rewritten when this model last changed at 1.1; expect that order again.

**New tests, one per claim this design makes:**

- a drained queue holds the last committed pose and does **not** switch to the idle clip;
- a held fighter's heading does not track the opponent;
- a commit ends at `strike_at + POSE_DWELL_TICKS` and the next becomes current exactly then;
- an armed pose converges within the measured admission tolerance over a replanned approach;
- a commit that cannot arrive times out, completes with `arrived = False`, and releases the queue;
- `strike_at` is populated for every commit that threw, so `league/scoring.py` still counts hits;
- a full match still produces a record that replays and re-scores.

**Acceptance:** `tools/run_match.py` produces a record whose commits all have non-null `strike_at`
where they threw; `tools/replay_match.py --rescore` agrees with the live score; the 575 existing
tests pass except those deliberately rewritten.

---

## Risks

**Pose fidelity loosens**, from ~4–5° mean to ~6° (see the note under "Numbers to measure" 2 — the
2–3° figure quoted elsewhere in the repo was measured without a placement and is not the right
baseline). This is inherent: the fighter converges on the pose instead of landing on it. It is the
one thing that could make this design look worse in the ring than the current one, and it will not
be visible until it is played.

**The library may need re-admitting.** If several poses fail the re-measurement, the work is larger
than the runtime change.

**Test churn is substantial** and concentrated in one file.

### Fallback

The conservative variant: arm the pose throughout the approach exactly as designed here, but end the
commit with one forced plan consumed whole, recovering 2–3° fidelity and a crisp `strike_at`. It
keeps the exact-consumption machinery this design deletes, and with it the class of bug that caused
the melting. It is a small delta from this design rather than a rewrite, so choosing it later costs
little.

---

## Not in scope

- The 1.3 s input latency (`docs/ASSUMPTIONS.md` §A24). 0.90 s of it is structural in the policy's
  lookahead and shortening the remaining 0.40 s is a separate, measurable trade.
- Queue depth. The sweep recommends 2 over 5; that is the owner's feel call and `M4-T4`'s job.
- `TARGET_COMMIT_RATE`, which §A22 already records as stale and which this design changes again by
  changing how long a commit takes.
