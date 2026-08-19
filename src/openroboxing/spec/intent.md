# intent.md — the staged-intent model

Version **2.2** · created 2026-08-07 · formalised 2026-08-07 by `M2-T4` · **remodelled 2026-08-08**
· one continuous intent 2026-08-13 · aimed a leg at a time, and ended by the move rather than by a
counter, 2026-08-17

Defines how a player's action reaches the generator. This is a cross-boundary structure
(`CLAUDE.md` invariant 7), so it was specified before `runtime/intents.py` existed and is respecified
here before that module is rewritten.

Status: the **model** and the **runtime semantics** below are decided and implemented in
`runtime/intents.py`. Several **numbers** are still marked *to be measured*; per standing rule 3 they
must come from measurement, not invention.

> **1.0 is a change of game, not a change of wording.** A commit used to be *a punch thrown from
> where you stand, one at a time*. It is now **a plan: go to this place and arrive in this pose** —
> and up to :data:`MAX_OUTSTANDING_COMMITS` of them may be outstanding at once. Walking stopped being
> a separate control and became the first half of every move. Decided by the project owner on
> 2026-08-08; see `docs/ASSUMPTIONS.md` §A19.

> **1.1 finishes the job 1.0 started.** 1.0 said "go here and arrive in this pose" but gave the move
> a **fixed length** — one plan of the pose's own token count — so a fighter walked for about a
> second, stopped wherever that left it, and the queue moved on. Pointing further did not go further.
> 1.1 makes the walk **open-ended: a commit runs until the fighter arrives**, and only then throws.
> Reported from play by the project owner on 2026-08-08; see `docs/ASSUMPTIONS.md` §A23.

---

## The player's loop

1. **Select a pose** from the loadout — a pre-authored, admitted pose record. This is the **final
   frame of the move**, not a button that fires one.
2. **Adjust it** within a bounded envelope (e.g. aim a punch's height/angle).
3. **Place the shadow** — drive a ghost of your own fighter to where the move should end: root
   position and heading, in world coordinates.
4. **Commit.** The plan is queued. The fighter **walks until it gets there** — however long that
   takes — and arrives in that pose on the last frame.
5. **Keep committing.** A commit may be issued while earlier ones are still running. They execute
   back to back, in the order issued.

A commit therefore has **no length you can read off the pose**. A jab thrown where you stand takes
about a second; the same jab placed across the ring takes six, because five of those seconds are the
walk. Choosing where to put the shadow is choosing how long you are committed for, and that is the
cost the queue charges.

**Run out of commits and the fighter stops.** An empty queue is not an idle animation that wanders —
it is the `idle` clip in place. Standing still is something you choose by *not acting*, and it costs
you the round.

### What happened to walking

There is no separate movement control, because there is no longer anything for it to do. In 0.2
`movement` was `in`/`out`/`left`/`right`, held on a key, steering the fighter continuously while
`placement` sat unused. In 1.0 the two collapsed: **a step is a commit whose pose is a stance**, and
distance management is expressed by where you put the shadow. One channel, one mechanism, one thing
to learn.

The 0.2 channel is **retired**, not deprecated — `CLAUDE.md` prefers deleting to disabling.

## Continuous staging — the timing model

The edit channels are **continuously staged while the fight runs**. There is no pause and no edit
mode. The player is always steering a staged intent; committing freezes whatever is staged at that
instant.

This matters because the edit takes seconds and the commit horizon is 0.6 s. Those are **not the
same clock**:

| Quantity | Applies to | Value |
|---|---|---|
| staging | selecting / adjusting / placing the shadow | unbounded — happens during play |
| `COMMIT_HORIZON_TICKS` | commit → execution, **as a minimum** | 30 ticks = 0.6 s — *inert, see below* |
| reference lookahead | commit → execution, **in practice** | **65 ticks = 1.3 s** |
| approach | walking to the placement | **open-ended** — distance ÷ `APPROACH_SPEED_M_S` |
| pose | the strike itself, once arrived | 6–16 tokens ≈ 0.8–2.1 s |
| `MAX_OUTSTANDING_COMMITS` | how many may be unfinished at once | **5** |

**Consequence:** every rate in `CLAUDE.md` stays as it is. The horizon was never a budget for the
edit; it is the *earliest* a committed move may begin — the window that makes it readable.

---

## Runtime semantics

Settled by `M2-T4`, remodelled 2026-08-08.

### A commit's span

A commit runs in **two phases**, and only the second has a length known in advance.

| Field | Meaning | Known |
|---|---|---|
| `issued_at` | the tick the player pressed commit | at commit |
| `commit_at` | the tick the **approach** begins — walking toward the placement | when it starts |
| `strike_at` | the tick the fighter **arrived** and the pose was armed | when it arrives |
| `end_tick` | the tick the **move finished** and the next may become current | when it finishes |
| `arrived` | whether `strike_at` came from arriving or from the timeout | when it arrives |
| `completed_by` | `settled` / `dwell` / `timeout` — what ended it | when it finishes |

**The queue is not a schedule.** Before 1.1 every commit's span was computed the moment it was
issued, so the whole queue's timing was known in advance. It cannot be any more: an approach ends
when the fighter gets there, which depends on physics. `commit_at`, `strike_at` and `end_tick` are
therefore **filled in as the commit runs** and are `None` before that. Anything reading them must
treat `None` as "still outstanding" rather than as zero.

**Only one commit is current at a time**, and it is always the oldest unfinished one. The next
becomes current the tick the previous one ends — no gap, so a combination does not
stutter. The `COMMIT_HORIZON_TICKS` floor still applies to each commit individually: it cannot become
current before `issued_at + COMMIT_HORIZON_TICKS`, which is what makes an isolated commit readable.
Committing behind a running move costs nothing extra, because the 0.6 s of warning elapses while you
watch the move in front of it.

#### What actually sets the floor

`COMMIT_HORIZON_TICKS` is **currently inert**, and the number a player feels is not 0.6 s.

The policy reads the reference motion **45 ticks (0.90 s) ahead of now** — that is what the encoder's
`10frame_step5` terms are — and the stream keeps a further `GENERATOR_MARGIN_FRAMES` (12 frames,
0.40 s) in front of that. So the earliest tick a commit can affect is **65 ticks ≈ 1.30 s** away:
everything sooner has already been generated, and re-generating it is exactly the replan that would
delete a strike.

| | ticks | seconds | |
|---|---|---|---|
| encoder lookahead | 45 | 0.90 | **structural** — the policy has to see the reference ahead of itself |
| generator margin | 20 | 0.40 | a *choice* (`runtime/reference.py`), and the only part that could shrink |
| `COMMIT_HORIZON_TICKS` | 30 | 0.60 | never binds, because it is less than the sum above |

This is not a regression and not new at 1.1 — it has been true since the reference stream existed —
but 1.1 is where it became measurable, because a commit's start is now recorded. Measured: a commit
issued at tick 10 begins at tick 75.

**It matters because latency is the whole feel of a fighting game.** 0.6 s of readable windup is a
design decision; 1.3 s of input lag is a different game. Whether to spend engineering on the 0.40 s
that *is* negotiable is a `M4-T4` question — see `docs/ASSUMPTIONS.md` §A24 — and the horizon should
not be "tuned" in the meantime, because turning a knob that does nothing is worse than leaving it.

#### Arrival

The approach ends when the fighter is within **`ARRIVAL_RADIUS_M`** of the placement, measured
pelvis-to-point on the ground plane. That is a **measured** number, not a chosen one — see
`spec/constants.py` and §Feasibility below.

The test is on the **real fighter under physics**, not on the generator's plan. The plan is
kinematic and arrives every time; the fighter tracking it is the thing that has to get there.

#### Where the plan is aimed is not where the commit ends

The commit ends at its placement. The **generator** is aimed at most `APPROACH_LEG_M` (1.0 m) along
the way there, re-derived from the fighter's true position on every frame.

The two are different because MotionBricks in-betweens toward its target as the plan's *last frame*.
Aimed at a placement three plans away, the plan arrives while the body is still walking — and a
reference standing at the goal has nothing left to pull the body forward with. Measured 2026-08-17
at 2.6 m off-axis: the plan sat 0.15 m from the placement while the body stalled at 0.685 m and then
drifted back out to 0.95 m. Aimed one leg ahead, plan and body stay coupled; the same eight-bearing
sweep completes an approach in 210–275 ticks instead of 229–405.

The leg is also the only speed command this stack has. Upstream's own `target_vel` (and each clip's
`avg_root_vel`) feeds `target_root_pos`, which `has_specific_target` then overwrites — so with a
specific target the clip's speed is inert, and how far ahead the plan is aimed over its own length
*is* the velocity being asked for. The idea is ARDY's (SIGGRAPH 2026): place kinematic constraints
where they can be reached and consume them as they pass.

#### A move ends when it is over, not when a counter says so

Since 2.2 the queue does not advance on a clock. A commit ends when whoever owns a body says the
move is finished, and `end_tick` is **stamped** at that tick rather than computed — `None` until
then, which is the same "not yet" every other span field means.

The test is `has_settled(commit)`, passed into `generator_intent` the way `has_arrived` already was,
and the world's answer is *the body has stopped closing on the pose*: the best pose error over the
last replan window is no better than the window before it, by more than
**`POSE_SETTLE_IMPROVEMENT_RAD`**. Two windows because one cannot tell "converged" from "still
falling"; a **replan** window because the reference is rebuilt at that cadence. Nothing settles
before two full windows exist, so a move always lasts at least 1.0 s — the visible strike the dwell
was protecting.

The counted dwell survives in exactly two places, both explicit:

- **no settle test passed** — the caller has no body to measure (a Studio rehearsal), and
  `POSE_DWELL_TICKS` applies as before;
- **`MAX_DWELL_TICKS`** — the guard. An event-driven end has one failure a counter did not: a pose
  the body never settles into would hold the whole queue forever.

Each commit records **which** of the three ended it in `completed_by` (`"settled"`, `"dwell"`,
`"timeout"`), because a replay should not have to guess.

Why the counter had to go: `POSE_DWELL_TICKS` is the settle time of the *slowest* pose in the
library, and the measured distribution is `[0, 0, 0, 0, 0, 2, 5, 12, 12, 74]` ticks — so nine moves
in ten were waiting out somebody else's worst case. The rule is the project owner's, 2026-08-17:
*pass targets to MotionBricks, and when the plan is finished and the robot is in position, pass the
next one.*

#### The approach cannot run forever

An approach that is not making progress — a fighter wedged in a corner, walking into its opponent,
or knocked down — would hold the whole queue behind it and the player would lose the round standing
still. So an approach is capped at **`APPROACH_TIMEOUT_TICKS`**, derived from the time to walk the
ring's diagonal at the measured approach speed.

When it fires the fighter **throws the pose where it stands** and the commit completes normally, with
`arrived = False` recorded. It is not dropped and not retried: a commit always executes
(§No cancellation), and a move thrown short is a visible, honest outcome the player can learn from.
The match record carries the flag so a replay can show which moves fell short.

Two predicates, and they are not the same:

- `is_scheduled(tick)` — the commit exists and has not finished: `issued_at ≤ tick`, and either
  `end_tick` is not yet known or `tick < end_tick`. This is what the queue bound counts.
- `is_executing(tick)` — `commit_at ≤ tick < end_tick`, with an unknown `end_tick` counting as still
  running. The move is under way and readable — walking counts, because walking *is* the move.

### A queue, bounded, in order

A commit is **accepted while fewer than `MAX_OUTSTANDING_COMMITS` are unfinished**, and appended.
Beyond that it is refused, and the error says when a slot next frees up.

**Five** is a game-feel decision by the project owner (`docs/ASSUMPTIONS.md` §A19).

> ⚠ **What five costs changed at 1.1.** It was chosen when a move was 0.8–2.1 s, making a full queue
> 4–10 s of pre-planned action against a 60 s round. Now a move is its walk plus its pose, so five
> moves placed across the ring is **20–30 s** — half a round committed in advance, unrecallable. That
> may be exactly the tension the no-cancellation rule is for, or it may be too much; it is a feel
> question for the first bracket (`M4-T4`), and the number has **not** been changed on a guess.
> `docs/ASSUMPTIONS.md` §A23.

The 0.1 rule this replaces — *"a second commit is refused, not deferred, because deferring one would
let a player buffer inputs and so remove the cost of committing early"* — was answered rather than
abandoned. Buffering still costs, and costs more: a queued plan is **five moves' worth of decisions
made before you knew what the opponent would do**, and none of them can be taken back.

### No cancellation, of anything

Once issued, a commit will execute — including one still waiting its turn. This is the founding rule
of the game held to its full extent, and it is what makes a deep queue a risk rather than a free
lookahead.

### Staging never touches a fired commit

`stage()` is legal at every tick, including mid-move, and changes only the channels it is passed.
A `StagedIntent` is immutable once returned, so a value read earlier cannot change underfoot.
`clear_pose()` unstages — it is not a cancellation, because nothing has fired.

### The horizon reaches the generator

`horizon_tokens` was inert until `M2-T4`: `control_signals` took `allowed_pred_num_tokens` from the
clip registry and ignored the pose. It is now passed through as a one-hot over the clip's 11-slot
mask (index `horizon_tokens - MIN_TOKENS`, covering 6…16 tokens). Asking for a length the clip does
not permit **raises** — a move that runs for a different length than the commit promised is a
scoring bug, not something to fall back from.

This is also what keeps the **pose phase** honest: because the token count is forced, that plan's
length is known before it is generated, so `end_tick` follows from `strike_at` by arithmetic rather
than by watching. The *approach* has no such guarantee and is not given one — it ends by arriving.

### Admission is enforced at construction

`IntentTimeline` validates every pose in the loadout when it is built, and by default refuses one
that is not `admitted`. The Studio passes `require_admitted=False` so a draft pose can be rehearsed
before it has been measured; a match never does.

---

## Channels

| Channel | What it carries | Upstream support |
|---|---|---|
| `pose_slot` | which loadout slot (pose record) — the move's **last frame** | needs **patch P0** (`M2-T1`) |
| `adjustment` | bounded deviation from the base pose | needs **patch P0** |
| `placement` | **where the move ends**: root position + heading (1.0: the primary control) | **free — already upstream** |
| `context` | style preset = clip `mode` one-hot | **free — already upstream** |
| `commit_at` | tick the move should begin, in 50 Hz ticks | ours |

*(0.2's `movement` channel was retired at 1.0 — see "What happened to walking".)*

### Placement is free, and is now the whole point

`full_agent._override_target_transforms` already accepts `specific_target_positions` and
`specific_target_headings`, blended by a `has_specific_target` mask
(`full_agent.py:298-319`, and the spring-model path at `:241-273`). No patch is required.

Two measurements, and they are about different things:

- **Kinematically** (`M2` / `runtime/generator.py`) a generator commanded to `(3, 2)` ends 0.12 m
  away, and 2.3 m from where it goes uncommanded.
- **Under physics** an open-ended approach also arrives. Measured 2026-08-08 over ten placements
  around a fighter — forward, lateral and behind — the worst closest approach was **0.30 m** and
  every placement was reached inside 4.3 s (`scratchpad/probe_arrival.py`).

  **That number did not survive 2.0, and the regression was invisible for four days.** Re-measured
  2026-08-17 with `tools/measure_approach.py` (1.5 m, eight bearings): only *forward* approaches
  still closed. Off-axis the body stalled at 0.38–0.54 m while the plan closed to 0.02–0.19 m every
  time, and four of seven bearings ran out the timeout and threw the pose short. Two causes, both
  fixed the same day:

  - the ambient context was `walk_boxing`, and upstream's lateral blendspace
    (`walk_left`/`walk_right`) only swaps in when the mode is `slow_walk` or `walk` — so in the
    boxing style a fighter has **no sideways gait at all**;
  - `GeneratorIntent.movement_angle` was never set, so upstream read "straight ahead" as the
    direction of travel no matter where the placement was.

  With the release's `walk` context, a real travel direction and a 1.0 m leg: **seven of eight**
  bearings arrive. The one that still does not is a placement directly behind (0.449 m against a
  0.40 m radius), which remains open.

So placement is **closed-loop, continuously**: the target is re-derived from where the fighter
actually is on every frame generated, and the generator's own buffer tail is what it is measured
*from*. That combination is what makes it converge, and the ordering is not interchangeable —

> **Anchor the conversion on the generator's buffer tail, not on the frame the robot is playing.**
> The tail is a lookahead ahead of the robot, so re-deriving the remaining distance from the robot
> each frame keeps pushing the plan forward until the robot catches up: an integrator, and it settles
> to ~0.1 m. Anchoring on the current frame instead cancels that lookahead and leaves a proportional
> controller whose steady-state error is the policy's tracking shortfall — measured at **27 % of the
> distance asked for** (0.22 m at 1 m, 0.49 m at 2 m, 0.81 m at 3 m). Both readings look principled;
> only one arrives. `docs/ASSUMPTIONS.md` §A23.

Before 1.1 this loop only closed **between** commits, and each commit was one fixed-length plan, so
it converged to about 0.75 m and stopped. `docs/ASSUMPTIONS.md` §A21.

Coordinates are **MuJoCo world `(x, y)` on the ground plane**, plus a heading in radians — the same
frame the arena, the shadow and the client all use.

**The generator does not plan in that frame**, and converting into its own is not optional:
`runtime/fight.py::to_generator_frame` expresses a placement as *from where the generator is, travel
the vector from the robot to the target*. Passing a world point through raw aims each fighter off by
wherever its clip happened to start — the same trap `facing_angle` has always handled for angles,
which had no counterpart for positions until 1.0.

Note also the axis swap in the upstream code:
`specific_target_positions[:, :, 1]` feeds row 0 and `[:, :, 0]` feeds row 1 (`:306-307`). The
wrapper in `runtime/generator.py` owns this and documents it; nothing above the wrapper knows.

### Pose and adjustment need the one patch

Steps 1–2 are what `M2-T1` buys: extending `_override_target_transforms` to accept explicit target
**joint** transforms. Currently it overrides root position and heading only, and the pose itself is
sampled from the clip library by one-hot `mode` × `random_seed`
(`_generate_target_joint_transforms`, `full_agent.py:321-391`). The injection point is the
`(global_joint_positions, global_joint_rotations)` pair returned at `:391`.

---

## The shadow

The staged intent is shown to its own player as a **ghost of their fighter**, standing at the staged
placement in the staged pose. It is the same data the generator will be given: the pose record's
joint angles under forward kinematics, rooted at the placement.

**It is computed once, by the host, from the pose record** — not re-derived in the client. A preview
that ran its own kinematics would be a second implementation of "what this move looks like", free to
disagree with the move that actually happens.

**The opponent never sees it.** `WORKPLAN` M4-T1's "no HUD on the fighters — the windup is the only
cue" is unchanged: staging is private, and a *committed* move becomes public only once it is
executing, which is the moment it is readable in the world anyway. A queued-but-not-started commit is
therefore **not** transmitted to the opponent — it has been paid for, but it has not yet been shown.

---

## Feasibility

Two different guards, because two different things can be infeasible.

**Reach — retired at 1.1.** A move used to carry a fighter only as far as its own duration allowed,
so a client was sent `reach_m` per slot and greyed out a shadow placed beyond it. An open-ended
approach has no such limit: **anywhere in the ring is reachable**, and what a distant placement costs
is *time*, not failure. The guard is replaced by an honest estimate — `APPROACH_SPEED_M_S`, measured,
lets a client say "this will take about 4 s" before the player commits to it, which is the thing they
actually need to know now that they cannot take it back.

**Placement** — the root backbone also models unreachability: it has an out-of-reach token
(`OUT_OF_REACH_NUM_TOKENS`, gated by `allow_pred_out_of_reach_num_tokens`,
`root_backbone.py:191-194`). If the commanded position cannot be reached within the allowed token
range, the generator can say so. Use this rather than inventing a distance limit; surface it to the
player as "can't get there".

This matters more at 1.0 than it did at 0.1, because placement is now how a player moves at all: a
shadow dragged across the ring is a request the generator may be unable to honour in the tokens the
pose allows, and the player must be told *before* committing, not discover it by not arriving.

**Pose + adjustment** — MotionBricks is kinematic and physically unaware and will happily emit
self-penetrating or torque-infeasible targets (`CLAUDE.md`, known traps). The base pose is admitted
**offline** in the Studio on measured telegraph window and tracking error. The live adjustment is
**not** individually admitted, so it must be constrained to an envelope whose corners were admitted
offline. Validating the envelope, not the individual adjustment, is what keeps admission meaningful.

An adjustment outside the envelope, or on a joint the envelope does not cover, **raises**. Clamping
it would silently produce a pose nobody has ever measured, which is precisely what admission exists
to prevent.

### The pose target is reached — and must not be replanned over

`M2-T3b` first reported the pose target as a soft constraint the model ignored. **That was wrong**;
see `spec/upstream_notes.md`. Authored poses are reached to 2–3° mean and under 11° worst, across
guards, jabs, hooks, uppercuts and slips.

The real constraint is temporal, not spatial. MotionBricks in-betweens from context to target, and
the target is the plan's **last** frame, so:

**While the pose phase is running, the fighter must not replan.** A replan discards the current
plan's tail, which is the strike.

This is why the two phases are generated differently, and the difference is the whole mechanism:

| Phase | Plan | Why |
|---|---|---|
| approach | **replanned** at the ambient cadence, target re-derived every frame | it has to steer; nothing is being aimed at yet |
| pose | **one forced plan, consumed whole** | the strike is the last frame and a replan would delete it |

So a commit is *n* throwaway plans followed by one that must not be thrown away. Getting that
backwards in either direction is a silent failure: replan the pose and no punch ever lands; force the
approach and the fighter walks one plan's worth and stops — which is exactly the 1.0 bug 1.1 fixes.

**Consequence for the reference stream.** Frames are generated ahead of the tick that plays them, so
the stream must ask what the fighter is doing **at the tick each frame will be consumed**, not at the
tick it happens to be filling on. Getting this wrong does not crash: it slides every move a fixed
lookahead late. See `runtime/reference.py`.

**And the forced plan must be consumed exactly.** A pose plan holds `horizon_tokens × 4` frames at
30 Hz while the pose phase occupies `duration_ticks` ticks at 50 Hz, and the two grids do not divide.
The stream must therefore bind a forced plan to the commit that forced it and discard whatever is
left when that commit ends — otherwise a leftover frame is played as the first frame of the *next*
commit, and the error accumulates one frame per move until a whole commit is somebody else's tail.

---

## Impact on the plan

| Task | Change |
|---|---|
| `M2-T1` | unchanged in kind, but must also support the **adjustment envelope**, not just a fixed pose |
| `M2-T2` | the pose record gains an **adjustment envelope** definition |
| `M2-T5` | admission must cover the envelope, not only the base pose |
| `M4-T1` | **larger than scoped.** Needs a 3-D client with a spatial picker (`spec/protocol.md` 0.4) |
| `M4-T4` | owns two feel questions now: ring size, **and whether a 5-deep queue is 20–30 s too long** |
| `M5-T2` | `TARGET_COMMIT_RATE` was calibrated against one-at-a-time commits and **needs re-measuring** |
| `S-T1` | unchanged — the Studio stays offline, and gains envelope authoring |

**Not a regression:** the "no HUD, the windup is the only cue" principle survives, and is now
load-bearing in a second way — a queue is only a risk if the opponent cannot read it.

---

## Changelog

- **2.2** (2026-08-17) — **a move ends when it is over.** `end_tick` is stamped rather than computed:
  `generator_intent` asks `has_settled(commit)` — has the body stopped closing on the pose? — and the
  queue advances on that instead of on `strike_at + POSE_DWELL_TICKS`. The counted dwell remains as
  the rule for a caller with no body (the Studio) and `MAX_DWELL_TICKS` as the guard against a pose
  that never settles; `completed_by` records which of the three ended each commit. The project
  owner's rule, in their words: *pass targets to MotionBricks, and when the plan is finished and the
  robot is in position, pass the next target.*
- **2.1** (2026-08-17) — **the fighter can walk sideways again.** The ambient context is the
  release's `walk` (`intents.TRAVEL_CONTEXT`) instead of the shadow-boxing loop, which is what gates
  upstream's lateral gaits; the runtime now tells the generator which way it is *travelling* as well
  as which way it faces; and an approach aims the generator at most `APPROACH_LEG_M` ahead rather
  than at the whole placement. Arrival, the timeout and the queue are untouched — what changed is
  what the generator is asked for. Measured before and after with `tools/measure_approach.py`
  (§Feasibility): three of seven bearings arriving became seven of eight, and the approach itself
  got about a third shorter. Occasioned by the project owner's report that a fighter "walks to
  position and then executes the motion".
- **2.0** (2026-08-13) — **a commit is one continuous intent**: the pose is armed on every replan for
  the commit's whole life instead of only after arrival, the generator picks its own plan length, and
  a completed commit is held rather than followed by an idle clip. Specified in
  `docs/superpowers/specs/2026-08-13-continuous-pose-targeting-design.md` and implemented in
  `runtime/intents.py`; this entry was written on 2026-08-17, when the file was next edited.
- **1.1** (2026-08-08) — **a commit runs until it arrives.** The move gained an open-ended approach
  phase in front of its pose phase, so `commit_at` / `strike_at` / `end_tick` are filled in as it
  runs instead of being computed at commit time, and the queue stopped being a schedule. Arrival is
  `ARRIVAL_RADIUS_M`, measured; a stalled approach is capped by `APPROACH_TIMEOUT_TICKS` and throws
  where it stands with `arrived = False`. The `reach_m` feasibility guard was retired — anywhere in
  the ring is reachable and distance now costs time. Recorded that the world→generator conversion
  must anchor on the generator's buffer tail, and that a forced plan must be bound to the commit that
  forced it. Reported from play by the project owner.
- **1.0** (2026-08-08) — the queued-plan model. A commit is now *place + pose*, the generator walks
  the fighter there and arrives in it, and up to `MAX_OUTSTANDING_COMMITS` = 5 may be outstanding.
  `commit_at` became `max(issued_at + horizon, tail_end)` so the horizon is a floor rather than a
  stutter between queued moves. The 0.2 `movement` channel was retired, since placement now expresses
  distance management. No cancellation, extended to queued-but-unstarted commits. Added §The shadow.
  Decided by the project owner; the version moves because the model did.
- **0.2** (2026-08-08) — added the `movement` channel. *Superseded by 1.0.*
- **0.1 corrected** (2026-08-07) — the "soft pose target" finding was a measurement artifact of
  replanning every 15 frames of a 24–64 frame plan. Authored poses *are* reached. Replaced with the
  rule that actually holds: a commit must not be replanned over.
- **0.1 formalised** (2026-08-07, `M2-T4`) — runtime semantics settled and implemented in
  `runtime/intents.py`: a commit's span and the two predicates, rejection rather than queueing,
  staging's isolation from a fired commit, `horizon_tokens` wired through to
  `allowed_pred_num_tokens`, admission enforced at construction.
- **0.1** (2026-08-07) — first draft. Staged-intent model, continuous staging, four channels,
  placement-is-free finding, envelope-based feasibility. Numbers deliberately absent.
