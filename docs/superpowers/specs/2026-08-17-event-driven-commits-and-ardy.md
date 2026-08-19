# Commits that end when the move ends — and what ARDY suggests

Design note · 2026-08-17 · status **implemented and measured** — see §5

The project owner asked for two things after a session on the sparring bench:

> «the idea is not to have a deterministic state machine for the commits — the idea is to pass
> targets to MotionBricks and when MotionBricks terminates the plan and the robot gets to position
> you pass the next target»

and, as inspiration for making the motion system more accurate and controllable,
[nv-tlabs/ardy](https://github.com/nv-tlabs/ardy).

This note records what the bench now measures, why the current design produces the behaviour the
owner objected to, what the change actually is, and — importantly — **why one part of it must not
be built first.**

---

## 1. What was measured

Two probes against a real `SparringWorld` (checkpoints loaded, physics on, seed 1234, `orthodox`,
slot 1). Both are reproducible; §6 turns them into a repo tool.

**The arrival test reads the body, not the plan.** Instrumenting `SparringWorld.has_arrived`
confirms it: `runtime/fight.py` measures `data.xpos[pelvis_body][:2]` — the pelvis under physics —
against the placement. The owner's suspicion that arrival is read off the MotionBricks ghost is
**not** what the code does. But the suspicion points at something real, below.

**One placement, seven bearings, 1.5 m away.** Bearing 0 is the generator's forward axis; the
approach's arrival radius is 0.40 m and its timeout 419 ticks (8.4 s).

| bearing | body, closest | plan, closest | arrived | approach ticks |
|---:|---:|---:|:--|---:|
| 0° | **0.007 m** | 0.017 m | yes | 115 |
| +45° | 0.382 m | 0.090 m | yes | 355 |
| +90° | 0.173 m | 0.119 m | **no** | 419 — timed out |
| +135° | 0.538 m | 0.190 m | **no** | 419 — timed out |
| 180° | 0.468 m | 0.082 m | **no** | 419 — timed out |
| −45° | 0.391 m | 0.182 m | yes | 172 |
| −90° | 0.490 m | 0.127 m | **no** | 419 — timed out |

Read it as two sentences:

- **The plan always arrives.** MotionBricks closes to 0.02–0.19 m at every bearing. The generator
  is not what fails.
- **The body only arrives forward.** Straight ahead it lands on the placement (7 mm). Off-axis it
  stalls at 0.4–0.5 m and, in four of seven cases, never enters the radius before the 8.4 s timeout
  fires the pose where it stands, with `arrived=False`.

A separate run at 2.6 m off-axis showed the same shape from the other side: the body closed to
0.685 m at tick 250, then **drifted back out to 0.95 m** and oscillated there while the plan sat at
0.15 m — the approach does not merely converge slowly, it stops converging.

## 2. Why this is what the owner saw

The owner reported "after the commit it walks to position and then executes the motion, I think
this is not right". Every piece of that follows from the numbers above.

- **The walk is long because the body is slow off-axis, not because the design sequences it.**
  Since `spec/intent.md` 2.0 the pose is armed on *every* replan, so the intent is one continuous
  "be there, in that pose" — there is no walk phase followed by a strike phase in the intent. What
  makes it look sequenced is that MotionBricks covers roughly half the remaining distance per plan,
  so while the distance is large the plan is mostly locomotion and the pose is a far attractor. The
  strike becomes visible only near the target.
- **The strike often fires on a timer, not on an arrival.** In four of seven bearings the pose was
  thrown by `approach_timeout_ticks`. That is the "deterministic state machine" the owner is
  reacting to — and it is doing real work today, because without it a stalled approach would hold
  every commit behind it forever.
- **The decision is read a lookahead early.** `has_arrived` is asked at generation time about a
  frame that plays ~65 ticks (1.3 s) later. That is deliberate and documented (an approach
  decelerates into its target), but it means `strike_at` is stamped in the future and the body may
  be somewhere else by the time the pose actually fires.

So the owner's instinct — *the arrival has something to do with the ghost and not the robot* — is
half right in the way that matters: the **plan** reaches the placement six seconds before the
**body** does, and the whole approach is spent waiting for a body that may never close.

## 3. The change the owner asked for

**A commit ends when its move is over, not when a counter says so.**

Today `Commit.end_tick` is a property: `strike_at + POSE_DWELL_TICKS` (74). The next commit becomes
current at that tick. The dwell is measured — it is how long the slowest pose in the library needs
to settle — but it is still a countdown, and it is what makes the queue advance on a clock.

Shipped as `spec/intent.md` **2.2**:

- `Commit.end_tick` stops being derived and becomes `ended_at`, **stamped** when the move completes,
  exactly as `commit_at` and `strike_at` already are. `None` keeps meaning "not yet".
- `IntentTimeline.generator_intent` gains a second callback beside `has_arrived`:

  ```python
  generator_intent(tick, *, facing_angle=0.0, has_arrived=None, has_settled=None)
  ```

  `has_settled(commit) -> bool` is asked once per frame after a commit has struck. Omitting it is
  legal and means *the counted dwell* — the only rule a caller with no body can use, which is what
  the Studio's rehearsals are. A world that has a body passes it, and then the queue advances on the
  move being over.
- The world owns both, because the timeline knows nothing about geometry or bodies. `has_settled`
  is *the body has stopped closing on the pose*: the best pose error over the last replan window
  (25 ticks) is no better than the window before it by more than `POSE_SETTLE_IMPROVEMENT_RAD`.

  Two windows rather than one, because a single window cannot tell "converged" from "still falling";
  a **replan** window rather than any other length, because the reference is rebuilt at that cadence
  and anything shorter measures the inside of a plan. `tools/measure_dwell.py`'s own definition of
  settling was not usable live — it reads the whole run to find the asymptote first — so this is its
  causal analogue, and nothing settles before two full windows exist, which floors a move at 1.0 s.
- The counted dwell **stays** in two explicit places — the bodyless rule above, and
  `MAX_DWELL_TICKS` (3 × the measured dwell) as the guard against a pose the body never settles
  into. Each commit records **which** ended it: `completed_by: "settled" | "dwell" | "timeout"`.

What does not change: the server stays authoritative, the timeline still knows nothing about where
a fighter is, nothing fails silently, and there is still no cancellation.

### 3.1 Why this must not be built first

**Under the measurements in §1, replacing the dwell with "settled" without fixing closure turns a
tick-limited stall into a permanent one.** Four of seven bearings never arrive; with completion
gated on arrival and no guard, those commits would hold the whole queue forever. The guard is what
makes the new design safe, and a guard that fires in 4 of 7 moves is not a guard — it *is* the
mechanism, renamed.

So the order is: **fix the closure, then make completion event-driven.** Otherwise the change is
cosmetic at best and a deadlock at worst.

## 4. What ARDY offers

**ARDY** — *Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion
Generation*, Kaifeng Zhao, Mathis Petrovich, Haotian Zhang, Tingwu Wang, Siyu Tang, Davis Rempe;
ACM TOG / SIGGRAPH 2026; NVIDIA Toronto AI Lab. Apache-2.0 code, NVIDIA Open Model licence on the
weights. It generates humanoid motion autoregressively from streaming text plus kinematic
constraints, and ships an interactive demo where a user places waypoints with the mouse and steers
with the keyboard while the character keeps moving.

Five things in it speak directly to §1's defect.

**(a) A target carries a *when*, not only a *where*.** ARDY's constraints are
`(frame_indices, value)` pairs inpainted into the generation window —
`Root2DConstraintSet(skeleton, frame_indices, root_2d, global_root_heading)`, plus full-body,
end-effector and per-limb variants. Our `GeneratorIntent.target_position` has no frame index: we say
"be there", never "be there by frame k". A schedule derived from the distance and the measured
sustained speed (`APPROACH_SPEED_M_S`, 0.83 m/s) would make plan speed a choice instead of an
inference — and would let the plan *stop being at the target* while the body is still 0.9 m away,
which is exactly the state the 2.6 m run got stuck in.

**(b) Waypoints, placed ahead and consumed as they pass.** ARDY's velocity steering synthesises
waypoints every `TARGET_VELOCITY_GOAL_FRAME_INTERVAL` frames into the future and removes each one
once passed (`remove_keyframe`). Decomposing an approach into legs — each no longer than what one
plan actually covers, each with a small heading change — is the most plausible fix for the off-axis
stall, because a 180° placement is currently one plan asked to turn the body around and cross 1.5 m
at once. **This is also the owner's own model** ("pass the next target when the previous is done"),
one level down: inside the approach, not only between commits.

**(c) The window stretches to the furthest constraint.** ARDY computes `max_window_len` from
`max_constraint_idx`, never below `history_length + gen_horizon_len`: a plan is exactly as long as
the constraint it has to reach. Ours leaves `horizon_tokens=None` and lets the model pick by argmax,
which is a good default precisely *because* nothing tells it a deadline. Give it a deadline and the
length follows from it.

**(d) Replan on event, not only on cadence.** ARDY replans when something changes —
`on_replan_trigger()`, `restart_from_now()`, and `skip_if_busy` so a frame-by-frame check cannot
pile up work. We replan every `REPLAN_DT` = 0.5 s unconditionally, whether or not anything changed.
Event-driven replanning (new commit, leg consumed, arrival) over a slower ambient cadence would cut
the churn where the target is re-derived every half second from a body that has barely moved.

**(e) Contact-consistent post-correction.** ARDY predicts foot-contact states and ships a C++
`MotionCorrection` pass for foot skating. Our bridge resamples 30 → 50 Hz and differentiates for
velocities; nothing checks that the reference's feet are consistent with contact. A policy asked to
track a skating reference will lag — which is one candidate explanation for why off-axis (turning)
references track so much worse than forward ones.

**And one strategic note, ruled out for now** — the project owner's call, 2026-08-17: transfer the
approaches, not the network. ARDY ships **ARDY-G1-RP-25FPS**, a Unitree G1 skeleton model
(25 FPS, 8- or 52-frame horizons), with ONNX and TensorRT export (`scripts/export_onnx.py`,
`ardy/model/trt.py`) and a MuJoCo export path (`ardy/exports/mujoco.py`). It speaks our robot
natively. Behind the `GeneratorIntent` seam it would be a candidate **second backend**, and the cleanest
experiment for the question §1 leaves open — *is a stall MotionBricks, the bridge, or the policy?*
It stays on the shelf: the ideas transferred without it, and §5 shows how far they went.

## 5. What was built, and what it measured

Everything below happened on 2026-08-17, in this order, each step measured with
`tools/measure_approach.py` (1.5 m, eight bearings, arrival radius 0.40 m).

| step | change | arrived | approach ticks |
|---|---|---:|---|
| baseline | `walk_boxing`, no travel direction, aimed at the placement | 3 of 7 | 229–419 |
| §4(a-ish) | travel direction supplied | 4 of 8 | |
| clips | the release's `walk` instead of the boxing loop | 5 of 8 | |
| both | | **7 of 8** | 229–405 |
| §4(b) | + a 1.0 m leg | **7 of 8** | **210–275** |

Then §3, event-driven completion, on top of that: every commit in the sweep ended `settled`, with a
dwell of **50–73 ticks** against the counted 74 — the floor being two settle windows (1.0 s), and the
two moves that needed longer taking longer, which is the whole point.

The one remaining failure is a placement directly behind the fighter: it reaches 0.449 m against a
0.40 m radius and ends on the approach timeout.

## 6. Order of work

1. ~~The bench separates body from plan~~ — done: `dist` and `dist_plan`, the arrival radius drawn
   across the strip, and a timed-out commit says so in the queue.
2. ~~Make the measurement a tool~~ — done: `tools/measure_approach.py`, with `--context`, `--leg`
   and `--travel-angle` so each lever can be ablated.
3. ~~Waypointed approach~~ — done: `APPROACH_LEG_M`, a bench knob, and the sweep above.
4. ~~A deadline on the target~~ — **answered, not built.** Upstream's `target_vel` and each clip's
   `avg_root_vel` feed a target root position that `has_specific_target` then overwrites, so with a
   specific target they are inert: *how far ahead the plan is aimed over its own length is the
   velocity being asked for*, and that is the leg. ARDY (a) and (b) are the same mechanism here.
5. ~~Event-driven completion~~ — done: `spec/intent.md` 2.2.
6. **Still open — the placement directly behind.** 0.449 m against a 0.40 m radius, ending on the
   timeout. Candidates in order of cheapness: let an intermediate leg face the way it travels so the
   turn is spread over the approach (ARDY's velocity transition); give the reference contact-aware
   post-correction (ARDY (e)); widen the radius, which is the answer only if the first two fail.
7. **Still open — replan on event** (ARDY (d)). The cadence is a flat 0.5 s whether or not anything
   changed; a new commit, a consumed leg and an arrival are all events that deserve a plan, and the
   half-second in between mostly does not.

## 7. Non-goals

- No change to the match protocol. The bench's extra fields are the bench's own.
- No cancellation of commits, at any point. Unchanged from `spec/intent.md`.
- No tuning of `POSE_DWELL_TICKS` as a workaround: the dwell is measured and correct for what it
  measures; the objection is to *ending a move on a counter*, not to the counter's value.
