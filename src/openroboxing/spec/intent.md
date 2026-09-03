# intent.md — the staged-intent model

Version **3.2** · created 2026-08-07 · formalised 2026-08-07 by `M2-T4` · remodelled 2026-08-08 ·
one continuous intent 2026-08-13 · aimed a leg at a time, and ended by the move rather than by a
counter, 2026-08-17 · **a commit is a combination, and the approach is gone, 2026-08-28** · `D6`
lands and `Loadout` is deleted, 2026-08-28 (`M6` Phase 3, task A6) · **a fighter always faces its
opponent, 2026-09-03 (owner), reversing D5's recorded heading** · **the keyframe is pinned in
absolute time and the hole in front of it shrinks, 2026-09-03 (owner)**

Defines how a player's action reaches the generator. This is a cross-boundary structure
(`CLAUDE.md` invariant 7), so it is specified before `runtime/intents.py` is rewritten against it.

Status: the model is decided — by the project owner,
`docs/superpowers/specs/2026-08-27-motion-combinations-design.md`, decisions D1–D6 — and is
implemented. `runtime/intents.py` reads `SPEC_VERSION = "3.2"`; a test pairs this file's
version with that constant.

**`D6` — no loadout: the whole library shared and paged, nine combinations at a time — is
implemented.** The client and protocol work it needed (`spec/protocol.md` 0.6, `M6` Phase 3) has
landed, and `Loadout` is deleted (`M6` Phase 3, task A6) — nothing in the codebase names it any
more. "Select a combination" below describes the paged picker directly.

`spec/pose_record.md` is unchanged by any of this — a combination *contains* pose records, one per
keyframe, and does not replace the schema they use.

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

> **3.0 is the second change of game.** 1.0 turned a commit from *a punch thrown from where you
> stand* into *a plan: walk here and arrive in this pose*. 3.0 removes the walk. A commit is now
> *play this recorded combination, starting wherever you already are, its last pose landing on the
> ghost you placed*. What a player buys with a commit is no longer a punch plus an unknown amount of
> walking — it is a specific timed piece of motion whose length was fixed the day it was captured,
> not by how far the ghost was dragged. Decided by the project owner, 2026-08-27; see
> `docs/superpowers/specs/2026-08-27-motion-combinations-design.md`.

---

## The player's loop

1. **Select a combination** — 2–3 recorded key poses with recorded timing (`spec/combination.md`
   0.2), from a library of 174 built from the mocap corpus under `motions/`. Both fighters carry
   the whole library and page through it nine at a time (`D6`) — there is no per-seat loadout to
   select from first. This replaces "select a pose": the unit of selection is no longer one key
   pose.
2. **Place the ghost** — drag it to where the combination's **last keyframe** should land: position
   only, world coordinates. There is no heading control any more — the ghost's heading is derived,
   never chosen, and since 3.1 what it is derived from is the **opponent**: a fighter always faces
   the fighter it is boxing (see "The ghost", below).
3. **Commit.** The plan is queued. The fighter runs the combination **starting from wherever it
   actually is** the tick the move begins — no approach, no walk to a start — and its recorded
   footwork carries it while whatever residual distance is left over to the ghost is added as an
   even drift across the whole combination.
4. **Keep committing.** A commit may be issued while earlier ones are still running. They execute
   back to back, in the order issued.

A commit's length is now **exactly the recording's length**, read straight off
`CombinationRecord.duration_ticks`, regardless of where the ghost is placed. Distance no longer buys
time the way it did under 1.0–2.2; it buys **drift speed** instead (see "Off-target execution",
below). Choosing where to put the ghost is choosing how hard the fighter has to run to reach it
inside a combination's own fixed duration, not how long the commit lasts.

**Run out of commits and the fighter stops.** Unchanged from 1.0: an empty queue is not an idle
animation that wanders — it is the `idle` clip in place, `OPENING_STANCE_CONTEXT` before a round's
first commit. Standing still is something you choose by *not acting*, and it costs you the round.

### What happened to the approach

There is nothing gradual about its removal: the whole walk-then-throw phase 1.0–2.2 built is gone in
one step, not shrunk. `TRAVEL_CONTEXT`, `ARRIVAL_RADIUS_M`'s use as an end-of-approach test,
`APPROACH_LEG_M`'s leg-ahead aiming, `approach_timeout_ticks` and `DEFAULT_APPROACH_TIMEOUT_TICKS`,
and the `has_arrived` callable all go with it — see "Removed at 3.0" for the complete list and what
replaced each.

**Why it is safe to remove rather than merely retire.** The approach existed because 1.0's only
control was a placement with no idea what motion should get the fighter there — walking was invented
to fill that gap. A combination fills it instead: it already contains the footwork (D4), so there is
nothing left for a generic walk to do. This is not a simplification for its own sake; it is what D3
requires — a combination starts in place because it is a specific *recorded* motion, and recorded
motions do not have an approach phase to their own start.

---

## Continuous staging — the timing model

Unchanged in kind from 2.2: the edit channels are staged continuously while the fight runs, and
committing freezes whatever is staged at that instant. What changed is what committing now buys,
because the phase that used to be open-ended no longer is:

| Quantity | Applies to | Value |
|---|---|---|
| staging | selecting / placing the ghost | unbounded — happens during play |
| `COMMIT_HORIZON_TICKS` | commit → execution, as a minimum | 30 ticks = 0.6 s — inert, see below |
| reference lookahead | commit → execution, in practice | 65 ticks = 1.3 s |
| combination | the whole move, position to ghost | **`record.duration_ticks`, fixed by the recording** — 1–2 legs of 0.8–3.2 s each (`COMBINATION_MIN/MAX_KEYFRAMES`, `MIN_TOKENS`–`MAX_TARGET_LEG_TOKENS`); measured 0.93–6.00 s over the rebuilt library, median 3.87 s |
| `MAX_OUTSTANDING_COMMITS` | how many may be unfinished at once | **5**, unchanged |

The approach row from 2.2 is simply gone: there is no phase left in this table that is open-ended,
which is the whole content of the reversal below.

---

## Runtime semantics

Settled by `M2-T4`, remodelled 2026-08-08, re-remodelled 2026-08-28 for motion combinations
(`M6-T4` / `M6-T5`).

### A commit's span — known when it starts, and this reverses 2.2

| Field | Meaning | Known |
|---|---|---|
| `issued_at` | the tick the player pressed commit | at commit |
| `commit_at` | the tick this commit becomes current and starts running | as soon as the previous commit's `end_tick` is known — see below |
| `end_tick` | the tick the combination finishes and the next may become current | the same tick, by arithmetic: `end_tick = commit_at + record.duration_ticks` |

**2.2 said "the queue is not a schedule."** Before 1.1 every commit's span was computed at issue
time; 1.1 made it open-ended because an approach's length depended on distance under physics and
could not be known until the fighter got there, so `commit_at` / `strike_at` / `end_tick` were
filled in only as each commit ran, and `None` had to mean "still outstanding."

**3.0 reverses that, in full.** With the approach gone, a combination's length is its recording's
length — `record.duration_ticks`, fixed the day it was captured and known before the commit is ever
issued. There is nothing left to watch for: `end_tick` is not stamped when a body settles, it is
computed the instant `commit_at` is. And because `commit_at = max(issued_at + COMMIT_HORIZON_TICKS,
previous.end_tick)` — the same floor arithmetic 1.0 introduced — and `previous.end_tick` is now
always already known, **the whole outstanding queue's timing is computable the moment a commit is
appended**, not only the currently-executing one. The queue *is* a schedule again, and this time the
thing that broke it — an approach whose length depended on distance — does not exist.

**This is not a claim that space is equally certain.** Time is committed the moment a commit is
appended; where the fighter is standing when that time arrives is not, because a fighter can be
pushed, knocked down, or simply have tracked its previous combination imperfectly. That is exactly
what "Off-target execution", below, is for — timing and placement are no longer the same question.

The two predicates from 2.2 simplify along with it, because there is no longer an unknown `end_tick`
to guard against:

- `is_scheduled(tick)` — the commit exists and has not finished: `issued_at ≤ tick < end_tick`.
- `is_executing(tick)` — `commit_at ≤ tick < end_tick`.

Both `end_tick`s here are always known, so neither predicate needs 2.2's "`None` means still
running" rule.

`arrived` and `completed_by` are gone with the phases and endings they distinguished — see "Removed
at 3.0".

#### What actually sets the floor

`COMMIT_HORIZON_TICKS` is still **currently inert**, and that did not change with the approach's
removal — it was never about the approach.

The policy reads the reference motion **45 ticks (0.90 s) ahead of now** — the encoder's
`10frame_step5` terms — and the stream keeps a further `GENERATOR_MARGIN_FRAMES` (12 frames, 0.40 s)
in front of that. So the earliest tick a commit can affect is still **65 ticks ≈ 1.30 s** away:
everything sooner has already been generated, and re-generating it is exactly the replan that would
delete the live leg.

| | ticks | seconds | |
|---|---|---|---|
| encoder lookahead | 45 | 0.90 | structural — the policy has to see the reference ahead of itself |
| generator margin | 20 | 0.40 | a choice (`runtime/reference.py`), the only part that could shrink |
| `COMMIT_HORIZON_TICKS` | 30 | 0.60 | never binds, because it is less than the sum above |

Unchanged from 2.2: this is not new at 3.0, and the horizon should not be "tuned" in the meantime —
see `docs/ASSUMPTIONS.md` §A24.

### Off-target execution: re-warped from wherever the fighter actually is

**Owner decision, 2026-08-28.** A queued combination's `commit_at` may arrive with the fighter
somewhere other than where the previous ghost was aimed — physics does not track a plan exactly, and
being hit is out of distribution for the policy (`CLAUDE.md`, known traps). A combination starting
off-target still runs, and still reaches its ghost.

Mechanically: `runtime/warp.py::warp()` is called **again, for real**, at `commit_at`, with the
fighter's *true* `(p₀, h₀)` at that tick as the anchor — not the position that was assumed when the
player placed the ghost. The residual (leftover travel to the ghost) is recomputed from that real
anchor, and the drift ramp runs however fast reaching the ghost within `record.duration_ticks` now
requires, **even above `APPROACH_SPEED_M_S`**. Nothing is clamped and nothing raises
(`speed_ceiling=None`); a fighter knocked far off tracks badly under physics, and physics decides the
rest.

**The achieved drift speed is recorded in the match record**, because it is now the signal that
tells a replay "this move was asked to run further than it was built for."

**The speed ceiling therefore validates the player's placement at issue time only.** When the player
commits, `warp()` is called with the *projected* anchor — wherever the fighter is expected to be when
this commit becomes current — and the default `speed_ceiling=APPROACH_SPEED_M_S`; exceeding it raises
`WarpError` and the client shows "can't get there" before the player pays for the commit. That check
says nothing about what happens if the fighter is somewhere else by the time the commit actually
runs — it cannot, because the real anchor does not exist yet. See "Feasibility", below.

### The drift gain

`DRIFT_GAIN = 0.803` — measured 2026-08-28 across 9 combinations spanning all three corpus families
at four drift distances, 36 (combination, distance) pairs: MotionBricks covers only about 80 % of a
commanded residual, so `warp()` divides the residual by it before ramping, asking for more so the
fighter actually lands on the ghost rather than 20 % short of it. Full method and the corrected
metric (`incremental_gain`, isolating the residual's own coverage from the recording's own travel)
are in `docs/perf/2026-08-28-drift-gain.md`.

**The old open-ended approach hid this.** An approach that walks until `has_arrived` says so cannot
undershoot — it just takes longer. Removing the approach is exactly what turns a hidden 20 %
shortfall into a visible one, so the gain correction is not new caution added on top of 3.0; it is
the price of the approach's removal, paid explicitly instead of by an approach nobody had to give a
length.

The gain applies to the **residual only, never to the recorded footwork** (D4, restated): gaining the
recording too would re-scale a 2 cm weight-shift the same way it re-scales a 2 m drift — exactly the
distortion D4 exists to prevent.

### The style is `walk_boxing`

Every leg runs in `walk_boxing` (`sequence.COMBINATION_CONTEXT`), not `walk` — measured 2026-08-28:
`walk` permits only 6–11 tokens (`narrow_allowed_tokens` raises on 12) while a recorded leg may run
to 16 tokens, so `walk_boxing` is the only clip whose mask can express every forced leg length.

**This resolves `CLAUDE.md`'s sideways-gait warning, rather than ignoring it.** The warning is that
`walk_boxing` gates out upstream's lateral blendspace (`walk_left` / `walk_right`), so a fighter
travelling in it has no sideways gait — true, and the reason 2.1 moved the *approach's* ambient
context to `walk`. It does not apply here, because a combination's travel no longer comes from the
gait remap at all: it comes from `target_position`, re-derived every leg from the warp. Measured: a
ghost 1 m to the side is reached as well as one 1 m ahead — **0.79 m either way**. The trap 2.1 fixed
and the mechanism 3.0 uses are different enough that the same clip is safe for one and was not for
the other.

### The keyframe is pinned; the hole in front of it shrinks

**Owner framing, 2026-09-03:** *time in MotionBricks is a continuous array that has to be filled
where there are holes, and the keyframes you put in it stay in place while the array moves forward.*

That is the model, and 3.0 through 3.1 violated it. MotionBricks fills a hole between its 4 context
frames and a target at the plan's last token. `CombinationRunner.intent_for` asked for
`leg.horizon_tokens` — the leg's **full** length — on every replan, so the target was re-aimed
`REPLAN_DT × GENERATOR_HZ` = 15 frames further out each time. A 12-token leg put its keyframe at
frame 48, then 63, then 78:

| replan at frame | requested | keyframe lands at |
|---|---|---|
| 0 | 12 tokens | 48 |
| 15 | 12 tokens | 63 |
| 30 | 12 tokens | 78 |

The keyframe **receded and never arrived at its boundary**. The fighter converged on it — which is
why holding a pose worked at all — but the recorded rhythm was stretched and the pose only ever
partially attained. This is the "motions broken in pieces" defect the owner reported.

3.2 asks for the hole that is actually left: `ceil(boundary_tick − tick)` in tokens. The keyframe
stays pinned in absolute time while the window slides and consumed frames are discarded. Three
regimes follow, each meaning something different:

| remaining hole | behaviour | why |
|---|---|---|
| `> MAX_TOKENS` | request `MAX_TOKENS`, **no pose target** | the keyframe is not reachable inside one plan, so nothing is aimed at it: ambient `walk_boxing` shaped only by the leg's `target_position` |
| `MIN_TOKENS`–`MAX_TOKENS` | request the remainder, keyframe as target | the real in-between; where the recorded pose lands |
| `< MIN_TOKENS` | **do not replan at all** | no plan that short exists, so re-filling would only push the keyframe past its own boundary |

**`ceil`, not `round`, and it is load-bearing.** A plan ending short of its boundary leaves the play
cursor clamped on its last frame, and `get_context_mujoco_qpos` then returns four copies of it
(`full_agent.py:503-521`) — a zero-velocity context telling the model the fighter is standing still
while it is mid-combination. `ceil` guarantees the plan always reaches at least the boundary,
overshooting by less than one token, which the next leg's replan writes over.
`tests/test_reference_replan_flag.py` asserts the context never collapses, and is verified to fail
under `floor`.

**This withdraws 3.0's "consumed exactly" contract.** That contract was specified and never
implemented: only the forcing half was built, and `tests/test_reference_forced_length.py` explicitly
asserts no replan is ever forced. 3.2 delivers the same guarantee — a leg lasts its recorded
duration — through a mechanism that forces nothing, so all three of those regression tests keep
passing unchanged. The three defects 3.0 feared are not reintroduced, because the machinery that
caused them is not reintroduced.

### A leg is no longer a plan

The corollary, and the change with the widest blast radius. Until 3.2 `leg_tokens ≤ MAX_TOKENS` held
**because a leg was exactly one plan**. A long leg is now an untargeted phase *plus* a landing plan,
so `MAX_TOKENS` bounds a **plan** and `MAX_TARGET_LEG_TOKENS` (24 tokens, 3.2 s) bounds a **leg**.
Three places enforced the old identity and all three moved: `segment.leg_tokens`' raise *and its
`min(MAX_TOKENS, …)` clamp*, and `combination_record`'s validator. The clamp was the dangerous one —
left alone it silently truncates every merged leg back to 16 tokens and the library looks rebuilt
while being unchanged.

### A queue, bounded, in order — unchanged

A commit is accepted while fewer than `MAX_OUTSTANDING_COMMITS` (**5**) are unfinished, and appended;
beyond that it is refused, with an error saying when a slot next frees. Unchanged since 1.0
(`docs/ASSUMPTIONS.md` §A19); the 2.2-era caveat about what five commits now cost in wall time still
applies and needs re-measuring against combination lengths rather than walk-plus-pose lengths — a
`M4-T4` question, not an `M6` one.

### No cancellation, of anything — unchanged

Once issued, a commit will execute — including one still waiting its turn, including a combination
that starts off-target and has to drift hard to reach its ghost. This is the founding rule of the
game held to its full extent, unchanged since 1.0, and it is what makes a deep queue a risk rather
than a free lookahead.

### Staging never touches a fired commit — unchanged in rule, changed in what is staged

`stage()` is legal at every tick, including mid-move, and changes only the channels it is passed. A
`StagedIntent` is immutable once returned, so a value read earlier cannot change underfoot. What is
staged is now a **combination** and a **ghost position**, rather than a pose, a bounded adjustment
and a placement that also carried a heading — see "Channels", below. Unstaging the combination
selection is not a cancellation, because nothing has fired.

### Admission is enforced at construction — unchanged in rule, extended in scope

`IntentTimeline` (or its `M6` successor) validates every combination against `spec/combination.md`'s
admission rule when it is built, and by default refuses one that is not `"admitted"`. The Studio
passes `require_admitted=False` so a draft combination can be rehearsed before it has been measured;
a match never does. Admission for a combination additionally checks **duration** — within one token
over the whole combination (D2) — which a single pose record never needed.

---

## Channels

| Channel | What it carries | Replaces |
|---|---|---|
| `combination` | which recorded combination — 3–6 keyframes with recorded timing, from the whole shared library (`D6`) | `pose_slot` |
| `ghost_position` | where the combination's **last keyframe** must land: world `(x, y)` only | `placement` (which also carried a player-set heading) |
| `commit_at` | tick the move should begin, in 50 Hz ticks | ours, unchanged |

Two channels are gone outright:

- **`adjustment`** — a combination carries no adjustment envelope (`spec/combination.md`: "not a
  superset of a pose record ... carries no `adjustment_envelope`"). Bounded live deviation was a
  single-pose feature; a combination's expressiveness comes from *which* combination is selected,
  not from bending one.
- **`context`** — the style preset is no longer player-facing. Every leg runs `walk_boxing` (see
  above), so there is nothing left for this channel to choose.

### Ghost heading is derived, not staged

`Placement.heading` is gone as a player-set field (see "Removed at 3.0"), and the player never sets a
heading directly. **Since 3.1 it is derived by facing the opponent** (owner, 2026-09-03):
`runtime/warp.py::ghost_heading(ghost_position, opponent_position)` is the bearing from the ghost to
the opponent, and while the combination runs the same bearing is re-measured **every tick** from the
opponent's live position (`runtime/fight.py::FightWorld.facing_angle`) and used for both headings
that reach MotionBricks — the target frame's `target_heading` and the `facing_angle` control signal.
3.0 derived it from the recording instead (`h₀ + recorded_heading_delta`, per keyframe
`heading_i = h₀ + heading_offset_i`); see "The ghost", below, for what that cost.

### Coordinates, unchanged

World `(x, y)` on the ground plane, the same frame the arena, the ghost and the client all use. The
generator still does not plan in that frame; `runtime/warp.py` owns the conversion the way
`runtime/generator.py` and `runtime/fight.py::to_generator_frame` did for a single placement.

---

## The ghost

Unchanged in purpose from 1.0: the staged intent is shown to its own player as a ghost of their
fighter, computed once by the host from the combination record — never re-derived in the client, for
the same reason a client-side kinematics pass was always rejected: a preview that ran its own
implementation of "what this move looks like" would be free to disagree with the move that actually
happens.

**What changed is what the player controls.** Before 3.0 the player set both the ghost's position
and its heading. The player drags **position only** (D5): the heading is derived, and dragging the
ghost to a new position moves where the *last keyframe* lands without the player ever aiming it.

**Where the derived heading points — reversed at 3.1 (owner, 2026-09-03).** It points at the
opponent, from wherever the fighter is, on every tick. 3.0 derived it from the recording instead, on
D5's reasoning that the corpus's travelling combinations turn by up to **158°** and a target-facing
ghost would discard the turn that *is* the motion. That reasoning holds for the *recording* and
fails for the *fight*: a boxer is turned towards the fighter they are boxing at all times, and a
fighter that inherits a recorded 158° turn ends the move facing the ropes with its back to the
opponent, which is worse than any turn it preserves. So the recorded turn still moves the body — it
is in the keyframe joint angles and in the footwork — it just no longer aims the fighter.

`facing_angle` (where the fighter looks) and `movement_angle` (where it travels) are still different
signals per leg — `CLAUDE.md`'s named trap — and the difference is now larger, not smaller: travel
comes from the warped footwork, facing comes from the opponent. Only a leg that does not travel at
all (`warp.STILL_LEG_M`) takes the bearing as its movement direction too, having no direction of its
own.

**The opponent never sees it.** Unchanged from 1.0: `WORKPLAN` M4-T1's "no HUD on the fighters — the
windup is the only cue" is unaffected — staging is private, and a committed move becomes public only
once it is executing.

---

## Feasibility

One guard now, where 1.0–2.2 needed two.

**Reach.** Under 1.1–2.2, "anywhere in the ring is reachable, and distance costs time" — true because
the approach was open-ended. It is no longer unconditionally true: a combination has a **fixed**
duration, so a ghost placed further than the combination can drift to within that duration, at
`APPROACH_SPEED_M_S`, is genuinely out of reach *for that combination*, not merely slow. This is
`runtime/warp.py::WarpError`, raised at issue time (`speed_ceiling=APPROACH_SPEED_M_S` by default).

Because reach now depends on the combination's own recorded duration, **the boundary differs per
combination and moves as the player pages through the library** — a jog combination with several
seconds to work with reaches further than a shadow-boxing combination with barely more than a second,
for the same drift speed. A client renders this the way 1.1's `reach_m` once did per slot, but
recomputed per combination rather than authored once.

**Placement, once issued, is not re-checked.** As "Off-target execution" states: the ceiling
validates the player's placement at the moment they commit, against the anchor assumed then. What
actually happens when the commit runs is not re-validated and cannot raise — a fighter dragged off
its intended path by a hit is not a feasibility failure, it is the match.

**Pose — narrower than before, because there is no live adjustment.** Each keyframe's `joint_angles`
is admitted exactly as a `pose_record.md` pose is: offline, on measured tracking error, because
MotionBricks remains kinematic and physically unaware and will happily emit self-penetrating or
torque-infeasible targets (`CLAUDE.md`, known traps). There is no envelope to validate live, because
there is no live adjustment (see "Channels"); a combination's only per-play variable is the ghost,
and that is exactly what the reach guard above covers.

---

## Removed at 3.0

Deleted, not deprecated — `CLAUDE.md` prefers deleting to disabling, and every one of these existed
only to serve the approach or the counted dwell, both gone.

| Removed | What replaced it |
|---|---|
| `TRAVEL_CONTEXT` | `sequence.COMBINATION_CONTEXT` (`walk_boxing`, always — see above) |
| the approach (walk-to-placement phase) | nothing — a combination starts in place (D3) |
| `approach_timeout_ticks`, `DEFAULT_APPROACH_TIMEOUT_TICKS` | nothing needed: `end_tick` is exact arithmetic, not a race against a runaway approach |
| `has_arrived` | nothing needed: there is no arrival event to test for |
| `has_settled` | nothing needed: `end_tick = commit_at + record.duration_ticks` |
| the counted dwell, `POSE_DWELL_TICKS` | the combination's own recorded pauses — legs with a small or zero `root_offset` are authored motion, not a wait |
| `MAX_DWELL_TICKS` | nothing needed: nothing can hang, because nothing is being waited for |
| `Placement.heading` as a player-set field | `runtime/warp.py::ghost_heading` — derived, never chosen (D5); since 3.1 derived by facing the opponent |
| the per-commit `slot` / `adjustment` as the unit of selection | the per-commit `combination` (a `CombinationRecord`) plus its anchor — see "Channels" |

`arrived` and `completed_by` go with the fields they distinguished, for the same reason: there is
exactly one way a 3.0 commit ends — its recorded duration elapses — so nothing is left to record
about *how* it ended.

`APPROACH_SPEED_M_S` itself **survives, repurposed**: it was the control that governed how far an
approach walked in a given time; it is now only the feasibility ceiling `warp()` checks a placement
against at issue time (see "Feasibility"). It measures the same physical quantity — how fast a
fighter can sustain travel — and nothing about that measurement changed; only what asks the question
did.

---

## Unchanged at 3.0

Worth restating so a reader of this file alone knows what survived, rather than inferring it from
absence:

- **No cancellation, of anything**, including a queued commit — the founding rule of the game.
- `MAX_OUTSTANDING_COMMITS` = 5.
- `COMMIT_HORIZON_TICKS` as a **floor**, not a delay — a commit queued behind a running move starts
  the instant that move ends, because the readable window has already elapsed while it waited.
- `OPENING_STANCE_CONTEXT` before the first commit of a round.
- Ticks are 50 Hz (`TICK_HZ`).
- Placement — now the ghost's position only — is MuJoCo world `(x, y)`.
- The Studio stays offline; `require_admitted=False` is still how it rehearses drafts.
- `test_generator_pose_override.py` against a pristine agent, and the whole runtime-installed
  override mechanism it protects (`CLAUDE.md` invariant 3) — combinations still reach MotionBricks
  through the same `_apply_target_pose_override` hook, one keyframe at a time.

---

## Impact on the plan

| Task | Change |
|---|---|
| `M6-T4` | this document |
| `M6-T5` | implements it: rewrites `runtime/intents.py`'s `Commit`, `stage`, `generator_intent` against `warp.py` / `sequence.py`; bumps `SPEC_VERSION` to `"3.0"` |
| `M6-T6` | the three forced-plan regression tests named above, against `runtime/reference.py` |
| `M4-T4` | still owns ring size and queue depth; now measured against combination lengths, not walk-plus-pose lengths |
| `M5-T2` | `TARGET_COMMIT_RATE` was calibrated against the 1.0–2.2 model and needs re-measuring again |
| Client / protocol (`D6`, `spec/protocol.md`) | `M6` Phase 3: whole-library paging, no loadout, ghost position only. Implemented; `Loadout` is deleted (task A6) |

---

## Changelog

- **3.2** (2026-09-03) — **a keyframe is pinned in absolute time.** `intent_for` requested the leg's
  full length on every replan, so the target was re-aimed 15 frames further out each time and never
  landed on its boundary — the "motions broken in pieces" defect. It now requests
  `ceil(boundary − tick)` in tokens, drops the pose target while the hole exceeds `MAX_TOKENS`, and
  stops replanning below `MIN_TOKENS` so the last plan lands exactly. **Withdraws 3.0's "consumed
  exactly" contract**, which was specified but never implemented; the same guarantee is met by a
  mechanism that forces nothing, so `tests/test_reference_forced_length.py` passes unchanged.
  A leg is no longer one plan, so `MAX_TOKENS` stops bounding a leg and `MAX_TARGET_LEG_TOKENS`
  does. The library is rebuilt on sparse targets: 174 combinations of 2–3 keyframes, median leg
  15 tokens (2.00 s) against 9 (1.20 s), 39 % of legs now longer than one plan.

- **3.1** (2026-09-03) — **a fighter always faces its opponent.** Owner decision, reversing the half
  of `D5` that said where a derived heading points (the half that says the player never sets it
  stands). Both headings that reach MotionBricks — the *target frame's* `target_heading` and the
  `facing_angle` control signal — are the bearing to the opponent, measured in the world and
  re-measured every tick, because the opponent moves while a 2.4–7.6 s combination runs. A leg's
  recorded heading is no longer what a fighter looks along; it survives only where there is no
  opponent to face (the Studio's rehearsal, the warp tools), and `runtime/sequence.py`'s
  `CombinationRunner.intent_for(tick, facing_angle)` is the single place the override happens. A
  still leg (`warp.STILL_LEG_M`) takes the bearing as its `movement_angle` too. `ghost_heading` now
  takes the ghost and the opponent's positions rather than a record and a heading, and the clients
  draw the same bearing for their preview (`client/app.js`, `client/sparring.js` via
  `ring.fighterPosition`). What did *not* change: the wire (`spec/protocol.md` stays 0.6 — `welcome`
  still carries each combination's `heading_delta`, which describes the recording and no longer
  drives a preview), the warp's geometry, the queue, and the rule that the player never sets a
  heading. `SPEC_VERSION` moves to `"3.1"`; the tests that pair the two move with it.
- **3.0** (2026-08-28) — **a commit is a combination, and the approach is gone.** Replaces the single
  authored key pose with a 3–6-keyframe recorded combination (`spec/combination.md` 0.1) selected
  from a library of ~120 built from the mocap corpus under `motions/`. A commit starts **in place**
  (D3) instead of walking to a placement; its span is `end_tick = commit_at + record.duration_ticks`,
  known the moment `commit_at` is, reversing 2.2's "the queue is not a schedule" — safe because an
  approach's length depended on distance under physics and could not be known in advance, and a
  combination's length is its recording's, which can be. The ghost's position is player-set; its
  heading is derived from the fighter's own heading plus the combination's recorded turn and is never
  chosen (D5) — the corpus turns by up to 158°, which a target-facing ghost would discard.
  Intermediate keyframes keep their recorded footwork at true size and ramp the leftover travel on
  elapsed time, not proportionally (D4) — proportional scaling needs 30–141× on shadow-boxing
  combinations, turning a 2 cm weight-shift into a 2.8 m lurch. A combination that starts
  off-target — because physics did not track the previous move exactly — is re-warped from the
  fighter's real position and reaches its ghost anyway, running whatever drift that needs even above
  `APPROACH_SPEED_M_S`; nothing is clamped and nothing raises, and the achieved drift speed is
  recorded (owner, 2026-08-28). The generator covers only `DRIFT_GAIN` = 0.803 of a commanded
  residual (measured, `docs/perf/2026-08-28-drift-gain.md`), a shortfall the old open-ended approach
  hid by walking until it arrived; `warp()` now corrects for it explicitly, on the residual only.
  Every leg runs `walk_boxing`, because `walk` cannot express the 12–16-token legs the corpus needs —
  this resolves `CLAUDE.md`'s sideways-gait warning rather than reintroducing it, because travel now
  comes from `target_position`, not the gait remap. Plan lengths are forced again, per leg, which is
  what makes a leg's duration hold; `runtime/reference.py` names three defects that forcing caused
  before, which `M6-T6` now guards as regression tests. Removed: `TRAVEL_CONTEXT`, the approach
  itself, `approach_timeout_ticks` / `DEFAULT_APPROACH_TIMEOUT_TICKS`, `has_arrived`, `has_settled`,
  `POSE_DWELL_TICKS`, `MAX_DWELL_TICKS`, `Placement.heading` as a player-set field, and the
  per-commit `slot` / `adjustment` as the unit of selection. `D6` (no loadout, whole library,
  nine-per-page paging) was owner-agreed but a later phase as of this entry — `Loadout` stayed
  unchanged in code until then. Designed in
  `docs/superpowers/specs/2026-08-27-motion-combinations-design.md`, decisions D1–D6;
  `SPEC_VERSION` in `intents.py` moves to `"3.0"` in the same change as the implementation
  (`M6-T5`), and a test pairs the two.
  **`D6` landed and `Loadout` was deleted in `M6` Phase 3 (task A6), 2026-08-28** — see this
  file's own Status note above. Every remaining importer (tools, server, client, tests) was
  ported to the whole shared library in the same phase; nothing in the codebase names `Loadout`
  any more.
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
