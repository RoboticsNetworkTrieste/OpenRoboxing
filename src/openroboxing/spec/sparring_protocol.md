# sparring_protocol.md — client ↔ sparring host

Version **0.3** · created 2026-08-17 · design: `docs/superpowers/specs/2026-08-17-sparring-tool-design.md`
· ported for motion combinations 2026-08-28 (`spec/intent.md` 3.0, B3)

The sparring bench (`tools/serve_sparring.py`, default port **8081**) runs the core motion stack
outside a match: one player fighter (red), one passive sacco (blue, `IdlePilot`), no rounds, no
scoring, no match record. This spec covers what crosses the client ↔ host boundary; everything it
does not restate is inherited from `spec/protocol.md` 0.6.

Inherited unchanged:

- the **binary frame** format (`ORBO` header + 7 floats per body), `GET /scene.json`,
  `GET /meshes.bin`;
- the staging subset of client messages: `intent`, `commit`, `clear`, `ping` — validated by the same
  `server/protocol.py` code, applied to the **red** seat only. `stage` and `place` do not exist any
  more at 0.6 — a combination has no slot for the first to name and its ghost carries no heading for
  the second to set (`spec/intent.md` `D5`/`D6`) — so 0.6 collapsed them into one `intent` message,
  and the bench inherits that collapse rather than keeping its own copy of the old pair;
- the host-authoritative rule. The bench adds controls a match must never have (pause, reset,
  teleport, knobs); it stays the case that the client decides nothing about what a commit means.

---

## Client → host (sparring additions)

```jsonc
{"type": "pause"}                                        // stop stepping; state still served
{"type": "resume"}
{"type": "pause_on_fall", "on": true}                    // auto-pause when a root height < 0.4 m
{"type": "reset", "seed": 1234}                          // seed optional; omitted = keep current
{"type": "teleport_sacco", "x": 1.0, "y": 0.5, "heading": 3.14}
```

Unknown types, bad coordinates and commits into a full queue produce an `error` message exactly as
in the match protocol. The queue bound here is **10** (`SPARRING_MAX_OUTSTANDING`, bench default —
the match canonical is 5, and the knob block below always reports the difference). Like the match
host (`server/host.py::MatchHost.handle`), a `commit` beyond a combination's own `reach_m` is refused
synchronously, before it ever queues — the bench mirrors the match's feasibility check rather than
inventing a looser one, because the whole point of a bench is that what it refuses is what a match
would refuse.

## Host → client

`state` messages are the red seat's own view (full queue — there is no opponent to hide from), plus
`error` / `pong` as in the match. Sparring adds one message, at **10 Hz**:

```jsonc
{"type": "debug",
 "tick": 4812, "paused": false,
 "machine": "RUNNING",                     // OPENING | WAITING | RUNNING | HOLD
 "commit_ordinal": 3,                      // index into the session's commit log, or -1
 "queue": [ /* full spans: combination, ghost, issued_at, commit_at, end_tick, executing —
              `server/protocol.py::queue_entry`'s own shape — plus the bench's own additions:
              drift_speed_m_s, leg_index, leg_count. null means "not yet", never zero. */ ],
 "plan_ghost": {"x": 1.1, "y": -0.3, "z": 0.79, "heading": 1.2, "angles": {"...29 by name...": 0.0}},
 "trail": [[1.1, -0.3], ...],              // root path, now -> plan tail, world frame
 "series_head": {"err_mean": 0.04, "err_max": 0.21, "dist": 1.3, "dist_plan": 0.2,
                 "err_by_joint": {"...29 by joint name...": 0.0},   // feeds the heatmap
                 "root_h_red": 0.79, "root_h_blue": 0.79, "step_ms": 7.7},
 "replans": [[4790, false, 44], ...],      // (tick, forced, plan_frames), recent window
 "keyframe_events": [[4655, 2, 1, 0.031, 0.089], ...],  // (tick, ordinal, leg, err_mean, err_max)
 "knobs": {"replan_dt": {"current": 0.5, "canonical": 0.5}, "...": {}},
 "recording": {"start_tick": 0, "end_tick": 4812},
 "drafts_allowed": true,                   // this session's require_admitted, inverted — see below
 "pilot_error": {"tick": 4800, "message": "cannot commit: no pose is staged"}}  // or null
```

**`plan_ghost` is not `queue[].ghost`.** The two are different things that 0.1-0.2 could get away
with calling the same name (`ghost`) because the committed target had no other name yet;
`spec/intent.md` 3.0 gave the player's committed target the name `ghost`, so this spec renames the
*visualisation* — the reference frame the encoder is chasing (see "The visualisation transform",
below) — to `plan_ghost` rather than let a wire message carry two unrelated things under one key. A
queue entry's own `ghost` is `server/protocol.py::queue_entry`'s field: the world `(x, y)` the
combination's *last keyframe* must land on, unchanged in shape from the match protocol.

**`drafts_allowed`** says whether this session's `IntentTimeline`s were built with
`require_admitted=False` (`spec/intent.md` "Admission is enforced at construction": *"The Studio
passes `require_admitted=False` so a draft combination can be rehearsed before it has been
measured... a match never does."*). The bench is exactly that Studio-side caller, and every combination
on disk today is `admission="draft"` (telegraph and tracking error are unmeasured, scheduled work),
so a session run the default way is rehearsing entirely unmeasured motion. A client shows this
persistently rather than let an unadmitted combination look indistinguishable from a match-ready one
— the bug this spec's own `strict_dumps` history warns about is exactly this shape: something true
about the session that was never wrong on the wire, only silent.

**`pilot_error`** is the red pilot's most recent refusal. The pilot applies queued messages a tick
after the socket received them, so a refusal cannot be a direct reply — it is latched here instead,
and a client shows it when the message changes. Without this a refused commit is indistinguishable
from a broken bench.

**Every socket drives.** There is no controller seat and no viewer role: the bench is one human on
localhost, and a gate here failed on the most common action (a page refresh can open the new socket
before the server processes the old one's close, permanently locking the live page out — found
2026-08-17). A match protects its seats; a bench protects a refresh.

**Every number on the wire is finite.** A missing value is `null`, never `NaN` — Python writes bare
`NaN` happily and `JSON.parse` refuses it, taking the whole message with it. The bench serialises
every HTTP body and every websocket message through `strict_dumps` (`allow_nan=False`), so a
non-finite value fails loudly in the server's log instead of silently blanking a panel. 0.1 shipped
`NaN` in `dist` on every un-approached tick: the charts' fetch threw, an empty `catch` swallowed it,
and the strip read "NO RECORDING YET" for a whole session (found 2026-08-17).

### The state machine

Derived read-only from the red timeline (`server/sparring_tap.py::derive_machine_state`); deriving
never advances it — `generator_intent` is the timeline's clock and only the world calls it.

| State | Condition |
|---|---|
| `OPENING` | no commit has completed and none is current |
| `WAITING` | a commit exists and is unfinished, but none is executing (horizon window, or queued) |
| `RUNNING` | a commit is executing — `Commit.is_executing(tick)` |
| `HOLD` | queue drained; a completed commit's intent is held |

**Two states, not four.** 0.1-0.2 split a running commit into `APPROACH` (walking to a placement)
and `DWELL` (holding a struck pose once arrived). `spec/intent.md` 3.0 removed the phase distinction
those names described — a combination has no walk-then-throw split, it is one continuous piece of
recorded motion — so there is exactly one running state left, `RUNNING`. What is still worth showing
about *where* inside that one state a commit is is not a two-way phase flag any more but the queue
entry's own `leg_index` / `leg_count` (below): a combination has 2-5 legs, and mid-combination is a
number, not a phase.

### The visualisation transform

The reference motion lives in the **generator frame**, and the encoder never consumes its root
position — so the reference has no canonical world position. For drawing only, the tap maps it as:

```
world_xy(k)  = robot_xy + R(apply_yaw) · (ref_xy[k] − ref_xy[tick])
world_yaw(k) = apply_yaw + yaw(ref_quat[k])
```

— the displacement the reference intends between now and frame *k*, applied from where the robot
really is: the coherent inverse of `fight.to_generator_frame`'s rotation. Joint angles and height
pass through unchanged. Unchanged in mechanism from 0.2 — 3.0 changed *what* is warped into a plan
(a combination's legs, not an approach's), not how a plan is drawn.

- `plan_ghost` is the transformed frame at `tick + LOOKAHEAD_TICKS` (45): **what the encoder is
  looking at**, i.e. what GEAR-SONIC is chasing right now.
- `trail` is the transformed root path from `tick` to the end of the **plan horizon**,
  `PLAN_HORIZON_TICKS` = 45 + 20 = 65 ticks — the lookahead plus the generator margin, which is
  exactly how far `ReferenceStream.ensure` fills. Beyond it the reference is not a plan, and a
  scrub that read to the end of its recording drew the rest of the session's path across the ring
  (0.1's "strange trail", found 2026-08-17). Live and scrub use the same bound, so the two modes
  cannot disagree about what a plan is.
- **Known limitation (0.1, unchanged):** the plan ghost is posed yaw-only — the client's shadow FK
  takes a heading, not a full quaternion — so reference pitch/roll is not shown.

## HTTP API

| Route | What |
|---|---|
| `GET /api/frame/{tick}` | scrub: `{"frame": "<base64 ORBO frame>", ...that tick's debug row...}`. The frame is rebuilt by `mj_forward` over the recorded qpos and packed by the same `Scene.pack`; the plan ghost and trail are recomputed from the recorded reference rows. 404 outside the recording window. |
| `GET /api/series?from=&to=&stride=` | downsampled traces for the charts: `tick[]`, `err_mean[]`, `err_max[]`, `dist[]`, `root_h_red[]`, `root_h_blue[]`, `step_ms[]`, `machine[]`, plus `replans` and `keyframe_events` in the window |
| `GET /api/session.npz` | the whole recording, `numpy.savez_compressed` — offline analysis |
| `GET /api/knobs` | every knob as `{"current": x, "canonical": y}` |
| `POST /api/knobs` | set any subset, e.g. `{"drift_gain": 0.75}`; response = the full updated block; unknown name or non-positive value → 400 |

## Knobs

| Knob | Canonical | Mechanism | Applies |
|---|---|---|---|
| `replan_dt` | 0.5 s (`reference.REPLAN_DT`) | red `ReferenceStream.replan_dt` | next replan |
| `horizon_ticks` | 30 (`COMMIT_HORIZON_TICKS`) | red timeline attribute | live |
| `max_outstanding` | 10 bench / 5 match | red timeline attribute | live |
| `drift_gain` | 0.803 (`constants.DRIFT_GAIN`) | **process-local override** of `runtime.warp.DRIFT_GAIN` | the next commit that *starts* — see below |

The dwell override 0.2 documented here — a process-local rewrite of `runtime.intents.
POSE_DWELL_TICKS` — went with the mechanism it tuned (`spec/intent.md` "Removed at 3.0": the counted
dwell is gone, along with `arrival_radius_m` and `approach_leg_m`, which governed a walk-to-placement
phase that no longer exists). `drift_gain` is 3.0's equivalent live-tunable correction — measured, not
chosen (`docs/perf/2026-08-28-drift-gain.md`), and exactly the kind of number a bench exists to
perturb. It is a module-global assignment made by sparring code inside the sparring process; the core
file (`runtime/warp.py`) is not edited, no other process is affected, and this table is its
declaration.

**Unlike 0.2's dwell knob, this one does not rewrite anything in flight.** `warp()` runs exactly once
per commit, at the tick it starts (`runtime/intents.py::IntentTimeline.generator_intent`), and its
result — the legs — is fixed from then on; turning this knob changes only what the *next* commit to
start computes its residual with. A running commit's legs do not move underfoot.

Every knob response carries the canonical value so a UI can mark any deviation.

## Recording

The tap holds a ring buffer (default 10 minutes at 50 Hz); when full, the oldest ticks fall off and
`recording.start_tick` moves. Event logs (`replans`, `keyframe_events`) keep absolute ticks and are
filtered to the window on the way out. A sparring session writes nothing to disk except the `.npz` a
user explicitly downloads.

### Two distances, not one — reframed for a ghost that is placed once and drifted toward, not walked to

`dist` is the **body's** distance to the commit's `ghost` — the pelvis under physics. `dist_plan` is
the **plan's**: the reference frame the encoder is chasing, against the same `ghost`, through the
same viz transform. Both are `null` when nothing is executing.

**What this pair diagnoses changed with the approach's removal.** 0.1-0.2 used it to tell an
*arriving* approach from a *stalled* one — a binary, watched-for event. 3.0 has no arrival to watch
for: `ghost` is where a combination's *last* keyframe should land, and every leg's own target already
converges toward it on elapsed time (`runtime/warp.py::warp`), so `dist` is expected to fall roughly
monotonically across a commit's whole span rather than snap to zero at one moment. What the pair still
answers, and the reason it survives the port: whether the **body** tracks that convergence as well as
the kinematic **plan** does — `dist_plan` decreasing smoothly while `dist` lags or plateaus is the
same *plan-in, body-out* signature 0.1-0.2 measured, now read as a trend across a commit rather than
a pass/fail at one radius.

**The arrival-radius overlay is gone, deliberately.** 0.2's charts drew `ARRIVAL_RADIUS_M` across the
distance strip because it was the threshold that decided whether a commit landed or timed out. There
is no such threshold at 3.0 — `end_tick` is exact arithmetic, not a race against a radius — so a chart
that still drew one would be showing a number nothing tests any more. Dropped rather than kept as
decoration (`CLAUDE.md`: delete, don't disable).

### Per-keyframe tracking error: the settle test's replacement

0.1-0.2 also measured, per placement, whether the body's tracking error stopped improving — the
counted dwell's read on "has the strike settled". `spec/intent.md` 3.0 deletes that test along with
the dwell it gated (`has_settled`, `POSE_SETTLE_IMPROVEMENT_RAD`'s use in the runtime): a leg ends
when its own recorded duration elapses, not when the body stops closing on it, so there is no settle
event left to watch for.

**What is still genuinely measurable, and interesting, is the tracking error at the instant each leg
*does* end** — a keyframe boundary is now exact arithmetic (`runtime/sequence.py::CombinationRunner.
leg_index`), so the bench logs `(tick, commit_ordinal, leg, err_mean, err_max)` to `keyframe_events`
the tick a commit's live leg advances, recording how close the body actually was to the recorded pose
the plan had just finished asking for. This is the direct 3.0 answer to "did the body settle into the
pose": not a test with a pass/fail radius, but the honest number at the one tick that used to be
guessed at by watching for convergence to stop.

## Changelog

- **0.3** (2026-08-28) — **ported to `spec/intent.md` 3.0 (B3).** `stage`/`place` are gone with the
  match protocol's own collapse into one `intent` message; a queue entry carries `combination` and
  `ghost` instead of `slot` and `placement`, and drops `arrived` / `completed_by` — there is no
  approach to time out and no dwell to distinguish, so a commit is not-started, `RUNNING`, or
  finished. The five-state machine becomes four (`APPROACH`/`DWELL` collapse to `RUNNING`); a queue
  entry instead carries `leg_index`/`leg_count`, and `drift_speed_m_s` — how hard an off-target
  commit had to run to still reach its ghost (`runtime/fight.py::_record_drift`), 3.0's own new
  failure mode now that reaching a target is drift rather than an open-ended walk. The plan
  visualisation is renamed `plan_ghost` because `ghost` now means the player's committed target.
  `dist`/`dist_plan` survive, reframed: not an arrival test but a trend the whole commit long, and
  the arrival-radius overlay they used to be checked against is deleted (`CLAUDE.md`: delete, don't
  disable — nothing tests that radius any more). New: `keyframe_events`, logging tracking error at
  each leg boundary — the concrete answer to "did the body settle into the pose" now that settling is
  not watched for. The knob table drops `arrival_radius_m`, `approach_leg_m`,
  `approach_timeout_ticks`, `pose_dwell_ticks` (their mechanisms are gone) and gains `drift_gain`, a
  process-local override of `runtime.warp.DRIFT_GAIN` in the same spirit as the old dwell override,
  but — unlike it — affecting only commits that have not yet started, never one already running.
  `drafts_allowed` is new on the `debug` message: every combination on disk today is `"draft"`
  (`spec/intent.md`: "The Studio passes `require_admitted=False`... a match never does"), and a bench
  session must say so on screen rather than let unmeasured motion look indistinguishable from a
  match-ready move.
- **0.2** (2026-08-17) — every number on the wire is finite (`null`, never `NaN`); the trail is
  bounded to `PLAN_HORIZON_TICKS` in both modes; a scrubbed tick carries its queue; `dist_plan`
  joins `dist` so an approach can be attributed to the plan or to the body; queue entries carry
  `arrived`.
- **0.1** (2026-08-17) — first version: sparring message set, debug message, state machine table,
  visualisation transform, HTTP API, knob table with the dwell-override declaration.
