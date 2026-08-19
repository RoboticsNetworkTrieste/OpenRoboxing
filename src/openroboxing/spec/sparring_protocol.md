# sparring_protocol.md — client ↔ sparring host

Version **0.2** · created 2026-08-17 · design: `docs/superpowers/specs/2026-08-17-sparring-tool-design.md`

The sparring bench (`tools/serve_sparring.py`, default port **8081**) runs the core motion stack
outside a match: one player fighter (red), one passive sacco (blue, `IdlePilot`), no rounds, no
scoring, no match record. This spec covers what crosses the client ↔ host boundary; everything it
does not restate is inherited from `spec/protocol.md` 0.5.

Inherited unchanged:

- the **binary frame** format (`ORBO` header + 7 floats per body), `GET /scene.json`,
  `GET /meshes.bin`;
- the staging subset of client messages: `stage`, `place`, `commit`, `clear`, `ping` — validated by
  the same `server/protocol.py` code, applied to the **red** seat only;
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
the match canonical is 5, and the knob block below always reports the difference).

## Host → client

`state` messages are the red seat's own view (full queue — there is no opponent to hide from), plus
`error` / `pong` as in the match. Sparring adds one message, at **10 Hz**:

```jsonc
{"type": "debug",
 "tick": 4812, "paused": false,
 "machine": "APPROACH",                    // OPENING | WAITING | APPROACH | DWELL | HOLD
 "commit_ordinal": 3,                      // index into the session's commit log, or -1
 "queue": [ /* full spans: issued_at, commit_at, strike_at, end_tick, arrived, slot, pose,
              placement — null means "not yet", never zero. `arrived` is the bench's own addition
              to the match's queue entry: false is a move whose approach ran out of time. */ ],
 "ghost": {"x": 1.1, "y": -0.3, "z": 0.79, "heading": 1.2, "angles": {"...29 by name...": 0.0}},
 "trail": [[1.1, -0.3], ...],              // root path, now -> plan tail, world frame
 "series_head": {"err_mean": 0.04, "err_max": 0.21, "dist": 1.3, "dist_plan": 0.2,
                 "err_by_joint": {"...29 by joint name...": 0.0},   // feeds the heatmap
                 "root_h_red": 0.79, "root_h_blue": 0.79, "step_ms": 7.7},
 "replans": [[4790, false, 44], ...],      // (tick, forced, plan_frames), recent window
 "knobs": {"replan_dt": {"current": 0.5, "canonical": 0.5}, "...": {}},
 "recording": {"start_tick": 0, "end_tick": 4812},
 "pilot_error": {"tick": 4800, "message": "cannot commit: no pose is staged"}}  // or null
```

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
| `APPROACH` | a commit is executing and `strike_at` is unset |
| `DWELL` | executing, `strike_at` set, `tick < end_tick` |
| `HOLD` | queue drained; a completed commit's intent is held |

### The visualisation transform

The reference motion lives in the **generator frame**, and the encoder never consumes its root
position — so the reference has no canonical world position. For drawing only, the tap maps it as:

```
world_xy(k)  = robot_xy + R(apply_yaw) · (ref_xy[k] − ref_xy[tick])
world_yaw(k) = apply_yaw + yaw(ref_quat[k])
```

— the displacement the reference intends between now and frame *k*, applied from where the robot
really is: the coherent inverse of `fight.to_generator_frame`'s rotation. Joint angles and height
pass through unchanged.

- `ghost` is the transformed frame at `tick + LOOKAHEAD_TICKS` (45): **what the encoder is looking
  at**, i.e. what GEAR-SONIC is chasing right now.
- `trail` is the transformed root path from `tick` to the end of the **plan horizon**,
  `PLAN_HORIZON_TICKS` = 45 + 20 = 65 ticks — the lookahead plus the generator margin, which is
  exactly how far `ReferenceStream.ensure` fills. Beyond it the reference is not a plan, and a
  scrub that read to the end of its recording drew the rest of the session's path across the ring
  (0.1's "strange trail", found 2026-08-17). Live and scrub use the same bound, so the two modes
  cannot disagree about what a plan is.
- **Known limitation (0.1):** the ghost is posed yaw-only — the client's shadow FK takes a heading,
  not a full quaternion — so reference pitch/roll is not shown.

## HTTP API

| Route | What |
|---|---|
| `GET /api/frame/{tick}` | scrub: `{"frame": "<base64 ORBO frame>", "debug": {…that tick's row…}}`. The frame is rebuilt by `mj_forward` over the recorded qpos and packed by the same `Scene.pack`; ghost and trail are recomputed from the recorded reference rows. 404 outside the recording window. |
| `GET /api/series?from=&to=&stride=` | downsampled traces for the charts: `tick[]`, `err_mean[]`, `err_max[]`, `dist[]`, `root_h_red[]`, `root_h_blue[]`, `step_ms[]`, `machine[]`, plus replan and commit events in the window |
| `GET /api/session.npz` | the whole recording, `numpy.savez_compressed` — offline analysis |
| `GET /api/knobs` | every knob as `{"current": x, "canonical": y}` |
| `POST /api/knobs` | set any subset, e.g. `{"arrival_radius_m": 0.3}`; response = the full updated block; unknown name or non-positive value → 400 |

## Knobs

| Knob | Canonical | Mechanism | Applies |
|---|---|---|---|
| `replan_dt` | 0.5 s (`reference.REPLAN_DT`) | red `ReferenceStream.replan_dt` | next replan |
| `horizon_ticks` | 30 (`COMMIT_HORIZON_TICKS`) | red timeline attribute | live |
| `max_outstanding` | 10 bench / 5 match | red timeline attribute | live |
| `arrival_radius_m` | 0.40 (`ARRIVAL_RADIUS_M`) | `SparringWorld.arrival_radius_m` | live |
| `approach_timeout_ticks` | ring-derived (`approach_timeout_ticks(ring)`) | red timeline attribute | live |
| `pose_dwell_ticks` | 74 (`POSE_DWELL_TICKS`) | **process-local override** of `runtime.intents.POSE_DWELL_TICKS` | live — rewrites in-flight `end_tick`s too; the UI warns |

The dwell override is a module-global assignment made by sparring code inside the sparring process.
The core file is not edited, no other process is affected, and this table is its declaration. Every
knob response carries the canonical value so a UI can mark any deviation.

## Recording

The tap holds a ring buffer (default 10 minutes at 50 Hz); when full, the oldest ticks fall off and
`recording.start_tick` moves. Event logs (replans, commits) keep absolute ticks and are filtered to
the window on the way out. A sparring session writes nothing to disk except the `.npz` a user
explicitly downloads.

### Two distances, not one

`dist` is the **body's** distance to the placement being approached — the pelvis under physics,
which is what `has_arrived` reads and therefore what decides the commit. `dist_plan` is the
**plan's**: the reference frame the encoder is chasing, against the same placement, through the
same viz transform. Both are `null` when nothing is being approached.

The pair is the diagnosis of an approach, and one number cannot give it. Measured over seven
bearings at 1.5 m (2026-08-17, `tools/measure_approach.py`): the plan closed to 0.02–0.19 m at every
bearing while the body closed to 0.007 m straight ahead and only 0.38–0.54 m off-axis — so four of
seven commits ended on `approach_timeout_ticks` with `arrived=false` rather than by arriving. The
charts draw the arrival radius across the strip so "the body never got inside" is visible rather
than inferred.

## Changelog

- **0.2** (2026-08-17) — every number on the wire is finite (`null`, never `NaN`); the trail is
  bounded to `PLAN_HORIZON_TICKS` in both modes; a scrubbed tick carries its queue; `dist_plan`
  joins `dist` so an approach can be attributed to the plan or to the body; queue entries carry
  `arrived`.
- **0.1** (2026-08-17) — first version: sparring message set, debug message, state machine table,
  visualisation transform, HTTP API, knob table with the dwell-override declaration.
