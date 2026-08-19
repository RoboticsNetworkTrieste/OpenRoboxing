# Sparring — a free-space debug bench for the match runtime

Date: 2026-08-17 · approved by the project owner in brainstorming (approach A, four scoping answers
recorded below) · spec review gate waived by the owner ("scrivila, committala e implementala").

## What it is

A tool that runs the **core motion stack** — intent timeline → MotionBricks → bridge → GEAR-SONIC →
MuJoCo — outside a match: no rounds, no scoring, no knockouts, one player fighter and a passive
target, driven from a browser. Its job is **debugging the humanoid's behaviour**: see what
MotionBricks generated, see how GEAR-SONIC executes it, read the intent state machine, and turn the
runtime's knobs while watching.

Everything is built **parallel to the core**: new files only, the core is imported and never edited.

## Owner's scoping decisions

| Question | Decision |
|---|---|
| How many robots | **One player fighter + a passive target** ("sacco"): blue on `IdlePilot` |
| Time control | **Real time + full-session recording with scrubbing**; live pause included |
| The plan ghost | **The reference frame the encoder is chasing (~0.9 s ahead) + a ground trail** to the plan's tail |
| Runtime knobs | **Live-editable from the UI**, every deviation marked against `spec/constants.py` |

## New files

```
src/openroboxing/
  server/sparring_app.py     aiohttp app: SparringHost, websocket, debug + scrub + knob APIs
  server/sparring_tap.py     DebugTap: per-tick recording, state derivation, viz transform
  client/sparring.html/.js/.css   the UI (reuses ring.js, overlay pattern, tokens/)
  tools/serve_sparring.py    entrypoint: http://localhost:8081/
  spec/sparring_protocol.md  v0.1 — debug message, knob API, scrub API, viz transform
  tests/test_sparring_tap.py, tests/test_sparring_app.py
```

## SparringWorld

`SparringWorld(FightWorld)`, defined in `server/sparring_app.py`:

- red = `QueuedPilot` (reused from `server/host.py`), blue = `IdlePilot` — the sacco stands in the
  opening stance and serves as an arrival/contact target.
- `max_outstanding=10` by default (a sparring queue is deeper than a match's 5).
- `require_admitted=False` by default: Studio drafts are testable. A `--admitted-only` flag restores
  the match rule.
- The **only override**: `has_arrived` reads `self.arrival_radius_m` (initialised to
  `ARRIVAL_RADIUS_M`) instead of the constant, so the knob can move it.
- Bench actions: **teleport the sacco** to `(x, y, heading)` (writes root qpos + `mj_forward`), and
  **reset with seed** (fresh timelines via `reset_round(0)` after reseeding the pool from the given
  seed; clears the tap).

## The loop

Modelled on `MatchHost.run` but sparring-shaped: 50 Hz against the wall clock, late ticks dropped and
counted, `broadcast` of binary frames at 30 FPS. No round clock — the session runs until reset.

- **Pause/resume**: paused, the loop stops stepping but keeps serving state, scrub and knobs.
- **Pause on fall** (optional, default off): auto-pause when a root height drops below 0.4 m.
- The debug JSON message goes out at **10 Hz** alongside the frames.

## DebugTap

Per-tick appends into preallocated-growing float32 arrays (~840 B/tick ≈ 12 MB per 5 min). Ring
buffer capped (default 10 minutes) — when full, oldest ticks are dropped and the UI shows the
recording window's start.

Recorded per tick:

- `qpos` of the full arena (both fighters) — the source for scrub reconstruction;
- red's **reference qpos** at the tick (what MotionBricks asked for), and blue's;
- red per-joint tracking error (29) = measured joints − reference joints;
- red raw action (29);
- root heights, `separation_m`, distance-to-placement (NaN when no placement), `step_ms`;
- **state machine state**, derived from the red timeline (see below);
- current commit ordinal (or −1).

Event logs (not per-tick):

- **replans**: `(tick, forced, plan_frames)` — captured by wrapping `generate` **on the red
  generator instance** at construction. An instance-level wrapper in the sparring process; the core
  never sees it.
- **commits**: the full span (`issued_at / commit_at / strike_at / end_tick / arrived`, slot, pose,
  placement) — re-read from `world.commits()` at the debug cadence, so spans fill in as they resolve.

### State machine derivation (red timeline, per tick)

| State | Condition |
|---|---|
| `OPENING` | no commit has completed and none is current |
| `WAITING` | a commit exists, not finished, not yet current (inside its horizon window or queued) |
| `APPROACH` | current commit executing, `strike_at is None` |
| `DWELL` | `strike_at` set, `tick < end_tick` |
| `HOLD` | queue drained, a completed commit's intent is held |

Derived read-only from `timeline.executing(tick)` / commit fields — never by advancing the timeline
(`generator_intent` is the clock and belongs to the world alone).

### The visualisation transform (plan ghost + trail)

The reference lives in the **generator frame**; the encoder never consumes its root position, so it
has no canonical world position. For drawing only:

```
world_xy(k)  = robot_xy + R(apply_yaw) · (ref_xy[k] − ref_xy[tick])
world_yaw(k) = apply_yaw + ref_yaw[k]
```

— the displacement the reference intends over the lookahead, applied from where the robot really is;
the coherent inverse of `to_generator_frame`. Joint angles and z pass through unchanged. The ghost is
drawn **yaw-only** (the client's shadow FK takes a heading, not a full quaternion); pitch/roll of the
reference root is a known, documented limitation of v0.1.

- **Plan ghost**: the transformed reference frame at `tick + LOOKAHEAD_TICKS` (45) — what the
  encoder is looking at.
- **Trail**: transformed root `(x, y)` of `stream.motion[tick:]` — from now to the plan's tail —
  sent at debug cadence, drawn as a ground polyline.

## Protocol (`spec/sparring_protocol.md` v0.1)

Client → host over the websocket, superset of the match protocol's staging subset:
`stage`, `place`, `commit`, `clear`, `ping`, plus sparring controls:

```jsonc
{"type": "pause"} {"type": "resume"} {"type": "reset", "seed": 1234}
{"type": "teleport_sacco", "x": 1.0, "y": 0.5, "heading": 3.14}
```

Host → client: binary frames (identical format to the match), `state` (seat view of red incl. full
queue), `error`, `pong`, and at 10 Hz:

```jsonc
{"type": "debug", "tick": 4812, "paused": false, "machine": "APPROACH",
 "commit_ordinal": 3, "queue": [ /* full spans */ ],
 "ghost": {"x":.., "y":.., "z":.., "heading":.., "angles": {..29..}},
 "trail": [[x,y], ...],
 "series_head": {"err_mean":.., "err_max":.., "dist":.., "root_h":.., "step_ms":..},
 "knobs": {"replan_dt": {"current": 0.5, "canonical": 0.5}, ...},
 "recording": {"start_tick":.., "end_tick":..}}
```

HTTP:

- `GET /scene.json`, `GET /meshes.bin` — as the match (from `Scene`).
- `GET /api/frame/{tick}` — scrub: `mj_forward` on the recorded qpos into a scratch `MjData`, packed
  in the **same binary format**, base64 in a JSON envelope together with that tick's debug data
  (machine state, errors, ghost, trail reconstructed from recorded reference).
- `GET /api/series?from=&to=&stride=` — downsampled trace arrays for the charts.
- `GET /api/session.npz` — the whole tap, numpy archive, for offline analysis.
- `POST /api/knobs` — set any subset; response always `{name: {current, canonical}}`.

## Knobs

| Knob | Canonical | Mechanism | Applies |
|---|---|---|---|
| `replan_dt` | 0.5 s | `red.stream.replan_dt` | next replan |
| `horizon_ticks` | 30 | timeline attribute | live |
| `max_outstanding` | 10 (bench default; match canonical 5) | timeline attribute | live |
| `arrival_radius_m` | 0.40 | `SparringWorld` attribute | live |
| `approach_timeout_ticks` | ring-derived | timeline attribute | live |
| `pose_dwell_ticks` | 74 | **process-local override** of `runtime.intents.POSE_DWELL_TICKS` | live — also rewrites in-flight `end_tick`s; the UI warns |

The dwell override is a module-global set from sparring code, confined to the sparring process and
declared here; the core file is not edited.

## Client (`client/sparring.html`)

TORC design system, ink ground. Layout:

- **Main**: 3-D view (reused `Ring`). Bodies: the two real fighters (streamed frames); the **aim
  ghost** (existing `showShadow('red', …)`); the **plan ghost** — `shadowFor('plan')` with its
  material recoloured clay (`0xDDAE86`), posed from the debug message; the **trail** and the
  **arrival-radius circle** drawn in the SVG overlay through `ring.project` (the `overlay.js`
  pattern); placement markers.
- **Heatmap toggle**: tint each red body's mesh materials (`emissive`) by its joint's |tracking
  error|, green→clay, fixed scale 0–0.5 rad. Mapping body→joint from `shadow_kinematics`; materials
  are per-drawable instances, so tinting touches only the red fighter. This is "how GEAR-SONIC
  executes it", painted on the silhouette.
- **Right column**: machine-state chip; the 10-slot queue with live spans (`null` shown as "—");
  staged slot + placement + anchor; the knobs, deviations in clay with a "canonical" reset per knob.
- **Bottom**: session scrubber (live/replay modes; dragging pauses following), phase bands
  (OPENING/WAITING/APPROACH/DWELL/HOLD as coloured bands), and canvas trace charts sharing the same
  x-axis: tracking error (mean+max), distance-to-target, root height, step ms, replan event ticks.
  One vertical cursor across all charts, following the scrub position.
- **Script panel**: a JSON sequence `[{"slot": "3", "x": 1.2, "y": -0.4, "heading": 1.57,
  "at_tick": 500}, …]` and *Run*: commits fire in order, respecting the queue bound (`at_tick`
  optional — absent means "as soon as a slot frees"). Save/load in `localStorage`. This is how the
  same 10-commit series is replayed exactly after turning a knob.
- **Keys**: `1–6` pose, `WASD` ghost, `SPACE` commit, `Q` unstage (red bindings, as the match);
  `P` pause, `R` reset, `←/→` scrub ±1 tick, `Shift+←/→` ±50.

## Testing

- **State derivation**: synthetic commits → expected phase sequence (pure Python, no GPU).
- **Viz transform**: coherence property against `to_generator_frame` on known placements.
- **Scrub**: frame packed from recorded qpos is byte-identical to the live frame of the same tick.
- **Knob API**: aiohttp test client over a stub world; every response carries `{current, canonical}`.
- The full GPU loop stays a manual check, command documented in the tool's `--help` (as for the
  match host).

## Non-goals

- No second pilotable seat, no scoring, no match records. A sparring session is not a match and
  writes none of `spec/match_record.md`.
- No core edits: `runtime/`, `server/host.py`, `client/ring.js`, `client/app.js` unchanged.
- No persistence of sessions beyond the `.npz` export.
