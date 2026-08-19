# Sparring Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A free-space debug bench (`serve_sparring`) that runs the core motion stack outside a
match: one player fighter + a passive sacco, live commits with a plan ghost, full-session recording
with scrubbing, and live runtime knobs — all in new files, zero core edits.

**Architecture:** `SparringHost` wraps a `SparringWorld(FightWorld)` (red = `QueuedPilot`, blue =
`IdlePilot`), records every tick into a `DebugTap`, and serves a browser UI that reuses `ring.js`.
The reference motion is made drawable by a documented visualisation transform; scrubbing replays
recorded qpos through `mj_forward` + the existing `Scene.pack`.

**Tech Stack:** Python 3.10, aiohttp, numpy, mujoco (CPU for tests), vanilla JS + three.js
(existing `client/vendor`), TORC tokens. Spec: `docs/superpowers/specs/2026-08-17-sparring-tool-design.md`.

**Verification environment:** run Python via `.venv_mb/bin/python` from the repository root.
Pure-Python/mujoco tests need no GPU; the full loop is a manual GPU check at the end.

---

## File map

| File | Responsibility |
|---|---|
| `src/openroboxing/spec/sparring_protocol.md` | v0.1 — message set, knob API, scrub API, viz transform (invariant 7) |
| `src/openroboxing/server/sparring_tap.py` | DebugTap storage; machine-state derivation; viz transform; npz export. Pure numpy — no aiohttp, no mujoco |
| `src/openroboxing/server/sparring_app.py` | `SparringWorld`, `KNOBS` registry, `SparringHost` (loop, debug msg, scrub, series), `build_sparring_app` (routes) |
| `src/openroboxing/tools/serve_sparring.py` | CLI entrypoint, port 8081 |
| `src/openroboxing/client/sparring.html` `.css` | UI skeleton + TORC ink styling |
| `src/openroboxing/client/sparring.js` | socket, 3-D view, ghosts, overlay, panels, knobs, script runner, keys |
| `src/openroboxing/client/sparring-charts.js` | canvas trace charts + phase bands + scrubber cursor |
| `tests/test_sparring_tap.py` | state derivation, viz transform, tap storage/series/npz |
| `tests/test_sparring_app.py` | knob API, debug message, handle(), scrub equality (mujoco CPU) |

Core files that must NOT change: anything under `runtime/`, `server/host.py`, `server/app.py`,
`server/protocol.py`, `client/ring.js`, `client/app.js`.

Facts the implementation relies on (verified against the tree at planning time):

- A real MotionBricks replan **reassigns** `agent.frames['mujoco_qpos']` (new tensor object); the
  cadence no-op returns early without assignment (`full_agent.py:122-128` vs `:166-168`). So
  `id(agent.frames['mujoco_qpos'])` changing across a `generate()` call ⇔ a replan happened.
- `Commit.end_tick` reads the module global `POSE_DWELL_TICKS` in `runtime/intents.py` at property
  call time, so `openroboxing.runtime.intents.POSE_DWELL_TICKS = n` takes effect live.
- `Ring.shadowFor(seat)` creates a ghost per seat name with `SHADOW_COLOUR[seat] ?? 0xffffff`; the
  returned `{material}` is mutable, so a `'plan'` ghost can be recoloured clay from sparring.js.
- `Scene.pack(tick, data)` reads only `data.xpos/xquat` — it works on a scratch `MjData` after
  `mj_forward`.
- `GeneratorPool.reset(round_index)` reseeds from `self.match_seed`, so setting
  `world.pool.match_seed = seed` before `world.reset_round(0)` reseeds a session.

---

### Task 1: `spec/sparring_protocol.md`

**Files:** Create `src/openroboxing/spec/sparring_protocol.md`

- [ ] **Step 1: Write the spec.** Version 0.1. Content: relationship to `spec/protocol.md` (same
  binary frame format, same staging subset `stage/place/commit/clear/ping`); the sparring-only
  client messages `pause`, `resume`, `{"type":"reset","seed":int}`,
  `{"type":"teleport_sacco","x","y","heading"}`; the 10 Hz `debug` message schema exactly as in the
  design spec (machine, commit_ordinal, queue, ghost, trail, series_head, knobs, recording,
  replans); HTTP endpoints `GET /api/frame/{tick}`, `GET /api/series?from&to&stride`,
  `GET /api/session.npz`, `GET/POST /api/knobs`; the knob table with canonical sources; the
  visualisation transform formulas and the yaw-only ghost limitation; the dwell override mechanism
  and its in-flight-commit caveat. State machine table (OPENING/WAITING/APPROACH/DWELL/HOLD) with
  conditions.
- [ ] **Step 2: Commit** — `docs(spec): sparring protocol v0.1`

### Task 2: `sparring_tap.py` — machine-state derivation (TDD)

**Files:** Create `src/openroboxing/server/sparring_tap.py`, `tests/test_sparring_tap.py`

- [ ] **Step 1: Failing tests.** Use a tiny stand-in commit (duck-typed like
  `runtime.intents.Commit`: fields `issued_at/commit_at/strike_at/arrived` + the three predicates —
  import the real `Commit` and a real `PoseRecord` fixture is heavier than needed; use the real
  `Commit` with a minimal `PoseRecord` built from `studio.pose_record` defaults if cheap, else a
  `SimpleNamespace` implementing `is_executing/is_scheduled/end_tick`). Cases:
  - no commits → `OPENING`
  - one commit issued at 10, tick 15, not current → `WAITING`
  - executing, `strike_at=None` → `APPROACH`
  - executing, `strike_at=50`, tick 60 < end → `DWELL`
  - finished (tick ≥ end_tick), queue empty → `HOLD`
  - finished + a second commit issued but waiting → `WAITING` (scheduled wins over hold)
- [ ] **Step 2: Run** `.venv_mb/bin/python -m pytest tests/test_sparring_tap.py -q` → FAIL (module missing)
- [ ] **Step 3: Implement.**

```python
MACHINE_STATES = ("OPENING", "WAITING", "APPROACH", "DWELL", "HOLD")
OPENING, WAITING, APPROACH, DWELL, HOLD = range(5)

def derive_machine_state(commits, tick: int) -> int:
    """Read-only: never advances the timeline (generator_intent is the clock, not this)."""
    for commit in commits:
        if commit.is_executing(tick):
            if commit.strike_at is None or tick < commit.strike_at:
                return APPROACH
            return DWELL
    if any(c.is_scheduled(tick) for c in commits):
        return WAITING
    for commit in reversed(commits):
        end = commit.end_tick
        if end is not None and tick >= end:
            return HOLD
    return OPENING
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: Commit** — `feat(sparring): machine-state derivation`

### Task 3: `sparring_tap.py` — viz transform (TDD)

- [ ] **Step 1: Failing tests.**
  - `yaw_of_quat_wxyz`: identity → 0; pure yaw π/2 quat → π/2.
  - `viz_world_path(ref_motion, tick, robot_xy, apply_yaw)`: with `apply_yaw=0`, row `tick` maps
    exactly onto `robot_xy`; a reference displacement of `(1, 0)` with `apply_yaw=π/2` maps to
    world `(0, 1)`.
  - Coherence with the core: for a placement `p_world`, robot at `r`, context tail at `c`,
    `to_generator_frame` computes `c + R(−yaw)(p−r)`; check that `viz_world_path` applied to a
    generator-frame point built that way returns `p_world` (round trip through both rotations,
    using base row = context tail).
  - `viz_ghost(...)`: returns dict with `x,y,z,heading,angles(29)`; clamps `tick+lookahead` at the
    end of the motion.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.**

```python
def yaw_of_quat_wxyz(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def viz_world_path(ref_motion, tick, robot_xy, apply_yaw):
    """Generator-frame reference rows [tick:], drawable: displacement-from-now applied at the robot.

    Visualisation only — the encoder never consumes the reference root position, so this world
    position is an interpretation, and it is the inverse of fight.to_generator_frame's rotation.
    """
    ref = np.asarray(ref_motion, dtype=np.float64)
    delta = ref[tick:, 0:2] - ref[tick, 0:2]
    c, s = math.cos(apply_yaw), math.sin(apply_yaw)
    rot = np.array([[c, -s], [s, c]])
    return np.asarray(robot_xy, dtype=np.float64) + delta @ rot.T

def viz_ghost(ref_motion, tick, lookahead, robot_xy, apply_yaw, joint_names):
    ref = np.asarray(ref_motion, dtype=np.float64)
    k = min(tick + lookahead, ref.shape[0] - 1)
    path = viz_world_path(ref_motion, tick, robot_xy, apply_yaw)
    xy = path[k - tick]
    frame = ref[k]
    return {
        "x": float(xy[0]), "y": float(xy[1]), "z": float(frame[2]),
        "heading": apply_yaw + yaw_of_quat_wxyz(frame[3:7]),
        "angles": {name: float(v) for name, v in zip(joint_names, frame[7:])},
    }
```

- [ ] **Step 4: PASS. Step 5: Commit** — `feat(sparring): the visualisation transform`

### Task 4: `sparring_tap.py` — DebugTap storage (TDD)

- [ ] **Step 1: Failing tests.** Append 100 synthetic ticks; assert `window() == (0, 99)`;
  `at(42)` returns the appended scalars/arrays; `series(0, 99, stride=10)` returns 10-point lists;
  cap: construct `DebugTap(max_ticks=50)`, append 100, `window() == (50, 99)`, `at(10)` raises
  `TapError`; `to_npz_bytes()` round-trips through `np.load` with the same keys and shapes;
  replan/commit event lists survive trimming (events keep absolute ticks; `events_in_window()`
  filters).
- [ ] **Step 2: FAIL. Step 3: Implement** — per-column `collections.deque(maxlen=max_ticks)` for:
  `tick, qpos(f4, nq), ref_red(f4,36), ref_blue(f4,36), err_red(f4,29), action_red(f4,29),
  root_h_red, root_h_blue, separation, dist_target(NaN allowed), step_ms, machine(i1),
  commit_ordinal(i2)`. `append(**kwargs)` validates shapes (fail loudly). `at(tick)` indexes by
  `tick - window_start`. `series(from_, to, stride)` → dict of plain lists for the chart endpoint
  (mean+max of `|err_red|` per tick precomputed here). `replans: list[(tick, forced, plan_frames)]`,
  appended by the host. `to_npz_bytes()` → `np.savez_compressed` into `io.BytesIO`.
- [ ] **Step 4: PASS. Step 5: Commit** — `feat(sparring): the DebugTap ring recorder`

### Task 5: `SparringWorld` + KNOBS registry (TDD on the registry)

**Files:** Create `src/openroboxing/server/sparring_app.py`, `tests/test_sparring_app.py`

- [ ] **Step 1: Failing tests** for the knob registry against a `FakeWorld`
  (`SimpleNamespace(fighters={'red': SimpleNamespace(timeline=..., stream=...)}, arrival_radius_m=0.40)`):
  `knob_values(world)` returns every knob as `{"current", "canonical"}` with canonical from
  `spec/constants.py`; `set_knobs(world, {"replan_dt": 0.3})` mutates `stream.replan_dt`; unknown
  knob name raises `SparringError`; non-finite/≤0 values raise; `pose_dwell_ticks` sets
  `openroboxing.runtime.intents.POSE_DWELL_TICKS` (assert via a real `Commit` whose `end_tick`
  moves, then restore in a `finally`).
- [ ] **Step 2: FAIL. Step 3: Implement.**

```python
@dataclass(frozen=True)
class Knob:
    canonical: float
    get: Callable[[Any], float]
    set: Callable[[Any, float], None]
    minimum: float = 1e-9

def _set_dwell(world, v):
    import openroboxing.runtime.intents as intents
    intents.POSE_DWELL_TICKS = int(v)

KNOBS: dict[str, Knob] = {
    "replan_dt": Knob(REPLAN_DT, lambda w: w.fighters["red"].stream.replan_dt,
                      lambda w, v: setattr(w.fighters["red"].stream, "replan_dt", float(v))),
    "horizon_ticks": Knob(COMMIT_HORIZON_TICKS, ..., lambda w, v: setattr(w.fighters["red"].timeline, "horizon_ticks", int(v)), minimum=0.0),
    "max_outstanding": Knob(SPARRING_MAX_OUTSTANDING, ..., minimum=1.0),
    "arrival_radius_m": Knob(ARRIVAL_RADIUS_M, lambda w: w.arrival_radius_m,
                             lambda w, v: setattr(w, "arrival_radius_m", float(v))),
    "approach_timeout_ticks": Knob(float(DEFAULT_APPROACH_TIMEOUT_TICKS), ..., minimum=1.0),
    "pose_dwell_ticks": Knob(float(POSE_DWELL_TICKS),
                             lambda w: float(intents_module.POSE_DWELL_TICKS), _set_dwell, minimum=1.0),
}
```

  (`SPARRING_MAX_OUTSTANDING = 10`, module constant with a comment: bench default, match canonical
  is `MAX_OUTSTANDING_COMMITS = 5` — deliberately different and always shown as a deviation.)
  Then `SparringWorld(FightWorld)`: `__init__(*args, **kwargs)` → `super().__init__`, then
  `self.arrival_radius_m = ARRIVAL_RADIUS_M`; `has_arrived` copied from core but reading
  `self.arrival_radius_m`; `teleport_sacco(x, y, heading)` writes blue's `root_qpos` (position
  `[x, y, 0.793]`, quat `[cos(h/2),0,0,sin(h/2)]`), zeroes blue's root dofs, `mj_forward`.
- [ ] **Step 4: PASS. Step 5: Commit** — `feat(sparring): SparringWorld and the knob registry`

### Task 6: `SparringHost` (TDD on fake world)

- [ ] **Step 1: Failing tests.** `FakeWorld` grows: `step(tick)` records calls; `data.qpos` a
  vector; `commits()` returns dicts; fighters carry real `IntentTimeline`s over the test loadout
  fixture used by existing intent tests (reuse the fixture pattern from
  `tests/test_intents*.py` — check its name at implementation and import the same
  helper). Tests: `step_once` advances `host.tick`, appends one tap row, measures `step_ms`;
  `pause()` makes `step_once` a no-op that still serves state; `reset(seed=7)` sets
  `world.pool.match_seed = 7`, calls `world.reset_round(0)`, clears the tap and zeroes `tick`;
  `debug_message()` carries `machine` (name, not int), full queue spans with `null` for unset,
  ghost+trail (from a fake stream with a known `motion` array), knob block, recording window;
  `handle()` accepts the staging subset via `protocol.parse` and the sparring types
  (`pause/resume/reset/teleport_sacco`) before/without it — unknown types still produce an
  `error` dict; commit into a full 10-queue → `error`.
- [ ] **Step 2: FAIL. Step 3: Implement** `SparringHost`:
  - `__init__(world, *, render=True)`: tap, `QueuedPilot` for red (reused import from
    `server.host`), replan detection wrap:

```python
gen = world.fighters["red"].generator
original = gen.generate
def _watched(intent, context_qpos, dt, *, force=False):
    before = id(gen.agent.frames["mujoco_qpos"])
    original(intent, context_qpos, dt, force=force)
    after_frames = gen.agent.frames["mujoco_qpos"]
    if id(after_frames) != before:
        self.tap.replans.append((self._filling_tick, bool(force), int(after_frames.shape[1])))
gen.generate = _watched
```

    (`self._filling_tick` is set by `step_once` before `world.step` — the tick being generated
    for; an instance-level wrapper, the core class untouched.)
  - `step_once()`: set red pilot anchor (as `MatchHost.step_once` does), `world.step(self.tick)`,
    read tap row values from the world (tracking error = red measured joints − 
    `red.stream.motion[tick, 7:]`; distance-to-placement from the executing commit or NaN), fall
    check → optional auto-pause, `tick += 1`.
  - `debug_message()`, `scrub_payload(tick)` (recorded row + ghost/trail recomputed from recorded
    `ref_red` rows), `series(from_, to, stride)`, `npz_bytes()`, `state_message()` (red seat view:
    reuse `protocol.seat_state`/`visible_queue` with `own=True`).
  - `handle(message)`: sparring types first, then `protocol.parse` for the staging subset routed
    to the red pilot's queue.
  - `async run()`: endless paced loop (deadline pacing and drop-count copied in shape from
    `MatchHost.run`), broadcast frames at 30 FPS + `debug` at 10 Hz; honours `self.paused`.
- [ ] **Step 4: PASS. Step 5: Commit** — `feat(sparring): the SparringHost`

### Task 7: scrub equality (mujoco CPU test) + aiohttp routes

- [ ] **Step 1: Failing tests.**
  - Scrub: `build_arena(ArenaConfig())` → `MjData`; nudge qpos, `mj_forward`, `Scene.pack(5, data)`
    live; save `qpos`; scramble `data`; restore via the host's `_repack(tick, qpos)` path (scratch
    `MjData`) → byte-identical bytes.
  - Routes (aiohttp `TestClient` over `build_sparring_app(host_with_fake_world)`):
    `GET /api/knobs` → all six with `{current, canonical}`; `POST /api/knobs` `{"replan_dt": 0.3}`
    → echoed, fake mutated; bad name → 400; `GET /api/series` → lists; `GET /api/frame/{tick}`
    outside the window → 404 with a message; `GET /api/session.npz` → `content_type
    application/octet-stream`, loadable by numpy.
- [ ] **Step 2: FAIL. Step 3: Implement** `build_sparring_app(host, client_dir=CLIENT_DIR)`:
  routes `/` → `sparring.html`, `/ws` (single red seat + unlimited read-only viewers), `/scene.json`,
  `/meshes.bin` (same closure-cached pattern as `server/app.py`), the four `/api/*` above, static.
  `serve_sparring(host, port)` mirrors `server.app.serve` without the wait-for-players gate.
- [ ] **Step 4: PASS. Step 5: Commit** — `feat(sparring): scrub repack and the HTTP surface`

### Task 8: `tools/serve_sparring.py`

- [ ] **Step 1: Write it** (no unit test — entrypoint; mirrors `tools/serve_match.py`): args
  `--port 8081`, `--loadout orthodox`, `--seed 1234`, `--admitted-only` (default off → drafts OK),
  `--record-minutes 10`. Build `Loadout` for both seats from `LOADOUT_DIR`, print slots, build
  `SparringWorld` (red `QueuedPilot`, blue default `IdlePilot`, `max_outstanding=10`,
  `require_admitted=args.admitted_only`), `SparringHost`, `asyncio.run(serve_sparring(...))`.
  `--help` documents the manual GPU check:
  `.venv_mb/bin/python -m openroboxing.tools.serve_sparring` then open `http://localhost:8081/`.
- [ ] **Step 2: Commit** — `feat(sparring): the serve_sparring entrypoint`

### Task 9: client — skeleton + 3-D + ghosts

**Files:** Create `client/sparring.html`, `client/sparring.css`, `client/sparring.js`

- [ ] **Step 1: HTML/CSS.** TORC ink ground (tokens already in `client/tokens/`); layout grid:
  main canvas + right column (state chip, queue list, knobs) + bottom strip (scrubber, phase bands,
  charts, script panel). Reuse `style.css` variables where the match client defines them; keep all
  sparring-specific rules in `sparring.css`.
- [ ] **Step 2: sparring.js — connection + view.** `new Ring(canvas)`, `await ring.load('')`,
  websocket `/ws`, `applyFrame` on binary; staging keys and the WASD ghost-drive loop **mirrored
  from `client/app.js`** (same bindings: `1–6`, `WASD`, `SPACE`, `Q`; read app.js and adapt the
  red-seat half only); aim ghost via `ring.showShadow('red', …)` exactly as the match does.
- [ ] **Step 3: the plan ghost + trail.** On each `debug` message:
  `const g = ring.shadowFor('plan'); g.material.color.set(0xDDAE86); g.material.opacity = 0.35;`
  then `ring.showShadow('plan', ghost.x, ghost.y, ghost.heading, ghost.angles, ghost.z)`. Trail +
  arrival-radius circle drawn in an SVG overlay positioned by `ring.project(x, y, 0)` per vertex
  (the `overlay.js` pattern — read it and follow its structure; do not import match-specific
  pieces). Heatmap toggle: map red bodies by name via `scene.bodies` prefix `red_` →
  `shadow_kinematics` joints; per debug message set `mesh.material.emissive` from
  `|err|/0.5` clamped, green→clay ramp; toggle off restores black emissive.
- [ ] **Step 4: Manual check** (no GPU needed for layout): `python -m http.server` is NOT enough
  (needs `/scene.json`) — verify page loads against a running host, or defer to Task 12's GPU run.
  Commit — `feat(sparring): client skeleton, ghosts, heatmap`

### Task 10: client — panels, knobs, script runner

- [ ] **Step 1: State + queue panel.** Machine chip (5 states, colour per state), 10-slot queue
  list with spans (`issued/commit/strike/end`, `—` for null, `arrived=false` flagged), staged slot,
  anchor. All driven by `debug` + `state` messages.
- [ ] **Step 2: Knobs panel.** One row per knob from the `debug.knobs` block: numeric input +
  "canonical" button; edits `POST /api/knobs`; deviation (`current !== canonical`) renders the row
  clay and shows both values. `pose_dwell_ticks` row carries the in-flight warning line from the
  spec. Extra controls: pause/resume (`P`), reset with seed field (`R`), teleport-sacco (click a
  "move sacco" toggle then click the ring: reuse the ground-picking used for the aim ghost),
  pause-on-fall checkbox, heatmap toggle, ghost/trail visibility toggles.
- [ ] **Step 3: Script runner.** Textarea (JSON array of `{slot, x, y, heading, at_tick?}`),
  Run/Stop, `localStorage` save/load by name. Runner sends `stage`+`place`+`commit` respecting
  `can_commit` from state messages and `at_tick` when given (compare against `debug.tick`).
- [ ] **Step 4: Commit** — `feat(sparring): panels, knobs, script runner`

### Task 11: client — charts + scrubber (`sparring-charts.js`)

- [ ] **Step 1: Load the `dataviz` skill before writing chart code** (per its trigger), then build:
  a shared-x canvas chart stack fed by `GET /api/series` (windowed, stride to ≤ 1200 points):
  tracking error mean+max, distance-to-target, root height (red+blue), step ms; phase bands row
  from the machine series (OPENING/WAITING/APPROACH/DWELL/HOLD as coloured bands); replan ticks as
  event markers. One vertical cursor across all charts.
- [ ] **Step 2: Scrubber.** Range input over `recording.window`; live mode follows `debug.tick`;
  dragging pauses following and fetches `GET /api/frame/{tick}` (debounced ~30 ms), rendering the
  base64 frame through the same `applyFrame` and the recorded debug row through the same panel
  update path. `←/→` ±1, `Shift+←/→` ±50. "LIVE" button returns to following.
- [ ] **Step 3: Commit** — `feat(sparring): trace charts and the session scrubber`

### Task 12: verification + docs

- [ ] **Step 1:** `.venv_mb/bin/python -m pytest tests -q` → all green.
- [ ] **Step 2:** `./lint.sh` → clean (fix what it flags).
- [ ] **Step 3: Manual GPU run** (the acceptance check):
  `.venv_mb/bin/python -m openroboxing.tools.serve_sparring` → open `http://localhost:8081/`;
  verify: 10 commits queue and execute; plan ghost leads the robot; trail + arrival circle; heatmap
  responds to a strike; pause, scrub back to a commit start, return to LIVE; change
  `arrival_radius_m` and see the deviation marked; script runner replays a saved sequence; npz
  downloads.
- [ ] **Step 4:** Add the tool to `README.md`'s command list
  (`serve_sparring — free-space debug bench`) and a one-line pointer in the layout section.
- [ ] **Step 5: Final commit** — `feat(sparring): the sparring bench` (README + anything residual).

---

## Self-review notes

- Spec coverage: every spec section maps to a task (protocol→1, tap→2-4, world/knobs→5,
  host/loop→6, scrub/API→7, entrypoint→8, client→9-11, tests/manual→2-7+12, non-goals enforced by
  the "must not change" list).
- The existing-fixture reference in Task 6 (loadout fixture) is intentionally resolved at
  implementation time by reading `tests/` — the fixture exists (the intents tests
  construct loadouts); reusing it beats duplicating a pose factory here.
- Type consistency: `derive_machine_state` returns an int index; the host serialises
  `MACHINE_STATES[i]` — tests in Task 2 assert on the int constants, the debug message on names.
