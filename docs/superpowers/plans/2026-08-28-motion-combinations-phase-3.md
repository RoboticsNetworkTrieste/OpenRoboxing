# Motion Combinations — Phase 3: the surface

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a combination selectable and playable from a browser — nine per page, paged through the whole library — and delete `Loadout`.

**Architecture:** Phase 2 left the match runtime driving combinations while everything around it still speaks the language of poses, slots and placements. This phase converts that surface. Part A is the match path (protocol, host, agent, client, replay) and is what makes the game playable. Part B is the sparring subsystem and the measurement tools, which are a separate app with their own spec.

**Tech Stack:** Python 3.10+, pytest, and vanilla browser JS with no build step. Run everything with `.venv_mb/bin/python`.

---

## Where the repo actually is

Phase 2 ended at **631 passed, 23 deselected, 2 failed, 3 errors**. The failures are all stale surface:

| File | Stale references | Part |
|---|---|---|
| `server/host.py` | 17 | A |
| `server/protocol.py` | 5 | A |
| `server/agent.py` | 6 | A |
| `server/client.py` | 3 | A |
| `tools/run_match.py` | 17 | A |
| `tools/serve_match.py` | 8 | A |
| `runtime/replay.py` + `test_manifest.py` | — | A |
| `client/app.js`, `index.html`, `ring.js` | the loadout bar | A |
| `server/sparring_app.py` | 13 | B |
| `server/sparring_tap.py`, `client/sparring.js`, `spec/sparring_protocol.md` | — | B |
| `tools/run_single.py` | 13 | B |
| `tools/tune.py`, `seat_fairness.py`, `latency_ab.py`, `serve_sparring.py` | 6/6/5/7 | B |
| `tools/measure_approach.py` | 11 | B — **delete** |
| `tools/measure_dwell.py` | 2 | B — **delete** |

Two tools measure things that no longer exist: `measure_approach.py` measures how a fighter closes on a placement, and `measure_dwell.py` measures the counted dwell. `spec/intent.md` 3.0 removed both concepts. **Porting them would mean inventing a question for them to answer**, so Part B deletes them and says so in the commit. Their findings already live in `docs/perf/` and in the constants they produced; nothing is lost but the scripts.

## The design this implements — D6, in full

**There is no loadout.** A fighter carries the whole ~120-combination library. The client shows **nine at a time** in a 3×3 grid with forward and back buttons paging through all of them (14 pages).

**What the client needs per combination**, because the shadow is drawn in the browser: the name, the duration in seconds, the **final keyframe's 29 joint angles** (that is the ghost's pose), and `recorded_heading_delta` (the ghost's heading is the fighter's heading plus that — never aimed at the opponent). At 120 records that is roughly 50 KB of JSON, sent once at `welcome`.

**Reach is now per-combination.** Under 1.1 anywhere in the ring was reachable and distance merely cost time. Under 3.0 a combination's duration is fixed by its recording, so its reach is `APPROACH_SPEED_M_S × duration` — between about 1.6 m and 6.6 m depending on the move. The client must show the *selected* move's reach, and it changes as the player pages.

---

# Part A — the match path

### Task A1: `protocol.py` sends a library, not a loadout

**Files:** `src/openroboxing/server/protocol.py`, `tests/test_server.py` (the protocol tests only)

Read `protocol.py`'s `welcome()` first — its docstring explains why poses are sent at all (the browser draws the shadow), and that reasoning survives unchanged.

- [ ] **Step 1:** Write failing tests: `welcome` carries a `combinations` list of `{name, seconds, heading_delta, reach_m, pose}`; every entry's `pose` has all 29 joints; the payload no longer has `loadout`, `horizons`, `pose_seconds` or `poses`; a spectator gets the same library (it is not secret — both fighters have all of it, which is D6's point and a change from the old per-seat loadout).
- [ ] **Step 2:** Run, watch fail.
- [ ] **Step 3:** Implement. `welcome(seat, library, match_format, arena, match_id)`. `reach_m` is `APPROACH_SPEED_M_S * duration_ticks / TICK_HZ` — the furthest ghost that combination can be given at issue time. Keep `approach_speed_m_s` in the payload; it is still the ceiling that validates a placement.
- [ ] **Step 4:** Run, green. **Step 5:** Commit `A1: welcome carries the combination library, not a per-seat loadout`.

### Task A2: `host.py` stages and commits combinations

**Files:** `src/openroboxing/server/host.py`, `tests/test_server.py`

Read `spec/protocol.md` — the intent message shape is specified there and must move with the code.

- [ ] **Step 1:** Failing tests: an `intent` message carrying `{combination, ghost:[x,y]}` stages it; committing queues it; an unknown combination name is rejected with a typed error and does **not** disconnect the client (`protocol.py` already has the hostile-input pattern — follow it); a ghost outside the selected combination's reach is rejected at issue time with the number.
- [ ] **Step 2–4:** Implement, green.
- [ ] **Step 5:** Update `spec/protocol.md` in the same commit — it is a versioned cross-boundary schema, so bump it.
- [ ] **Step 6:** Commit `A2: host stages and commits combinations`.

### Task A3: The scripted agent picks combinations

**Files:** `src/openroboxing/server/agent.py`, `tests/test_agent.py`

`agent.py` currently learns "which slot is a stance and which are strikes" from the loadout it was dealt (`:172-183`). With no loadout it must choose from the library.

- [ ] **Step 1:** Failing tests: the agent picks only combinations it can reach from where it stands; it varies its choices rather than repeating one; it never commits into a full queue.
- [ ] **Step 2–4:** Implement. Classification by name prefix (`shadow-boxing` / `ib-dodge` / `ib-combat-turn-jog`) is honest and available; anything cleverer is out of scope. Say in the docstring that this is a placeholder opponent, not a trained one.
- [ ] **Step 5:** Commit `A3: the scripted agent chooses from the combination library`.

### Task A4: The nine-per-page picker

**Files:** `src/openroboxing/client/app.js`, `src/openroboxing/client/index.html`, `src/openroboxing/client/ring.js`

Vanilla JS, no build step, no framework — match the existing file's style exactly.

- [ ] **Step 1:** Replace the six-slot loadout bar with a **3×3 grid** plus **prev/next page buttons** and a page indicator (`3 / 14`). Each cell shows the combination's name, its duration, and its select key.
- [ ] **Step 2:** Keys. Red selects with `1`–`9` and pages with `[` / `]`; blue selects with `U I O J K L M , .` and pages with `;` / `'`. Update `SEATS` and the on-screen key hints in `index.html`, which currently read "loadout · 6 slots" and "1 — 6".
- [ ] **Step 3:** The ghost. Its **heading is derived** — the fighter's heading plus the selected combination's `heading_delta` — so the player drags position only. Remove any heading control. `ring.js` draws the shadow from the selected combination's final-keyframe pose.
- [ ] **Step 4:** Reach. The existing cost block shows a walk time and a throw time; under 3.0 there is no walk. Replace it with the selected combination's duration and its reach, and mark a ghost beyond that reach as rejected — because the host will reject it.
- [ ] **Step 5:** Load it in a browser and confirm paging, selection, ghost placement and commit all work. There is no JS test harness in this repo, so this step is manual and its result goes in the commit message.
- [ ] **Step 6:** Commit `A4: nine-per-page combination picker; the ghost's heading is derived`.

### Task A5: `replay.py` and the record

**Files:** `src/openroboxing/runtime/replay.py`, `src/openroboxing/spec/match_record.md`, `tests/test_replay.py`, `tests/test_manifest.py`

- [ ] A commit in the record now names a combination and a ghost, and carries the achieved drift speed phase 2 logs. `match_record.md` is versioned — bump it. Re-derive the rules from the trace as before.
- [ ] Commit `A5: the match record carries combinations and the achieved drift`.

### Task A6: Delete `Loadout`

**Files:** `src/openroboxing/runtime/intents.py`, `src/openroboxing/paths.py`, `src/openroboxing/poses/loadouts/`, and every remaining importer

- [ ] Only once A1–A5 are green. Delete the class, `LOADOUT_DIR`, `orthodox.json`, and the module comment that says it survives until Plan 3.
- [ ] `poses/v0.1/` **stays** — the pose-level tests and the golden library depend on it.
- [ ] Commit `A6: delete Loadout - the library is the move set (D6)`.

**Part A definition of done:** a match runs from a browser, nine combinations per page, paging through all 120; `tests/test_server.py`, `test_agent.py`, `test_replay.py`, `test_manifest.py` green; `Loadout` gone.

---

# Part B — sparring and the tools

### Task B1: Delete the two tools that measure deleted concepts

- [ ] Delete `tools/measure_approach.py` and `tools/measure_dwell.py`. Grep for importers first — `test_manifest.py` imports `DEFAULT_CONTEXT` from `run_match.py`, so this family has cross-references. Commit `B1: delete the approach and dwell measurement tools with what they measured`.

### Task B2: Port the remaining tools

- [ ] `run_match.py`, `serve_match.py`, `run_single.py`, `tune.py`, `seat_fairness.py`, `latency_ab.py`, `serve_sparring.py`. Each takes a combination library where it took a loadout, and a combination name where it took a slot. `--help` must still work for each; run every one of them and say so.
- [ ] Commit `B2: the tools take a combination library`.

### Task B3: The sparring subsystem

**Files:** `server/sparring_app.py`, `server/sparring_tap.py`, `client/sparring.js`, `spec/sparring_protocol.md`, `tests/test_sparring_app.py`, `tests/test_sparring_tap.py`

- [ ] Read `spec/sparring_protocol.md` first. It is built on `has_arrived` and `approach_timeout_ticks`, both gone. Decide, and **state the decision in the commit**: port it to combinations, or retire it. If retiring, delete the subsystem and its spec together rather than leaving a broken app in the tree.
- [ ] Commit `B3: sparring against spec/intent.md 3.0`.

**Part B definition of done:** the full suite green with no `--continue-on-collection-errors`, and every tool's `--help` runs.

---

## Rules for every task

- `./lint.sh` has ~95 pre-existing findings; the bar is **no new findings in files you touch** (`uvx ruff@0.16.2 check`).
- Do not modify `external/gr00t-wbc`.
- Do not delete a failing test to make the suite green. Port it, or delete it deliberately and say why in the commit message.
- `CLAUDE.md` is updated in the same commit as anything that makes it untrue — its layout section still lists `client/` as holding a Pose Studio and `poses/loadouts/`.
