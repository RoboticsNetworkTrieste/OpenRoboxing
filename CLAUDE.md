# CLAUDE.md — OpenRoboxing

Persistent context for Claude Code. Read this before touching anything. If a task in `WORKPLAN.md`
contradicts this file, this file wins and the contradiction is a bug to report.

---

## What this repository is

A **standalone repository** that adds a fighting game and a league on top of
`NVlabs/GR00T-WholeBodyControl`, which it **consumes** as the submodule `external/gr00t-wbc`,
tracking NVlabs `main` and **never modified**. This is not a fork: there is no fork of upstream
anywhere, and nothing to rebase. All of our code lives under `src/openroboxing/`; the submodule is
read-only, full stop — see "Upstream policy" below.

**The product:** two humanoids box in a physics simulation. A player picks a pose, places a ghost of
their fighter where the move should happen, and commits ~0.6 s ahead; **MotionBricks** (upstream,
generative) walks the fighter there and arrives in that pose; the **GEAR-SONIC** policy (upstream,
finetuned by us) executes it in **MuJoCo**; physics decides the hit. Up to five commits queue up and
run back to back. No joypad, no HUD, no cancellation after a commit — including a queued one.

**Naming discipline — get this right in code, comments and docs:**
- The motion generator is **MotionBricks**. It is *not* GR00T N1.5 (a different NVIDIA model).
- The whole-body policy is **GEAR-SONIC**, part of GR00T-WBC.
- Our own things: `openroboxing`, the *Studio*, the *ring client*, the *league site*.

---

## Architecture invariants

Violating any of these is a design regression, not a shortcut:

1. **One OS process per match runtime.** MuJoCo physics, **one MotionBricks instance per fighter**,
   one shared policy. **No DDS. No C++ deploy stack. No inter-process physics.**

   *Corrected 2026-08-08 (M3-T2); line numbers re-verified 2026-09-03 at upstream `v1.1`.* This
   invariant said "one batched MotionBricks instance (both fighters, `batch=2`, shared weights)".
   **Upstream cannot do that** — `full_agent.py` hardcodes `batch_size = 1` (`:395`, the first line
   of `_generate_inbetween_frames`), reads `frames['mujoco_qpos'][0, ...]` (`:506`) and calls
   `num_pred_frames.item()` (`:163-164`), which raises on a two-element tensor. Supporting it means a
   second patch to upstream, which invariant 3 makes a stop-and-ask. Measured, it is not worth it:
   two generators cost 1513 MiB of a 49 GiB card and 29.6 ms per replan, which over a 0.5 s interval
   is 1.18 ms of a 20 ms tick. Full reasoning in `runtime/pool.py`.

   The policy *is* shared, and safely: `GearSonicPolicy` holds only its ONNX sessions, with all
   per-fighter state in each fighter's `ObservationBuilder`.
2. **Server-authoritative.** Clients send intents and receive state. Never run physics or generation
   client-side, even "just for prediction".
3. **The generator is upstream and unmodified.** No exception — the one behaviour we add, the
   target-pose override, is installed on the agent instance at runtime and leaves the submodule
   untouched (see "The one thing we add to upstream, at runtime"). If a change that cannot be
   installed at runtime seems necessary, stop and ask.
4. **Every index mapping is derived by joint/body *name*** from the authoritative name lists, and
   asserted invertible at construction. Never hard-code a permutation array, never copy one from
   documentation. This is the single most likely source of a silent catastrophic bug.
5. **Fail loudly.** No silent fallbacks, no `try/except: pass`, no clamping that hides a bad input. If
   an observation term cannot be built, raise.
6. **Determinism is recorded, not assumed.** Every match records seeds, versions (pose library, policy
   weights, rules, robot model) and a state trace. Bit-exact re-simulation is *not* promised.
7. **Specs before implementations.** Anything crossing a boundary (intent, pose record, combination,
   match record, rules, agent API) is a versioned schema in `src/openroboxing/spec/` first.

## Canonical rates — use these constants, never literals

| Quantity | Value | Notes |
|---|---|---|
| Tick | **50 Hz** | the policy rate; all `tick` / `commit_at` fields are in these units |
| Generator | **30 Hz** | MotionBricks native output; resampled 30→50 Hz in the bridge |
| Intent queue service | **30 Hz** | server-side queue processing |
| Ring stream to clients | **30 FPS** | display only |
| Commit horizon (default) | **30 ticks = 0.6 s** | per-match parameter; a floor, not a gap between queued moves |
| Plan length | **the hole left to the next keyframe**, 6–16 tokens | `ceil(boundary − tick)` since `spec/intent.md` 3.2; the keyframe is pinned in absolute time, so this shrinks as it is approached |
| Combination style | **`walk_boxing`** | the only clip allowing 6–16 tokens; `walk` caps at 11 |
| Leg | **0.8–3.2 s** | `MIN_TOKENS`–`MAX_TARGET_LEG_TOKENS` × 4 frames at 30 Hz. **A leg is not a plan**: 39 % are longer than one, and run untargeted then land |
| Combination | **2–3 keyframes, 0.93–6.0 s** | measured over the 174-record v0.2 library, median 3.87 s; median leg 15 tokens (2.00 s) |
| Move end | **`commit_at + duration_ticks`** | known when it starts — `spec/intent.md` 3.0 |
| Drift gain | **0.935** | the generator covers this fraction of a commanded residual; `warp.py` divides by it. Re-measured 2026-09-03 under the pinned-keyframe schedule; the +/-0.10 consistency bar is not met — see docs/perf/2026-09-03-drift-gain-pinned.md |
| Sustained walk | **0.83 m/s** | measured; validates a player's placement at issue time only |

---

## Repository layout (ours)

```
CLAUDE.md        this file — at the root, so an agent session loads it without being told
README.md        what the game is, how to install it, how to run it
LICENSING.md     ours is Apache-2.0; the weights are NVIDIA's and are downloaded, not re-hosted
install.sh       venv, submodule, LFS, the GEAR-SONIC checkpoints, editable install
motions/         the mocap corpus: 19 takes x 2 mirrors, Maya-style CSV at 30 fps (M5)
pyproject.toml   the package, its dependencies, and [tool.pytest.ini_options]
external/
  gr00t-wbc/     NVlabs/GR00T-WholeBodyControl as a submodule. Pristine. Never edited.
src/openroboxing/
  paths.py       GR00T_ROOT and every artefact under it; OPENROBOXING_GR00T_ROOT overrides
  spec/          versioned schemas + rates.md + constants.py + the upstream registry
  parity/        observation parity harness vs the C++ reference (M1)
  runtime/
    intents.py   timeline and commit queue; a commit is a combination (spec/intent.md 3.0)
    sequence.py  CombinationRunner: which leg is live at a tick, and the pinned-keyframe horizon
    generator.py MotionBricks wrapper: intents -> qpos @30 Hz, and P0 installed at runtime
    reference.py the reference stream: pull, resample, stay ahead of the tick; the ambient replan cadence
    bridge.py    qpos -> policy inputs: name-derived remap, 30->50 Hz, velocities
    obs.py       observation assembly (the risky part — see parity/)
    policy.py    GEAR-SONIC via ONNX Runtime; effort limits read from the model
    warp.py      places a recorded combination: footwork at true size, travel ramped (M5)
    world.py     one fighter under physics (M1)
    arena.py     the ring: two fighters, ropes, gloves, cameras, lights
    pool.py      one generator per fighter, and the isolation guarantee
    contact.py   hit attribution and the state trace
    fight.py     two fighters under physics — the world a match runs on
    match.py     rounds, clock, knockdowns, the record. Rules live here and nowhere else.
    replay.py    play a record back: re-derive the rules, or render it
  studio/        pose authoring, telegraph, regression gate (S-T3)
                 motion_import.py + segment.py + combination_record.py build the library (M5);
                 segment.thin_targets selects the sparse targets a plan is aimed at (3.2)
  client/        ring client, fight-night screen, Pose Studio (all vanilla, no build step)
  server/        match host, protocol, agent API, Studio API
  league/        scoring, Glicko-2, Swiss pairing, season, freeze manifest
  poses/         pose library (data, versioned) · dev/ holds Studio drafts
                 v0.1/ single key poses · v0.2/combinations/ the 174 built combinations
  tools/         CLI entrypoints — `python -m openroboxing.tools.<name>`
tests/           unit + golden tests, fixtures under tests/fixtures/
docs/            ASSUMPTIONS.md (every decision taken that was really the owner's),
                 WORKPLAN.md, perf/, playtest/, superpowers/ (plans + specs)
```

The package directory moved to `src/`, but the **package name did not**: it is still
`openroboxing`, and every entry point is still `python -m openroboxing.tools.<name>`. Paths written
below as `spec/…`, `runtime/…`, `tools/…` are relative to `src/openroboxing/`.

## Traps found the hard way (read before debugging physics)

- **`g1_29dof.xml` is not a simulation model** and looks exactly like one: same masses, joint ranges
  and torque limits as `g1_29dof_old.xml`, but **zero rotor armature, zero joint damping**. A stiff
  PD on it collapses a fighter in half a second. Compose scenes from `paths.G1_29DOF_SIM_XML`.
- **A limit read from an unset field becomes zero, and zero is a valid-looking number.** The same
  file declares torque limits on the *joint*, not the motor, so reading `actuator_ctrlrange` returned
  zeros and clipped every torque to nothing — silently. `policy.effort_limits` now raises instead.
- Both are written up in `spec/upstream_notes.md`.
- **A clip in `motions/` is not the clip upstream trained on, despite the identical name.**
  `walk_boxing`'s `clip_id` is `shadow_boxing_R_003__A360_M` and that file is in the corpus, but
  upstream's cached `mujoco_qpos` for it has **elbows of the opposite sign** and a 22.9 deg
  worst-joint error at the best alignment anywhere in the take — it is a different retarget.
  Measured 2026-08-27. Do not build a golden fixture on the assumption that they match; what
  establishes the corpus is in the robot's convention is that every value of all 38 takes falls
  inside the G1's own `jnt_range` (`tests/test_motion_import.py`).

## Upstream files you will read constantly

All under the submodule, so the paths below are openable as written. `paths.GR00T_ROOT` resolves the
same tree at runtime, honouring `OPENROBOXING_GR00T_ROOT` when another checkout is named.

| Path | Why |
|---|---|
| `external/gr00t-wbc/motionbricks/motionbricks/motion_backbone/demo/full_agent.py` | the generator API: `generate_new_frames`, `get_next_frame`, `_override_target_transforms`, `_generate_target_joint_transforms` |
| `external/gr00t-wbc/motionbricks/motionbricks/motion_backbone/demo/clips.py` | `CLIPS` registry (15 clips, one boxing), clip cache loading |
| `external/gr00t-wbc/motionbricks/motionbricks/motion_backbone/demo/controllers.py` | replan cadence, key bindings, mode resolution by name |
| `external/gr00t-wbc/motionbricks/motionbricks/helper/mujoco_helper.py` | `mujoco_qpos_converter`, 36-dim qpos, quaternion reorder, canonicalisation |
| `external/gr00t-wbc/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/observation_config.hpp` | **the specification for `obs.py`** — the single definition point for all observations |
| `external/gr00t-wbc/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` | observation registry wiring (~lines 364–410), `policy_input_file_` (line ~307) |
| `external/gr00t-wbc/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/state_logger.hpp`, `file_sink.hpp` | per-signal CSV dumps used as golden fixtures |
| `external/gr00t-wbc/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/motion_data_reader.hpp` | reference-motion CSV format (`joint_pos.csv`, `body_quat.csv`, …, `metadata.txt`) |
| `external/gr00t-wbc/gear_sonic/scripts/run_sim_loop.py` | the upstream MuJoCo sim loop (single robot, DDS) — reference only, we do not use DDS |
| `external/gr00t-wbc/docs/source/references/observation_config.md`, `external/gr00t-wbc/docs/source/tutorials/zmq.md` | prose specs for the above |

## The one thing we add to upstream, at runtime

Upstream samples the pose target from the clip library by one-hot `mode` + `random_seed`, and
`_override_target_transforms` overrides **only root position and heading** — and only under
`BYPASS_SPRING_MODEL`, which is off. What the game needs is **explicit target joint transforms**, so
that a player-authored key pose drives generation instead of a clip's.

Nothing is patched into upstream to get it. The behaviour is installed on the agent *instance* at
construction, in `src/openroboxing/runtime/generator.py`:

- **`_wrap_target_transforms`** — the call site. It wraps `_generate_target_joint_transforms` so the
  override runs immediately after the clip-sampled target exists.
- **`_apply_target_pose_override`** — the body. Additive and optional: new keys in the input dict,
  existing behaviour unchanged when they are absent.

`tests/test_generator_pose_override.py` proves it works against a **pristine** agent — that test is
the thing that will catch a changed hook signature after a submodule bump. Registry entry and
history: `src/openroboxing/spec/upstream_patches.md`.

## Upstream policy

- **The submodule is read-only. Full stop.** There is no allowlist and no registered diff — nothing
  in `external/gr00t-wbc` is edited, ever, which is exactly what lets it track NVlabs `main`.
  Anything we need from upstream is installed at runtime or it does not happen.
- **Bumping upstream is a deliberate command**, not a rebase:

  ```bash
  git submodule update --remote external/gr00t-wbc
  ```

  **After every bump:** run the test suite, and re-verify the observation-registry offsets recorded
  in `src/openroboxing/spec/upstream_notes.md` — a bump invalidates every line number in it.
- **Never re-host upstream checkpoints.** Fetch at install time; `install.sh` does exactly this,
  pulling the GEAR-SONIC policy from `nvidia/GEAR-SONIC` with upstream's own `download_from_hf.py`.
  Finetuned derivatives we publish are a separate, licence-reviewed matter.

---

## Conventions

- **Python 3.10+**, type hints on all public functions, `ruff` + `black` (repo has `lint.sh`,
  `pyproject.toml`).
- **No notebooks in the repo.** Experiments go in `tools/` as CLI scripts with `--help`.
- **Config**: dataclasses or pydantic models, not dicts passed around. Every magic number is a named
  constant with a comment saying where it came from.
- **Logging** over printing; one structured event log per match, and that log *is* the match record.
- **Tests**: pytest. Golden fixtures under `tests/fixtures/` committed as small CSV/npz. Any function
  that transforms between conventions (joint order, quaternion order, frame rate) gets a round-trip
  property test.
- **Commits**: conventional-commit prefixes, one logical change each. Reference the workplan task ID
  (`M1-T3: …`).
- **Docs**: every module gets a docstring saying what convention its inputs and outputs are in
  (which joint order, which quaternion order, which frame rate, which coordinate frame). Most bugs in
  this project will be convention bugs.

## Definition of done (applies to every task)

1. Code + type hints + module docstring stating conventions.
2. Tests, including a round-trip or golden test where a conversion is involved.
3. `lint.sh` clean.
4. The task's explicit acceptance criterion from `WORKPLAN.md` demonstrably met, with the command to
   reproduce it written in the PR body.
5. Any new cross-boundary structure added to `spec/` with a version bump.
6. If anything in this file became untrue, this file is updated in the same PR.

## Things that are known traps

- **Joint ordering**: MuJoCo order ≠ IsaacLab order. Derive by name; assert `remap(unremap(x)) == x`.
- **Quaternion ordering**: MuJoCo is `wxyz`, most other code is `xyzw`. `mujoco_helper.py` already
  handles this — read it before writing your own.
- **The 36-dim qpos** is root position (3) + root quaternion (4) + 29 joints.
- **Velocities**: the bridge finite-differences positions after resampling, not before.
- **MotionBricks is kinematic and physically unaware.** It will produce self-penetrating or
  torque-infeasible motion. The Studio must reject such poses on measured tracking error; do not
  "fix" them in the bridge.
- **Being hit is out of distribution** for the upstream policy — it was trained penalising contact
  outside feet, hands and elbows. Expect odd behaviour on impact; that is a research track, not a bug
  to patch in the runtime.
- **The walk approach is gone** (`spec/intent.md` 3.0). A commit no longer travels to a placement:
  it starts where the fighter stands and its recorded motion carries it to the ghost. `TRAVEL_CONTEXT`,
  `has_arrived`, `has_settled`, the counted dwell and `approach_timeout_ticks` no longer exist. If you
  are reading older prose that describes a fighter walking to a placement, it predates 3.0.
- **Only one boxing clip exists** in the public `CLIPS` registry. Strikes come from authored key
  poses, not from clips — and a fighter **travels in `walk`**, not in that boxing clip. Upstream's
  lateral gaits (`walk_left`/`walk_right`) are swapped in only when the mode is `slow_walk` or
  `walk`, so travelling in `walk_boxing` leaves a fighter with no sideways gait at all. Measured
  2026-08-17: it is why off-axis approaches did not close. See `intents.TRAVEL_CONTEXT`.
- **`GeneratorIntent` has two directions and they are different signals.** `facing_angle` is where
  the fighter looks; `movement_angle` is where it travels, and the difference is what selects the
  gait. Leaving the second at its default says "straight ahead, always".
- **A fighter always faces its opponent** (owner, 2026-09-03, `spec/intent.md` 3.1, reversing half of
  design D5). Both headings that reach MotionBricks — the **target frame's** `target_heading` and the
  `facing_angle` control signal — are the live bearing to the opponent, measured in world frame by
  `fight.py::FightWorld.facing_angle` and re-measured every tick, because the opponent moves while a
  2.4–7.6 s combination runs. A combination's recorded turn moves the *body*, never the aim. The
  recorded heading in `warp.py` survives only where there is no opponent to face — the Studio's
  rehearsal and the warp tools — and `sequence.py::CombinationRunner.intent_for` is the one place the
  override happens. Prose saying a ghost's heading is the fighter's heading plus the combination's
  recorded turn predates 3.1.
