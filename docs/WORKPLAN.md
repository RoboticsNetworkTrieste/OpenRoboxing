# WORKPLAN.md — OpenRoboxing, six months to Season 0

Sequenced tasks for Claude Code. Read `CLAUDE.md` first. Every task has an **acceptance criterion**
that is a command someone else can run; if it does not, the task is badly specified and should be
pushed back on.

**Task IDs** are `M<milestone>-T<n>`. Reference them in commits and PRs.

**Rule for the whole plan:** if a task's acceptance criterion cannot be met, **stop and report**
rather than working around it. The workarounds in this project are all expensive and all need a human
decision.

---

## M0 · Ground truth (before week 1, ~2 days)

### M0-T1 · Environment and checkpoints
Acquire upstream, verify its stack runs, fetch checkpoints.
- Consume `NVlabs/GR00T-WholeBodyControl` as the **pristine submodule** `external/gr00t-wbc`
  tracking `main`, and create the `src/openroboxing/` skeleton per `CLAUDE.md`. **No fork.** This
  task originally required forking NVlabs so that patch P0 could live in git; installing P0 at
  runtime instead (`runtime/generator.py`) removed the reason, and the submodule is never modified.
- Run upstream's own `external/gr00t-wbc/check_environment.py`; resolve what it reports.
- Fetch model checkpoints with upstream's `download_from_hf.py` — `install.sh` does this from
  `nvidia/GEAR-SONIC`. **Do not commit weights.**
- Record how upstream is tracked — submodule, branch, and the `OPENROBOXING_GR00T_ROOT` override —
  in `src/openroboxing/spec/upstream_patches.md`.

**Acceptance:** `bash install.sh` completes, and
`.venv_mb/bin/python -m openroboxing.tools.env_report --quick` prints the submodule's HEAD and its
distance from `origin/main`, checkpoint paths and hashes, GPU, MuJoCo and ONNX Runtime versions.

### M0-T2 · Upstream reality check
Before designing anything, confirm three facts that the plan depends on. Report findings in
`src/openroboxing/spec/upstream_notes.md`.
1. Has the **MotionBricks release integrated into the GEAR-SONIC pipeline** landed upstream since our
   snapshot? If yes, much of M1 may already exist — **stop and report before continuing.**
2. Enumerate the observation terms actually active in the shipped config
   (`observation_config.hpp` + the registry wiring in `g1_deploy_onnx_ref.cpp`). Produce a table:
   term name, dimension, source signal, history length.
3. Confirm how `policy_input_file_` is enabled and what exactly it writes (order, dtype, one row per
   control tick?).

**Acceptance:** `upstream_notes.md` contains the three answers, the observation table, and the exact
flags/config needed to produce a `policy_input` dump.

### M0-T3 · Run the MotionBricks interactive demo
Get the generator running standalone, unmodified, and capture its output.
- Run the upstream interactive demo; confirm the 15-clip registry loads and `walk_boxing` works.
- Write `tools/dump_generation.py`: drive `full_navigation_agent` headlessly with a scripted sequence
  of `mode` / `movement_direction` / `facing_direction` / `specific_target_positions`, and dump the
  resulting 36-dim MuJoCo qpos stream to npz.

**Acceptance:** `python -m openroboxing.tools.dump_generation --style walk_boxing --seconds 10 --out
gen.npz` produces a qpos array of shape `(300, 36)` (10 s at 30 Hz) that renders as plausible motion
in a MuJoCo viewer.

---

## M1 · "It moves" — weeks 1–4
**Goal:** MotionBricks output drives the GEAR-SONIC policy under MuJoCo physics, in one Python
process, with observation parity proven against the C++ reference.

This milestone is the project's gate. Do it in the order below; the parity harness comes *before* the
runtime, because everything downstream is worthless if the observations are wrong.

### M1-T1 · Name-derived index mappings
`runtime/conventions.py`: build MuJoCo↔IsaacLab joint and body mappings **by name**, from the
authoritative name lists in the model and the upstream config.
- Expose `to_isaaclab(x)`, `to_mujoco(x)` for joint vectors and body arrays.
- Assert invertibility at import; raise on any unmapped or duplicated name.
- Handle the `wxyz`↔`xyzw` quaternion convention explicitly, reusing `mujoco_helper.py` where possible.

**Acceptance:** property test over 1000 random vectors: `to_mujoco(to_isaaclab(x)) == x` exactly;
constructing the mapping with a deliberately renamed joint raises.

**Do not:** hard-code an index array, or copy one from documentation.

### M1-T2 · Golden capture from the C++ reference
Produce the fixtures that define correctness.
- Build the upstream C++ deploy (sim2sim path) per upstream docs.
- Feed it a **fixed reference motion** (the CSV format in `motion_data_reader.hpp`) and capture, per
  control tick: the reference motion inputs, the robot state, the **policy input vector**, and the
  policy output actions.
- Commit a trimmed fixture (a few hundred ticks) to `tests/fixtures/golden_policy_io/`, plus a README
  stating exactly how it was produced and the upstream SHA.

**Acceptance:** `tests/fixtures/golden_policy_io/` contains aligned arrays and the reproduction
command; a smoke test loads them and checks shapes against the observation table from M0-T2.

**If the C++ build is not feasible in the available environment, stop and report** — the fallback
(deriving observations from the training code and validating behaviourally) is a different plan with
a different risk profile and needs a human decision.

### M1-T3 · Observation assembly in Python
`runtime/obs.py`: implement every active observation term from the M0-T2 table — history buffers,
token state, reference-motion terms, projected gravity, joint state, previous actions, whatever the
registry lists.
- One function per term, each with a docstring naming its convention and source.
- Assembled in the registry's order, into a single vector.

**Acceptance — the M1 gate:** `pytest tests/test_obs_parity.py` replays the golden fixture and matches
the C++ `policy_input` vector to **max absolute error < 1e-4** on every tick. Report per-term error so
a failure localises immediately.

### M1-T4 · Policy runner
`runtime/policy.py`: GEAR-SONIC via ONNX Runtime — load, warm up, step at 50 Hz, expose action output
in the documented convention and scaling.

**Acceptance:** replaying the golden observations through the Python policy reproduces the C++ actions
to < 1e-3 max absolute error.

### M1-T5 · The bridge
`runtime/bridge.py`: generator output → policy reference inputs. MuJoCo joint order → IsaacLab (via
M1-T1), 30 → 50 Hz resampling (linear for positions, slerp for quaternions — mirror the upstream
planner's behaviour), finite-difference velocities **after** resampling.

**Acceptance:** feeding a known analytic qpos ramp produces the expected resampled series; a
round-trip test on a real generated clip shows no NaNs, no discontinuities at segment boundaries, and
velocities consistent with finite differences of the positions.

### M1-T6 · Single-fighter runtime
`runtime/world.py` + a `tools/run_single.py` entrypoint: one G1 in MuJoCo, physics on, policy at
50 Hz, reference motion supplied by the generator through the bridge, robot state fed back as
generator context each replan.

**Acceptance:** `python -m openroboxing.tools.run_single --style walk_boxing --seconds 30` runs a G1
that walks and shadow-boxes under physics without falling, and writes a run log with tracking error
per body.

### M1-T7 · Performance measurement (do not skip)
`tools/bench_world.py`: measure real-time factor for **two** 29-DOF humanoids in the arena scene with
contact enabled, at several timesteps and collision-geometry configurations, on the target machine.

**Acceptance:** a short report in `docs/perf/m1_mujoco.md` with real-time factors and a recommended
timestep, and an explicit verdict on whether two fighters at 50 Hz control hold real time. **If they
do not, report before M3 begins.**

---

## M2 · "It obeys" — weeks 5–8
**Goal:** a human commits a key pose and the robot performs it. The repository goes public.

### M2-T1 · Target-pose override
Make the generator accept explicit target joint transforms, so a player-authored key pose can drive
generation instead of one sampled from the clip library. Additive and optional: behaviour is
unchanged when the keys are absent.

Two things this task said that turned out to be wrong, both corrected in place:

- **The hook is not `_override_target_transforms`.** That one runs *before* the joint transforms
  exist and is gated behind `BYPASS_SPRING_MODEL`, which the demo never enables. The override is a
  separate `_override_target_joint_transforms`, applied after `_generate_target_joint_transforms`.
- **It is not a patch.** Upstream is a pristine submodule and is never edited. The hook's body and
  its call site are installed on the agent instance at construction — see `_apply_target_pose_override`
  and `_wrap_target_transforms` in `src/openroboxing/runtime/generator.py`.

- Verify the pose-constraint tensors feeding `_generate_inbetween_frames` accept an external pose with
  the same mask semantics the model was trained with. **This is the assumption the game rests on — if
  it does not hold, stop and report.** *(Answered: yes. Structurally, the pose reaches the model
  through `local_poses` with a `has_local_poses` mask that exists because training saw pose
  constraints present and absent; empirically, an authored pose moves the result and stays
  continuous.)*
- Document in `spec/upstream_patches.md`, and cover it with `tests/test_generator_pose_override.py`,
  which runs against an agent that has no hook — so a submodule bump cannot silently disarm it.

**Acceptance:** `tools/commit_pose.py --pose poses/dev/guard_high.json` generates motion that reaches
the commanded pose at the commanded tick, within a stated joint-error tolerance, and the in-between
motion is continuous.

### M2-T2 · Pose record spec and library format
`spec/pose_record.md` + loader: name, source clip and frame range, keyframes on the 29-DOF skeleton,
horizon in tokens, measured telegraph window, admission status, library version.

**Acceptance:** schema validation round-trips; an invalid record fails with a specific message naming
the offending field.

### M2-T3 · Telegraph measurement
`studio/telegraph.py`: for a generated motion, compute the window between the first frame where the
motion is distinguishable from the neutral/guard baseline and the contact frame. Start with a
deterministic geometric proxy (end-effector displacement and body-signal divergence above threshold);
leave a hook for a learned classifier later.

**Acceptance:** `tools/measure_telegraph.py --pose poses/dev/hook_R.json` reports a window in ms and a
pass/fail against a configurable floor; two obviously different poses (a slow hook and a snap jab)
produce clearly different windows.

### M2-T4 · Intent timeline v0
`runtime/intents.py` + `spec/intent.md`: the commit queue. One active commit, no cancellation,
`commit_at` in 50 Hz ticks, style and move channels, slot resolution through a loadout.

**Acceptance:** a scripted intent sequence drives a single fighter end to end; issuing a second commit
while one is active is rejected with a specific error; `spec/intent.md` is versioned at 0.1.

### M2-T5 · Authored pose library v0.1
6–10 boxing poses: guard, jab L, jab R, hook L, hook R, slip, cover. Each with a measured telegraph
window and a recorded tracking error from a runtime trial.

**Acceptance:** `poses/v0.1/` validates, every pose has both measurements recorded, and
`tools/run_single.py --loadout poses/v0.1/orthodox.json` executes all six on command.

### M2-T6 · Public release
README that is honest about maturity, LICENSE and attribution (GEAR-SONIC and MotionBricks credited
explicitly), CONTRIBUTING, the six workstream issues opened with `good-first-issue` where true,
install instructions verified on a clean machine, and no checkpoints committed. What is published is
this repository as itself — not a fork of NVlabs; upstream arrives as the `external/gr00t-wbc`
submodule and a reader clones it with `--recurse-submodules`.

**Acceptance:** a person who has never seen the repo follows the README on a clean machine and reaches
a running `run_single` within 30 minutes; record the transcript of that attempt in the PR.

---

## M3 · "There are two of them" — weeks 9–13
**Goal:** two fighters, one world, contact, a match loop, replays.

### M3-T1 · Arena scene
MJCF with ring geometry, two G1 instances, padded-glove collision geoms, cameras. Parameterised by the
timestep and collision configuration chosen in M1-T7.

**Acceptance:** both fighters spawn in stance facing each other; `bench_world.py` on this scene meets
the real-time target.

### M3-T2 · Batched generation
One MotionBricks instance serving both fighters with `batch=2` and shared weights; per-fighter context,
seeds and replan scheduling.

**Acceptance:** two fighters generate independent motion in one instance; a test asserts VRAM use and
per-replan latency stay within the M1-T7 budget, and that fighter A's intents never affect fighter B's
output (swap-order test).

### M3-T3 · Contact sensing and hit attribution
Contact/impulse extraction from MuJoCo: which body of which fighter, magnitude, location, time.
Distance, guard state and ring-position traces.

**Acceptance:** a scripted scenario (A throws a hook, B stands still) yields exactly one attributed hit
event with a plausible impulse; a scenario with a blocked strike attributes contact to the guarding
arm, not the head.

### M3-T4 · Match loop
`runtime/match.py`: rounds, clock, knockdown detection (root height plus contact), the count, get-up
window, end conditions. **Scoring hooks only** — emit events, do not compute a winner yet.

**Acceptance:** a full three-round match runs headless and produces a match record containing every
field in `spec/match_record.md`.

### M3-T5 · Replays
Record intent log + seeds + state trace; a player that reconstructs the visual from the trace.

**Acceptance:** a recorded match replays visually identically from the trace; the intent log alone is
under a few hundred kilobytes for a full match.

---

## M4 · "It's playable" — weeks 14–18
**Goal:** a human plays it, on a laptop, and it feels like something. First meetup bracket.

### M4-T1 · Ring client v0 (local)
Browser client: ring view at 30 FPS, the intent timeline (style / move / commit rows), six-key
loadout bar with durations, round clock and score, ping indicator. **No HUD on the fighters** — the
windup is the only cue.

**Acceptance:** a hotseat match is playable end to end by two people on one machine; a first-time
player understands the six keys without being told more than one sentence.

### M4-T2 · Match host
`server/`: a process that owns one match, accepts two clients over websocket, services the intent
queue at 30 Hz, streams state at 30 FPS.

**Acceptance:** two browsers on the LAN play a full match; artificially injecting 200 ms latency does
not change match outcomes systematically (run a scripted-agent A/B and compare win rates).

### M4-T3 · One-command install
Packaging for Linux and Windows/WSL. Checkpoint fetch, dependency pinning, a smoke test.

**Acceptance:** on a clean Windows machine with WSL, one documented command reaches a running hotseat
match; the transcript goes in the PR.

### M4-T4 · Playtest and tune
Run the first meetup bracket. Instrument it: match length, commits per minute, hit rates, how often
matches devolve into circling (the passivity failure mode).

**Acceptance:** `docs/playtest/first_bracket.md` reports the numbers, the qualitative reactions, and a
concrete list of parameter changes (commit horizon, commit window, loadout constraints, telegraph
floor).

---

## M5 · "It's a league" — weeks 19–23
**Goal:** the site, the table, the ladder.

### M5-T1 · League services
Registration with GitHub sign-in, fighter handles, loadout selection, Swiss pairing, Glicko-2 with
confidence intervals, fixtures, the public replay archive.

**Acceptance:** a simulated 32-player, 10-week season runs end to end from scripted clients and
produces a sane table; ratings converge and the 8-match threshold behaves as specified.

### M5-T2 · Scoring v0.9
Implement the four dimensions on top of the M3-T4 event stream: landed impulses, knockdowns, ring
control, aggression. **Aggression and ring control need concrete definitions first** — that is
workstream 03's job; if the definitions are not ready, implement the interface and a placeholder, and
flag it.

**Acceptance:** replaying ten recorded matches produces scores that a human watching the replays
agrees with in at least eight cases; disagreements are documented as rule bugs, not code bugs.

### M5-T3 · Fight-night screen
Projector view: bracket, live match, table, replay highlights.

**Acceptance:** driven live at a meetup without an operator touching a terminal.

### M5-T4 · Agent API
Same intent structure over the same transport, sandboxed, rate-limited, with a compute budget per
tick. Exhibition track, outside the Season 0 table.

**Acceptance:** a scripted baseline agent plays a full ranked-format match against a human client and
appears in the exhibition results, not the table.

---

## M6 · "Season 0" — weeks 24–26

### M6-T1 · Freeze
Rules v1.0, pose library v1.0, policy weights for the season, robot model — all versioned, all
published together, with a changelog and the reproduction config for the weights.

**Acceptance:** a released season manifest pins every asset by version and hash; a match record can be
traced to the exact assets that produced it.

### M6-T2 · Licence review before publishing weights
**Blocking.** Publishing finetuned derivatives of NVIDIA-licensed checkpoints is redistribution of
derived models. Confirm the terms, produce the attribution text, and document the base checkpoint and
job config for every published weight set.

**Acceptance:** `LICENSING.md` states the position, every release carries the attribution, and a human
has signed off before the first weight publication.

### M6-T3 · Launch
Registration opens, the Trieste Open is dated, the docs are complete, the contribution guide is real.

**Acceptance:** an external entrant registers, plays a ranked match and appears on the table without
any manual intervention.

---

## Parallel: the Studio (M2 → M5)

Pose Studio ships with M2 (it is how the library is authored). The finetuning half starts once there
are enough poses to justify a run.

### S-T1 · Pose Studio web tool
Keyframe editing on the G1 skeleton, source clip and frame range selection, horizon selection,
telegraph measurement, admission check, "add to library".

**Acceptance:** an author with no repo access creates a valid, admitted pose record through the browser.

### S-T2 · Finetune job runner
Isaac Lab job: base = upstream GEAR-SONIC, target = a set of pose records, objective = **pose-tracking
fidelity** (not "under contact" — see the project definition, correction C4). Report mean joint error.
Runs on TORC hardware, not a laptop.

**Acceptance:** a finetune run reproduces from its recorded config alone and reports mean joint error
on a held-out set.

### S-T3 · Regression suite
Before any weight set ships: a fixed battery of upstream motions checked for tracking regression, so a
pose-specific finetune cannot silently degrade general behaviour.

**Acceptance:** a deliberately over-fitted checkpoint fails the regression gate.

---

## Standing rules for Claude Code

1. **Read `CLAUDE.md` at the start of every session.** Conventions in this project are the whole game.
2. **Stop-and-report triggers** (do not work around): the observation parity gate fails · the pose
   override does not behave as the model expects · MuJoCo cannot hold real time with two fighters ·
   the upstream MotionBricks/GEAR-SONIC integration has landed and duplicates our work · the C++
   golden capture cannot be produced.
3. **Never invent a number.** Tolerances, rates and dimensions come from the specs or from
   measurement. If you need one that does not exist, add it to `spec/` with its justification.
4. **Prefer deleting to disabling.** No dead code paths guarded by flags nobody sets.
5. **Every PR states how to reproduce its acceptance criterion.** A PR without that is not done.
6. **When a task reveals the plan is wrong, say so.** The plan is a hypothesis about a codebase none
   of us has fully executed yet; contradicting it with evidence is the most valuable thing you can do.
