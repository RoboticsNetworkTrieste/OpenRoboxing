# upstream_notes.md — M0-T2 upstream reality check

**Status:** complete · **Date:** 2026-08-07 · **Task:** `M0-T2`

Snapshot under test: `a9d20b2ac0949244d94461a1a3263f38c5027c4a` (see `upstream_patches.md`).
All line references are against that snapshot, and **a submodule bump invalidates every one of
them** — re-verify after `git submodule update --remote external/gr00t-wbc`.

**Re-verified 2026-09-03 at `a0732b642c0333077e127a2f56ab0014c196bca4` (upstream `v1.1`).** The bump
from `c374bae` to `v1.1` is **documentation only** — `README.md` and `docs/source/*`, five files,
nothing under `motionbricks/`, `gear_sonic/` or `gear_sonic_deploy/` — so it moved no line in this
file. The re-verification did catch drift left by the *previous* bump (`a9d20b2` → `c374bae`), which
added three registry entries inside `GetObservationRegistry` (the 4-frame low-latency SONIC variants
`motion_joint_positions_wrists_4frame_step1`, `smpl_joints_4frame_step1`,
`smpl_anchor_orientation_4frame_step1`) and shifted every `g1_deploy_onnx_ref.cpp` line after ~1741
by **+3**. Those numbers are corrected below. Nothing else changed: the three added entries are
*available* terms in the registry, not *active* ones — `policy/release/observation_config.yaml` is
untouched, so the assembled observation is bit-identical and `tests/test_obs_parity.py` still passes.
`full_agent.py` has not changed since the original snapshot at all.

Paths written bare below (`gear_sonic_deploy/…`, `motionbricks/…`, `g1_deploy_onnx_ref.cpp`) are
upstream's own, relative to `GR00T_ROOT`: the submodule at `external/gr00t-wbc`, or the checkout
named by `OPENROBOXING_GR00T_ROOT`.

---

## Q1 — Has the MotionBricks → GEAR-SONIC integration landed upstream?

**No. The project is not duplicated; proceed.**

`origin/main` is at `1983e88888217f6c69283cf3a9d1af01e87f07af`, **46 commits ahead** of our snapshot.
Reviewed all 46. They are: Isaac Teleop (Pico/CloudXR/XR camera streaming), Jetson Thor + JetPack 7
support, an OAK-D W camera mount, PPO/GC training perf work, docs, and a `sonic wrist v1 1` config.
Nothing wires MotionBricks into the GEAR-SONIC runtime. No `motionbricks/` file was touched.

Three upstream commits **are** relevant to M1 and should be read before `obs.py` is written, because
they touch observation ordering — the exact thing M1-T3 must reproduce:

| Commit | Why it matters |
|---|---|
| `69ac0d4` | *"Make ONNX export use a deterministic tokenizer-observation order"* — implies pre-fix exports had **non-deterministic** tokenizer ordering. Our shipped `release/` ONNX predates this. |
| `5933a12` | Adds "low-latency SONIC observation support" — a second observation path. |
| `ed39915` | Caches tokenizer observation slice metadata. |

**Consequence:** `69ac0d4` is a live risk to M1-T3. If parity fails on the *encoder* input, suspect
tokenizer term ordering before suspecting term maths. The deploy C++ builds encoder input in the
order the YAML lists; the exported ONNX must agree, and upstream shipped a fix saying it did not
always.

**Deploy C++ changed by only 4 insertions across all 46 commits** — the reference implementation we
are matching is effectively stable.

---

## Q2 — Observation terms active in the shipped config

### Where the specification actually lives

`CLAUDE.md` states `observation_config.hpp` is *"the single definition point for all observations."*
**This is not accurate and the file should be corrected.** `observation_config.hpp` is only a
hand-rolled YAML parser plus a fallback default list that the shipped config never uses.

The real definition is three-part:

| Concern | Location |
|---|---|
| Term name → dimension → gather function | `g1_deploy_onnx_ref.cpp:1704-1794` (`GetObservationRegistry`) — ~70 terms available |
| Which terms are *active*, and their order | `gear_sonic_deploy/policy/release/observation_config.yaml` |
| Concatenation order + dimension validation | `g1_deploy_onnx_ref.cpp:1798-1840` (`InitializeObservationFunctions`) |

Order is **the YAML's listing order**, not the registry's. Offsets accumulate in that order and the
total is asserted against the ONNX input dimension at startup (`:1835-1840`).

### Which config file is authoritative

`policy/release/` ships **two** configs with disjoint term vocabularies. Verified against the ONNX
files directly:

| File | Encoder dim it implies | Loadable by the C++? | Verdict |
|---|---|---|---|
| `observation_config.yaml` | **1762** | yes | **authoritative** |
| `observation_config_sonic_release.yaml` | 1751 (per its own header) | **no** | belongs to a different export |

`observation_config_sonic_release.yaml` uses Python/IsaacLab term names (`encoder_index`,
`command_multi_future_nonflat`, `motion_anchor_ori_b_mf_nonflat`, …) that appear in **neither** our
snapshot's registry **nor** `origin/main`'s. Feeding it to the deploy throws
*"Unknown observation function"* (`:1817-1820`). Treat it as documentation of a newer sibling export,
not as config.

Ground truth from the shipped weights:

```
model_encoder.onnx : obs_dict [1, 1762] -> encoded_tokens [1, 64]
model_decoder.onnx : obs_dict [1,  994] -> action         [1,  29]
```

Both match `observation_config.yaml` exactly. **The `# Total dimension: 436` comment at the top of
that file is stale — the real total is 994.** Never trust that comment.

### Policy (decoder) input — 994 dims, in order

| # | Term | Dim | Source signal | History |
|---|---|---|---|---|
| 1 | `token_state` | 64 | encoder/tokenizer output | current |
| 2 | `his_base_angular_velocity_10frame_step1` | 30 | base IMU gyro (3) | 10 frames, step 1 |
| 3 | `his_body_joint_positions_10frame_step1` | 290 | measured joint pos (29) | 10 frames, step 1 |
| 4 | `his_body_joint_velocities_10frame_step1` | 290 | measured joint vel (29) | 10 frames, step 1 |
| 5 | `his_last_actions_10frame_step1` | 290 | previous policy actions (29) | 10 frames, step 1 |
| 6 | `his_gravity_dir_10frame_step1` | 30 | projected gravity (3) | 10 frames, step 1 |

**Total 994.** Note every robot-state term is a 10-deep history — there is no single-frame term in
the policy input. `obs.py` needs five ring buffers of depth 10, all filled at 50 Hz.

### Encoder (tokenizer) input — 1762 dims, in order

| # | Term | Dim | Required in `g1` mode? |
|---|---|---|---|
| 1 | `encoder_mode_4` | 4 | **yes** |
| 2 | `motion_joint_positions_10frame_step5` | 290 | **yes** |
| 3 | `motion_joint_velocities_10frame_step5` | 290 | **yes** |
| 4 | `motion_root_z_position_10frame_step5` | 10 | no → zero |
| 5 | `motion_root_z_position` | 1 | no → zero |
| 6 | `motion_anchor_orientation` | 6 | no → zero |
| 7 | `motion_anchor_orientation_10frame_step5` | 60 | **yes** |
| 8 | `motion_joint_positions_lowerbody_10frame_step5` | 120 | no → zero |
| 9 | `motion_joint_velocities_lowerbody_10frame_step5` | 120 | no → zero |
| 10 | `vr_3point_local_target` | 9 | no → zero |
| 11 | `vr_3point_local_orn_target` | 12 | no → zero |
| 12 | `smpl_joints_10frame_step1` | 720 | no → zero |
| 13 | `smpl_anchor_orientation_10frame_step1` | 60 | no → zero |
| 14 | `motion_joint_positions_wrists_10frame_step1` | 60 | no → zero |

**Total 1762.**

### The per-mode zero-fill — the single biggest M1-T3 simplification

Verified in `g1_deploy_onnx_ref.cpp:2011` and `:2044-2092`: the encoder buffer is zeroed, the active
mode's `required_observations` list is looked up, and **only those terms are computed**. Everything
else is left at zero (`:2092` — *"Not required for this mode — leave as zero"*).

OpenRoboxing drives the robot from a kinematic reference motion, i.e. **encoder mode `g1`
(`mode_id: 0`)**. So of 1762 encoder dims, only **644 are ever non-zero**:

```
encoder_mode_4                          4
motion_joint_positions_10frame_step5  290
motion_joint_velocities_10frame_step5 290
motion_anchor_orientation_10frame_step5 60
                                    -----
                                      644   (the other 1118 dims are structurally zero)
```

**We never need to implement the SMPL or VR-3point terms** — 912 dims of the hardest terms drop out.

**Confirmed against the golden capture (2026-08-07), not just from reading code.** Of 1762 encoder
dims, exactly **640 are ever non-zero**, and `encoder_mode_4` is among the zeros: it writes the mode
id as a scalar (0 for `g1`) plus three fills, so it is identically zero and is **not a one-hot**. So
`obs.py` implements **three** encoder terms, totalling 640 informative dims. Asserted in
`tests/test_golden_fixture.py`.

### History layout — measured, not guessed

The five policy history terms are **frame-major `[frame, joint]`, oldest-first**: index 0 is the
oldest frame, index `HISTORY_LEN-1` the current one. Each tick the window shifts and the new sample
enters at the last index.

Established from the capture by checking that frame `k` at tick `t+1` equals frame `k+1` at tick `t`
— residual exactly `0.0`, while the three competing layouts (`[joint, frame]`, and either order's
reverse) give residuals of 0.5–2.6. Pinned in `tests/test_golden_fixture.py`.

Two further notes:
- `encoder_mode_4` is dim 4 but produced by `GatherEncoderMode(buf, offset, 3)` — a one-hot plus 3
  zero-fills (`:1687-1689`). The `encoder_mode` term (dim 3, fill 2) is the older variant; the
  shipped config uses `encoder_mode_4`.
- Mode ids: `g1: 0`, `teleop: 1`, `smpl: 2`.

---

## Q3 — `policy_input_file_`: how to enable it and what it writes

**Flag:** `--policy-input-logfile <path>` (`:4223`, documented at `:4115`).
Constructor parameter `policy_input_file_path`, 10th positional (`:2141`).

**Enable semantics** (`:2209-2217`): non-empty path → the file is truncated once at construction,
then reopened in append mode. Empty (the default) → the `unique_ptr` stays null and nothing is
written. No other flag is required.

**What it writes** (`:4022-4030`):

```cpp
if(policy_input_file_) {
  for(auto d : obs_buffer_) { (*policy_input_file_) << d << ","; }
  (*policy_input_file_) << std::endl;
}
```

| Property | Value |
|---|---|
| Contents | `obs_buffer_` — the **policy/decoder** input (994 floats). **Not** the encoder input. |
| Order | exactly the YAML order in the table above |
| Rows | one per control tick, `control_dt_ = 0.02` → **50 Hz** (`:2162`) |
| Format | ASCII decimal, comma-separated, **trailing comma before the newline** → naive parsers yield a spurious 995th empty field. Strip it. |
| Precision | default `ostream` formatting — **~6 significant digits**, not full double |
| Header | none |
| Written | after inference, inside the control loop, alongside `target_motion_file_` |

### Precision caveat — affects the M1 gate tolerance

M1-T3 demands max abs error `< 1e-4` against this dump. Default `operator<<` gives ~6 significant
digits, so a value near 1.0 quantises at ~1e-6 and a value near 100 at ~1e-4. Joint positions and
gravity are O(1) and safe. **`his_body_joint_velocities` can reach O(10–100)**, where dump
quantisation alone approaches the tolerance.

**Recommendation:** raise the dump precision before capturing fixtures — one line, additive:
`(*policy_input_file_) << std::setprecision(9);` at construction. Register it in
`upstream_patches.md` as a fixture-only patch. Otherwise the M1 gate may fail on the *reference's*
rounding rather than on our maths.

### Related dumps available for the same run

| Flag | Contents |
|---|---|
| `--policy-input-logfile` | 994-dim policy input, 50 Hz |
| `--target-motion-logfile` | root pos (3) + root quat (4) + 29 joints, **remapped `isaaclab_to_mujoco`** (`:4015-4017`) |
| `--planner-motion-logfile` | closed-loop planner output, same remap (`:3277`) |

The C++ `policy_input` dump does **not** include the encoder input. To get parity on the tokenizer we
must either add an encoder dump (a second fixture-only patch) or validate the encoder indirectly via
`token_state`, which is dims 0–63 of the policy input.

---

## Consequences for the plan

1. **Correct `CLAUDE.md`**: `observation_config.hpp` is a parser, not the observation spec. The spec
   is `g1_deploy_onnx_ref.cpp:1704-1794` + `policy/release/observation_config.yaml`.
2. **M1-T3 is smaller than scoped** — 6 policy terms and 4 encoder terms, not the full registry.
3. **Two fixture-only C++ patches are wanted** before M1-T2: dump precision, and an encoder-input
   dump. Both additive, both to be registered in `upstream_patches.md`.
4. **`69ac0d4` (tokenizer order determinism) is the top parity risk.** Read it before debugging.

---

# Findings from M2 (skeleton and pose override)

## The generator and the policy are different G1 revisions

The generator was trained against `motionbricks/assets/skeletons/g1/g1.xml`; the policy and our
MuJoCo world use `gear_sonic_deploy/g1/g1_29dof.xml`. Both carry the same 29 hinge joints **in the
same order** (asserted at construction in `studio/skeleton_fk.py`), but they place three bodies
differently:

| Joint | generator | deploy | delta |
|---|---|---|---|
| `waist_roll_joint` | `z = 0.044` | `z = 0.035` | 9 mm |
| `waist_pitch_joint` | *(no offset)* | `z = 0.019` | 19 mm |
| `left/right_shoulder_pitch_joint` | `z = 0.24778` | `z = 0.23778` | 10 mm |

At identical joint angles the two models' Cartesian frames therefore disagree by up to **9.3 mm**
(measured at the joint anchors). Joint angles themselves carry across exactly, which is why this is
cosmetic rather than fatal — the policy tracks angles, not positions.

**The rule this implies:** compute on the generator's model whatever the *generator* consumes
(`studio/skeleton_fk.py`); compute on the deploy model whatever describes the *physical robot*
(`studio/telegraph.py`). Neither file may use the other's.

Nobody should "fix" this by pointing both at one XML: the generator's weights encode its own
skeleton's proportions.

## Upstream already implements skeleton forward kinematics

`mujoco_helper.mujoco_qpos_converter.convert_mujoco_qpos_to_motion_transforms` turns a 36-dim qpos
into `g1skel34` global transforms, and it is the same call the agent makes on its own context frames
(`full_agent.py:182`). It handles joint-axis resolution, rest-pose body quaternions, the z-up→y-up
change of basis and the dead-joint fill.

Verified against MuJoCo's own solver on the same model: **max 5.5e-08 m** across joint amplitudes
from 0 to 0.8 rad. Two independent implementations, so this is evidence rather than a restatement.

Build it with `mujoco_qpos_converter(...)` directly, **never** `get_mujoco_converter(...)`: the
latter caches one converter in a module-level global and would hand our XML to the next agent that
asks for one.

It needs only `motion_rep.skeleton`, so forward kinematics runs with no checkpoint and no GPU.

## The pose override reaches its target — and how the measurement fooled me once

**Corrected 2026-08-07.** An earlier version of this section claimed the pose target was a *soft*
constraint that the model pulled back toward the clip manifold, citing 68.7° error on a commanded
shoulder. **That was a measurement artifact and the claim was wrong.** It is recorded here rather
than deleted, because the way it went wrong is a trap anyone measuring this pipeline will hit.

### What the model actually does

MotionBricks is an **in-betweening** model. A plan runs from the current context to the target pose,
and the target lands on the plan's **last** frame. Commanded poses are reached:

| Authored pose | mean error | worst joint |
|---|---|---|
| guard, both hands up | 2.0° | 6.8° |
| jab L, straight arm | 2.6° | 8.0° |
| jab R, straight arm | 2.4° | 8.7° |
| hook L, arm across | 2.7° | 9.6° |
| uppercut R | 2.6° | 8.6° |
| slip, waist lean | 3.2° | 10.8° |
| deep squat | 2.5° | 7.0° |
| *both arms straight up (deliberately absurd)* | 4.5° | 22.5° |

Measured at the plan endpoint, 8 tokens, `walk_boxing`. Reachability is flat across the horizon
range: the same jab measures 2.5–2.7° mean at 6, 8, 12 and 16 tokens.

So **poses can be authored directly in joint space.** The library is not confined to configurations
harvested from the one boxing clip, and strikes do not need a finetune to exist.

### How the earlier measurement went wrong

`generate_new_frames` early-returns until the plan cursor passes `controller_dt × fps` frames
(`full_agent.py:122-124`). Driving it as `next_frame()` then `generate(dt=0.5)` every frame therefore
replans every 15 frames — while a plan is 24–64 frames long. Every sample came from the first quarter
of a plan, and the target is at the end. The same jab measures **65.3° at frame 15 and 7.8° at frame
60**.

The control rules out coincidence: with no pose armed, the same plan's endpoint is 74.5° from the
target rather than 7.8°.

### The runtime rule this implies

**A committed move must not replan while it is executing.** Replanning discards the plan's tail,
which is the move — the strike simply never lands. A commit owns the timeline in
`runtime/intents.py`; the generator must be left alone for its duration. Use
`studio.rehearsal.rehearse_commit` (one forced plan, read whole) for anything that measures a move,
and `rehearse` only for ambient motion where there is no target to arrive at.

## `horizon_tokens` on a pose record is not yet wired to anything

`control_signals` takes `allowed_pred_num_tokens` from the clip registry
(`CLIPS[style]['allowed_pred_num_tokens']`), so a record's `horizon_tokens` is currently inert. The
commit queue (M2-T4) owns the horizon and should pass it through.

## Placement is free — and how, exactly

`spec/intent.md` claimed placement needs no patch. Verified, and the mechanism is worth writing down
because the obvious reading of the code says otherwise.

There are **two** placement overrides upstream, and only one of them runs:

1. **The spring model reads the target directly** (`full_agent.py:244-247`):
   `target_root_pos = ... * (1 - has_specific_target) + specific_target_positions[:, -1, [1, 0]] * has_specific_target`.
   Unconditional. This is the path that works.
2. `_override_target_transforms` (`:301-319`) is an alternative, gated on `BYPASS_SPRING_MODEL`,
   which `navigation_demo` never enables. Reading only this one suggests placement is inert. It is
   not, because of (1).

Measured with the default config: a fighter commanded to `(3.0, 2.0)` ends at `(2.90, 1.94)` —
0.12 m from the target, and 2.31 m from where the same seed goes uncommanded.

**Shape:** `specific_target_positions` must be `[batch, N, 3]` in MuJoCo world coordinates, not a
ground-plane pair — canonicalisation subtracts a 3-D origin and rotates by a 3x3 (`:613-615`). Our
wrapper passed `[1, 1, 2]` from M1 until `M2-T4`; nothing caught it because no test drove placement
through the real generator. The height component is carried through and never read.

The `[1, 0]` index pair is the MuJoCo→motion change of basis (MuJoCo `y` is motion `x`, MuJoCo `x` is
motion `z`), not an arbitrary quirk — the earlier note in `runtime/generator.py` calling it an "axis
swap the module owns" was misleading and is corrected.

**If anyone ever enables `BYPASS_SPRING_MODEL`:** `_override_target_transforms` reshapes
`specific_target_headings` to `target_root_headings.shape`, which is `[batch, NUM_FRAMES_PER_TOKEN]`.
Our one-frame signals raise there. Path (2) needs per-frame targets; path (1) does not.

---

# M2-T5: the telegraph proxy does not survive real motion

**Open problem, found 2026-08-07.** The pose library's *reachability* half works: all 10 authored
poses are produced by the generator at 2.8–4.0° mean and 8.2–14.1° worst-joint error, comfortably
inside the 20° tolerance. The *telegraph* half does not.

`studio/telegraph.py` (M2-T3) decides a move is distinguishable once its bodies leave the baseline's
own spread by 3σ. That was validated against a **synthetic** baseline: a static guard with 2 mm of
jitter. Against real generated motion it breaks:

| pose | reachability worst | telegraph |
|---|---|---|
| hook-left | 13.8° | *never distinguishable* |
| hook-right | 13.7° | 100 ms |
| uppercut-left | 9.8° | *never distinguishable* |
| uppercut-right | 9.5° | 433 ms |

Mirrored poses have near-identical reachability, so they are the same move handed differently; a
measurement that gives one 433 ms and the other nothing is measuring noise. Changing the baseline
from the guard's own plan to 6 s of ambient motion changed *which* poses reported a window but did
not fix the asymmetry.

**Why it fails.** A real fighter's ambient motion is shadow-boxing: the hands travel far and
constantly. The 3σ bar off that spread is larger than the displacement a committed strike adds, so
most moves never cross it, and which ones do is down to where the ambient run happened to be.

**The fix is a different question, not a tuned σ.** "Distinguishable from a baseline" is the wrong
formulation once the baseline is itself moving. What the opponent actually has to do is tell *this*
move from **the other moves that could have been committed** — so the telegraph window should be
measured against the library's own sibling plans: the frame at which this move's trajectory separates
from every other move's. That needs the library to exist first, which it now does.

This is a change to a specified definition (`WORKPLAN` M2-T3 says "distinguishable from the
neutral/guard baseline"), so it wants a human decision before implementation.

**Until then:** every pose stays `draft`. Admission requires a telegraph window
(`spec/pose_record.md`), and none of these has a trustworthy one. The three that produced a number in
one run are not admitted, because the number is not believable.

## `g1_29dof.xml` is not a simulation model, and looks exactly like one

Found 2026-08-08 (`M3-T4`), after both fighters collapsed within half a second in a ring that had
passed every M3-T1 test.

There are two 29-DOF G1 files in `gear_sonic_deploy/g1/`, and the shipped simulation scene uses the
one whose name suggests it is obsolete:

| | `g1_29dof.xml` | `g1_29dof_old.xml` |
|---|---|---|
| included by `scene_29dof.xml` | no | **yes** |
| body masses, inertias, frames | identical | identical |
| joint ranges | identical | identical |
| torque limits | 88/139/50/25/5 N·m, on the **joint** (`actuatorfrcrange`) | the same, on the **motor** (`ctrlrange`) |
| **rotor armature** | **0.0** | 0.01 |
| **joint damping** | **0.0** | 0.05 |
| **friction loss** | **0.0** | 0.2 |
| hand collision meshes | 2 extra | absent |

Everything a diff would draw attention to matches. What differs is the actuator dynamics, and
without rotor armature a stiff PD controller — GEAR-SONIC runs `kp = armature x omega^2`, up to
99 N·m/rad — is unstable at any timestep the budget allows. Measured: a single fighter driven by the
**known-good** M1-T6 loop stands and walks 7.75 m on `scene_29dof.xml` and falls at tick 49 on
`g1_29dof.xml`. Bisected: not the friction (0.5 vs 1.0), not the hand meshes — both were patched out
and it still fell.

**Rule.** Compose scenes from `paths.G1_29DOF_SIM_XML`. `G1_29DOF_XML` is kinematics and meshes;
treat it as an asset for rendering and FK, not as something to put actuators on.

### A second failure hid behind the first

The arena also read each motor's effort limit from `actuator_ctrlrange`, which `g1_29dof.xml` does
not set. That yielded a limit of **zero**, and clipping every torque to zero does not raise, log or
look wrong — the fighter simply melts, exactly like a physics problem. `runtime/policy.effort_limits`
now reads whichever field the model declares (`forcerange`, then `ctrlrange`, then the joint's
`actuatorfrcrange`) and **raises** when none of them is set.

Worth stating as a pattern, because both bugs are the same bug: *a limit read from an unset field
becomes zero, and zero is a valid-looking number.* Anywhere a physical bound is read out of a model,
finding none must raise rather than default.

### What this changed downstream

- `ArenaConfig.friction` is now explicit (0.5, the value `scene_29dof.xml` sets), applied to **every**
  geom including the canvas — MuJoCo combines a contact pair's friction by taking the maximum, so
  setting only the fighters would have left every footstep at the canvas's default.
- The arena no longer needs a floor removed: the sim model does not carry one. The removal is kept,
  now matched by geom **type** rather than by the name `floor`, which the sim model's unnamed geoms
  would never have matched.
- `SCENARIO_SEPARATION` in `tests/test_contact.py` moved 0.17 -> 0.15 m. The sim model's hands have no
  mesh geoms of their own, so a glove reaches about 2 cm less far. Re-scanned, not nudged.
