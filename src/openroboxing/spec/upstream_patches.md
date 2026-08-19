# upstream_patches.md — the registry of what we need from upstream

Version 0.2 · created 2026-08-07 (`M0-T1`) · rewritten 2026-08-19 (the extraction)

`CLAUDE.md` makes the submodule **read-only, full stop**. Nothing in `external/gr00t-wbc` is
modified, so nothing here is a live diff any more. What this file records is the history of the
three patches that once existed, and where each of them ended up.

---

## Upstream tracking

| Field | Value |
|---|---|
| Submodule | `external/gr00t-wbc` → `https://github.com/NVlabs/GR00T-WholeBodyControl.git` |
| Branch tracked | `main` |
| Override | `OPENROBOXING_GR00T_ROOT` names another checkout, bypassing the submodule |

The submodule is **pristine**. Nothing in it is edited, ever — that is what lets it track `main`.

A submodule always pins a SHA in the superproject; git has no follow-the-branch checkout mode. So
"tracking `main`" means the bump is a deliberate command:

    git submodule update --remote external/gr00t-wbc

**After every bump, re-verify:** run the test suite (`tests/test_generator_pose_override.py` is the
one that catches a changed hook signature), and re-check the observation-registry offsets recorded in
`upstream_notes.md` — a rebase invalidates every line number in it.

---

## Registered patches

### P0 — target-pose override (`M2-T1`) — *installed at runtime*

| Field | Value |
|---|---|
| File | `src/openroboxing/runtime/generator.py` — **ours**, not upstream's |
| Symbol | `_wrap_target_transforms` (the call site) · `_apply_target_pose_override` (the body) |
| Hook wrapped | `full_agent._generate_target_joint_transforms`, on the agent *instance* |
| Kind | additive; new optional keys, existing behaviour unchanged when absent |
| Status | **installed at runtime** 2026-08-19 — was *applied* as a diff 2026-08-07 |

No longer a diff. The hook's body and its call site are installed on the agent instance at
construction, so upstream needs no modification. See `_wrap_target_transforms` and
`_apply_target_pose_override` in `src/openroboxing/runtime/generator.py`;
`tests/test_generator_pose_override.py` proves it against a pristine agent.

#### The gate: does the model accept an externally authored pose?

**Yes.** Answered two ways.

*Structurally.* The target pose reaches the model through `_generate_inbetween_frames` as part of
`local_poses`, alongside a boolean `has_local_poses` mask (`full_agent.py:449-456`). That mask exists
because the model was trained with pose constraints present or absent. `_generate_target_joint_transforms`
only *sources* the tensor from the clip library — the model cannot tell where it came from, so an
authored pose travels the identical code path with identical mask semantics.

*Empirically.* Capturing the clip-sampled target, raising every non-root joint 30 cm and feeding it
back produced finite motion that differed from the baseline by up to **0.92 rad**, with the largest
per-frame step **0.197 rad** against a **0.029 rad** median — motion that responds to the override
and stays continuous.

#### Correction to `CLAUDE.md`

`CLAUDE.md` says to extend `_override_target_transforms`. **That is the wrong hook**, for two reasons:

1. It runs at `:149`, *before* `_generate_target_joint_transforms` produces the joint transforms at
   `:153` — there is nothing to override yet.
2. It is gated behind `self.BYPASS_SPRING_MODEL` (`:148`), so it does not even run in the default
   configuration.

The correct hook is immediately **after** `:153`. `_override_target_transforms` is left untouched;
the patch adds a separate, clearly named method rather than overloading it.

This also refines `spec/intent.md`: placement via `specific_target_positions` is free **only when
`BYPASS_SPRING_MODEL` is set**. Otherwise the spring model owns the root and placement must be
expressed through `movement_direction` / `facing_direction`.

#### The pose is on a 34-joint skeleton, not the 29-DOF robot

Measured: `target_global_joint_positions` is `[1, 4, 34, 3]` and `..._rotations` is `[1, 4, 34, 3, 3]`
— the `g1skel34` skeleton, four frames (one token). `CLAUDE.md` and `M2-T2` describe pose records as
"keyframes on the 29-DOF skeleton"; **the generator's pose space is 34 skeleton joints**, and the
29 robot DOFs are downstream of it. The pose record spec must say which space it stores, and a
conversion belongs in the Studio, not in the runtime.

Current behaviour, confirmed by reading: the method returns immediately unless **both**
`specific_target_positions` and `specific_target_headings` are present (`:301-302`), and then
overrides **only** root position and root heading, blended by a `has_specific_target` mask
(`:304-318`). It never touches joint transforms.

The pose target itself is produced by `_generate_target_joint_transforms` (`:321-391`), which selects
a pose from the clip library by one-hot `mode` × `random_seed` (`:326-344`) and then realigns it to
the target heading. **The injection point for an authored key pose is the
`(global_joint_positions, global_joint_rotations)` pair returned at `:391`** — after clip selection,
so an override must bypass the one-hot gather rather than blend with it.

Open question that `M2-T1` must answer before this patch is written: whether the pose-constraint
tensors consumed by `_generate_inbetween_frames` (`:393+`) accept an arbitrary externally-authored
pose with the same mask semantics the model was trained on. `CLAUDE.md` flags this as the assumption
the whole game rests on. **If it does not hold, stop and report.**

### P1 — policy-input dump precision (fixture-only) — *upstream-side, fixture capture only*

| Field | Value |
|---|---|
| File | `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` |
| Site | constructor, in the `policy_input_file_` block |
| Change | `policy_input_file_->precision(9);` |
| Kind | additive; affects only the debug dump |
| Status | **upstream-side, fixture capture only** — applied 2026-08-07, compiles clean |

Applied in a GR00T-WBC working copy used to capture golden fixtures, never in the submodule. The
fixture they produced (`tests/fixtures/golden_policy_io/golden.npz`) is committed, so nothing here
needs them to run. Point `capture_run.sh` at that working copy with `OPENROBOXING_DEPLOY_DIR`.

Rationale in `upstream_notes.md` §Q3: the dump uses default `ostream` precision (~6 significant
digits). For joint-velocity terms of magnitude O(10–100) the quantisation error alone approaches the
`1e-4` M1 gate tolerance, so the gate could fail on the reference's rounding rather than on our
observation maths.

Uses the stream's `precision()` method rather than `std::setprecision` so no new `#include
<iomanip>` is needed — the file includes `<fstream>` only.

### P2 — encoder-input dump (fixture-only) — *upstream-side, fixture capture only*

| Field | Value |
|---|---|
| File | `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` |
| Flag | `--encoder-input-logfile <path>` |
| Kind | additive; inert unless the flag is passed |
| Status | **upstream-side, fixture capture only** — applied 2026-08-07, compiles clean |

Applied in a GR00T-WBC working copy used to capture golden fixtures, never in the submodule. The
fixture they produced (`tests/fixtures/golden_policy_io/golden.npz`) is committed, so nothing here
needs them to run. Point `capture_run.sh` at that working copy with `OPENROBOXING_DEPLOY_DIR`.

The shipped dump covers only the 994-dim policy input. Without an encoder dump, the 1762-dim
tokenizer input can only be validated indirectly through `token_state` (policy dims 0–63), which
localises a parity failure poorly — exactly what `M1-T3` says to avoid.

Touches seven sites, all additive:

| Site | Change |
|---|---|
| member declarations | `std::unique_ptr<std::ofstream> encoder_input_file_;` |
| `G1Deploy` ctor signature | `std::string encoder_input_file_path = ""` **appended last**, so existing call sites compile unchanged |
| ctor body | truncate-then-append, mirroring `policy_input_file_`; `precision(9)` |
| control loop | dump `encoder_obs_buffer_` immediately after the `policy_input_file_` block |
| `main()` locals | `std::string encoderInputLogfile = "";` |
| `--help` text | one line |
| argv parsing | one `else if` branch |
| construction call site | `encoderInputLogfile` appended last |

**Row alignment:** the dump sits directly after the policy-input dump in the control loop, so the two
files are row-aligned tick-for-tick. This is correct because `GatherObservations()` has already run,
and its `token_state` term is what invokes the encoder — leaving `encoder_obs_buffer_` holding
exactly the vector that produced this tick's tokens (`GatherTokenState` → `GatherEncoderObservations`
→ `Encode`).

### Build note — ROS2 is not required

The reference builds and links with `HAS_ROS2=0` (`src/g1/g1_deploy_onnx_ref/CMakeLists.txt:3-24`);
ROS2 is not installed on this machine and is not needed. The prebuilt binary in `target/release/` was
produced elsewhere **with** ROS2 enabled and therefore cannot run here (`librclcpp.so` missing) —
ignore it and build locally.

`target/release/` and `build/` were root-owned from an earlier `sudo` build, which breaks linking.
One-time fix:

```bash
sudo chown -R $USER:$USER gear_sonic_deploy/target gear_sonic_deploy/build
```

---

## Local modifications already present (pre-existing, not ours)

Recorded 2026-08-07, about the shared `GR00T-WholeBodyControl` checkout OpenRoboxing was developed
inside — **not** about the `external/gr00t-wbc` submodule, which is pristine. That checkout carried
substantial uncommitted 23-DOF work unrelated to OpenRoboxing. It was **not** registered here and
**must not** be treated as an OpenRoboxing patch. Of note:

| Path | State |
|---|---|
| `motionbricks/scripts/interactive_demo_g1.py` | +3 lines: target-skeleton viewer overlay. Harmless; viewer-only. |
| `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/` | **pristine vs HEAD** — the reference implementation is clean. Good. |
| `gear_sonic_deploy/src/g1/g1_deploy_onnx_23dof{,_originale}/` | untracked 23-DOF ports; ignore. |
| `gear_sonic_deploy/policy/my23dof/` | untracked 23-DOF weights; ignore. |

OpenRoboxing targets the **29-DOF** G1 and the `policy/release/` weights throughout.

---

## Closed: the fork is not needed

`M0-T1` required this checkout to become a fork of NVlabs, so that P0 could live in git. Installing
P0 at runtime removes the reason. There is no fork, and no plan for one: OpenRoboxing consumes
upstream and never modifies it.
