# golden_policy_io — observation-parity fixtures (M1-T2)

`golden.npz` is the definition of correctness for `runtime/obs.py`. It is a 400-tick window of the
**C++ reference** deploy running under MuJoCo, capturing exactly what the shipped policy was fed.

## Contents

**Outputs** — what the reference fed its networks. These are what `obs.py` must reproduce.

| Array | Shape | What it is |
|---|---|---|
| `policy_input` | `(400, 994)` | the decoder input, `obs_buffer_`, in `observation_config.yaml` order |
| `encoder_input` | `(400, 1762)` | the tokenizer input, `encoder_obs_buffer_` |
| `target_motion` | `(400, 36)` | root pos (3) + root quat wxyz (4) + 29 joints, **remapped to MuJoCo order** |

**Inputs** — the raw signals those were built from, via `--enable-csv-logs`. Without these a "parity"
test would only check that our code re-derives the golden vector *from itself*.

| Array | Shape | Convention |
|---|---|---|
| `state_q` | `(400, 29)` | joint positions, **hardware/MuJoCo order, default offsets applied** |
| `state_dq` | `(400, 29)` | joint velocities, hardware order |
| `state_action` | `(400, 29)` | policy actions, hardware order, scaled + offset — **the M1-T4 target** |
| `state_base_ang_vel` | `(400, 3)` | base IMU angular velocity |
| `state_base_quat` | `(400, 4)` | base IMU orientation, `wxyz`, unit norm |
| `state_token_state` | `(400, 64)` | encoder output |
| `state_encoder_mode` | `(400, 1)` | active encoder mode — all `0` (`g1`) |

| Scalar | Value |
|---|---|
| `start_tick` | 300 — offset into the original 1549-tick capture |
| `num_ticks` | 400 |
| `capture_length` | 1549 |
| `upstream_sha` | commit the capture was produced at |

Every array is row-aligned: row `i` is the same 50 Hz control tick.

### The alignment is proven, not assumed

The dump flags and `StateLogger` are separate code paths, so their row correspondence has to be
established. `token_state` appears in both — as `policy_input[:, 0:64]` and as `state_token_state` —
and they agree with **max abs error `0.000e+00`** at zero shift, while ±1 and ±2 tick shifts give
errors of 0.125 and 0.188. Asserted in `tests/test_golden_fixture.py`.

### The transform obs.py must implement

`state_q` is raw hardware: MuJoCo/motor order, with default standing angles included. The
observation history stores something different — `g1_deploy_onnx_ref.cpp:2827`:

```cpp
body_q[i] = joint_state[mujoco_to_isaaclab[i]].q() - default_angles[mujoco_to_isaaclab[i]];
```

i.e. **reorder to IsaacLab, then subtract the default angles**. Reproducing that from `state_q` and
matching `policy_input`'s `his_body_joint_positions` block is precisely the M1 gate.

Captured at `4cf41a932235019dde9b92a63271c3b6bf80c379` (pinned upstream `a9d20b2` + patches P1, P2).

## Why ticks 300–700

The full capture is 1549 ticks. The reference motion is static until ~tick 100 and finishes by
~tick 1000; 300–700 sits in the middle of continuous motion (mean |Δq| ≈ 0.28–0.44 rad/tick summed
over 29 joints). A window in the static region would pass a parity test trivially.

## Reproducing

Two processes. Both patches (`spec/upstream_patches.md` P1, P2) must be applied — in a separate
**GR00T-WBC working copy**, never in the `external/gr00t-wbc` submodule, which stays pristine.
Every path below is relative to that working copy's root; `OPENROBOXING_DEPLOY_DIR` is how
`capture_run.sh` is pointed at its `gear_sonic_deploy/`. None of this is needed to run the tests:
`golden.npz` is committed.

```bash
# 0. one-time: target/ and build/ must be writable (an earlier sudo build left them root-owned)
sudo chown -R $USER:$USER gear_sonic_deploy/target gear_sonic_deploy/build

# 1. build the reference. ROS2 is optional and NOT needed; the prebuilt binary shipped in
#    target/release/ was built elsewhere with ROS2 on and cannot run here.
cd gear_sonic_deploy
source scripts/setup_env.sh
export HAS_ROS2=0
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target g1_deploy_onnx_ref -j$(nproc)

# 2. terminal A — MuJoCo sim, headless, DDS on loopback
.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py --no-enable-onscreen --interface lo

# 3. terminal B — the reference deploy, with both dumps enabled
cd gear_sonic_deploy
./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --input-type manager \
  --disable-crc-check \
  --policy-input-logfile  <out>/policy_input.csv \
  --encoder-input-logfile <out>/encoder_input.csv \
  --target-motion-logfile <out>/target_motion.csv
```

Then drive the operator state machine. The keyboard handler reads `STDIN_FILENO` with non-blocking
reads, so keys can be piped from a FIFO — see `src/openroboxing/parity/capture_run.sh`:

- `]` once `Init Done` appears → `ProgramState::CONTROL`
- `T` after ~3 s → play the reference motion to its end

CSV rows have a **trailing comma**, so a naive split yields a spurious final empty field. Strip it.

## Gotchas that cost time

- `--input-type manager` still reports *"Switched to: KEYBOARD"*. That is the manager selecting its
  keyboard sub-interface, not a failure — but it does mean playback will not start on its own.
- Opening the key FIFO write-only (`exec 3>fifo`) deadlocks: a FIFO open for writing blocks until a
  reader appears, and the reader is the deploy started afterwards. Use `exec 3<>fifo`.
- Do not run the capture script under `set -u`: `scripts/setup_env.sh` dereferences unset variables
  and aborts the script the moment it is sourced.

## What the capture confirmed

Facts now established from data rather than from reading code — all recorded in
`spec/upstream_notes.md`:

- Shapes are **994 / 1762 / 36**, matching the shipped ONNX exactly.
- **The per-mode zero-fill is real.** In `g1` mode only three encoder terms are non-zero
  (`motion_joint_positions_10frame_step5`, `motion_joint_velocities_10frame_step5`,
  `motion_anchor_orientation_10frame_step5`) — **640 non-zero dims of 1762**. The 720-dim SMPL block
  and all VR-3point terms are identically zero and must not be implemented.
- **`encoder_mode_4` is all-zero** in `g1` mode: it writes the mode id as a scalar (0 for g1) plus
  three zero fills. It is *not* a one-hot.
- **History layout is frame-major `[frame, joint]`, oldest-first**: index 0 is the oldest frame,
  index 9 the current one. Established by checking that frame `k` at tick `t+1` equals frame `k+1` at
  tick `t`, residual exactly `0.0`.
- **Patch P1 was necessary.** `his_body_joint_velocities` spans −33.07 … 29.14. At the stock ~6
  significant digits a value of 33.0694 quantises at ~1e-4 — the same order as the M1 gate tolerance.
  The fixture carries 9 significant digits.

## Not captured

Policy **actions** are not dumped by the reference. M1-T4 compares against actions; they can be
recovered from `his_last_actions_10frame_step1` (policy dims 674–964), whose newest frame at tick
`t+1` is the action emitted at tick `t`. If that proves insufficient, re-capture with
`--enable-csv-logs`, which writes per-signal CSVs including actions.
