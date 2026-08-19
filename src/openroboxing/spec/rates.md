# rates.md — canonical rates and dimensions

Version 0.1 · created 2026-08-07

`CLAUDE.md`: *"Never invent a number."* Every constant here is either measured or cited to the file
that defines it. Code imports these from `src/openroboxing/spec/constants.py`; **no literals in code.**

---

## Rates

| Constant | Value | Source |
|---|---|---|
| `TICK_HZ` | 50 | policy rate; `g1_deploy_onnx_ref.cpp:2162` (`control_dt_ = 0.02`) |
| `GENERATOR_HZ` | 30 | `external/gr00t-wbc/motionbricks/out/motionbricks_root/version_1/config.yaml:209` (`fps: 30`) |
| `INTENT_QUEUE_HZ` | 30 | design decision, `CLAUDE.md` |
| `RING_STREAM_FPS` | 30 | display only, `CLAUDE.md` |
| `PUBLISH_HZ` | 500 | `publish_dt_ = 0.002` (`:2161`) — motor command rate; **not used by us** (no DDS) |
| `PLANNER_HZ` | 10 | `planner_dt_ = 0.1` (`:2163`) — upstream replan cadence, reference only |

The bridge resamples 30 → 50 Hz. Ratio 5:3 — every 3 generator frames become 5 control ticks.

## Tokens

| Constant | Value | Source |
|---|---|---|
| `NUM_FRAMES_PER_TOKEN` | 4 | `full_agent.py:34`, `clips.py:16` |
| `MIN_TOKENS` | 6 | `motionbricks_root/version_1/config.yaml:48` |
| `MAX_TOKENS` | 16 | `motionbricks_root/version_1/config.yaml:49` |
| `NUM_TIME_TOKENS` | 11 | derived: `MAX_TOKENS - MIN_TOKENS + 1`; `root_backbone.py:197` |
| Seconds per token | 4/30 = 0.1333 s | derived |
| Pose phase length | 0.800 s – 2.133 s | derived from 6–16 tokens |
| `APPROACH_SPEED_M_S` | 0.83 m/s | measured; `scratchpad/probe_arrival.py`, 2026-08-08 |
| `ARRIVAL_RADIUS_M` | 0.40 m | measured; worst closest approach over ten placements was 0.30 m |

A commit's *total* length is not in this table because it is not a rate: since `spec/intent.md` 1.1 a
move walks to its placement before it throws, so it lasts however long that takes plus the pose
phase. Measured at **3–5 s** in agent play in a 4.90 m ring.

**`allowed_pred_num_tokens` is an 11-slot mask over the inclusive range 6…16**, index `i` ↔
`MIN_TOKENS + i` tokens (`root_backbone.py:195-202`). This confirms `CLAUDE.md`'s rates table — the
11-slot length is not a discrepancy.

`walk_boxing` is one of only two clips whose mask is all-ones, i.e. the full 6–16 range is available
(`clips.py:152-156`). Most locomotion clips are restricted to `[1]*6 + [0]*5` → 6–11 tokens.

| Constant | Value | Source |
|---|---|---|
| `COMMIT_HORIZON_TICKS` | 30 (= 0.6 s @ 50 Hz) | `CLAUDE.md`, per-match parameter |
| `POSE_DWELL_TICKS` | 74 (= 1.48 s @ 50 Hz) | measured; `tools/measure_dwell.py`, 2026-08-13 |
| `GENERATOR_POSE_TOLERANCE_RAD` | 0.1795 rad (= 10.3°) | measured; `tools/measure_dwell.py`, 2026-08-13 |

Both come from one run, reproducible to identical numbers on an independent run:

```
.venv_mb/bin/python -m openroboxing.tools.measure_dwell
```

(seed 1234, placement 2.5 m away, 8 s per pose, all ten `poses/v0.1` records.)

**`POSE_DWELL_TICKS` — how long a fighter holds its committed pose after arriving**, before the next
queued commit becomes current. The measured settle times over the library are
`[0, 0, 0, 0, 0, 2, 5, 12, 12, 74]` ticks — min 0, median 1, p90 19, max 74 (0.00 s … 1.48 s). The
constant is the **library maximum, deliberately**: it guarantees every pose has settled before the
next commit starts, and it errs toward the strike being *visible*, which is the symptom that prompted
the redesign ("commits melt together"). A game that feels slightly slow is recoverable; one that
reproduces the reported bug is not. Per-pose dwells are the known refinement — but they need a
positive floor and a validator that raises on a non-positive dwell, because five of ten poses
measured exactly 0 and a dwell of 0 silently reintroduces the melting while still scoring the hit.
74 ticks exceeds any forced plan in v0.1 (6 tokens = 40 ticks, 8 tokens = 53), which is the point;
a pose authored at `MAX_TOKENS` = 16 would be 107 ticks and would be cut short, but there is no such
pose in v0.1.

**`GENERATOR_POSE_TOLERANCE_RAD` — how close the generator must get to an authored pose** for that
pose to be admitted. 0.1795 rad is the worst pose's worst frame over its final 15-frame replan cycle
(`hook-left` / `jab-left`); the library mean over that same final cycle is 7.7°. It is measured over
a **full replan cycle**, not one frame, because the error oscillates within each cycle and a
single-frame reading flatters or overstates it depending on phase. The same value came out under
three different settle-detection rules, so it does not depend on that metric. It is looser than the
figure `poses/v0.1` was originally admitted against (2–3° with no placement, 3.7–5.0° with one)
because a continuously-armed pose is converged on rather than landed exactly — see `pose_record.md`.

## Robot dimensions

| Constant | Value | Source |
|---|---|---|
| `NUM_JOINTS` | 29 | `policy_parameters.hpp` |
| `QPOS_DIM` | 36 | 3 root pos + 4 root quat + 29 joints; `clips.py:114` |
| `NUM_TRACKED_BODIES` | 14 | reference-motion `metadata.txt` |
| `ACTION_DIM` | 29 | `model_decoder.onnx` output |

## Policy I/O

| Constant | Value | Source |
|---|---|---|
| `POLICY_INPUT_DIM` | 994 | `model_decoder.onnx` input, measured |
| `ENCODER_INPUT_DIM` | 1762 | `model_encoder.onnx` input, measured |
| `TOKEN_DIM` | 64 | `model_encoder.onnx` output; `observation_config.yaml` `dimension: 64` |
| `ENCODER_INPUT_ACTIVE_DIM_G1` | 644 | derived; see `upstream_notes.md` §Q2 per-mode zero-fill |
| `HISTORY_LEN` | 10 | every policy history term is `10frame_step1` |

Encoder mode ids: `g1: 0`, `teleop: 1`, `smpl: 2`.

## Conventions

| Item | Value | Source |
|---|---|---|
| MuJoCo quaternion order | `wxyz` | MuJoCo convention |
| Other code quaternion order | `xyzw` | `mujoco_helper.py` handles the swap |
| Action scaling | `0.25 × effort_limit / stiffness` | `policy_parameters.hpp:27, :109` |
| Joint target | `action × action_scale + default_angle` | `policy_parameters.hpp:29` |
| PD stiffness | `armature × ω²`, ω = 2π·10 Hz | `policy_parameters.hpp:21, :46` |
| PD damping | `2 × ζ × armature × ω`, ζ = 2.0 | `policy_parameters.hpp:22, :47` |

## Joint order permutations — reference values only

`CLAUDE.md` invariant 4: these **must be derived by name** at runtime (`M1-T1`). They are recorded
here **only as the assertion target** — deriving by name and then checking equality against these is
correct; copying them into code is not.

From `policy_parameters.hpp:100-104`:

```
isaaclab_to_mujoco = [0, 3, 6, 9,13,17, 1, 4, 7,10,14,18, 2, 5, 8,
                     11,15,19,21,23,25,27,12,16,20,22,24,26,28]
mujoco_to_isaaclab = [0, 6,12, 1, 7,13, 2, 8,14, 3, 9,15,22, 4,10,
                     16,23, 5,11,17,24,18,25,19,26,20,27,21,28]
```

Verified 2026-08-07: both are valid permutations of `range(29)` and mutually inverse in both
directions.

Subset index arrays (lower body, wrists, upper body, VR points) are at `policy_parameters.hpp:68-97`.

## Tolerances

| Gate | Value | Source |
|---|---|---|
| M1-T3 observation parity | max abs err < 1e-4 | `WORKPLAN.md` |
| M1-T4 action parity | max abs err < 1e-3 | `WORKPLAN.md` |

**Caveat:** the C++ `policy_input` dump carries only ~6 significant digits by default. See
`upstream_notes.md` §Q3 and patch P1 — capture fixtures at higher precision or the 1e-4 gate is
partly measuring the reference's own rounding.
