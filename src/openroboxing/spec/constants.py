"""Canonical constants for OpenRoboxing.

`CLAUDE.md`: *"Every magic number is a named constant with a comment saying where it came from."*
Every value here is cited to its source in ``src/openroboxing/spec/rates.md``; change them there first.

Conventions
-----------
- Rates are in hertz, durations in seconds, tick counts in 50 Hz control ticks.
- Joint-order permutations are **deliberately absent** from this module. `CLAUDE.md` invariant 4
  requires them to be derived by joint *name* at runtime (``runtime/conventions.py``, M1-T1).
  ``rates.md`` records the C++ arrays as an assertion target only.
"""

from __future__ import annotations

# --- Rates -------------------------------------------------------------------------------------
TICK_HZ: int = 50
"""Policy control rate. Source: g1_deploy_onnx_ref.cpp:2162 (control_dt_ = 0.02)."""

GENERATOR_HZ: int = 30
"""MotionBricks native output rate. Source: motionbricks_root/version_1/config.yaml:209."""

INTENT_QUEUE_HZ: int = 30
"""Server-side intent queue service rate. Source: CLAUDE.md."""

RING_STREAM_FPS: int = 30
"""Client display stream rate. Display only. Source: CLAUDE.md."""

TICK_DT: float = 1.0 / TICK_HZ
GENERATOR_DT: float = 1.0 / GENERATOR_HZ

# --- Tokens ------------------------------------------------------------------------------------
NUM_FRAMES_PER_TOKEN: int = 4
"""Source: full_agent.py:34, clips.py:16."""

MIN_TOKENS: int = 6
"""Source: motionbricks_root/version_1/config.yaml:48."""

MAX_TOKENS: int = 16
"""Source: motionbricks_root/version_1/config.yaml:49."""

NUM_TIME_TOKENS: int = MAX_TOKENS - MIN_TOKENS + 1
"""Length of the `allowed_pred_num_tokens` mask. Source: root_backbone.py:197."""

SECONDS_PER_TOKEN: float = NUM_FRAMES_PER_TOKEN / GENERATOR_HZ

COMMIT_HORIZON_TICKS: int = 30
"""Minimum lead from commit to execution = 0.6 s at 50 Hz. Per-match parameter.

A *floor*, not a fixed delay: a commit queued behind a running move starts the instant that move
ends, because the readable window has already elapsed. Source: CLAUDE.md, spec/intent.md 1.0.

**Currently inert, and known to be.** The policy reads the reference 45 ticks ahead and the stream
buffers 20 more, so nothing a player commits can reach the fighter sooner than **65 ticks (1.3 s)** -
and 30 < 65. Raising this above 65 would start to bind; lowering it changes nothing at all. Measured
2026-08-08; see docs/ASSUMPTIONS.md A24 before tuning it.
"""

RING_SIZE_M: float = 4.90
"""Side length inside the ropes, metres. A World Boxing / AIBA elite ring is 4.90 m square.

The default for `ArenaConfig.ring_size`, which is what a match actually reads - this is here so that
anything deriving a distance or a duration from the ring (`intents.approach_timeout_ticks`) shares
one number with the geometry rather than repeating it. Whether a G1-sized ring should be smaller is
a feel question for M4-T4, which is why the arena keeps it a parameter.
"""

APPROACH_SPEED_M_S: float = 0.83
"""How fast a fighter closes on a placement, sustained, under physics.

**Measured, not chosen** (2026-08-08, scratchpad/probe_arrival.py): an open-ended approach reaches a
target 1.0 m away in 2.3 s and one 2.5 m away in 4.1 s, so the marginal rate is 1.5 m / 1.8 s =
0.83 m/s once walking, after about 1.0 s of getting started.

Not to be confused with the ~1.6 m/s of a single forced plan dashing (spec/intent.md 1.0's retired
`MOVE_TRAVEL_SPEED_M_S`): a sustained approach replans every half second and pays for it. This is
the number a client uses to tell a player how long a placement will cost them, and the number
APPROACH_TIMEOUT_TICKS is derived from.
"""

ARRIVAL_RADIUS_M: float = 0.40
"""How near a placement counts as arrived, ending a commit's approach phase and throwing the pose.

**Measured, not chosen** (2026-08-08, scratchpad/probe_arrival.py): over ten placements around a
fighter - forward, lateral, behind, 0.3 m to 2.5 m - the *worst* closest approach was **0.30 m** (a
target directly behind, which the fighter has to walk around to). Every one of the ten reached 0.40 m
and did so within 4.3 s, so this is the tightest radius that always closes, plus a margin.

It is also half the measured CONTACT_RANGE_M, which is the sanity check that matters in play: a
player who placed the shadow at punching distance arrives inside punching distance.

Tighter would be worse, not better - at 0.25 m one of the ten never arrived and would have run to
APPROACH_TIMEOUT_TICKS, stalling everything queued behind it.
"""

APPROACH_LEG_M: float = 1.0
"""How far ahead the generator is aimed during an approach, at most. Zero means "the whole way".

A placement is where the *commit* ends; this is where the *plan* is asked to end. The distinction
exists because MotionBricks in-betweens toward the target as its plan's last frame: give it a target
it can reach inside one plan and the plan travels; give it one three plans away and the plan arrives
at the target while the body is still walking, after which the reference has nowhere left to pull.
Measured 2026-08-17 at 2.6 m off-axis: the plan sat 0.15 m from the placement while the body stalled
at 0.685 m and then *drifted back out* to 0.95 m.

**Derived from the plan's own reach.** A plan is 11-16 tokens (the model's own choice, ~1.5-2.1 s at
GENERATOR_HZ) and a fighter sustains APPROACH_SPEED_M_S, so one plan covers 1.2-1.7 m. A 1.0 m leg
is inside that at every plan length, which is the property that matters: the plan's end must be
somewhere the body can actually arrive.

The idea is ARDY's (SIGGRAPH 2026, nv-tlabs): kinematic constraints are placed at reachable points
along the path and consumed as they are passed. Here the leg is re-derived from the fighter's true
position every frame, so "consumed" needs no bookkeeping - the leg slides forward as the body moves
and shrinks onto the placement itself once the fighter is within one leg of it.
"""

MAX_OUTSTANDING_COMMITS: int = 5
"""How many commits may be unfinished at once. Source: project owner, 2026-08-08 (spec/intent.md 1.0).

A game-feel decision, not a derived quantity. It was taken when a move was 0.8-2.1 s, making a full
queue 4-10 s of pre-planned action against a 60 s round.

Since spec/intent.md 1.1 a move is its walk plus its pose - measured at 3-5 s - so five is now
**20-30 s**, half a round committed in advance and unrecallable. That may be exactly the tension the
no-cancellation rule is for, or far too much. Not changed on a guess: it is a feel question for
M4-T4, and `tools/tune.py --knob queue_depth` is what measures it. See docs/ASSUMPTIONS.md A23.
"""

POSE_DWELL_TICKS: int = 74
"""How long a fighter stands in its committed pose after arriving, before the next queued commit
becomes current. 74 ticks = 1.48 s at 50 Hz.

**Measured, not chosen** (2026-08-13, `tools/measure_dwell.py`): the settle time of each of the ten
`poses/v0.1` records, thrown at a placement 2.5 m away, expressed in ticks after first arrival. The
distribution over the library is min 0, median 1, p90 19, max **74** - all ten:
`[0, 0, 0, 0, 0, 2, 5, 12, 12, 74]`. This constant is the **library maximum**.

The maximum and not the p90 on purpose. It guarantees *every* pose in the library has settled before
the next commit starts, and it errs toward the strike being **visible** - which is the exact symptom
the project owner reported ("commits melt together"). A game that feels slightly slow is a
recoverable error; one that reproduces the reported bug is not.

Why a dwell exists at all: without it the next queued commit becomes current at the instant of
arrival, so the strike is cut short and never reads as a strike. That is the melting. See
spec/intent.md 2.0.

**The known refinement, and what it needs.** Per-pose dwells recorded in each pose record would fit
this codebase - pose records already carry `telegraph_ms` and `generator_error_rad` - and the spread
above (0 to 74 for the same 1.48 s constant) says it would help. But it needs a positive floor and a
validator that *raises* on a non-positive dwell: five of the ten poses measured exactly 0, and a
dwell of 0 makes `end_tick == strike_at`, so `is_scheduled(strike_at)` is False, the pose is never
held for a single tick, `current()` skips straight to the next commit - melting reintroduced,
silently - while `strike_at` is still populated, so `league/scoring.py` counts a hit that was never
displayed.

One interaction to know about: 74 ticks exceeds any forced plan in v0.1 (a 6-token plan is 40 ticks,
an 8-token plan 53), which is fine, because holding past the plan is the intent. A pose authored at
`MAX_TOKENS` = 16 would be 107 ticks, and a 74-tick dwell would cut *it* short. Nothing in v0.1 is
affected.
"""

POSE_SETTLE_IMPROVEMENT_RAD: float = 0.01
"""How much a pose must still be closing by for a move to count as unfinished.

A commit ends when the body has stopped getting nearer the pose, not when a counter runs out
(`spec/intent.md` 2.2). "Stopped" needs a floor, and this is it: the best error over the last replan
window must beat the window before it by more than this, or the move is done.

**Derived from the two errors that are already measured**, not chosen. The pose library is admitted
at a ``generator_error_rad`` around 0.1 rad, and the policy tracks its reference at 0.04-0.07 rad
mean in steady state (sparring bench, 2026-08-17). An improvement an order of magnitude below either
of those is not improvement — it is the residual oscillation of a fighter already standing in the
pose, and waiting for it costs the queue behind it real time.
"""

MAX_DWELL_TICKS: int = 3 * POSE_DWELL_TICKS
"""The longest a commit may hold its pose before it ends regardless. A guard, not a mechanism.

An event-driven completion has one failure the counted dwell did not: a pose the body never settles
into holds the whole queue **forever**. So the measured dwell of the slowest pose in the library
(:data:`POSE_DWELL_TICKS`) is given a factor of three, the same shape of margin
``APPROACH_TIMEOUT_TICKS`` has over the slowest observed arrival, and a commit that reaches it ends
with ``completed_by = "timeout"`` so a bench and a match record can tell the two apart.
"""

GENERATOR_POSE_TOLERANCE_RAD: float = 0.1795
"""How close the generator must get to an authored pose for that pose to be admitted.

**Measured, not chosen** (2026-08-13, `tools/measure_dwell.py`): 0.1795 rad = **10.3 degrees**, the
worst pose's worst frame over its final 15-frame replan cycle (`hook-left` / `jab-left`). Mean error
over the same final cycle, across the library, is 7.7 degrees.

Measured over a **full replan cycle**, not at a single frame: the error oscillates within each cycle,
so a single-frame reading flatters or overstates it depending on phase.

Independent check worth recording: this value came out **identical under three different
settle-detection rules**, so it never depended on that metric.

It is looser than the forced-plan figure `poses/v0.1` was originally admitted against - 2-3 degrees
measured with **no** placement, 3.7-5.0 degrees with one - because a continuously-armed pose is
*converged on* rather than landed exactly. See spec/pose_record.md.
"""

# --- Robot -------------------------------------------------------------------------------------
NUM_JOINTS: int = 29
"""Source: policy_parameters.hpp."""

QPOS_DIM: int = 36
"""3 root position + 4 root quaternion (wxyz) + 29 joints. Source: clips.py:114."""

NUM_TRACKED_BODIES: int = 14
"""Bodies in a reference motion. Source: reference/example/*/metadata.txt."""

ACTION_DIM: int = 29
"""Source: model_decoder.onnx output shape."""

# --- Policy I/O --------------------------------------------------------------------------------
POLICY_INPUT_DIM: int = 994
"""Measured from model_decoder.onnx. NOTE: the 436 in observation_config.yaml's header is stale."""

ENCODER_INPUT_DIM: int = 1762
"""Measured from model_encoder.onnx."""

TOKEN_DIM: int = 64
"""Encoder output width. Source: observation_config.yaml `dimension: 64`."""

ENCODER_INPUT_REQUIRED_DIM_G1: int = 644
"""Encoder dims the g1 mode's `required_observations` list computes: 4 + 290 + 290 + 60.

The rest are left at zero by the per-mode fill (g1_deploy_onnx_ref.cpp:2041-2089).
"""

ENCODER_INPUT_NONZERO_DIM_G1: int = 640
"""Encoder dims that actually carry information in g1 mode.

644 minus `encoder_mode_4`'s 4 dims, which are identically zero because that term writes the mode id
as a scalar (0 for g1) followed by three zero fills - it is not a one-hot.

Measured from the golden capture, not derived: see tests/test_golden_fixture.py.
"""

HISTORY_LEN: int = 10
"""Every policy history term is `10frame_step1`."""

ENCODER_MODE_G1: int = 0
ENCODER_MODE_TELEOP: int = 1
ENCODER_MODE_SMPL: int = 2

# --- Default standing pose -----------------------------------------------------------------------
# Source: policy_parameters.hpp:210-240 (`default_angles`), which stores them as a bare array in
# MuJoCo/hardware order. Keyed by joint name here so callers reorder via runtime.conventions instead
# of trusting an index - CLAUDE.md invariant 4.
#
# Used two ways by the reference: subtracted from measured joint positions before they enter the
# observation history (g1_deploy_onnx_ref.cpp:2847), and added back to form the motor target
# (`target = action * action_scale + default_angle`, :3122).
DEFAULT_ANGLES_BY_JOINT: dict[str, float] = {
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# --- Actuators -----------------------------------------------------------------------------------
# Source: policy_parameters.hpp:37-65. Gains are derived from these, not transcribed:
#   stiffness    = armature * omega^2                (omega = 2*pi*10 Hz)
#   damping      = 2 * zeta * armature * omega       (zeta = 2.0)
#   action_scale = 0.25 * effort_limit / stiffness
# and the joint target is `action * action_scale + default_angle` (:29).
NATURAL_FREQ: float = 10.0 * 2.0 * 3.1415926535
"""rad/s. Deliberately reproduces the C++ literal, including its truncated pi (:46)."""

DAMPING_RATIO: float = 2.0

MOTOR_ARMATURE: dict[str, float] = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}

MOTOR_EFFORT_LIMIT: dict[str, float] = {
    "5020": 25.0,
    "7520_14": 88.0,
    "7520_22": 139.0,
    "4010": 5.0,
}

# Which motor drives which joint. Source: the per-line comments on `g1_action_scale`
# (policy_parameters.hpp:109-139). Keyed by name so ordering goes through runtime.conventions.
MOTOR_TYPE_BY_JOINT: dict[str, str] = {
    "left_hip_pitch_joint": "7520_22",
    "left_hip_roll_joint": "7520_22",
    "left_hip_yaw_joint": "7520_14",
    "left_knee_joint": "7520_22",
    "left_ankle_pitch_joint": "5020",
    "left_ankle_roll_joint": "5020",
    "right_hip_pitch_joint": "7520_22",
    "right_hip_roll_joint": "7520_22",
    "right_hip_yaw_joint": "7520_14",
    "right_knee_joint": "7520_22",
    "right_ankle_pitch_joint": "5020",
    "right_ankle_roll_joint": "5020",
    "waist_yaw_joint": "7520_14",
    "waist_roll_joint": "5020",
    "waist_pitch_joint": "5020",
    "left_shoulder_pitch_joint": "5020",
    "left_shoulder_roll_joint": "5020",
    "left_shoulder_yaw_joint": "5020",
    "left_elbow_joint": "5020",
    "left_wrist_roll_joint": "5020",
    "left_wrist_pitch_joint": "4010",
    "left_wrist_yaw_joint": "4010",
    "right_shoulder_pitch_joint": "5020",
    "right_shoulder_roll_joint": "5020",
    "right_shoulder_yaw_joint": "5020",
    "right_elbow_joint": "5020",
    "right_wrist_roll_joint": "5020",
    "right_wrist_pitch_joint": "4010",
    "right_wrist_yaw_joint": "4010",
}

# Six joints carry a 2x multiplier on their PD gains (policy_parameters.hpp kps:148-149,154-155,158-159
# and the matching kds). The multiplier applies ONLY to kp/kd - `g1_action_scale` uses the unmultiplied
# stiffness, so action scaling and PD stiffness genuinely disagree for these joints.
PD_GAIN_MULTIPLIER_BY_JOINT: dict[str, float] = {
    "left_ankle_pitch_joint": 2.0,
    "left_ankle_roll_joint": 2.0,
    "right_ankle_pitch_joint": 2.0,
    "right_ankle_roll_joint": 2.0,
    "waist_roll_joint": 2.0,
    "waist_pitch_joint": 2.0,
}

# --- Tolerances --------------------------------------------------------------------------------
OBS_PARITY_TOL: float = 1e-4
"""M1-T3 gate. Source: WORKPLAN.md. See rates.md for the dump-precision caveat."""

ACTION_PARITY_TOL: float = 1e-3
"""M1-T4 gate. Source: WORKPLAN.md."""
