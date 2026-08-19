# OpenRoboxing — project definition v0.8

**TORC · Trieste · aligned to `OpenRoboxing_project_presentation.pdf` (14 slides, 03 Aug 2026)**
Supersedes PUGIL v0.7. The deck now carries the narrative; this document carries the substance.

---

## 0. What the deck locks — and eight corrections it needs

### 0.1 Locked by the deck
| # | Decision |
|---|---|
| 1 | **Name: OpenRoboxing**, under TORC. `torc.it/openroboxing`, `ciao@torc.community` |
| 2 | **Humans play it.** Season 0 is one division, one table, and every match on the table was fought by a person |
| 3 | **Six keys.** A player's loadout is six bound moves with durations, not free-form pose authoring |
| 4 | **Browser client, server runtime.** The same client runs a hotseat match at a meetup and a ranked match online |
| 5 | **Studio: author a motion, then finetune the policy to execute it.** Two steps, one web tool |
| 6 | **Season 0 = 10 weeks**, Swiss pairing, Glicko-2, 8 matches to rank, top 4 playoff, one belt at the Trieste Open |
| 7 | **Scoring exists in outline**: landed impulses, knockdowns, ring control, aggression — published and versioned as Rules v1.0 |
| 8 | **Fight nights from month 4** at TORC meetups; the finale is a public night in Trieste |
| 9 | **Hardware later, through the gates**, and the laptop generates while only the policy runs on the Jetson |
| 10 | **Fork public at week 8**, Apache 2.0, weights from upstream |
| 11 | **Six workstreams**: web · events · rules · RL · data · packaging |

### 0.2 Corrections to make before the deck is shown again

**C1 — `GR00T N1.5` on slide 7 is the wrong model, and it is the one error that costs credibility.**
GR00T N1.5 is NVIDIA's vision-language-action foundation model. The motion generator here is
**MotionBricks**, and the whole-body policy is **GEAR-SONIC**, both from
`NVlabs/GR00T-WholeBodyControl`. An expert audience catches this in seconds. Fix the layer card to
read *Motion generator — MotionBricks · upstream*.

**C2 — slide 5 "base policy: GR00T WBC v1.0 · upstream"** is acceptable as our version label, but say
*GEAR-SONIC (GR00T-WBC)* at least once so the lineage is unambiguous.

**C3 — tick rate is inconsistent.** Slide 4 shows contact at `t+30` labelled 0.6 s, which is 50 Hz.
Slide 7 says the intent server runs at 30 Hz. Resolve as: **ticks are 50 Hz** (the policy rate), the
generator is natively 30 Hz, the ring stream is 30 FPS, and the intent server *services the queue* at
30 Hz. Put one canonical rate table in the spec so nobody has to reverse-engineer it.

**C4 — "objective: pose tracking under contact" understates a research problem.** Contact-aware
finetuning is the impact-robustness question — the policy penalises contact outside feet, hands and
elbows during upstream training, so *being hit is out of distribution*. That is not a 4-hour job.
For Season 0 the honest objective is **pose-tracking fidelity on our library**; contact robustness is
a research track. Either change the mockup label or be ready to explain the distinction in the room.

**C5 — keep the footnote "finetuning teaches the policy to execute a motion, not to fight — no
self-play, no opponents."** It is the most credibility-preserving sentence in the deck. Do not let it
get edited out for space.

**C6 — "policy weights published per season" changes the licence posture.** Publishing finetuned
derivatives of NVIDIA-licensed weights is a redistribution of derived models, not a link to upstream.
The NVIDIA Open Model License review moves from *advisable* to *blocking before the first weight
publication* (§12).

**C7 — slide 8 says every match was fought by a person; workstream 04 is agent baselines.** State
where agents live: an **exhibition/baseline track outside the Season 0 table**. Otherwise the first
question from the room is whether bots can farm the ladder.

**C8 — "anyone with a laptop can enter" is true for *entering*, not for *hosting*.** Playing ranked
needs a browser. Hosting a match — local hotseat, a meetup bracket, the league server — needs a
gaming GPU. Say it on slide 13 or in the site FAQ so nobody arrives expecting to self-host on a
thin client.

---

## 1. What OpenRoboxing is

**The first league where autonomous humanoids fight and nobody holds the controller.**

Two people, one ring. Each chooses *moves*, not joint angles: you place a motion in the near future,
a generative model fills in the trajectory, a whole-body policy executes it under physics, and two
bodies collide. Nobody steers the robot mid-motion, and nothing arbitrates the hit except physics.

Everything under the game already exists as open source:

| Layer | Source | Status |
|---|---|---|
| Whole-body policy tracking arbitrary motion, hardware-proven | **GEAR-SONIC** | upstream checkpoint, finetuned on our poses |
| Real-time generative in-betweening from key poses | **MotionBricks** | upstream, unchanged |
| Physics, ring, two fighters | **MuJoCo** | ours to assemble |
| Retargeted human motion at scale | **Bones-SEED** (142K clips, ~288 h) | source for clip mining |
| **The game, the studio, the league** | **us** | six months |

The contribution is the **interface and the sport**, not the robot control.

---

## 2. Landscape — say it before they Google it

**Teleoperated robot fights already draw crowds.** Unitree's *Iron Fist King: Awakening* (May 2025)
ran a four-G1 bracket with engineers on the remotes. CMG's Hangzhou tournament the same month had
operators driving G1s by remote and voice — humans chose the attacks. At CES 2026 two G1s traded
punches in headgear and gloves. Unitree now markets the G1 for boxing.

**Autonomous humanoid boxing exists in research.** *RoboStriker* (arXiv, Jan 2026) presents
hierarchical decision-making for autonomous humanoid boxing on the 29-DOF G1, trained in Isaac Lab
with domain randomisation, **on a single RTX 4090**.

**Nobody built the middle layer.** Events have no autonomy. Research has agents but no game, no
players, no table. NVIDIA shipped seven boxing modes inside its planner and left them as a demo GIF.

Three consequences: the sport already has a reference ruleset (CMG: three two-minute rounds, kicks
above punches, falls penalised, eight-second count) · RoboStriker is a baseline and a possible
collaborator, not a competitor · and the open, human-playable, bring-your-own-fighter position is
unoccupied.

---

## 3. Gameplay

### 3.1 The loop
1. **Commit.** Choose a motion from your loadout, aim where it ends (step-in vector, heading),
   commit it. No take-backs.
2. **Windup.** The generator begins producing the approach immediately: `t+0` weight shifts,
   `t+8` shoulder loads, `t+18` guard drops, `t+30` contact.
3. **Reading.** Your opponent sees that on the body — there is no HUD on the fighters. Roughly half a
   second to slip, block or trade, and they must commit before they are certain.
4. **The hit.** Nobody arbitrates. Two bodies collide and physics settles it.
5. **Scoring.** Clean hits, knockdowns, ring control, aggression. Replays are public.

**The whole game is choosing the right move early enough.**

### 3.2 Six keys, and a loadout
A player holds **six bound moves**, each with a duration — e.g. `jab L 0.8 s`, `jab R 0.8 s`,
`hook L 1.6 s`, `hook R 2.1 s`, `slip 0.5 s`, `guard hold`. Plus a persistent **style**
(`orthodox · pressure`, `southpaw`) and a **move** channel (step-in and heading) that manages
distance.

Two design consequences worth protecting:
- **Depth lives in authoring, not in match input.** Free-form pose constraints belong in the Studio.
  In the ring you press one of six keys. This is what makes the game learnable in a minute and still
  deep, and it is a better split than the "experts send weighted hand targets mid-match" idea it
  replaces.
- **The loadout is deck-building.** Six slots from a growing library is a real competitive lever:
  fast-and-cheap versus slow-and-heavy, and a metagame that shifts when the library is patched.

### 3.3 Duration is the risk dial
Commit length maps directly onto the generator's `allowed_pred_num_tokens` (6–16 tokens ≈ 0.8–2.1 s).
A long commit buys a heavier, more grounded motion at the cost of a longer window in which you cannot
answer. A short commit is a jab. **The mechanic was already in the model's API; we exposed it rather
than invented it.**

### 3.4 Readability is a shipped feature, not an aspiration
Because the body is the only real-time channel, *how visible a windup is* decides both fairness and
watchability. The Studio measures a **telegraph window** per move (the deck shows `240 ms · passes
floor`) and Rules v1.0 publishes the **floor**. A move with a near-zero windup is not a strong move —
it is an unbalanced one, and it fails admission.

Feints emerge for free: a commit whose windup resembles another move *is* a feint, and nobody
implements feinting.

### 3.5 Commits are final
No cancellation. This creates a passivity incentive — whoever commits later has the advantage — which
the deck already answers by putting **aggression** in the scoring dimensions. Keep it there; it is
the anti-passivity mechanism, and it needs a concrete definition in Rules v1.0 rather than a slot in
a list.

---

## 4. The Studio — author a motion, then teach the policy to execute it

New in v0.8, from slide 5, and it is the piece that turns the pose library into a product.

**Step 1 · Pose Studio.** Author a key-pose motion on the 29-DOF G1 skeleton: keyframes, a `source`
clip and frame range (`shadow_boxing_R_003 · fr 25–35`), a horizon in tokens (`16 tok · 2.1 s`), and a
**measured telegraph window** checked against the published floor. Output: a versioned **pose record**
added to the library.

**Step 2 · Policy Finetune.** Take the upstream GEAR-SONIC checkpoint as base, finetune on the new
motion plus variants, objective **pose-tracking fidelity** (see C4), report mean joint error, publish
weights per season.

Three things to be explicit about, because the mockup implies them without saying them:
- **This runs on TORC hardware, not a laptop.** Finetuning needs Isaac Lab and a real GPU. Playing
  needs a browser. Authoring sits between the two.
- **Finetuning teaches execution, not fighting.** No self-play, no opponents, no strategy. The deck's
  footnote is exactly right and should survive every future edit.
- **Every published weight set is a licensed derivative** (§12), and every one must be reproducible
  from a recorded job config, or the season's results are not auditable.

---

## 5. Architecture — four layers

```
SURFACES          ring client (browser) · league site · fight-night screen (projector)
────────────────────────────────────────────────────────────────────────────────────
LEAGUE SERVICES   registration + loadout · Swiss pairing · Glicko-2 · replay archive
────────────────────────────────────────────────────────────────────────────────────
MATCH RUNTIME     intent server (queue @30 Hz, ticks @50 Hz)
                  → MotionBricks generator (upstream, batch = 2 fighters)
                  → bridge (order remap · 30→50 Hz · velocities)
                  → observation assembly + GEAR-SONIC policy (ONNX Runtime) ×2 @50 Hz
                  → MuJoCo physics, two fighters, contacts, match state
────────────────────────────────────────────────────────────────────────────────────
VERSIONED ASSETS  pose library v1.0 · policy weights (per season) · rules v1.0 · G1 29-DOF model
```

Invariants:
1. **One process per match runtime.** MuJoCo, both generators (one batched instance), both policies.
   No DDS, no C++ deployment stack, no inter-process physics.
2. **Server-authoritative.** Clients send intents and receive state. No client physics, no client
   generation — this is the anti-cheat design, the fairness design, and the reason ranked play needs
   only a browser.
3. **Two hosting modes, one codebase.** Local hotseat on a gaming GPU (dev, meetups) and the hosted
   league server. Same runtime, different front door.
4. **Everything above the runtime is ordinary web work** — and none of it needs a robot in the room.

### 5.1 Why it fits one GPU
MotionBricks is ~224 M parameters (VQVAE 23.5 M + pose 150 M + root 50 M) and ~2 ms per inference;
both fighters share the same weights because styles are data, so **one instance with `batch=2`** runs
the whole match — roughly 1 GB in fp16 at a ~4 Hz replan. The policy runs at 50 Hz on a Jetson Orin,
so two instances are trivial. The real question is **MuJoCo with two 29-DOF humanoids in contact**,
which is CPU-bound and must be measured in week 1. RoboStriker's single-4090 result is independent
evidence the envelope is right.

### 5.2 The one real cost
**There is no Python-side policy runner upstream** — no `onnxruntime` import anywhere in
`gear_sonic`. The policy only runs inside the C++ stack, which assembles observations in C++. We
reimplement observation assembly in Python. Estimated 3–6 weeks, and the failure mode is silent: a
wrong permutation yields a policy that tracks garbage without ever erroring.

**But it is a diff, not a gamble.** The C++ stack has an observation registry
(`include/observation_config.hpp` — "a single place to define all observations") and dumps the
**policy input vector** to file (`policy_input_file_`, `g1_deploy_onnx_ref.cpp:307`), plus per-signal
CSV sinks (`state_logger.hpp`, `file_sink.hpp`). So: run the C++ deploy on a known reference motion,
capture the golden policy inputs, replay identical inputs through Python, diff to tolerance. That is
M1's acceptance test.

---

## 6. Upstream: keep, replace, patch

**Keep** — GEAR-SONIC checkpoint (as finetune base) · MotionBricks checkpoints and the whole Python
generation path · the MuJoCo G1 model · the motion↔qpos converter (`helper/mujoco_helper.py`:
36-dim qpos, quaternion reorder, canonicalisation) · `observation_config.hpp` as the specification
for §5.2 · the reference-motion CSV format (`motion_data_reader.hpp`: `joint_pos.csv`,
`joint_vel.csv`, `body_pos.csv`, `body_quat.csv`, `body_lin_vel.csv`, `body_ang_vel.csv`,
`metadata.txt`).

**Replace** — the kinematic planner ONNX (11 inputs, 27 frozen modes, clip library baked into the
graph, boxing replan pinned at 1.0 s); on the Python path replan cadence is a call argument
(`controller_dt`, `force_generation`) and `_should_regenerate` is ours to override · the C++ input
interfaces (keyboard, gamepad, ZMQ manager) · the `has_specific_target` C++ patch, which we do not
need: `generate_new_frames` auto-enables it when `specific_target_positions` is supplied.

**The one patch that matters** — `_override_target_transforms` overrides *only* root position and
heading; the pose target still comes from `_generate_target_joint_transforms`, which samples a random
frame from the clip library by one-hot `mode` and `random_seed`. **Extend it to accept explicit target
joint transforms.** One function, in Python. Everything the game promises hangs off it. `⚠` Verify the
pose-constraint tensors feeding `_generate_inbetween_frames` accept an external pose with the same
mask semantics the model was trained with.

**Styles are a dict with one boxing clip in it** — `demo/clips.py` `CLIPS` (line 131) holds 15 clips;
exactly one is boxing (`walk_boxing` → `shadow_boxing_R_003__A360_M`, frames 25–35). The seven boxing
modes of the deploy ONNX exist only inside that sealed graph, and `G1-clip.ckpt` is a pre-baked cache
of just those 15. Adding clips needs the dataset plus `reprocess_clips=True`.
→ **Season 0 needs no new clips**: with the patch above, strikes are authored key poses and the single
shadow-boxing clip supplies the style prior. Clip mining is workstream 05, not a blocker.

---

## 7. Specs to version from day one

External code and contributors depend on these; breaking them carelessly is how the project loses its
community. All live in `src/openroboxing/spec/`, semver'd, with a changelog.

| Spec | Contents |
|---|---|
| **`rates.md`** | The canonical rate table (C3): ticks 50 Hz · generator 30 Hz · intent queue 30 Hz · stream 30 FPS |
| **`intent`** | `tick`, `commit_at`, `style`, `move{step_in, heading_rad, speed}`, `slot` (1–6), and the resolved `pose_ref` + `horizon_tokens`. No `cancel` field |
| **`pose_record`** | name, source clip + frame range, keyframes, horizon tokens, measured telegraph window, admission status, library version |
| **`loadout`** | six slots → pose refs, style, library version |
| **`match_record`** | intent log (canonical), state trace, seeds, contact/impulse events, knockdowns, ring-position trace, per-round scores, policy + library + rules versions |
| **`rules`** | scoring weights, telegraph floor, commit-window rule, round length and count, count-out |
| **`agent_api`** | same intent structure as a human client, for the exhibition track |

---

## 8. The league

**Season 0** — 10 weeks · one division · one table · unlimited open entry · Swiss pairing · register
Monday, fight by Wednesday · 8 matches to appear on the table · Glicko-2 with confidence interval ·
top 4 at week 10 fight for the belt at **the Trieste Open**.

**Scoring a round (Rules v1.0)** — landed impulses, knockdowns, ring control, aggression. Published
and versioned. Two things to nail down before week 26: how aggression is computed (it is the
anti-passivity lever, §3.5), and how ring control is measured from the position trace.

**Balance is a patch** — pose library, telegraph floor, commit rules, all frozen per season and
changed between seasons with a changelog, exactly like a fighting game.

**Replays are the memory** — every match is a small intent log; the archive is public. `⚠` Never
promise bit-exact re-simulation: MuJoCo determinism across machines is not guaranteed and GPU
inference adds noise, so the state trace ships alongside the log.

**Agents** — exhibition and baseline track, outside the Season 0 table (C7). Same intent API, which is
what makes human-vs-agent exhibitions and imitation learning from human matches free.

---

## 9. Fight nights, then hardware

**From month 4**, hotseat brackets at TORC meetups: a projector, a bracket, an hour of it, nothing on
the record. From week 19 the online ladder runs and the table moves live during meetups. The finale is
a public night in Trieste.

**Then hardware, through the gates.** The route is settled and it is the one the deck shows: the
laptop generates, only the policy runs on the Jetson — motion streams over WiFi, the C++ stack buffers
~30 s (`ReserveCapacity(1500, …)`) and degrades gracefully on stale input. The anticipatory design is
*why* this works over a wireless link.

Two gates before any contact event, public or private:
- **Gate A — impact robustness.** Being hit is out of distribution for the upstream policy (§0.2 C4).
  Adversarial-impulse and contact finetuning, validated on a padded rig before two robots touch.
- **Gate B — rules and safety published.** Weight and reach classes (an H2 knee lifted a G1 off the
  ground in Unitree's own footage), legal target zones, contact-energy ceiling and how it is measured,
  mandatory instrumentation, inspection checklist, stoppage authority, barrier geometry, redundant
  **physical** E-stop per robot independent of any software, LiPo and thermal protocol.

And two obligations that come with being open: **publish the safety spec as a versioned artifact**,
because someone with two G1s and our repo will run their own match; and **reserve the name for
sanctioned events** — anyone may fork the code, nobody may call an uninspected contact match an
OpenRoboxing event.

---

## 10. Roadmap — six months to Season 0

| Weeks | Milestone | Content |
|---|---|---|
| 1–4 | **It moves** | Generator → policy → physics, one G1, one process. Observation parity harness green. MuJoCo two-humanoid real-time measured |
| 5–8 | **It obeys** | Key-pose commit patch, intent timeline v0, 6–10 authored poses. **Fork goes public** |
| 9–13 | **There are two of them** | One world, two fighters, contact sensing, hit attribution, match loop, replays |
| 14–18 | **It's playable** | Timeline UI, hotseat, one-command install, **first meetup bracket** |
| 19–23 | **It's a league** | The site: registration, ratings, table, fixtures, replays. Online ladder opens |
| 24–26 | **Season 0** | Rules frozen at v1.0, registration opens, the Trieste Open is dated |

*Everything before week 18 is a laptop and a repository. Everything after it is a sport.* Season 0
stays small — one embodiment, one arena — small enough to finish, complete enough to have a champion.

**The Studio** (§4) spans M2–M5: Pose Studio with M2 (it is how the pose library gets authored),
finetuning from M3 once there are enough poses to justify a run.

**Six workstreams**, independent and off the critical path: 01 web (the league site) · 02 events
(fight nights, venue, format, the belt) · 03 rules (turn a match record into a result people accept) ·
04 RL (agent baselines in a committed, non-cancellable action space) · 05 data (clip mining in 288
hours — the ideal first issue) · 06 packaging (Windows and WSL, clean-machine install — *this decides
who can enter*).

---

## 11. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Python observation assembly diverges from the C++ reference, silently | **Critical — M1 gate** | Golden-diff harness against `policy_input` dumps; derive every mapping by joint *name* with round-trip assertions; never trust published index arrays |
| R2 | MuJoCo with two humanoids in contact is not real time on a gaming GPU/CPU | High | Measure week 1; larger timestep, simplified collision geometry, decouple physics from render |
| R3 | MotionBricks is kinematic and physically unaware — self-penetration and torque-infeasible output are acknowledged upstream; the policy will refuse to track some poses | High | Studio admission gate: lint against joint limits, reject on measured tracking error; closed-loop context feed keeps the generator anchored to physics |
| R4 | The interface is not fun | High | Hotseat playtests at meetups from month 4, with people who did not build it; horizon, commit window and loadout all tunable |
| R5 | Scoring nobody accepts — the league's legitimacy rests on it | High | Rules v1.0 published and versioned; aggression and ring control defined concretely, not listed; replays public so disputes are checkable |
| R6 | Finetuning quality: pose-tracking finetunes degrade general behaviour | Medium | Hold out a regression suite of upstream motions; publish mean joint error per release; never ship a weight set without the regression run |
| R7 | Six months with a small team | High | Fork public at week 8; six independent workstreams; the deck exists to parallelise M1 |
| R8 | Upstream churn — MotionBricks is a preview and a GEAR-SONIC-integrated release was targeted for mid-2026 | Medium | Fork with upstream as a remote, rebase deliberately. `⚠` **Check whether it landed — it could shrink M1** |
| R9 | Licence exposure from publishing derived weights (C6) | Medium–High | Review before the first publication; attribution and terms in every release; document the base checkpoint and job config |
| R10 | Nobody shows up | Medium | A real design idea, six bounded issues, honesty about what does not exist, and a dated public night |
| R11 | Someone ships the game first — the events genre is warming | Medium | Six months, and the open + human-playable + browser position is unoccupied |

---

## 12. Licensing, credit, fork hygiene

- **Fork of `NVlabs/GR00T-WholeBodyControl`**, upstream as a remote, rebased deliberately.
- **Code** Apache 2.0, ours likewise. **Weights** under the **NVIDIA Open Model License** — commercial
  use with attribution, subject to NVIDIA's Trustworthy AI terms. Base checkpoints are fetched from
  upstream at install time, never re-hosted. **Finetuned derivatives we publish per season are a
  different matter and need the review in C6 before the first release.**
- **Bones-SEED terms** must be checked before redistributing anything derived, if workstream 05 uses it.
- **Credit loudly and specifically.** GEAR-SONIC and MotionBricks are NVIDIA GEAR's work and the
  project is unbuildable without them. Say so on the layer slide. For this audience accuracy *is*
  credibility.
- **Repo layout** — ours under `src/openroboxing/`; upstream trees touched only through a documented
  allowlist so rebases stay survivable.

---

## 13. Open questions

**Settled by the deck:** name · human-first single division · six keys and loadouts · browser client
with server runtime · Studio with finetuning · 10-week season, Swiss, Glicko-2, one belt · scoring
dimensions · fight nights from month 4 · hardware via off-board generation, gated · fork public at
week 8 · six workstreams.

1. **Aggression, defined.** The anti-passivity lever needs a formula, not a bullet. Commits per
   minute? Forward pressure integrated over ring position? Landed-attempt ratio?
2. **Ring control, defined.** From the position trace — centre occupancy, or opponent displacement?
3. **Commit window.** May the next intent be issued shortly *before* the current resolves? This is
   where combinations come from, and the difference between turn-based and flowing feel.
4. **Loadout constraints.** Six free slots, or a budget (total duration, or one heavy move max)? This
   is the deck-building rule and it decides the metagame.
5. **Who owns which workstream**, and which two of the six are covered by the people in the room.
6. **Trieste Open date.** Fixing it publicly is the single most effective schedule-forcing device
   available, and it should be chosen before week 8.
7. **RoboStriker authors** — worth contacting for the exhibition baseline?

---

## 14. Appendix — verified against upstream `main` (Aug 2026)

| Claim | Source |
|---|---|
| `full_navigation_agent` API: `generate_new_frames(input, controller_dt, force_generation)`, `get_next_frame`, `get_context_mujoco_qpos` | `motionbricks/motion_backbone/demo/full_agent.py:109–135, 503–522` |
| `has_specific_target` auto-enabled when `specific_target_positions` supplied | `full_agent.py:131–133` |
| `_override_target_transforms` blends root position + heading only | `full_agent.py:298–320` |
| Target pose sampled from clips via one-hot `mode` + `random_seed` | `full_agent.py:321–393` |
| `CLIPS` = 15-entry dict; only `walk_boxing` is boxing (`shadow_boxing_R_003__A360_M`, fr 25–35) | `demo/clips.py:131–228` |
| Clip cache pre-baked; adding clips needs the dataset + `reprocess_clips` | `clips.py:19–72` |
| Demo replans every 8 frames at 30 FPS (~0.27 s) | `demo/controllers.py` |
| qpos converter: 36-dim layout, quaternion reorder, canonicalisation | `helper/mujoco_helper.py:108–430` |
| TensorRT/ONNX-safe rewrites of jit/einsum ops (export is walkable) | `helper/mujoco_helper.py:21–49` |
| **No Python-side ONNX policy runner exists** | no `onnxruntime`/`InferenceSession` in `gear_sonic/**.py` |
| **Observation registry — single definition point** | `include/observation_config.hpp`; `g1_deploy_onnx_ref.cpp:369–410` |
| **Policy input vector dumped to file** | `g1_deploy_onnx_ref.cpp:307` (`policy_input_file_`) |
| Per-signal CSV sinks | `include/state_logger.hpp`, `include/file_sink.hpp`, `src/state_logger.cpp` |
| Reference-motion CSV format | `include/motion_data_reader.hpp:532–660` |
| Sim bridge is DDS-based; `DOMAIN_ID` configurable; `ChannelFactoryInitialize` process-global | `gear_sonic/utils/mujoco_sim/simulator_factory.py:6–21` |
| Single-robot sim loop | `gear_sonic/scripts/run_sim_loop.py` (67 lines) |
| `LocalMotionPlannerBase` abstract base; `LocalMotionPlannerONNX` shipped impl | `include/localmotion_kplanner.hpp:236` |
| Planner motion buffer ≈30 s at 50 Hz | `localmotion_kplanner.hpp:245` |
| Planner ONNX: 11 inputs, 27 modes, boxing replan 1.0 s, 8-frame blend, 30→50 Hz | `docs/references/planner_onnx.html` |
| 7 boxing modes only inside the deploy ONNX (modes 9–16) | `planner_onnx.html`; `localmotion_kplanner.hpp:78–106` |
| Bones-SEED 142K+ clips, ~288 h, G1-retargeted | repo README |
| Apache 2.0 code / NVIDIA Open Model License weights | repo README, `LICENSE` |

**External sources:** Unitree *Iron Fist King: Awakening*, May 2025 (Popular Science) · CMG Hangzhou
tournament and 2026 combat competition, G1 marketed for boxing (eWeek) · CES 2026 G1 boxing demo,
teleoperation as a bridge to autonomy (Interesting Engineering) · H2 vs G1 sparring, Dec 2025
(Interesting Engineering) · *RoboStriker*, arXiv, Jan 2026.
