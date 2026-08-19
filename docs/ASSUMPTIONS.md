# ASSUMPTIONS — decisions taken without you

Every entry here is a **design decision that was yours to make** and that I took anyway, to keep
moving. They are all reversible, and each says what to change if you disagree.

Read this when you next have time. Nothing here is blocking *me*; the entries marked **⚠ WANTS YOU**
are the ones where my guess is least likely to match your intent, and where being wrong costs the
most rework.

Started 2026-08-08, after M3 completed. Conventions: `CLAUDE.md` standing rule 3 says never invent a
number — where I had to, the number is here with its reasoning, and it went into a `spec/` file too.

---

## Legend

| Mark | Meaning |
|---|---|
| ⚠ **WANTS YOU** | A real design choice about how the game feels or what it means. Please review. |
| ○ | Mechanical: a default that follows from something already decided. Review if bored. |
| ✎ | Written into a versioned `spec/` file, so changing it means bumping that spec. |

---

## A1 ⚠ ✎ · What "aggression" and "ring control" mean

`WORKPLAN` M5-T2 says these two scoring dimensions "need concrete definitions first — that is
workstream 03's job; if the definitions are not ready, implement the interface and a placeholder,
and flag it." They are not ready. I implemented them for real rather than as placeholders, because a
placeholder that always returns 0.5 makes the M5-T2 acceptance criterion (ten replays, a human agrees
with eight) impossible to even attempt.

Definitions are in `spec/scoring.md` v0.1. In short:

- **Aggression** = commits issued per minute that were *thrown at a reachable opponent*, normalised
  against a target rate. Committing into thin air is not aggression, and neither is standing at
  range doing nothing.
- **Ring control** = the fraction of the round a fighter spent closer to the ring's centre than its
  opponent, weighted by how close the two were. Circling at distance does not earn control.

**If you disagree:** these are the two knobs most likely to be wrong, because they encode what the
game rewards. Everything downstream reads `spec/scoring.md`; changing the definitions there and
re-running `tools/score_match.py` over the archive re-scores every recorded match without
re-simulating anything.

## A2 ⚠ ✎ · What a knockout is worth

`spec/match_record.md` v0.1 listed this as Open. A round won by knockout scores **10–7** against the
round loser's **7**, and a knockdown that does not become a knockout is **10–8**. These are boxing's
own numbers (the "10-point must" system), chosen for that reason and not derived.

**If you disagree:** `spec/scoring.md` §Round scores. The alternative worth considering is that a
knockout should dominate the *match* rather than just its round — but that contradicts the rule you
already chose (a knockout ends the round, not the match), so I kept your rule intact.

## A3 ○ ✎ · Rest between rounds, and the stance each round starts from

`spec/match_record.md` v0.1 left both Open. v0.1 behaviour stands: **no fatigue or damage carries
across rounds**, and **both fighters reset to the same starting stance** regardless of what happened.
A knocked-out fighter starts round 2 exactly as fresh as the one who knocked them out.

This follows from there being no damage model. Inventing one would be a game-design decision far
larger than anything else in this file.

## A4 ○ · The scripted pilots are not an opponent AI

`runtime/fight.py`'s `ScriptedPilot` commits from a fixed list of ticks. It exists so a match can run
headless, and it is deliberately not clever — `M4-T1` puts a human on those controls and `M5-T4` an
agent. Nothing about how well it fights should be read as how well the *game* plays.

## A5 ✓ SUPERSEDED · Fighters face their opponent *while holding*

Was: the ambient facing angle points each fighter at the other every tick, because nobody steers it
and a boxer who does not turn walks out of the ring.

**Superseded by §A19.** A committed move now carries its own heading in its placement, so the bearing
only decides which way a fighter looks when its queue has run dry. The guess in the original entry —
"this is arguably a control the player should hold" — turned out to be right, and the player holds it
now.

---

## A6 ○ ✎ · The hotseat key layout

`M4-T1` asks for "a hotseat match playable by two people on one machine" but not which keys. Red gets
**1–6 and SPACE**, blue gets **U I O J K L and ENTER** — two hands, two halves of the keyboard, no
overlap and no focus management. One browser page opens two sockets; the host cannot tell hotseat
from two machines (`spec/protocol.md` §Hotseat).

**If you disagree:** `client/app.js`, the `SEATS` constant, six lines.

## A7 ✗ OVERRULED · The browser renders the ring

I chose server-side JPEG rendering and called it "the largest architectural bet in M4". **The project
owner overruled it on 2026-08-08**: the game needs a **shadow you can drive around the ring**, and
you cannot place a ghost in space by looking at a flat video of it.

The client now runs three.js (`spec/protocol.md` 0.4). It went better than the entry feared:

| | 0.3 (JPEG) | 0.4 (transforms) |
|---|---|---|
| per frame | ~25 kB at 640×360 | **1.7 kB** |
| per viewer | ~750 kB/s | **~55 kB/s** |
| one-off | none | 8.9 MB of geometry, cached |

The bandwidth pays the meshes back in about 12 seconds of play, and the entry's own worry — "it will
not survive a public online ladder with spectators" — is answered rather than deferred.

Two things made it cheap. Geometry is read out of the **compiled MuJoCo model** rather than the STL
files, so there is no mesh-name-to-filename mapping to go stale and no STL parser in the browser; and
red's and blue's identical copies are deduplicated, 72 meshes down to 36.

**The host still does the fighters' kinematics** — it sends world transforms it had already computed
to step physics — so the client cannot drift from the simulation. The one exception is the shadow,
and §A20 is about that.

## A8 ✓ DECIDED · Players just type a name

`M5-T1` says "registration with GitHub sign-in". **The project owner decided on 2026-08-08 to keep
typed names for now.** A client sends a handle and the host believes it.

That is right for a room where you know everybody, and wrong the moment strangers can play — anyone
can claim any name. Revisit before an open online ladder. Everything else in `league/` is real, and
bolting identity on later touches only the socket handshake, so nothing is being built around this.

## A9 ○ · A late tick is dropped, not chased

When the host overruns its 20 ms budget it drops the deficit rather than running two ticks to catch
up. A fight that speeds up after a stall is worse than one that stutters, and the count is reported
(`serve_match` prints it; measured 0.5% at 20 s).

## A10 ○ · Four seconds between rounds

`spec/match_record.md` leaves rest between rounds Open and no damage model exists, so this is only
long enough to read the round result. It is `ROUND_BREAK_S` in `server/host.py`.

## A11 ✗ OVERRULED · The score **is** shown during a round

I withheld it. **The project owner decided on 2026-08-08 that the score should be visible**, and it
now is — in the ring client and on the fight-night screen.

My worry was that a live number would be a *different* quantity from `spec/scoring.md`'s and would
disagree with the official result at the bell. The implementation removes that worry rather than
accepting it: the live score runs `league/scoring.score_round` over the events so far — same weights,
same definitions, same code — so it cannot disagree. It is labelled **provisional**, and `points`
counts completed rounds only, because the 10-point must depends on how a round *finished*.

`spec/protocol.md` 0.2. Reversing it means deleting the `score` block; nothing else reads it.

---

## A12 ✓ CLOSED · Red is not a worse seat than blue

The M4-T2 latency A/B incidentally measured red winning **5 of 16** baseline matches, which raised
the possibility that the seat itself was worth points — in which case Swiss pairing plus Glicko-2
would have faithfully rated which side of the ring somebody stood on.

**Settled by measurement**, not by argument: `tools/seat_fairness.py`, 40 further matches with the
agents swapped between seats and the *same* match seeds in both conditions. Pooled, red wins
**20 / 35 = 0.57, interval [0.41, 0.72]** — which contains 0.5. And red scored 0.57–0.65 there against
0.33 in the original A/B, on a different block of seeds, which makes the original number look like
variance rather than a handed arena.

It bounds the asymmetry at roughly ±16 points rather than proving the seats identical. Full write-up
in `docs/perf/m4_latency_ab.md`. Nothing to do.

## A13 ○ · The installed dependency list was measured

`requirements-runtime.txt` was produced by importing the whole match runtime, building a real
generator, and listing every third-party module that ended up in `sys.modules` — not by guessing.
My first attempt at the list had five packages in it and was wrong; the real answer is twenty-odd,
because MotionBricks pulls lightning, hydra, transformers and a vector-quantiser.

## A14 ○ · What M4-T3 has and has not been proven on

`install.sh` was verified end to end on Linux against a throwaway venv — platform check, Python
version check, venv creation, `uv pip install -r`, checkpoint check, smoke test, and the correct
refusal when checkpoints are absent. It has **not** been run on a clean Windows/WSL machine, which is
what M4-T3's acceptance actually asks for, and I cannot do that from here. The transcript that
criterion wants is yours to produce.

---

## A15 ⚠ · The Pose Studio authors, it does not admit

`S-T1` asks that "an author with no repo access creates a valid, **admitted** pose record through the
browser". The served Studio does everything except the last word: it edits, renders, reports reach,
validates against the real `studio/pose_record.py`, and saves a **draft**.

**Why:** admission requires a measured `generator_error_rad` — asking MotionBricks to actually reach
the pose, which loads a checkpoint and takes seconds per attempt. Putting that behind a browser
button is possible; putting it there *well* means deciding what an author sees while it runs and what
happens when it fails, which is a product question.

**What it means today:** save a draft in the browser, then run `tools/build_library.py` to measure and
admit it. Drafts are refused by a match (`Loadout.validate`), so nothing unmeasured can reach a ring.

## A16 ○ · Drafts are stamped `library_version: "dev"`

Not `v0.1`. Promoting a draft into a versioned library is a deliberate act with a review; stamping a
real version at save time would let an unmeasured pose claim membership of a library it has not been
admitted to.

---

## A17 ✓ DECIDED · Weights may be published; the code stays private for now

Four decisions taken by the project owner on **2026-08-08**:

| Question | Decision | Consequence |
|---|---|---|
| Publish the policy weights? | **Yes** | `LICENSING.md` written; the release gate can now be opened |
| Public GitHub fork? | **Not yet** | `M2-T6` stays deferred; nothing changes |
| How do players identify? | **Typed names** | see §A8 |
| Playtest preparation? | **Solo tuning tools first** | `tools/tune.py`; no playtest kit yet |

On the weights: the decision turned out to be **supported by the repository's own `LICENSE`**, which
puts the checkpoints under the **NVIDIA Open Model License** — permissive about exactly what M6-T2
was worried about (§2.2 grants the right to create and distribute derivative models; §2.4 says "You
own Your Derivative Models"). Three conditions apply, all easy: ship the agreement, carry a verbatim
attribution notice, and a Cosmos-only clause that on a plain reading does not apply to a whole-body
control policy. That last point is the one thing on the page worth a five-minute check with NVIDIA.

**Nothing has been published.** The gate in `league/manifest.py` stays: knowing the answer makes
publishing possible, requiring an acknowledgement keeps it deliberate.

---

## A18 ✓ SUPERSEDED · Players can now walk

Found by the first tuning sweep, not by design review: `GeneratorIntent.movement_angle` existed and
**nothing ever set it**, so fighters turned to face each other and then walked in a fixed direction
until they hit the ropes.

Fixed on 2026-08-08, because the game was not playable without it (`spec/intent.md` 0.2,
`spec/protocol.md` 0.3).

| | before | after |
|---|---|---|
| circling (round spent > 1.8 m apart) | 67–80 % | **26 %** |
| engagement | 0.19–0.32 | **0.69** |
| mean separation | 3.2–3.8 m | **1.24 m** |

**How it works.** Steering is one of `in` / `out` / `left` / `right`, always relative to the opponent,
**held** rather than tapped. Standing still is the `idle` clip (`avg_root_vel` 0.0); stepping is
`slow_walk` (0.6 m/s). Both are measured properties of the clip library, not numbers I picked — and
the old ambient clip carried **2.0 m/s**, which is the whole story of why fighters ended up in
opposite corners.

Two fighters holding position settle at exactly **0.99 m** and stay there, against a contact range of
0.80 m. So at rest you stand just out of reach and closing is a decision, which is the shape a boxing
game wants.

> **Superseded the same day by §A19.** The project owner remodelled the game around placing a shadow,
> which makes walking a *commit* rather than a held key. The `movement` channel was retired hours
> after it was added. The finding above still stands and still mattered — it is why anyone looked at
> distance at all — but the fix in this entry is gone.

The three feel questions this entry left open were answered by the remodel rather than by playing:
a committed move no longer overrides steering because there is no steering; holding is still `idle`,
now as "you ran out of commits"; and circling is now just placing the ghost to one side.

---

*(entries below this line were added as the work continued)*

## A19 ✗ OVERRULED · A commit is a plan, not a punch

I built the game around *one active commit, no queueing* and around held keys for movement. **The
project owner remodelled it on 2026-08-08**, and four decisions came with it:

| Question | Decision | Where it lives |
|---|---|---|
| What does a move button select? | **The pose you end in** | `spec/intent.md` 1.0 |
| Does a commit include walking there? | **Yes — one action** | §"What happened to walking" |
| Keep the 0.6 s readable pause? | **Yes, as a floor** | §"A commit's span" |
| How many may be queued? | **5** | `MAX_OUTSTANDING_COMMITS` |
| Can a queued move be taken back? | **No** | §"No cancellation, of anything" |

So: pick a pose, drive a ghost of yourself to where the move should end, commit. The generator
in-betweens from wherever you are to that placement and arrives in that pose. Up to five stack up and
run back to back; run out and your fighter stops.

**The horizon is a floor, not a gap.** A commit into an empty queue costs 0.6 s of readable
hesitation; one queued behind a running move starts the instant that move ends, because the warning
already elapsed while the opponent watched the first one. A fixed pause between queued moves would
put a stutter in every combination.

**The `movement` channel from §A18 was retired**, not deprecated — `CLAUDE.md` prefers deleting to
disabling, and two ways to move would be two things to learn and two things to balance.

**What this cost, honestly:** `spec/intent.md` went to 1.0, `spec/protocol.md` to 0.4, and 27 tests
that encoded the old rules were rewritten to encode the new ones. Nothing was disabled or left
behind a flag.

## A20 ○ · The shadow is the client's, and it is the only thing that is

The project owner asked for the shadow to live entirely in the browser. It does: posed there from the
joint angles in `welcome` and a kinematic tree in `/scene.json`, moved there, and **never transmitted
while it is being aimed**. The host learns a placement only when one is committed.

That is a real exception to "the client is a view and a keyboard", and it is bounded on purpose: the
client decides where the ghost is *drawn*, the host decides what a commit *means*. A preview that
round-tripped before it moved would be unusable, and where a player is *thinking* of standing is not
the server's business.

Because the browser now does forward kinematics for the first time, the exported tree is checked
against MuJoCo's own: `tests/test_scene.py` re-implements `client/ring.js`'s solver in numpy and
compares it to `mj_kinematics`. They agree to **8.5e-07 m**, which is the 6-decimal rounding the tree
ships with and not an error in the algorithm.

## A21 ✓ SUPERSEDED by §A23 · Placement is closed-loop across moves, and it does not land exactly

Two numbers a first game will run straight into, both **measured** (`scratchpad/probe_reach.py`,
2026-08-08) rather than assumed:

**A move's reach is its duration.** Told to travel 6 m — far more than any move can cover — a fighter
reaches **1.25–1.40 m on a 6-token move** (0.80 s) and **1.72–1.75 m on an 8-token move** (1.07 s).
Pointing the ghost further does not go further. The client says so: the ghost greys out and reads
"too far for this move", from `MOVE_TRAVEL_SPEED_M_S = 1.6`, which is the measured lower bound so a
promise shown to a player is one the move can keep.

**A queue converges on a target; it does not arrive at it.** Aimed at a point 2.2 m away: one move
lands 1.38 m short, three land 0.83 m short, five land **0.75 m** short. It converges because each
move re-derives the remaining distance from where the fighter *actually is* — MotionBricks is
kinematic and its plan arrives while the policy tracking it under physics lands short. The residual
is the lookahead: a move is planned ~0.9 s before it plays, so the correction is always one move
behind.

**This was a bug before it was a limitation.** Placement was passed to the generator in *world*
coordinates, but the generator plans in its own frame — the same trap `facing_angle` has always
handled for angles, with no counterpart for positions. Anchored to the generator's own belief about
itself, the second move in a queue thought it had already arrived: five moves at one target got no
further than one did. `FightWorld.to_generator_frame` is the fix.

**Whether 0.75 m matters is a feel question, and yours.** The contact range is 0.80 m, so aiming the
ghost *at* your opponent lands you roughly in range — which is convenient and slightly accidental.
Play it before deciding whether to chase it.

> **Answered by playing it, 2026-08-08: it mattered.** Both limitations above turned out to be
> consequences of a commit having a fixed length, not of placement being hard. `spec/intent.md` 1.1
> made a commit run until it arrives and both went away — see §A23. The reach cap and the 0.75 m
> residual are history; the frame-conversion finding in the paragraph above still stands.

## A22 ○ · `TARGET_COMMIT_RATE` was calibrated against the old rule and needs re-measuring

`spec/scoring.md`'s aggression dimension normalises against **12 commits/minute**, derived when a
fighter could hold only one commit at a time. Under a five-deep queue a player can issue far more,
and a *step* is now a commit too — so the same number now rewards walking about.

Not changed blind. It wants a tuning sweep with humans playing the new model, which is `M4-T4`. Until
then aggression is comparable *within* a season and not across the change; `league/manifest.py`
already pins a season to the code that scored it.

> **Now demonstrated, not just predicted (2026-08-08, after `spec/intent.md` 1.1).** A fighter
> committing continuously manages roughly **20 commits/minute**, against a target of 12 — so the
> ratio clamps at `MAX_AGGRESSION` and the dimension returns **1.50 for both fighters in every
> round**. Scored that way, a quarter of the match score carries no information at all.
>
> It is not universally dead: the reference agent, which waits a beat between commits, still scores
> 0.33–0.58. The dimension discriminates between *paced* fighters and stops discriminating above
> about 18/minute — which is now an easy rate to hit rather than a hard one. That is the concrete
> version of the warning above, and the number the playtest should move.

## A23 ✗ OVERRULED · A commit runs until it arrives, however long that takes

**The owner played it and reported the game was wrong.** From `spec/intent.md` 1.0: *"the animation
does not land at the committed point but lives just for a second and then the ghost resets and the
robot also stays in place [...] multiple commits are consumed very quickly and do not concatenate"*.

Every symptom was one decision. 1.0 gave a commit **one forced plan**, `horizon_tokens` long — 0.8 s
for a jab. So a fighter walked for about a second, stopped wherever that left it, and the queue moved
on; five commits were spent in five seconds; and the ghost, which hangs off the end of the queue,
snapped back to a fighter standing metres from where it had been pointed. The design read as
plausible and played as broken.

**1.1: the walk is open-ended.** A commit is now an approach that replans toward the placement for as
long as it takes, then a fixed-length pose phase that throws. `commit_at` / `strike_at` / `end_tick`
are filled in as the move runs, so the queue stopped being a schedule.

### Measured before it was written

| | 1.0 | 1.1 |
|---|---|---|
| one commit 2.4 m away | ~1.4 m, ~1 s | **2.84 m travelled, 5.0 s**, 0.44 m from the point |
| five commits round a square | consumed in ~5 s | **24 s, all five arrived**, gaps `[0,0,1,0]` |
| worst closest approach, 10 placements | — | **0.30 m** (a target directly behind) |
| reach | capped by the pose's duration | **anywhere in the ring** |

`ARRIVAL_RADIUS_M = 0.40` is that worst case with margin, and every one of the ten placements reached
it inside 4.3 s. **Tighter is not better**: at 0.25 m one of the ten never arrived, and a commit that
cannot finish holds the whole queue behind it until the timeout.

### Two things this cost, both for you to judge by playing

**A five-deep queue is now 20–30 s.** It was 4–10 s when a move was one plan. Five moves is half a
round committed in advance and unrecallable — which may be exactly the tension the no-cancellation
rule is for, or may be far too much. `MAX_OUTSTANDING_COMMITS` has **not** been changed on a guess;
it is a feel question for `M4-T4`.

> **Swept, 2026-08-08** (`docs/playtest/queue_depth_sweep.md`, 20 agent rounds). A deeper queue lands
> *fewer* punches — 109 hits/min at depth 1 against 66–75 from depth 3 up — because a move issued
> five deep is aimed at where the opponent stood seconds ago and walks there anyway. And at depth 5,
> **six commits per round are still unfinished at the bell** (twelve at depth 8, half a commit at
> depths 1–2): moves a player paid for and never saw. The shortlist says try **2**; the decision is
> still yours, and the sweep understates depth 8 because the reference agent fills a queue far more
> sedately than a person would.

**`TARGET_COMMIT_RATE` moved further out of date.** §A22 already flagged it. A move is now 3–5 s, so
a fighter committing constantly manages 12–20/minute against a target of 12 — the number now reads as
"commit constantly" rather than "commit often". Same answer: measure it with humans, do not guess.

### Arrival is coarse relative to punching distance — worth knowing before you play

`ARRIVAL_RADIUS_M` (0.40 m) is **half** `CONTACT_RANGE_M` (0.80 m), and two fighters each arriving
within their own radius scatter independently. Measured on a 40 s scripted round, planning the two a
full contact range apart landed **nothing at all**; planning 0.40 m apart landed 17 but a third were
*leg* contacts, i.e. treading on each other; **0.60 m landed 26, none of them legs**.

So a punch placed exactly at the edge of reach is close to a coin flip, while one thrown from a spot
you are already standing on lands reliably — which is why the reference agent, which commits from
where it stands, lands 61 hits over three rounds and a fixed script that walks to every commit lands
none. In play this reads as *step in, then punch* being much stronger than *punch while stepping in*.
That may be a virtue — it is roughly what boxing is — or it may mean the radius wants tightening. It
is the second thing to judge by playing, after queue depth.

### And one finding worth keeping

Converting a world placement into the generator's frame, **anchor it on the generator's buffer tail,
not on the frame the robot is playing.** Anchoring on the current frame cancels the lookahead and
looks more principled; it turns the loop proportional and leaves a steady-state error equal to the
policy's tracking shortfall — 0.22 m at 1 m, 0.49 m at 2 m, **0.81 m at 3 m**. The tail is a lookahead
ahead of the robot, so re-deriving from the robot each frame integrates, and it settles at ~0.1 m.
I predicted the opposite and measured it before changing it (`scratchpad/probe_approach.py`).

## A24 ⚠ · The commit horizon does nothing; the real one is 1.3 s and mostly structural

`COMMIT_HORIZON_TICKS = 30` (0.6 s) is documented everywhere as the readable window between
committing and the move starting. **It has never bound anything.**

The policy reads the reference motion 45 ticks (0.90 s) in front of now — the encoder's
`10frame_step5` terms — and `runtime/reference.py` keeps a further 12 generator frames (0.40 s)
buffered on top. Frames that far ahead are already generated by the time a commit arrives, and
re-generating them is precisely the replan that deletes a strike. So the earliest tick a commit can
reach is **65 ticks ≈ 1.30 s**, and 30 < 65.

Measured, now that 1.1 records when a move starts: a commit issued at tick 10 begins at tick **75**.

**Why it is worth your attention.** A fighting game is its latency. 0.6 s of deliberate windup is the
design; 1.3 s between pressing a key and the fighter moving is a different game, and it is what you
were playing. Of that:

- **0.90 s is structural.** The policy is trained to track a reference it can see ahead of itself.
  Shortening it means retraining, not a constant.
- **0.40 s is a choice** — `GENERATOR_MARGIN_FRAMES`, the buffer above the lookahead. It is the only
  part that could shrink, and shrinking it trades latency for more frequent generator calls and less
  slack before `ReferenceStream.require` raises.

**Not changed here.** It is a feel-versus-throughput trade with a hard floor beneath it, which makes
it an `M4-T4` question rather than a fix. What has been changed is the documentation: `spec/intent.md`
now says the horizon is inert rather than describing a window nobody experiences, so nobody tunes a
knob that does nothing.
