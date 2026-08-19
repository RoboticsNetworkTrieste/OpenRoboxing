# match_record.md — what a match writes down

Version **0.1** · created 2026-08-07 · task `M3-T4`

A match record is the *only* output of a match. Scoring, replays, the league table and any dispute
are all resolved from it, so it is specified before `runtime/match.py` exists (`CLAUDE.md`
invariant 7).

Format decisions on this page were made by the project owner and are recorded as decisions, not
derived; the measured numbers are marked as measured and cite where from.

---

## Format

| Quantity | Value | Source |
|---|---|---|
| Rounds | **3** | decided |
| Round length | **60 s** = 3000 ticks at `TICK_HZ` | decided |
| Get-up window | **8 s** = 400 ticks | decided |
| Rest between rounds | not simulated in v0.1 | see *Open* |

A committed move is an open-ended walk to its placement followed by a pose phase of 6–16 tokens
≈ 0.8–2.1 s (`spec/rates.md`), and one move executes at a time — measured at **3–5 s all in**. So a
60 s round holds roughly **12–20 exchanges** per fighter. Long enough to have a shape, short enough
that a bad round is not a long punishment.

*(Before `spec/intent.md` 1.1 a move was its pose phase alone and this said 30–70. Walking to a
placement used to be free, and is now most of what a commit does.)*

The record carries the format it was fought under. A match fought to different numbers must still be
readable, so `format` is data, not an assumption baked into the reader.

## A knockout ends the **round**, not the match

This is the rule that makes OpenRoboxing not-quite-boxing, and it is deliberate.

- A fighter who is **down** past the get-up window is **knocked out for that round**.
- The round ends there. The score for that round reflects it.
- **The match continues** to the next round, both fighters reset to their starting stance.
- A match therefore always runs its full 3 rounds. Every bout takes the same wall-clock time, which
  is what a bracket needs, and no single mistake ends a fight.

Consequences worth stating plainly:

- There is **no count-out that ends a match**, and no three-knockdown rule.
- A fighter can be knocked out in all three rounds and still be a fighter who turned up; how badly
  that loses is the scorer's business, not the match loop's.
- A knockout in round 3 ends round 3, which is also the end of the match. No special case.

## Down, and knocked out

**Down** is measured, not judged. From `runtime/contact.py`, itself measured on the arena:

| | Standing | Collapsed | Threshold |
|---|---|---|---|
| torso height | 0.847 m | 0.058 m | below **50 %** of standing (0.42 m) |
| torso upright (cosine of tilt) | 1.000 | 0.051 | below **0.5** (60° from vertical) |

Either condition alone counts. A fighter can be folded over with its torso still high, or flat on its
back with the torso barely below stance, and both are down.

**Knocked out** is *down continuously for the get-up window*: **8 s**, or **400 ticks** at
`TICK_HZ`. This is boxing's standard eight-count, chosen for that reason rather than derived.

It has a consequence worth seeing before a playtest does: 8 s is **13 % of a 60 s round**. A fighter
who goes down and gets up at 7 s has lost an eighth of the round to the canvas without being knocked
out, which is a real cost and probably the right one — but it also means a round can hold at most
seven full counts, and that a knockdown late in a round may be cut short by the bell rather than
resolved. The match loop must therefore handle *down when the bell goes*, and it does: the round ends
by `"bell"`, not `"knockout"`, and the knockdown is recorded with `became_knockout = false`.

Still a feel parameter. `WORKPLAN` M4-T4 may move it, and `format.get_up_window_ticks` travels in
every record so a match is always readable against the number it was fought under.

---

## The record

```
MatchRecord
  schema_version   str
  match_id         str
  format           {rounds, round_ticks, get_up_window_ticks, tick_hz}
  arena            ArenaConfig, verbatim          <- 0.2
  fighters         {red: FighterEntry, blue: FighterEntry}
  versions         {policy, pose_library, robot_model, rules, upstream_sha, openroboxing_sha}
  seeds            {match_seed, red, blue}
  rounds           [RoundRecord, ...]

FighterEntry
  handle           str
  loadout          {name, version, slots: {slot: pose_name}}

RoundRecord
  index            int, 0-based
  ticks            int, how long it actually ran
  ended_by         "bell" | "knockout"
  knocked_out      null | "red" | "blue"
  hits             [HitEvent, ...]
  knockdowns       [KnockdownEvent, ...]
  commits          [CommitEvent, ...]
  trace            StateTrace
```

### The ring travels too (0.2)

`arena` carries `runtime/arena.ArenaConfig` verbatim: ring size, rope heights, glove radius and
padding, canvas friction, physics timestep.

For the same reason `format` travels. A trace is `qpos` and nothing else, so replaying it means
rebuilding the ring around it — and a fight replayed in a 4.90 m ring that was fought in a 3.50 m one
shows fighters standing inside the ropes. `M4-T4` exists to change these numbers after a playtest,
which makes this the difference between an archive and a pile of unreadable files.

A record written before 0.2 is read as having been fought in the defaults, because at the time that
is all there was.

### Events

`HitEvent` is `runtime/contact.py`'s, verbatim: `attacker`, `defender`, `attacker_body`,
`defender_body`, `region`, `start_tick`, `end_tick`, `peak_force_n`, `impulse_ns`, `position`.

**Every glove contact on an opponent is recorded**, with no minimum force. Deciding what counts as a
scoring blow is the rules layer's job, and it must be possible to change that decision without
re-simulating every archived match. Nothing is thrown away at record time.

Glove-on-glove (a parry) and body-on-body (a clinch) are not hits and are not recorded as such.

`KnockdownEvent`: `fighter`, `start_tick`, `end_tick`, `lowest_torso_height_m`, `min_upright`,
`became_knockout` (bool).

`CommitEvent`: `fighter`, `slot`, `pose_name`, `issued_at`, `commit_at`, `strike_at`, `end_tick`,
`arrived`, `placement`, `adjustment`. This is `runtime/intents.Commit` — the record of what the
*player* did, as opposed to what the physics did.

Since `spec/intent.md` 1.1 a commit walks to its placement before it throws, so three of those fields
are **null until the move reaches that stage**, and a commit still walking at the bell has all three
null. `strike_at` is the tick the punch was actually thrown — the one to score aggression on, since
`commit_at` is now the start of a walk that may run for seconds. `arrived` is false when the approach
timed out and the fighter threw where it stood, which is how a replay can show a move that fell
short. Records written before 1.1 have no `strike_at` or `arrived` at all; for those `commit_at`
*was* the moment the move fired.

### The state trace

**Full state, every tick.** `qpos` for both fighters — 2 × 36 floats at `TICK_HZ`.

Stored as `float32`: a match is 3 × 3000 ticks × 72 floats ≈ **2.6 MB**, compressed less. That is
heavy for a public archive and it is accepted deliberately, because a full trace replays *exactly,
forever*, regardless of policy weights, pose library or physics drifting underneath it. Seeds alone
would be a hundredth the size and would stop reproducing the moment anything upstream changed — and a
league whose old matches cannot be replayed cannot settle a dispute about them.

Seeds and the intent log are recorded **as well**, so a match can also be re-simulated. When
re-simulation and the trace disagree, the trace is authoritative and the disagreement is the useful
signal: something in the stack moved.

---

## Determinism

`CLAUDE.md` invariant 6: determinism is *recorded*, not assumed. Bit-exact re-simulation is not
promised. What the record guarantees is that the trace shows what happened.

`versions` and `seeds` exist so a match can be *traced to the assets that produced it*
(`WORKPLAN` M6-T1), not so it can be re-derived from them.

---

## Open — to be decided, not invented

- **Rest between rounds.** Not simulated in v0.1. Whether a fighter carries fatigue or damage across
  rounds is a rules question that needs a damage model, which does not exist.
- **What a knockout is worth.** The match loop emits the event; scoring is M5-T2.
- **Starting stance each round.** v0.1 resets both fighters to the arena's starting stance. Whether a
  knocked-out fighter starts the next round disadvantaged is a rules decision.

## What replays, and what is a reconstruction

`runtime/replay.py` plays a record back. The distinction matters when a result is disputed:

| | replays | why |
|---|---|---|
| positions, separation, ring control | **to float32** | functions of `qpos`, which *is* the trace |
| torso height and orientation | **to float32** | likewise |
| **knockdowns and knockouts** | **exactly** | derived from the two rows above, by `runtime/match.py` |
| the picture | 1 pixel in 153,600, by 1/255 | measured; see below |
| hit *occurrence* | closely | contact geometry follows from `qpos` |
| **`peak_force_n`, `impulse_ns`** | **no** | they depend on velocity and acceleration, which the trace does not carry |

"To float32" is a quantisation of at most **4.8 × 10⁻⁸** in any joint angle or coordinate — eight
orders of magnitude below the knockdown thresholds, so a knockdown re-derives exactly unless a torso
sits within 50 nanometres of the line. Verified on a real 3-round match: every knockdown agreed, and
hit *counts* came back 16/16, 30 vs 28, 71/71.

Visually, rendering is bit-exact for a given `qpos` — two separately built rings render byte-identical
frames — so the float32 quantum is the *only* source of difference, and it reaches exactly one pixel
in 153,600 at one part in 255. `tests/test_replay.py` pins those bounds; loosening them is a decision
about this format, not a test fix.

Velocities are reconstructed with `mj_differentiatePos`, which gets close but is not the recorded
number. **The forces in the record are the authoritative ones** — they were measured while the fight
was simulated. A replay must never overwrite them, and `ReplayWorld` does not.

The consequence worth stating: a *knockdown* can be settled from a trace by anyone, with no GPU. An
argument about how hard a punch landed is settled by the record, or not at all.

## Changelog

- **0.2** (2026-08-08) — added `arena`, the `ArenaConfig` a match was fought in, so a record stays
  replayable once `M4-T4` starts changing the ring. Records without it are read as the defaults.
  Added *What replays, and what is a reconstruction* above; no field changed meaning.
- **0.1** (2026-08-07) — first version. Format 3 × 60 s with an 8 s get-up window; a knockout ends
  the round and the match continues; every glove contact recorded unfiltered; full state trace as the
  authoritative replay.
