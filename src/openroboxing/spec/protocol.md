# protocol.md — client ↔ match host

Version **0.6** · created 2026-08-08 · tasks `M4-T1`, `M4-T2`, `M6-T8`

One websocket per player. The host owns the match; the client owns nothing and is never trusted.

---

## The rule that shapes the protocol

**The client is a view and a keyboard.** It does not simulate, does not decide when a commit is legal,
and does not hold the clock. `spec/intent.md` says the queue rule "is enforced here rather than in
the UI, because a client cannot be trusted to enforce it" — that is a statement about *this*
protocol. A client may show a commit as rejected only because the host said so.

0.4 adds one carefully bounded exception, and names it as one: **the shadow lives entirely in the
client**. The ghost a player aims with is drawn in the browser, moved in the browser, and the host
never sees it — a preview that round-tripped before it moved would be unusable, and where a player is
*thinking* of standing is not the server's business. The host learns a ghost only when one is
**committed**. See §The shadow.

0.6 adds a second exception the client-side shadow already implied but 0.4-0.5 did not have to name:
**the library is not secret.** `spec/intent.md` 3.0's `D6` retires the per-seat loadout — both
fighters have identical, complete access to every combination — so `welcome` now ships the whole
library to a seat *and* to a spectator, rather than the single loadout a seat brought. See §Host →
client and §"Feasibility".

## Rates

| | Rate | Source |
|---|---|---|
| Simulation | 50 Hz | `spec/rates.md` |
| Intent service | **30 Hz** | `WORKPLAN` M4-T2 |
| State stream | **30 FPS** | `WORKPLAN` M4-T2 |

Intents arrive whenever the player presses a key and are applied at the next 30 Hz service tick, so a
keypress lands within 33 ms. The host never blocks the simulation waiting for a client.

## Transport

WebSocket. **Text frames are JSON control messages; binary frames are body transforms.** A binary
frame always refers to the most recent `state` message, so a client that renders the two together is
never more than one frame out of step.

### The client renders (0.4)

Until 0.4 a binary frame was a **server-rendered JPEG** of the ring, and the client was one `<canvas>`
with no 3-D stack. That was a deliberate bet — cheap client, no meshes to ship — and the project owner
overruled it on 2026-08-08: the game needs a real 3-D view because it needs a **shadow you can drive
around the ring**, and you cannot place a ghost in space by looking at a flat video of it.

So the client now runs three.js and the host streams poses:

| | 0.3 (JPEG) | 0.4 (transforms) |
|---|---|---|
| per frame | ~25 kB at 640×360 | **~1.8 kB** |
| per viewer | ~750 kB/s | **~55 kB/s** |
| one-off | none | ~18.5 MB of meshes, cached |
| client | a canvas | three.js + an STL loader |

The bandwidth pays for the meshes back in about 25 seconds of play, and the view is no longer capped
at the server's chosen resolution or camera.

**The host does the fighters' kinematics; the client does the shadow's.** Frames carry *world*
transforms per body — already computed, since MuJoCo computed them to step physics — so the client
never owns the fighters' kinematic tree and cannot drift from the simulation. Meshes are
free-floating objects positioned each frame.

The shadow is the deliberate exception: it is posed in the browser from the joint angles in `welcome`
and the kinematic tree in `/scene.json`. That is the only forward kinematics the client runs, and it
touches nothing the simulation owns.

### Scene description

Sent once, as JSON, at `GET /scene.json`. Everything static: which drawables exist, what shape they
are, which body they hang off, and their local offset within it.

```jsonc
{
  "bodies": ["red_pelvis", "red_left_hip_pitch_link", ..., "blue_..."],   // streamed, in order
  "shadow_bodies": ["pelvis", "left_hip_pitch_link", ...],               // one fighter's worth
  "shadow_kinematics": {
    "bodies": [{"parent": -1, "pos": [...], "quat": [...],
                "joints": [{"name": "left_hip_pitch_joint", "axis": [0,1,0], "pos": [...]}]}],
    "joints": ["left_hip_pitch_joint", ...]                              // MuJoCo order
  },
  "meshes": [{"name": "pelvis", "verts": 1234, "faces": 2400}, ...],     // /meshes.bin order
  "drawables": [
    {"body": 3, "type": "mesh", "mesh": 7,
     "pos": [0,0,0], "quat": [1,0,0,0], "size": [...], "rgba": [0.7,0.7,0.7,1]},
    {"body": -1, "type": "capsule", "pos": [...], "quat": [...], "size": [0.02, 2.45],
     "rgba": [0.9,0.9,0.85,1]}                                            // body -1 = world, static
  ],
  "meshes_url": "/meshes.bin",
  "arena": {"ring_size": 4.90, ...}
}
```

`body` indexes `bodies`; `-1` means the world body, which never moves and is placed once. A drawable
is a mesh (`mesh` indexes `meshes`) or a MuJoCo primitive (`sphere`, `capsule`, `cylinder`, `box`,
`plane`) — the ring is primitives, the fighters are meshes.

Every G1 joint is a **hinge**, so the shadow's FK needs only an axis and an anchor per joint. A model
that grew a slide or ball joint would pose wrong in a way nobody would notice, so `/scene.json`
refuses to describe one rather than emitting something the client would silently mis-draw.

### Mesh geometry

`GET /meshes.bin` — every unique mesh concatenated in `meshes` order. Per mesh: `float32[verts*3]`
positions, `float32[verts*3]` normals, `uint32[faces*3]` indices. Offsets are the running sum of the
counts already in the description, so they are not sent twice.

Red's and blue's geometry is identical, so it is **deduplicated and instanced**: 72 compiled meshes
become 36 shipped ones, about 10 MB, fetched once and cached. Normals ship rather than being
recomputed in the browser because MuJoCo's carry the model's hard edges.

### Binary frame layout

Little-endian. Header, then a flat `float32` array of `(px, py, pz, qw, qx, qy, qz)` per body:

```
offset  type      field
0       uint32    magic 0x4F42524F  ("ORBO")
4       uint32    tick
8       uint16    n_bodies         — matches len(scene.bodies)
10      uint16    reserved (0)
12      float32[n_bodies * 7]
```

**The same bytes go to every viewer.** A frame holds only the two real fighters, so there is nothing
private in it and one pack serves the whole room. Privacy lives in the JSON.

Quaternions are **`wxyz`**, MuJoCo's convention, not three.js's `xyzw`. The client converts on the way
in; that is the only place in the client where the difference exists.

A frame whose `n_bodies` disagrees with the scene description is a bug, not something to render
partially — the client must refuse it loudly (`CLAUDE.md` invariant 5).

---

## Client → host

```jsonc
{"type": "join",   "handle": "carlo", "seat": "red"}          // seat may be omitted; host assigns
{"type": "intent", "combination": "hook-left", "ghost": [1.2, -0.4]}   // 0.6 — what, and where it ends
{"type": "commit"}                                              // queue whatever is staged
{"type": "clear"}                                               // unstage; not a cancellation
{"type": "ping",   "t": 1712345678901}                         // client clock, echoed back
```

`intent` and `commit` are separate because they are separate in `spec/intent.md`: staging is
continuous and free, committing is the irreversible act. A client that only ever sends `commit` is
legal and means "fire whatever was last staged".

**0.6 collapses 0.4's separate `stage` (which slot) and `place` (where, with a player-set heading)
into one `intent` message.** `spec/intent.md` 3.0's `D6`/`D5` removed both of the reasons they needed
to be two: a combination has no slot to name — it is selected by name, from the whole shared library
— and its ghost carries no heading any more, since the ghost's heading is *derived* (the fighter's own
heading plus the combination's recorded turn) and never player-set. There is exactly one thing left to
stage, not two, so there is exactly one message.

`ghost` is **absolute** MuJoCo world `(x, y)` — the position the client's own shadow, drawn and moved
entirely in the browser (see §The shadow), has already been dragged to. The host never sees the
shadow move; it only ever sees where an `intent` says it currently is.

**Every message is validated.** An unknown `type`, an unknown `combination`, a non-finite or
malformed `ghost`, a `ghost` beyond the *combination's own* `reach_m` (see §"Feasibility"), or a
`commit` with a full queue produces an `error` message; it never mutates the match and never closes
the socket.

*(0.3's `move` message was removed at 0.4. 0.4's `stage` and `place` were removed at 0.6, folded into
`intent` as above.)*

## Host → client

```jsonc
{"type": "welcome", "seat": "red", "combinations": [{...}, ...],
                    "format": {...}, "arena": {...}, "match_id": "..."}
{"type": "state",   "tick": 412, "round": 1, "clock_ticks": 2588,
                    "seats": {"red": {...}, "blue": {...}}, "phase": "fighting",
                    "score": {...}, "separation_m": 1.42}
{"type": "event",   "event": "hit"|"knockdown"|"knockout"|"round_end"|"match_end", ...}
{"type": "error",   "message": "...", "rejected": "commit"}
{"type": "pong",    "t": 1712345678901}                    // the client's own clock, echoed
```

### `welcome`'s combination library (0.6)

```jsonc
{"type": "welcome", "spec_version": "0.6", "seat": "red", "match_id": "...",
  "combinations": [
    {"name": "hook-left-cross", "seconds": 1.6, "heading_delta": -0.12, "reach_m": 1.33,
     "pose": {"left_hip_pitch_joint": -0.31, ...}},   // all 29, the *final* keyframe only
    ...
  ],
  "approach_speed_m_s": 0.83,
  "format": {...}, "arena": {...}
}
```

**No more `loadout`.** `spec/intent.md`'s `D6` retires the per-seat six-slot loadout: both fighters
have identical, complete access to the whole library, and — because the library is not secret — a
spectator's `welcome` carries the same `combinations` a seat's does (unlike 1.0-2.2's loadout, which a
spectator's `welcome` withheld). `horizons`, `pose_seconds` and `poses` are gone with it; `pose` (one
entry's own final-keyframe joint angles) and `seconds` (`duration_ticks / TICK_HZ`) replace them,
one-for-one per combination rather than once per slot.

Sorted by `name`, so the client can page through it deterministically nine at a time (`D6`) without
inventing its own ordering.

`pose` carries only the **final keyframe's** joint angles — the pose the ghost is drawn in, not the
whole combination's motion — because that is the one MotionBricks-independent fact the client's
shadow needs to pose itself (see §The shadow, unchanged in principle since 0.4: the shadow is drawn in
the browser, and a ghost that had to ask the server where its elbow goes could not be aimed with).

`reach_m` is carried **per combination**, replacing 1.1's single ring-wide "everywhere is reachable" —
see §"Feasibility".

### Seat state, per fighter

```jsonc
{
  "handle": "carlo",
  "staged": "hook-left-cross",              // combination name, or null
  "position": {"x": -1.1, "y": 0.02},       // 0.6 — where the fighter is standing right now
  "ghost":    {"x": 1.2, "y": -0.4},        // 0.6 — where the shadow is, or null
  "anchor":   {"x": 0.9, "y": 0.0},         // 0.6 — where a commit issued now would start from
  "queue": [                                             // scheduled, executing first
    {"combination": "hook-left-cross", "ghost": {"x": 1.2, "y": -0.4}, "issued_at": 400,
     "commit_at": 430, "end_tick": 510, "executing": true},
    {"combination": "jab-left", "ghost": {"x": 0.9, "y": 0.0}, "issued_at": 455,
     "commit_at": null, "end_tick": null, "executing": false}    // has not started
  ],
  "queue_depth": 2,               // len(queue)
  "can_commit": true,             // queue_depth < MAX_OUTSTANDING_COMMITS, decided by the host
  "hits_landed": 7,
  "torso_height_m": 0.84,
  "down": false
}
```

`can_commit` exists so the client can grey out a key **without knowing the rule**. If the host's
rule changes, no client changes.

**`commit_at` and `end_tick` are `null` until the commit becomes current.** Unlike 1.1-2.2, this is
not "unknown, ask again later": `spec/intent.md` 3.0 gives every combination a **fixed** duration
(`record.duration_ticks`, read straight off the recording), so the instant `commit_at` is stamped,
`end_tick = commit_at + record.duration_ticks` is exact arithmetic and is never revised. `null` still
means *"not yet"* — the commit is queued but its turn has not come — never zero and never "already
over". There is no `strike_at` / `approaching` distinction any more: 3.0 deleted the walk-then-throw
phase they existed to tell apart, so a commit is simply not started, running (`executing`), or
finished.

**A seat sees its own `queue` in full. The opponent's `queue` carries only entries that are
executing.** A queued-but-unstarted commit has been paid for but not yet shown, and showing it would
hand the opponent a readable list of your next four moves — which is exactly the risk the queue is
supposed to be (`spec/intent.md` §The shadow).

### The shadow (0.4, ghost-only since 0.6)

**The shadow is the client's.** It is posed in the browser — the final keyframe's joint angles from
`welcome`, the kinematic tree from `/scene.json` — moved in the browser, and never transmitted while
it is being aimed. The host learns a ghost only when `intent` (or a bare `commit`) sends one. A ghost
that asked the server where to stand before it could move would be unusable, and a half-formed plan is
not the server's business.

**Position only, since 0.6.** Before `spec/intent.md` 3.0 a placement carried a player-set `heading`
too; a ghost's heading is now *derived* — the fighter's own heading plus the combination's recorded
turn (`runtime/warp.py::ghost_heading`) — and is never chosen by the player (design `D5`, because the
corpus's travelling combinations turn by up to 158° and a target-facing ghost would discard that
turn). The client may still compute and draw the derived heading for its own preview, but it has
nothing to send the host about it.

`anchor` is **where a commit issued right now would start from**: the last queued commit's ghost — a
combination's whole premise is that its final keyframe lands exactly there — or the fighter's current
position when the queue is empty. It is a *projection*, not a promise (see §"Feasibility" — a fighter
knocked off course still reaches its ghost, by drifting harder, never by the commit being refused
after the fact).

It is the anchor and not the fighter's live `position` that a client should judge a new ghost against,
for the same reason 0.4 gave: the next move starts from the end of the queue, and the anchor is
*stable*, where the live position moves for the whole duration of every move.

`ghost` in seat state is what the host holds as this seat's most recently staged ghost, echoed back so
a reconnecting client can recover it. It is *not* a live feed of the shadow's on-screen position,
which the host cannot see while it is being dragged.

### Feasibility: `reach_m` (0.6)

`spec/intent.md` "Feasibility": since a combination's duration is fixed by its recording (unlike
1.1-2.2, where anywhere in the ring was reachable and distance only cost time), how far its ghost may
sit from the anchor is bounded, and the bound **differs per combination** — roughly 1.6-6.6 m across
the library. `welcome`'s `reach_m` is that bound, computed once per entry
(`approach_speed_m_s * duration_ticks / TICK_HZ`); the host enforces the identical number when a
commit is issued (`server/protocol.py::check_reach`), so the two can never disagree, and a client that
shows it before a placement is attempted is never showing a number the host will contradict.

**This is the one place the speed ceiling is enforced.** It runs once, when the player commits — never
again. A commit that starts off-target because physics did not track a previous move exactly still
runs and still reaches its ghost, at whatever drift that needs, even above `approach_speed_m_s`
(`spec/intent.md` "Off-target execution"); nothing about that is re-checked or refused. A client that
shows `reach_m` is helping a player avoid the one rejection that *can* happen, not describing every
outcome that can follow a commit.

### Movement and range

`separation_m` is the distance between the two fighters. **It is sent because range is not secret** —
both fighters can see how far apart they are by looking — and because a player who cannot judge
distance cannot manage it, which is most of boxing. Measured: two fighters standing settle at
**0.99 m** against a contact range of 0.80 m, so at rest you are just out of reach and closing is a
decision.

### The live score (0.2)

```jsonc
"score": {
  "share":      {"red": 0.58, "blue": 0.42},   // this round so far, weighted, sums to 1
  "leading":    "red",                          // or null while it is level
  "dimensions": {"red": {"damage": 3.4, "control": 0.21, "aggression": 0.55}, "blue": {...}},
  "points":     {"red": 10, "blue": 9},         // completed rounds only
  "rounds_won": {"red": 1, "blue": 0}
}
```

**It is the real score, not a second scoreboard.** `share` is `spec/scoring.md`'s own weighted round
score, run over the events so far — same weights, same definitions, same code
(`league/scoring.score_round`). The alternative, a bespoke "who is winning" number invented for the
UI, would have been a second scoring system that disagrees with the official one at the worst moment.

Two honest limits, both stated so a client can present them properly:

- **It is provisional.** A round is scored when it ends; damage is normalised *within* the round, so
  an early share swings hard on one landed punch and settles as the round fills up.
- **`points` covers completed rounds only.** The round in progress contributes nothing to it until
  the bell, because the 10-point must depends on how the round *finished*.

Recomputed at `SCORE_INTERVAL_TICKS`, not every frame: ring control integrates over the whole
trace, and a client cannot see 25 recomputes a second anyway.

### What is deliberately not sent

- **No fighter HUD.** `WORKPLAN` M4-T1: "No HUD on the fighters — the windup is the only cue." The
  opponent's staged combination, shadow and unstarted queue entries are *never* transmitted.
- **No joint angles, only body transforms.** The client cannot reconstruct a fighter's internal state
  from what it draws, and does not need to.

## Phases

`fighting` → `round_over` → (`fighting` | `match_over`). The host drives it; a client that misses a
transition recovers from the next `state`, which always carries the phase.

## Hotseat ○

`M4-T1`'s acceptance is two people on one machine. One browser opens **two** sockets, one per seat,
and maps two key groups (see `client/`). The host cannot tell hotseat from two machines and does not
need to. Logged in `docs/ASSUMPTIONS.md` §A6.

## Latency

`ping`/`pong` echo the client's own clock so round-trip is measured without clock sync. `WORKPLAN`
M4-T2 requires that injected latency does not systematically change outcomes: because the host
services intents on its own 30 Hz tick and never waits for a client, latency delays *when a commit
lands*, and does not advantage either seat.

## Changelog

- **0.6** (2026-08-28, `M6-T8`) — **the protocol catches up with `spec/intent.md` 3.0: a commit is a
  combination.** `stage` (which slot) and `place` (where, with a player-set heading) collapse into one
  `intent` message (`{"combination": ..., "ghost": [x, y]}`) — a combination has no slot to name and
  its ghost has no heading to carry any more (`D5`/`D6`). `welcome` drops the per-seat `loadout` and
  ships `combinations`: the whole shared library, sorted by name, one entry per combination carrying
  `seconds`, `heading_delta`, a per-combination `reach_m`, and the *final keyframe's* `pose` — because
  under `D6` there is no loadout left to be secret, a spectator's `welcome` now carries exactly what a
  seat's does. `reach_m` replaces 1.1's retired assumption that anywhere in the ring is reachable: a
  combination's duration is fixed by its recording, so how far its ghost may sit from the anchor is
  bounded and differs move to move (`spec/intent.md` "Feasibility") — the host enforces the identical
  number the instant a commit is issued (`server/protocol.py::check_reach`), the one place the speed
  ceiling is enforced; a commit that starts off-target still reaches its ghost anyway, running
  whatever drift that needs, and is never refused after the fact. Seat state's `placement` is renamed
  `ghost` and drops `heading`; `queue` entries drop `slot`, `pose`, `strike_at` and `approaching` —
  3.0 deleted the walk-then-throw phase they distinguished — and gain `combination` and `ghost`.
  `Loadout` and `Placement` are gone from the runtime types this module speaks against.
- **0.5** (2026-08-08) — **a commit's span is settled as it runs.** `commit_at`, `strike_at` and
  `end_tick` in a queue entry are `null` until the move reaches each stage, and `approaching` says
  whether it is still walking. `welcome` dropped `reach_m` — since `spec/intent.md` 1.1 anywhere in
  the ring is reachable — and gained `pose_seconds` and `approach_speed_m_s`, which let a client
  estimate what a placement will *cost in time* rather than refusing it.
- **0.4** (2026-08-08) — the client renders. Binary frames became packed body transforms instead of
  JPEGs (~750 kB/s → ~55 kB/s), with a one-off scene description at `/scene.json` and geometry at
  `/meshes.bin`. Added `place`, `queue`, `placement`, `anchor`, and `poses` + `shadow_kinematics` so
  the browser can pose its own ghost; removed `move`. State JSON became per viewer so a queue cannot
  leak. The project owner overruled the server-rendering decision (`docs/ASSUMPTIONS.md` §A7) because
  the game needs a shadow you can drive around the ring, which a flat video cannot give you, and
  placed the shadow wholly client-side so aiming costs no round trip.
- **0.3** (2026-08-08) — added `move` and `separation_m`. *`move` superseded by 0.4.*
- **0.2** (2026-08-08) — added the live `score` block to `state`.
- **0.1** (2026-08-08) — first version.
