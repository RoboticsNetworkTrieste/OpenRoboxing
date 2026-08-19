# protocol.md — client ↔ match host

Version **0.5** · created 2026-08-08 · tasks `M4-T1`, `M4-T2`

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
*thinking* of standing is not the server's business. The host learns a placement only when one is
**committed**. See §The shadow.

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
{"type": "join",   "handle": "carlo", "seat": "red"}      // seat may be omitted; host assigns
{"type": "stage",  "slot": "3"}                            // choose the move's final pose
{"type": "place",  "x": 1.2, "y": -0.4, "heading": 1.57}   // 0.4 — where the shadow stands
{"type": "commit"}                                          // queue whatever is staged
{"type": "clear"}                                           // unstage; not a cancellation
{"type": "ping",   "t": 1712345678901}                     // client clock, echoed back
```

`stage` and `commit` are separate because they are separate in `spec/intent.md`: staging is
continuous and free, committing is the irreversible act. A client that only ever sends `commit` is
legal and means "fire the current stage, wherever the shadow last was".

**Every message is validated.** An unknown `type`, a `slot` outside the loadout, a non-finite
coordinate, or a `commit` with a full queue produces an `error` message; it never mutates the match
and never closes the socket.

*(0.3's `move` message was removed at 0.4. Steering is gone — `spec/intent.md` §"What happened to
walking".)*

## Host → client

```jsonc
{"type": "welcome", "seat": "red", "loadout": {"1": "jab-left", ...},
                    "format": {...}, "arena": {...}, "match_id": "..."}
{"type": "state",   "tick": 412, "round": 1, "clock_ticks": 2588,
                    "seats": {"red": {...}, "blue": {...}}, "phase": "fighting",
                    "score": {...}, "separation_m": 1.42}
{"type": "event",   "event": "hit"|"knockdown"|"knockout"|"round_end"|"match_end", ...}
{"type": "error",   "message": "...", "rejected": "commit"}
{"type": "pong",    "t": 1712345678901}                    // the client's own clock, echoed
```

### Seat state, per fighter

```jsonc
{
  "handle": "carlo",
  "staged": "3",                  // slot, or null
  "placement": {"x": 1.2, "y": -0.4, "heading": 1.57},   // 0.4 — where the shadow is, or null
  "anchor":    {"x": 0.9, "y": 0.0, "heading": 0.0},     // 0.4 — where the queue leaves you
  "queue": [                                             // 0.4 — scheduled, executing first
    {"slot": "3", "pose": "uppercut-right", "issued_at": 400,
     "commit_at": 430, "strike_at": null, "end_tick": null,   // 0.5 — still walking there
     "executing": true, "approaching": true},
    {"slot": "1", "pose": "jab-left", "issued_at": 455,
     "commit_at": null, "strike_at": null, "end_tick": null,  // 0.5 — has not started
     "executing": false, "approaching": false}
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

**`commit_at`, `strike_at` and `end_tick` are `null` until the move reaches each stage (0.5).** A
commit runs until it arrives (`spec/intent.md` 1.1), so its span is not known when it is issued and
the queue is not a schedule. A client must read `null` as *"not yet"* — never as zero, and never as
"already over", which would make a fighter walking across the ring look idle. `approaching` is the
one flag worth drawing: the move has started and is walking, and the punch has not been thrown.

**A seat sees its own `queue` in full. The opponent's `queue` carries only entries that are
executing.** A queued-but-unstarted commit has been paid for but not yet shown, and showing it would
hand the opponent a readable list of your next four moves — which is exactly the risk the queue is
supposed to be (`spec/intent.md` §The shadow).

### The shadow (0.4)

**The shadow is the client's.** It is posed in the browser — pose angles from `welcome`, kinematic
tree from `/scene.json` — moved in the browser, and never transmitted while it is being aimed. The
host learns a placement only when `commit` fires. A ghost that asked the server where to stand before
it could move would be unusable, and a half-formed plan is not the server's business.

`anchor` is **where the fighter will be standing when everything it has committed has finished** —
the last queued commit's placement, or the fighter's current root pose when the queue is empty.

Since 1.1 that promise is a real one: a commit walks until it arrives, so the anchor is where the
fighter will actually be rather than a point it was aimed at and fell short of.

It is the anchor and not the fighter's live position on purpose. The next move starts from the end of
the queue, so that is what a placement should be judged against; and it is *stable*, where the live
position moves under the player's cursor for the whole duration of every move.

So the client draws its shadow at `anchor + local offset` and edits the offset with no round trip.
On `commit` it sends the resulting **absolute** placement and resets the offset to zero. A commit
with no `place` since the last one means "do this where I will be" — the host resolves it against the
anchor — which is what a pure strike wants.

`heading` defaults to facing the opponent from the anchor. Distances are metres in **MuJoCo world
coordinates**, the frame the arena, the scene description and the binary frames all use.

`placement` in seat state is what the host holds for that seat, echoed back so a reconnecting client
can recover. It is *not* a live feed of the ghost, which the host cannot see.

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
  opponent's staged slot, shadow and unstarted queue entries are *never* transmitted.
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
