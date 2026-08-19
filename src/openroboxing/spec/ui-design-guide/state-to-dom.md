# State → DOM mapping

Every readout in the design, the exact field it comes from, how it is formatted, and what it does
when the field is `null`. The client never derives a rule — where the server sends a decision, the UI
obeys it.

Transport: `welcome` once on connect, `state` at 30 Hz, `event` broadcast, `error` on rejection,
`pong` for the ping readout, `/static/table.json` polled every 30 s.

---

## From `welcome` (once, on connect)

| Field | Where it lands | Formatting |
|---|---|---|
| `seat` | which panel gets the key hints and the ghost; `spectator` → the `/screen` layout | — |
| `loadout["1".."6"]` | the 6 loadout slot names | verbatim, 11px, lowercase as sent |
| `pose_seconds["1".."6"]` | the per-slot duration | `toFixed(2) + " s"`, mono tabular |
| `horizons["1".."6"]` | Pose Studio token count and its 16-cell strip | integer; `tokens × 4 / 30 = seconds` |
| `poses` | the ghost's joint angles, run through `/scene.json` locally | never displayed as numbers |
| `approach_speed_m_s` | denominator of the walk estimate | not displayed |
| `format.rounds` | `round N / 3` | integer |
| `format.round_ticks`, `tick_hz` | clock conversion | `clock_ticks / 50` seconds |
| `arena.ring_size` | ring dimension label, map scale (100 units = 1 m) | `4.90 m` |
| `handles` (spectator only) | corner names on `/screen` | verbatim |

A **spectator's** welcome carries no `loadout` and no `poses` — a projector must not leak what each
fighter has available. Build `/screen` so it never reads those fields.

---

## From `state` (30 Hz)

### Header / scoreline

| Field | Readout | Rule |
|---|---|---|
| `clock_ticks` | `0:51` | `ticks / 50`, floor to `m:ss`. Under 10 s the clock and the LIVE pill go `--status-warn`; that is the only urgency cue |
| `round` | `round 2 / 3` | 1-based, with `format.rounds` |
| `phase` | which layout is live | `fighting` → normal · `round_over` → the between-rounds state · `match_over` → the match banner |
| `tick` | not displayed | cut in the minimal pass |
| `pong.t` round trip | `ping 24 ms` | `Math.round(now - t)`, mono hud label. Small, dim, always available, never alarming |

### Separation card (public, both seats)

| Field | Readout | Rule |
|---|---|---|
| `separation_m` | `1.18 m` + the band word | `toFixed(2)`. Band from the scorer's own contact range: `≤0.80 IN RANGE` · `≤1.30 CLOSING` · `≤2.00 OUT OF RANGE` · else `MILES APART`. The band drives the word, its colour, and which of the four strip segments is filled |

The metres are secondary to the word. `1.42 m` means nothing mid-fight; `CLOSING` does.

### Score card

| Field | Readout | Rule |
|---|---|---|
| `score.share.red/blue` | `58%` / `42%` and the two-segment bar | `Math.round(share * 100)`. The pair sums to 1. Recomputed twice a second by design — **let it settle, do not animate it** |
| `score.leading` | `leading · red` | `null` while inside the draw margin → print `—` in `--text-muted`. Never guess a leader |
| `score.dimensions.*.damage / control / aggression` | the three bars | bar widths from the red/blue ratio per dimension. **The raw values are not printed** |
| `score.points` | `points 10 — 9` | integers. **Completed rounds only** |
| `score.rounds_won` | `rounds 1 — 0` | integers |

`share` is the official scorer's own weighted number over the round so far, not a second scoreboard.
The `provisional` chip must be present whenever it is shown. Damage is normalised *within* the round,
so an early share swings hard on one punch and settles as the round fills — the copy under the bar
says so.

### Per seat — `seats.red` / `seats.blue`

| Field | Readout | Rule |
|---|---|---|
| `handle` | the corner name | verbatim |
| `hits_landed` | the hit tally | integer |
| `staged` | which loadout slot renders staged | `null` → no slot is staged, and the cost block shows `—` / `nothing staged` |
| `position` | the fighter's live root pose in the canvas | **public for both seats** |
| `placement` | your own aimed placement | own seat only; sent to the server only at the moment of commit |
| `anchor` | the anchor marker, and the origin of the walk measurement | own seat only. **This is where the queue leaves you** — the last queued commit's placement, or your current position when the queue is empty. It is stable while a move plays out, unlike `position` |
| `queue[]` | the five queue rows | see below |
| `queue_depth` | `queue · N of 5` and `N left` | integer |
| `can_commit` | the commit chip's enabled state | **host-decided. Never re-derive it.** `false` → chip greys to `opacity:.42`, label becomes `commit — queue full` |
| `torso_height_m` | the knockdown banner only | `toFixed(2) + " m"` |
| `down` | the knockdown banner + the flattened fighter in the diagram | `true` → banner over the ring, never on the fighter |

### Queue cells — `queue[i]`

Five cells are **always drawn**, so the room you have left is visible rather than inferred from a
number. Cell state is derived only from the flags the server sends:

| Server state | Cell |
|---|---|
| no entry at index `i` | `empty` — dashed border, no fill |
| entry present, `executing:false` | `waiting` — paid for, not started |
| `executing:true` **and** `approaching:true` | `walking` — 1.5px accent, accent-quiet fill |
| `executing:true` **and** `approaching:false` | `striking` — 1.5px seat colour, seat-quiet fill |

`approaching:true` is the state most worth drawing: the move has started and is *walking*; the punch
has not been thrown.

| Field | Readout | Rule |
|---|---|---|
| `pose` | the cell's pose name | verbatim, ellipsis on overflow |
| `slot` | not printed in the cell | the loadout bar already carries the key |
| `commit_at`, `strike_at`, `end_tick` | the timing cell | **`null` until the move reaches each stage.** `null` renders `—`, never `0`, never "done". Only the walking cell prints anything: `arrives —` |
| `issued_at` | not displayed | — |
| `placement` | the walk path target for the walking entry | own seat only |

**Visibility.** You see your own queue in full. Of the opponent you see **only entries that are
already executing** — a queued-but-unstarted commit is private. Away from hotseat, render the
opponent's remaining cells as the hatched `private` variant: withheld, not unknown.

---

## The commit cost — computed locally, never sent

```
walk_distance   = distance(anchor, ghost_placement)          // metres, from anchor, not position
walk_seconds    = walk_distance / approach_speed_m_s          // welcome.approach_speed_m_s
throw_seconds   = pose_seconds[staged]                        // welcome.pose_seconds
estimate        = walk_seconds + throw_seconds
```

Rendered as `~{estimate.toFixed(1)} s` with the split `{walk.toFixed(1)} s walk` /
`{throw.toFixed(1)} s throw` and a two-segment bar in the same proportion.

Recompute inside the ghost's own `requestAnimationFrame` loop at 60 fps — **not** on the 30 Hz state
message. This is the decision in the game and it must be impossible to miss while the player's eyes
are on the ring. When `staged` is `null` the block shows `—`.

---

## From `event` (broadcast to everyone)

| `event` | Effect |
|---|---|
| `hit` | a row in the `/screen` live feed, colour-coded to the actor; `hits_landed` updates from the next `state` |
| `knockdown` | the knockdown banner over the ring, carrying the handle, the get-up window (`format.get_up_window_ticks / 50`) and `torso_height_m`; a `--status-warn` row in the feed |
| `knockout` | the match banner, with `knocked_out` naming the seat |
| `round_end` | the between-rounds state; `ended_by` is `bell` or `knockout`; the round's points come from the next `state.score.points` |
| `match_end` | the final banner; `hits` carries the closing tallies |

Banners sit **over** the ring, never on the fighters. No HUD on the fighters — the windup is the only
cue.

## From `error`

```jsonc
{"type": "error", "message": "5 moves are already queued; no cancellation", "rejected": "commit"}
```

Render `message` **verbatim** in the rejecting seat's cost block as an `Alert tone="danger"`. Do not
paraphrase it, do not steal focus, do not block the fight, do not use a modal. `rejected` tells you
which affordance to attach it to.

## From `/static/table.json` (polled every 30 s, may be absent)

| Field | Column |
|---|---|
| `handle` | handle |
| `rating` | rating, `toFixed(1)` |
| `rd` | rd, `toFixed(1)`, `--text-muted` |
| `won` / `drawn` / `lost` | `w-d-l` as `6-1-2` |
| `conservative`, `played` | not shown in this design; available if a column is wanted |

**The league may not be running.** If the fetch fails or returns no `table`, hide the card and let the
live feed take the space. A screen with no table is still a screen; a screen that breaks because the
league is down is not.

---

## Client → server (all the input there is)

```jsonc
{"type": "join",   "handle": "carlo", "seat": "red"}
{"type": "stage",  "slot": "3"}                            // pick the move's final pose
{"type": "place",  "x": 1.2, "y": -0.4, "heading": 1.57}   // sent ONLY at the moment of commit
{"type": "commit"}                                          // irreversible
{"type": "clear"}                                           // unstage; NOT a cancellation
{"type": "ping",   "t": 1712345678901}
```

The ghost's position is **never transmitted while being aimed**. `place` fires once, immediately
before `commit`, after which the local offset resets to zero — so an untouched ghost plus a commit
means "do this where I will be".

`clear` unstages. It is not a cancellation: there is no cancellation.
