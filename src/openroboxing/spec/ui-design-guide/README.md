# Handoff: OpenRoboxing web frontend

## Overview

OpenRoboxing is a physics-simulated humanoid boxing game. Two Unitree G1 robots (1.3 m) box in a
4.90 m ring inside a MuJoCo simulation. A player does not press a punch button: they pick the pose
the move **ends in**, drive a translucent ghost of their own fighter to the spot in the ring where
that move should happen, and commit. The fighter then *walks* there — however long that takes — and
arrives in that pose on the last frame. Physics decides whether it landed. Up to 5 commits queue
back to back and **nothing can be cancelled**.

The design therefore has one job above all others: **make the cost of a commit legible before the
player presses commit, and make the state of the unrecallable queue legible after.** Everything in
these mocks is subordinate to that.

Three surfaces are designed here:

| Surface | Route | Audience | Input |
|---|---|---|---|
| Player view | `/` | 1–2 players at one keyboard (hotseat), or one player + remote agent | keyboard only |
| Fight-night screen | `/screen` | a projector at a meetup, no operator | none at all |
| Pose Studio | `/` of a second server | an author building poses offline | mouse (sliders) |

Plus six player-view states: connecting, waiting for the second seat, disconnected, between rounds,
knockdown banner, and the remote-opponent (private queue) variant.

## About the design files

**The files in this bundle are design references created in HTML.** They are prototypes showing
intended look and behaviour — not production code to copy. `design-reference-standalone.html` is a
single self-contained file: open it in any browser, no server needed. It is a *canvas* containing
all four frames stacked vertically; pan and zoom to read them.

The target implementation is **plain HTML + CSS + ES modules with no build step, no framework and no
CDN** (see Hard constraints). So unlike a normal handoff, this one is close to its target: recreate
the markup as static HTML in the real pages and drive it from the WebSocket state, rather than
porting to a component framework. Do not lift the mock's inline styles wholesale — move them into a
stylesheet with the custom properties listed under Design tokens.

`three.js` is already vendored and already draws the 3-D ring into a `<canvas>`. **You inherit that
canvas; do not redesign the 3-D renderer.** Where the mocks show a wireframe perspective arena, that
is a stand-in for the real canvas — a hand-drawn diagram, not a target.

## Fidelity

**High-fidelity.** Final colours, typography, spacing, layout and states. All three surfaces are
authored at exactly **1920 × 1080** and verified to fit with no overflow. Recreate them
pixel-accurately. The only deliberately low-fidelity elements are:

- the arena wireframe (stand-in for the inherited three.js canvas),
- the Pose Studio preview plate (stand-in for the server-rendered PNG from `/api/render`),
- the top-down placement map, which **is** a real UI element but is drawn schematically.

## Hard constraints (non-negotiable)

- **No build step, no framework, no CDN.** Plain HTML + CSS + ES modules, served from a directory
  someone copied off a USB stick.
- **Dark theme.** The projector is the target. Legible from the back of a room.
- **Keyboard only during a fight.** No focus management, no click-to-activate, no mouse target that
  matters mid-round — two people share one keyboard and nothing may steal focus.
- **The client is a view and a keyboard.** It never decides whether an action is legal. The server
  sends `can_commit`; the UI greys a key out because it was told to, never because it computed a rule.
- **Latency budget.** State JSON at **30 Hz**; the 3-D frame at 30 FPS; the ghost driven locally at
  **60 fps** with no round trip. Anything that must feel instant (ghost motion, key highlight,
  estimated cost) is local. Anything authoritative (queue, score, range) redraws at 30 Hz.
- **Numbers that change 25×/s cannot be read.** The live score is recomputed twice a second by
  design. Readouts settle; they do not flicker.

## What must never be shown

- **No HUD on the fighters. The windup is the only cue.** The opponent's staged slot, their ghost and
  their queued-but-unstarted commits are never transmitted and must never be inferred or faked.
- **No fabricated certainty.** Never render `null` timing fields as `0` or as "done".
- **No second scoring system.** Do not invent a "who's winning" number; use `share`.
- On the projector: **no controls at all**, and no loadouts.

---

# Screen 01 — Player view (`/`)

**Purpose.** Two players at one keyboard fight a 3 × 60 s match. Each seat picks a pose, aims a
ghost, and commits, watching what the commit costs.

**Frame.** 1920 × 1080, `overflow:hidden`. Background: TORC ink `#08130F` with the `survey` field
ground (blueprint plate art under a scrim) and `data-instrument="quiet"`. Border 1px
`rgba(234,242,237,.08)`, radius 3px.

**Layout — four rows, flex column:**

| Row | Height | Contents |
|---|---|---|
| Header | 60px, flex-shrink 0 | wordmark · clock · round · LIVE pill · ping (right) |
| Top grid | ~695px, `padding:12px 16px`, `align-items:start` | `grid-template-columns: minmax(0,1fr) 420px 360px`, gap 16 |
| Seat row | flex 1 (~247px), `padding:0 16px` | `grid-template-columns: 1fr 1fr`, gap 16 |
| Teaching bar | 39px, `margin-top:12px` | kicker + one sentence, top hairline, glass fill |

### Header (60px)

- Left: `OpenRoboxing` — mono 12px, 600, `letter-spacing:.14em`, uppercase, `--text-body`.
  **No logo mark here.** The TORC ring mark is a delivered brand asset and must not be re-drawn or
  set beside another product name.
- 1px × 24px divider `--border-subtle`.
- Clock: Red Hat Display **32px / 700**, `letter-spacing:-.02em`, `tabular-nums`, `--text-strong`.
  Text `0:51`. Beside it `round 2 / 3` in mono 11px `.14em` uppercase `--text-muted`.
- LIVE pill: `--accent-quiet` fill, `border-radius:999px`, `padding:5px 10px`; 6px accent dot with
  `--glow-dot`, pulsing 1 → .45 over 2s `--ease-servo`; label mono 10px `.22em` `--text-accent`.
- Right: `ping 24 ms` only, `.torc-hud-label` (mono 10px, `.22em`, uppercase, `--text-muted`).
  Header telemetry was deliberately cut to one number — see Editorial decisions.
- Background `--surface-header` (88% ink) + `backdrop-filter:blur(14px)`, bottom hairline.

### Arena plate — column 1 of the top grid (~1076px wide)

Card: `--surface-1` fill, 1px hairline, radius 3px, `padding:12px`, plus `.torc-hud` corner brackets.

- Label row, `margin-bottom:6px`: `FIG. 01` in accent · `arena · three.js · unitree g1 · fixed
  camera · 16:9` · `30 fps` right-aligned. All `.torc-hud-label`.
- **The canvas.** In the mock this is an SVG `viewBox="0 0 992 558"` (16:9). In the real page it is
  the inherited three.js `<canvas>` at the same 16:9 box, `width:100%`, `background:#050D0A`,
  radius 2px. What the mock draws — perspective floor, posts, three rope levels, two G1 figures, the
  translucent ghost — the renderer already draws.
  **What the page must add on top of the canvas** (as absolutely-positioned overlays or as part of
  the scene, your call):
  - the dashed accent walk path from `anchor` to the ghost, 1.5px, `dasharray 7 6`, arrow head;
  - the anchor marker: 16px square rotated 45°, 1.5px `--status-danger`, no fill;
  - the contact-range ellipse around the opponent, r 0.80 m, 1px `--status-info`, `dasharray 5 6`,
    opacity .5;
  - the walk distance label (`2.62 m walk`, mono 11px `--text-accent`) on an opaque `#050D0A`
    backing rect, and the ghost callout (`ghost` / `end pose` / `faces the opponent`).
    **Both must be collision-tested against the fighter name labels** — in the mock the fighter band
    occupies y 381–406 of the 558-unit box, so the walk label sits at y 330–345 and the ghost callout
    at x 826, 15-unit line pitch, entirely inside the 992-unit width.
- Legend row under the canvas, top hairline, `gap:6px 22px`, three items in `.torc-hud-label` with
  swatches: filled red dot `live position · G1 mesh`; rotated red square outline `anchor — where the
  queue leaves you`; dashed accent circle `your ghost — local, 60 fps`.

### Placement map + separation — column 2 (420px)

**Placement map card.** `--surface-1`, hairline, radius 3, `padding:16px`.
Label `placement map · top view` / `4.90 m` right. Then a top-down SVG, `viewBox="-24 -24 538 538"`,
ring drawn 490 × 490 (**100 units = 1 m**):

- blueprint grid pattern, 49px pitch (0.49 m), stroke `--border-grid`;
- ring boundary 2.5px `--border-strong`;
- contact circle r 80 around the opponent, 1.5px `--status-info`, `dasharray 6 7`;
- dashed accent path anchor → ghost, 2.5px, `dasharray 9 7`, arrow head;
- anchor: 16px square rotated 45°, 2px `--status-danger`;
- ghost: r 17 circle, `--accent-quiet` fill, 2.5px dashed accent;
- fighters: r 14 filled dots, `--status-danger` / `--status-info`, joined by a 1.5px muted line.

Caption, 12px `--text-muted`: "Where the ghost stands, read from above — the one thing the fixed
camera hides." This card exists **because** the camera is fixed and perspective hides depth.

**Separation card.** Label `separation · public`. Then the band word in Display **28px / 700**
coloured by band, with the metres right-aligned in mono 16px `tabular-nums`. Below: a 6px band strip
(40% / 25% / 20% / 15% widths, `gap:1px`) where the active band is filled `--status-warn` and the
rest `--surface-3`, then four 9px labels `in range` / `closing` / `out` / `miles`.

Bands come from the scorer's own measured contact range (0.80 m):

| Band | Rule | Colour |
|---|---|---|
| `IN RANGE` | ≤ 0.80 m | `--status-ok` |
| `CLOSING` | ≤ 1.30 m | `--status-warn` |
| `OUT OF RANGE` | ≤ 2.00 m | `--text-secondary` |
| `MILES APART` | > 2.00 m | `--text-muted` |

Two standing fighters settle at 0.99 m, so at rest you are *just* out of reach and closing is a
decision. Mock shows `CLOSING · 1.18 m`.

### Score — column 3 (360px)

One card, `--surface-1`, hairline, radius 3, `padding:16px`:

1. `score` kicker (mono 10px `.22em` `--text-accent`) + a `provisional` chip right-aligned
   (`--status-warn-quiet` fill, `--status-warn-quiet-fg` text, 9px `.18em`, radius 2px).
   **The chip is not optional.** `share` is the official scorer's own weighted number over the round
   so far and must never read as final.
2. Share row: `58%` in Display 26/700 `--status-danger`, `leading · red` centred as a hud label,
   `42%` in `--status-info`. Then a 10px two-segment bar, `gap:2px`, widths = the two shares.
   `leading` may be `null` inside the draw margin — then print `—` and colour it `--text-muted`.
3. One 12px `--text-muted` line: "The scorer's own weighted share of round 2 so far. It swings early
   and settles late." (Damage is normalised *within* the round, so an early share swings hard on one
   punch.)
4. Hairline, then the three dimensions — `damage`, `ring control`, `aggression` — each as a mono 10px
   `.14em` uppercase label over a 4px two-segment bar at 80% opacity. **The raw dimension values are
   deliberately not printed** (see Editorial decisions); the bars carry the comparison.
5. Hairline, then one quiet line: `rounds 1 — 0` left, `points 10 — 9` right, both `.torc-hud-label`.
   `points` covers **completed rounds only**.

### Seat panels — the seat row (2 × 936px, ~245px tall)

Each panel: `--surface-1`, 1px hairline, **`border-top:1.5px solid`** the seat colour, radius 3px,
`padding:14px 16px`, flex column, `gap:10px`.

**Panel header row** (26px): 9px seat square · `RED CORNER` mono 10px `.22em` in seat colour ·
handle in Display 18/600 `--text-strong` · hits landed (Display 18/700 + hud label). Then, pushed
right, the key hint: the commit key as a **filled accent chip** (mono 11px 600, `--text-on-accent`
on `--accent`, `padding:5px 10px`, radius 2px) + label `commit`, then drive keys and unstage key as
outline chips (`--surface-3` fill, 1px `--border-strong`).
When `can_commit === false` the commit chip goes `--surface-3` / `--text-muted`, `opacity:.42`,
`cursor:not-allowed`, and the label reads `commit — queue full`. **Show it on the commit affordance
itself.**

**Inner grid**: `grid-template-columns: 344px 268px minmax(0,1fr)`, gap 16, `flex:1`, `min-height:0`.

1. **Loadout** (344px). Label `loadout · 6 slots` + right note (`1 — 6`, or `locked — queue full`).
   `grid-template-columns:repeat(3,1fr)`, gap 6 → 2 rows of 3. Each slot: 1px hairline, radius 2,
   `padding:7px 8px`, `--surface-glass` fill; key in mono 12px 600; pose name 11px/1.2
   `--text-secondary`, `margin-top:3px`, `min-height:20px`; duration in mono 10px `tabular-nums`
   (`pose_seconds[slot]`, 2 dp, ` s`).
   **Staged slot**: border `1.5px --accent`, fill `--accent-quiet`, key and duration in
   `--text-accent`, name in `--text-strong`, plus a `staged` tab (8px `.18em`, `--accent` fill,
   `--text-on-accent`) pinned `top:-7px;right:-1px`.
   When the queue is full the whole grid drops to `opacity:.42`.
2. **Cost of this commit** (268px) — *the most important block on the page.*
   Border 1px `--border-accent` with `border-top:1.5px solid --accent`, fill `--accent-quiet`,
   radius 3, `padding:12px 14px`, flex column.
   - Label `cost of this commit`, mono 10px `.22em`, `--text-accent`.
   - The estimate in Display **38px / 700**, `letter-spacing:-.025em`, `tabular-nums`,
     `--text-strong`, line-height 1 — e.g. `~4.9 s` — with `to walk / and throw` beside it at 12px.
   - A 7px two-segment bar: walk share at `opacity:.45`, throw share solid.
   - Breakdown line, mono 9px `.1em` uppercase `--text-secondary`: `3.2 s walk` / `1.7 s throw`.
   - Formula: `walk_distance / approach_speed_m_s + pose_seconds[slot]`, recomputed **locally at
     60 fps** as the ghost moves. `walk_distance` is measured from `anchor`, not from `position`.
   - **Unstaged state**: plain card (`--surface-glass` fill, hairline, no accent), the figure becomes
     `—` in `--text-muted`, sub-label `nothing staged`, and any `error` message renders here as a
     TORC `Alert tone="danger"` pinned to the bottom (`margin-top:auto`).
3. **Queue** (remaining ~284px). Label `queue · N of 5 · no cancellation` + right note (`3 left` in
   `--text-accent`, or `full` in `--status-danger-quiet-fg`). **Five rows always drawn**, flex column,
   `gap:4px`, each **26px** tall, radius 2, `padding:0 10px`, three cells:
   state (mono 9px `.18em` uppercase, fixed 66px) · pose name (11px, ellipsis) · timing
   (mono 9px `tabular-nums`, right).

   | State | Border | Fill | State text colour |
   |---|---|---|---|
   | `empty` | 1px dashed `--border-default` | none | `--text-muted` |
   | `waiting` (paid for, not started) | 1px `--border-strong` | `--surface-3` | `--text-secondary` |
   | `walking` (`approaching:true`) | **1.5px** `--accent` | `--accent-quiet` | `--text-accent` |
   | `striking` (`executing`) | **1.5px** seat colour | seat `-quiet` | seat colour |

   Only the walking row prints a timing, and it prints **`arrives —`** — `commit_at`, `strike_at` and
   `end_tick` are `null` until the move reaches each stage, and `null` means *not yet*. A fighter
   walking across the ring for 4 s must not read as idle.

**Teaching bar** (39px, full width): `first fight` kicker + one 13px sentence — "Pick the pose your
move ends in, drive your ghost to the spot in the ring where it should happen, then commit — your
fighter walks there and arrives in that pose, and the walk is time you cannot take back."
A first-time player at a meetup gets no tutorial; this line is the tutorial.

---

# Screen 02 — Fight-night screen (`/screen`)

**Purpose.** A projector at a meetup, no operator, zero controls. Self-reconnecting. Degrades
gracefully when the league is absent.

**Frame.** 1920 × 1080 ink, field `scan` (lidar range rings) with `data-field-glow="lit"` and
`data-field-depth="deep"`.

**Rows:**

1. **Scoreline**, `padding:34px 48px 26px`, `grid-template-columns:1fr auto 1fr`:
   - each corner: an 18 × 60px seat-colour block, then `RED CORNER` in mono 13px `.22em` over the
     handle in Display **62px / 700** (`letter-spacing:-.025em`), then the hit tally in Display
     **84px / 700** with `hits` beneath in mono 12px. Blue mirrors right-aligned.
   - centre: the clock in Display **96px / 700**, `letter-spacing:-.03em`, `tabular-nums`, with a
     LIVE pill under it reading `round 2 of 3` at 13px.
2. **Provisional bar**, `padding:0 48px 22px`: shares in mono 20px 600 at the edges, the
   `provisional · round in progress` chip centred, then a **16px** two-segment bar, `gap:3px`.
   Under it, two mono 12px `.14em` lines: `rounds 1 — 0 · points 10 — 9` and
   `damage · ring control · aggression`.
3. **Body**, `padding:0 48px`, `grid-template-columns:1fr 560px`, gap 28, `flex:1`:
   - **Arena plate** (`.torc-hud`): `FIG. 02` + `arena · unitree g1 · separation 1.18 m · closing`,
     then the same inherited canvas at 16:9 with heavier strokes and the two handles set as 15px mono
     labels under the fighters. **No ghost, no anchor, no walk path** — a spectator is never sent
     them.
   - **Live feed** card: `live feed` kicker, then rows of `padding:9px 0` separated by hairlines —
     mono 12px `tabular-nums` timestamp (52px), an 8px square colour-coded by actor
     (seat colour / `--status-warn` for knockdown / `--accent` for bell / `--text-muted` for
     resolved), then 15px prose. Newest first. Events come from the `event` broadcast.
   - **Season table** card: `season table` kicker + `polled every 30 s`.
     `grid-template-columns:1fr 78px 62px 78px` — handle / rating / rd / w-d-l. Header cells in mono
     11px `.14em` uppercase `--text-muted` on a hairline; body rows 15px handle, mono 14px
     `tabular-nums` figures, `rd` in `--text-muted`. Source `/static/table.json`.
     **A screen with no table is still a screen** — if the fetch fails or the league is not running,
     hide the card and let the feed take the space. Never break the screen.
4. **Status ticker** (bottom, `padding:16px 48px`, `--surface-header`, top hairline): accent dot +
   `connected` in mono 13px `.22em`; divider; a 13px `.14em` line naming the venue, the time range
   (`18:30 → 21:00`) and the next bout; then the **delivered TORC wordmark** as an `<img>` 20px tall
   at `opacity:.7`, right-aligned. Use `assets/torc-wordmark-dark.svg` — never re-draw the mark.

---

# Screen 03 — Pose Studio (second server)

**Purpose.** An author builds a pose offline, mouse-driven, not part of a fight. Every rule lives on
the server; the page draws sliders and shows what came back.

**Frame.** 1920 × 1080 ink, field `schematic` (45° PCB traces), `data-field-depth="open"`.

- **Header** (64px): `Pose Studio` mono 12px `.22em` · divider · pose name in Display 19/600 ·
  an `unsaved` chip in the clay secondary (`--secondary-quiet` fill, `--secondary` text) ·
  right: `g1 · 1.3 m · 23 joints` and `localhost:8081` as hud labels.
- **Body**: `grid-template-columns: 420px minmax(0,1fr) 380px`, `flex:1`, `min-height:0`, each column
  divided by a 1px vertical hairline.
  - **Joints rail** (420px, `padding:20px`, `gap:18px`): `joints` kicker + `rad · from /api/joints`.
    Groups `left arm` (4) / `right arm` (4) / `waist` (3), each opening on a mono 11px `.18em`
    uppercase heading over a hairline with the count right-aligned; `left leg` and `right leg` are
    collapsed rows (`6 · collapsed`).
    Each slider: a label row (mono 11px `--text-secondary` name, value in `--text-strong`
    `tabular-nums`), then a 4px `--surface-3` track, radius 2, with an accent fill to the thumb and a
    12px square thumb (radius 2, `--torc-mist-100` fill, 1px `--border-strong`).
    **Out-of-limit joints** turn value, fill and thumb border `--status-warn`.
  - **Preview** (centre, `padding:20px`): `preview` kicker + `/api/render · png · redraws on drag`,
    then the TORC **`ImagePlate`** (`index="FIG. 03"`, caption `uppercut-right · /api/render ·
    640 × 640`) filling the column. With no `src` it renders an explicitly labelled empty plate —
    that is the correct placeholder; swap in the server PNG as `src`. Footer row: `fig. 03 ·
    uppercut-right · frame 13 / 13` and `camera · fixed · 3/4 front`.
  - **Right rail** (380px, `padding:20px`, `gap:20px`):
    - `horizon` + `6 → 16 tokens`; the token count in Display 36/700 with `tokens` beside it and the
      seconds in mono 16px `--text-accent` right-aligned; then a 16-cell strip (each `flex:1`, 20px
      tall, `gap:3px`) with filled cells `--accent` at `opacity:.9` and the rest `--surface-3`;
      footnote `tokens × 4 / 30 = seconds`.
    - `hand reach` + `/api/reach`: two rows, mono 12px, left/right, the active side in
      `--text-accent`.
    - `status`: a TORC **`Alert tone="warn"`** carrying the server's own sentence, e.g. "right elbow
      2.09 exceeds limit 1.92 — from /api/check".
    - Actions, pinned bottom (`margin-top:auto`), TORC **`Button`** mounts, `block`, `mono`, 40px:
      `Check` (secondary), `Save pose` (primary, **disabled until check passes**), `Reset` (ghost).
      Footnote centred: `save is blocked until check passes`.

---

# Screen 04 — Player-view states

Six cards, `grid-template-columns:repeat(3,minmax(0,1fr))` across 1920, gap 20, each 300px tall,
ink ground with a field, hairline, radius 3, `padding:22px`, opening on a `state 0N` hud label.

1. **Connecting** — TORC `Spinner` (30px, the Lucide `loader-circle` on an 1100 ms linear spin),
   `Connecting` in Display 22/600, `ws://localhost:8080 · attempt 1` in mono 11px. Footer: "No state
   received yet. The ring is drawn, empty."
2. **Waiting for the blue seat** — two half-cards: the seated one carries a 1.5px seat-colour top
   border and `seated` in `--text-accent`; the open one is 1px **dashed** `--border-default` with
   `open` / `no join yet` in `--text-muted`. Footer: "One of two seats taken. The clock does not
   start."
3. **Disconnected** — red dot + `disconnected` mono 11px `.22em`, `Connection lost` in Display 22/600,
   then "Last state at tick 412. Retrying in 2 s. The queue on the server is unaffected." and a 4px
   progress track filled `--status-danger`. Footer: "Readouts hold their last value, dimmed. Nothing
   is faked."
4. **Between rounds** — field `strata`. `round 1 · ended by bell`, the two point totals in Display
   44/700 in seat colours with `round to carlo` in `--text-accent`, hairline, then "Queues cleared.
   Fighters reset to 1.20 m separation." Footer: `round 2 begins in 8 s`.
5. **Knockdown** — field `scan`, depth `hero`. A mini ring diagram at `opacity:.5` with the downed
   fighter drawn as a flattened ellipse, and a **banner across the middle**: `--surface-scrim` fill +
   `backdrop-filter:blur(6px)`, `border-top:1.5px solid --status-warn`, carrying `event · knockdown`,
   `Nadia is down` in Display 30/700, and `get-up window · 8 s · torso 0.31 m`. Footer: "Banner sits
   over the ring, never on the fighters." Same treatment for round-end and match-end.
6. **Remote opponent** — the private-queue variant. Only the executing entry renders normally; the
   other four cells are 1px hairline with a **45° hatch** fill
   (`repeating-linear-gradient(45deg, var(--surface-glass) 0 5px, transparent 5px 10px)`) labelled
   `private`. Footer: `no loadout · no ghost · no staged slot`.
   A queued-but-unstarted commit is withheld by design — showing it would hand the opponent a
   readable list of your next four moves, which is exactly the risk the queue is meant to be. Draw it
   as **withheld**, not as unknown.

---

## Interactions & behaviour

**Keys, hotseat, two seats on one keyboard:**

| | pick the pose | drive the ghost | commit | unstage |
|---|---|---|---|---|
| **red** | `1`–`6` | `W A S D` (held) | `SPACE` | `Q` |
| **blue** | `U I O J K L` | `↑ ↓ ← →` (held) | `ENTER` | `P` |

Drive keys move the ghost in **screen** directions — the camera is fixed, so nobody has to think
about whose forward is whose. The ghost's heading is **not a control**: it always faces the opponent
from wherever it stands.

**The ghost is entirely client-side.** It is posed in the browser from `poses` (joint angles) run
through the kinematic tree in `/scene.json`, drawn at `anchor + local offset`, and **never
transmitted while being aimed**. The server learns a placement only on commit, at which point the
offset resets to zero. So an untouched ghost + commit means *"do this where I will be"*. Aiming costs
zero round trips and must feel like direct manipulation.

**Loops.** Ghost transform + cost readout: `requestAnimationFrame`, 60 fps, no network. Queue, score,
range, clock: on `state` message, 30 Hz. Never animate anything that assumes 60 Hz truth.

**Motion.** `--ease-servo` `cubic-bezier(.33,0,.15,1)`; 140 ms hover/press, 220 ms enter/exit,
1100 ms for a full rotation. No overshoot, no bounce, no scale-in. `prefers-reduced-motion` kills
everything. The only animation in these mocks is the live dot (1 → .45 over 2 s) and the spinner.

**States.** Hover: surfaces lighten one step to `--surface-glass`, borders to strong or accent.
Press: `filter:brightness(.92)` — no nudge, no scale. Focus: 2px accent outline at 2px offset, never
removed. Disabled: `opacity:.42`, `cursor:not-allowed`.
None of these apply to keyboard-only fight controls, which have no hover target at all.

**Errors.** An `error` message renders in the seat's own cost block as an `Alert tone="danger"`
carrying the server's sentence verbatim. It must not steal focus, must not block the fight, and must
not be a modal.

**Responsive.** Single-column seats below 720 px. The 1920 × 1080 frame is the design target; scale
the canvas and let the rails stack.

## State → DOM mapping

See **`state-to-dom.md`** in this folder for the field-by-field table: every readout, the exact
`state` field it comes from, its formatting rule, and its `null` behaviour.

## Design tokens

All values are TORC design-system tokens. The full definitions ship in **`tokens/`** — link
`semantic.css` and design against the aliases, never the raw ramp.

**Ground and surfaces (ink — the operator ground):**

| Token | Value | Use |
|---|---|---|
| `--bg-page` | `#08130F` | the frame |
| `--bg-page-deep` | `#050D0A` | canvas well, label backing rects |
| `--surface-1` | `#101E18` | every card and panel |
| `--surface-2` | `#16281F` | raised, hover |
| `--surface-3` | `#1E3529` | inset, slider track, queue `waiting` fill, key chips |
| `--surface-glass` | `rgba(234,242,237,.04)` | loadout slots, teaching bar |
| `--surface-header` | `rgba(8,19,15,.88)` + `blur(14px)` | header, ticker |
| `--surface-scrim` | `rgba(8,19,15,.60)` | the knockdown banner |

**Text:** `--text-strong` `#FFFFFF` · `--text-body` `#EAF2ED` · `--text-secondary` `#B7C9BE` ·
`--text-muted` `#6E8C7E` · `--text-accent` `#4FD1A0` · `--text-on-accent` `#08130F`.

**Borders:** `--border-subtle` `rgba(234,242,237,.08)` (almost everything) ·
`--border-default` `#1E3529` (dashed empty cells) · `--border-strong` `#2C4A3C` ·
`--border-accent` `rgba(79,209,160,.40)` · `--border-grid` `rgba(234,242,237,.06)` ·
`--border-tick` `.22` · `--border-hud` `.28`.

**Accent and signals:**

| Token | Value | Meaning here |
|---|---|---|
| `--accent` | `#4FD1A0` | the ghost, the walk, the commit cost, the commit key, LIVE |
| `--accent-quiet` | `rgba(79,209,160,.12)` | staged slot, walking cell, cost block |
| `--status-danger` | `#FF5247` | **red seat identity** |
| `--status-info` | `#3FA9F5` | **blue seat identity** |
| `--status-warn` | `#FFB020` | `CLOSING`, knockdown, joint out of limit |
| `--status-*-quiet` / `-quiet-fg` | see `semantic.css` | chips and rows, where a label must not out-punch a real button |
| `--secondary` (clay) | `#DDAE86` | the `unsaved` chip only — never a status, never an action |

Seat red and blue are the TORC signal LEDs, deliberately far from the accent hue so seat identity
never reads as brand and brand never reads as status.

**Type — Red Hat, one family, three cuts** (self-hosted woff2, `tokens/fonts.css`):

| Cut | Where |
|---|---|
| **Red Hat Display** 500/600/700 | clock, handles, big figures, the cost estimate, card titles. Track `-0.025em` at 38px+, `-0.015em` below |
| **Red Hat Text** 400/500/600 | prose, pose names, teaching copy. 12–15px in this design |
| **Red Hat Mono** 400/500/600 | the telemetry voice: every kicker, label, badge, timing, metre and second |

All mono is set with `font-variant-numeric: tabular-nums` so columns of figures align.
Two mono label recipes carry most of the UI: `.torc-kicker` (10px, `.22em`, uppercase, accent) and
`.torc-hud-label` (10px, `.22em`, uppercase, `--text-muted`, tabular).

**Space.** 4px base. Frame padding 16 (48 on the projector). Card padding 16, seat panels `14px 16px`,
gaps 16 between cards, 6 inside grids, 4 between queue rows. Control heights 26 (queue row), 40
(button), 60/64 (header).

**Shape.** **Nothing is soft.** 2px on controls and slots, **3px canonical** for cards and plates,
999px on the LIVE pill only. No shadows on cards — hairlines do the work. `--glow-dot` on the live
indicator is the only glow in the design.

**Annotation — the HUD layer.** `.torc-hud` (machined corner brackets on the arena plate),
`.torc-hud-label`, `.torc-field` grounds (`survey` for content, `scan` for the projector and CTA,
`schematic` for technical surfaces, `strata` for number bands), and `data-instrument="quiet|full|off"`
to scale how loud the grid, ticks and brackets run. The mock exposes `instrument` as its one tweak.

## Assets

| Asset | Source | Use |
|---|---|---|
| `assets/torc-wordmark-dark.svg` | TORC design system, delivered brand asset | attribution on the projector footer |
| `assets/torc-icon-dark.svg` | same | favicon / compact header, if needed |
| Red Hat Display / Text / Mono | `tokens/fonts.css`, self-hosted woff2 (latin + latin-ext) | all type |
| Lucide icons (82 vendored glyphs) | TORC design system `assets/icons/` | every icon. **Never hand-draw an icon**; if a glyph is missing, add the Lucide SVG |
| Field background art | TORC `assets/backgrounds/` | the `.torc-field` grounds |

The brand mark is a **delivered asset**: never redraw, re-trace, recolour or approximate it, and
never set it beside another product name as if it were that product's logo. There is no photography
yet — `ImagePlate` renders a labelled empty plate, which is correct until real shots exist.

## Editorial decisions worth preserving

The player view was deliberately pared back after review — it read as too many numbers. What was cut,
and why, so it does not creep back:

- Header telemetry reduced to `ping` alone. Tick count and the 50/30/60 Hz rates are implementation
  facts, not decisions.
- The torso-height / standing card was removed entirely. `torso_height_m` still matters for the
  knockdown banner; it does not need a permanent readout.
- The three score dimensions kept their bars and labels but **lost their numeric values** — the
  comparison is the point, not the figure.
- `points` and `rounds_won` collapsed from two tiles to one quiet line.
- Four of five queue rows lost their timing text; only `arrives —` survives, because it is the only
  one that says something (`null` = not yet).
- Inside the arena, the separation figure and the ring dimensions were dropped — both are already
  stated elsewhere on the page.

What stayed is the set of numbers that decide something: the commit cost with its walk/throw split,
the separation in metres beside its word, the two shares, hits landed, and per-slot pose durations.

## Files in this bundle

| File | What it is |
|---|---|
| `design-reference-standalone.html` | **Start here.** All four frames, self-contained, opens offline in any browser. Pan and zoom. |
| `state-to-dom.md` | Field-by-field mapping from the server contract to every readout. |
| `source/OpenRoboxing Frontend Mockup.dc.html` | The design source. Needs `source/support.js` beside it and the token sheets to render. |
| `source/support.js` | Runtime for the source file. Not part of the implementation. |
| `tokens/*.css` | The TORC token sheets — the authoritative values for everything above. |
| `assets/*.svg` | The delivered TORC brand marks. |
