/* The two drawings the three.js renderer does not do (spec/ui-design-guide/, screen 01).
 *
 *  - `drawArena` annotates the arena canvas: the dashed walk path from the anchor to the ghost, the
 *    anchor marker, the contact ring around each fighter, the walk distance, the ghost callout. It
 *    is SVG over the canvas rather than geometry in the scene, so the renderer stays exactly what it
 *    was — the host's meshes, the host's transforms — and the annotation layer can be redrawn at
 *    60 fps without touching it. Every point goes through `ring.project`, i.e. the renderer's own
 *    camera, so an annotation cannot drift from the thing it annotates.
 *
 *  - `drawMap` draws the top view. The camera is fixed and perspective hides depth, so where the
 *    ghost stands is the one thing the arena cannot show; the map exists for that and nothing else.
 *
 * Neither function decides anything. They are given what the host sent plus the local ghost, and
 * they draw it.
 */

'use strict';

/* From spec/scoring.md, which measures it rather than choosing it: the G1's hand reaches 0.38 m
   forward of its own pelvis, so two fighters exchange at a pelvis separation up to ~0.76 m. */
const CONTACT_RANGE_M = 0.80;

/* The map is drawn 100 units to the metre, so a 4.90 m ring is 490 units square. */
const UNITS_PER_M = 100;

const SEAT_COLOUR = { red: 'var(--status-danger)', blue: 'var(--status-info)' };

function escapeText(value) {
  return String(value).replace(/[<>&]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' })[c]);
}

/* ---- the arena overlay ---------------------------------------------------------------------------- */

function contactRing(ring, at, colour) {
  /* Sampled rather than drawn as an ellipse: a circle on the ground under a perspective camera is
     an ellipse only if the camera is level, and the camera's tilt is `frameRing`'s business, not
     this file's. Sixteen points through the same projection is right whatever it does. */
  const points = [];
  for (let i = 0; i <= 16; i += 1) {
    const angle = (i / 16) * Math.PI * 2;
    const point = ring.project(
      at.x + Math.cos(angle) * CONTACT_RANGE_M,
      at.y + Math.sin(angle) * CONTACT_RANGE_M,
      0.02,
    );
    points.push(`${point.x.toFixed(1)},${point.y.toFixed(1)}`);
  }
  return `<polyline points="${points.join(' ')}" fill="none" stroke="${colour}" stroke-width="1"
    stroke-dasharray="5 6" opacity=".5"></polyline>`;
}

/* What a fighter is doing, in the words the queue uses. Sourced only from the entries the host was
   willing to send for that seat — of an opponent that is the executing commit and nothing else, so
   this can say "walking" without ever leaking what is queued behind it. */
function fighterState(seat) {
  const executing = (seat.queue || []).find((entry) => entry.executing);
  if (seat.down) return 'down';
  if (!executing) return 'idle';
  return executing.approaching ? 'walking' : 'striking';
}

/**
 * @param svg   the <svg> laid over the canvas; its viewBox is kept at the canvas's pixel box
 * @param ring  the Ring, for `project`
 * @param view  {seats, plans}: `seats` is the host's seat map, `plans` is one entry per seat whose
 *              ghost is currently placed — {seat, anchor, ghost, walkM}
 */
export function drawArena(svg, ring, view) {
  const box = svg.getBoundingClientRect();
  const width = Math.round(box.width);
  const height = Math.round(box.height);
  if (!width || !height) return;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

  const parts = [`<defs><marker id="ov-arrow" markerWidth="7" markerHeight="7" refX="5.5" refY="3"
    orient="auto"><path d="M0 0 6 3 0 6z" fill="var(--accent)"></path></marker></defs>`];

  for (const [name, seat] of Object.entries(view.seats || {})) {
    if (!seat.position) continue;
    const colour = SEAT_COLOUR[name] || 'var(--text-muted)';
    parts.push(contactRing(ring, seat.position, colour));

    /* The fighter's own label, under its feet. Never a HUD *on* the fighter: the windup stays the
       only cue for what is coming, and this says only what has already happened.
       Red hangs its label to the left and blue to the right, so two fighters in a clinch — which is
       most of a round — do not print one label on top of the other. */
    const foot = ring.project(seat.position.x, seat.position.y, 0);
    const left = name === 'red';
    parts.push(
      `<text x="${(foot.x + (left ? -10 : 10)).toFixed(1)}" y="${(foot.y + 20).toFixed(1)}"
        text-anchor="${left ? 'end' : 'start'}" fill="${colour}" font-family="var(--font-mono)"
        font-size="11" letter-spacing="1.4"
       >${escapeText(name.toUpperCase())} · ${fighterState(seat)}</text>`,
    );
  }

  for (const plan of view.plans || []) {
    const colour = SEAT_COLOUR[plan.seat] || 'var(--accent)';
    const from = ring.project(plan.anchor.x, plan.anchor.y, 0.02);
    const to = ring.project(plan.ghost.x, plan.ghost.y, 0.02);

    parts.push(
      `<path d="M${from.x.toFixed(1)} ${from.y.toFixed(1)} L${to.x.toFixed(1)} ${to.y.toFixed(1)}"
        stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="7 6" marker-end="url(#ov-arrow)"
        opacity=".85" fill="none"></path>`,
      `<rect x="${(from.x - 7).toFixed(1)}" y="${(from.y - 7).toFixed(1)}" width="14" height="14"
        fill="none" stroke="${colour}" stroke-width="1.5"
        transform="rotate(45 ${from.x.toFixed(1)} ${from.y.toFixed(1)})"></rect>`,
    );

    /* How far apart the anchor and the ghost ended up *on screen*, which is what decides whether
       there is room for an annotation between them — the metres do not, since a metre across the
       ring is a handful of pixels and a metre at the camera is a hundred. */
    const spread = Math.hypot(to.x - from.x, to.y - from.y);

    /* The walk, in metres, on an opaque backing so it stays readable over a fighter. Sat a third of
       the way along the path — the anchor end, away from the ghost — and pushed clear of the line
       itself along the normal, so it never sits on the path it is measuring. Suppressed when the
       ghost is on its anchor: "0.00 m walk" is noise. */
    if (spread > 34) {
      const normalX = -(to.y - from.y) / spread;
      const normalY = (to.x - from.x) / spread;
      const side = normalY > 0 ? -1 : 1;          // always push the label up the screen
      const labelX = from.x + (to.x - from.x) * 0.38 + normalX * side * 24;
      const labelY = from.y + (to.y - from.y) * 0.38 + normalY * side * 24;
      parts.push(
        `<rect x="${(labelX - 44).toFixed(1)}" y="${(labelY - 13).toFixed(1)}" width="88"
          height="18" rx="2" fill="var(--bg-page-deep)"></rect>`,
        `<text x="${labelX.toFixed(1)}" y="${labelY.toFixed(1)}" text-anchor="middle"
          fill="var(--text-accent)" font-family="var(--font-mono)" font-size="11" letter-spacing="1"
         >${plan.walkM.toFixed(2)} m walk</text>`,
      );
    }

    /* The ghost callout: three lines naming what the ghost is. It is a teaching label, so it gives
       way to the walk figure whenever the two are anywhere near each other — the figure is the
       number the player is deciding on, and the callout is the thing they know by round two. */
    if (spread <= 120) continue;
    const right = to.x < width - 190;
    const lineX = right ? to.x + 34 : to.x - 34;
    const textX = right ? to.x + 42 : to.x - 42;
    const anchorAttr = right ? 'start' : 'end';
    parts.push(
      `<path d="M${(right ? to.x + 14 : to.x - 14).toFixed(1)} ${(to.y - 34).toFixed(1)}
        L${lineX.toFixed(1)} ${(to.y - 34).toFixed(1)}" stroke="var(--accent)" stroke-width="1"
        opacity=".5"></path>`,
      `<text x="${textX.toFixed(1)}" y="${(to.y - 40).toFixed(1)}" text-anchor="${anchorAttr}"
        fill="var(--text-accent)" font-family="var(--font-mono)" font-size="11"
        letter-spacing="1.4">ghost</text>`,
      `<text x="${textX.toFixed(1)}" y="${(to.y - 25).toFixed(1)}" text-anchor="${anchorAttr}"
        fill="var(--text-muted)" font-family="var(--font-mono)" font-size="10"
        letter-spacing="1.1">end pose</text>`,
      `<text x="${textX.toFixed(1)}" y="${(to.y - 10).toFixed(1)}" text-anchor="${anchorAttr}"
        fill="var(--text-muted)" font-family="var(--font-mono)" font-size="10"
        letter-spacing="1.1">faces the opponent</text>`,
    );
  }

  svg.innerHTML = parts.join('');
}

/* The projector's version of the same layer, and the difference is the point: **no ghost, no
 * anchor, no walk path.** A spectator is never sent them, and anybody in the room can read a
 * projector. What is left is what is already visible in the ring — who is standing where, and how
 * far apart they are.
 */
export function drawSpectatorArena(svg, ring, seats, separation) {
  const box = svg.getBoundingClientRect();
  const width = Math.round(box.width);
  const height = Math.round(box.height);
  if (!width || !height) return;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

  const parts = [];
  const feet = [];

  for (const [name, seat] of Object.entries(seats || {})) {
    if (!seat.position) continue;
    const colour = SEAT_COLOUR[name] || 'var(--text-muted)';
    parts.push(contactRing(ring, seat.position, colour));

    const foot = ring.project(seat.position.x, seat.position.y, 0);
    feet.push(foot);
    parts.push(
      `<text x="${foot.x.toFixed(1)}" y="${(foot.y + 26).toFixed(1)}" text-anchor="middle"
        fill="${colour}" font-family="var(--font-mono)" font-size="15" letter-spacing="2"
       >${escapeText((seat.handle || name).toUpperCase())}</text>`,
    );
  }

  if (feet.length === 2 && separation !== null && separation !== undefined) {
    const midX = (feet[0].x + feet[1].x) / 2;
    const midY = (feet[0].y + feet[1].y) / 2;
    parts.push(
      `<path d="M${feet[0].x.toFixed(1)} ${feet[0].y.toFixed(1)}
        L${feet[1].x.toFixed(1)} ${feet[1].y.toFixed(1)}" stroke="var(--text-muted)"
        stroke-width="1.5" opacity=".7" fill="none"></path>`,
      `<rect x="${(midX - 37).toFixed(1)}" y="${(midY - 15).toFixed(1)}" width="74" height="20"
        rx="2" fill="var(--bg-page-deep)"></rect>`,
      `<text x="${midX.toFixed(1)}" y="${midY.toFixed(1)}" text-anchor="middle"
        fill="var(--text-secondary)" font-family="var(--font-mono)" font-size="14"
        letter-spacing="1">${separation.toFixed(2)} m</text>`,
    );
  }

  svg.innerHTML = parts.join('');
}

/* ---- the top view ----------------------------------------------------------------------------------
 * MuJoCo world (x, y) with x to the right and y up the page, which is the same orientation the fixed
 * camera looks along — so a player reading the map does not have to rotate it in their head.
 */
export function drawMap(group, view, ringHalf) {
  const toMap = (x, y) => ({
    x: (ringHalf + x) * UNITS_PER_M,
    y: (ringHalf - y) * UNITS_PER_M,
  });

  const parts = [];
  const positions = [];

  for (const [name, seat] of Object.entries(view.seats || {})) {
    if (!seat.position) continue;
    const at = toMap(seat.position.x, seat.position.y);
    positions.push(at);
    parts.push(
      `<circle cx="${at.x.toFixed(1)}" cy="${at.y.toFixed(1)}" r="${CONTACT_RANGE_M * UNITS_PER_M}"
        fill="none" stroke="${SEAT_COLOUR[name]}" stroke-width="1.5" stroke-dasharray="6 7"
        opacity=".5"></circle>`,
    );
  }

  if (positions.length === 2) {
    parts.push(
      `<path d="M${positions[0].x.toFixed(1)} ${positions[0].y.toFixed(1)}
        L${positions[1].x.toFixed(1)} ${positions[1].y.toFixed(1)}" stroke="var(--text-muted)"
        stroke-width="1.5" opacity=".7"></path>`,
    );
  }

  for (const plan of view.plans || []) {
    const from = toMap(plan.anchor.x, plan.anchor.y);
    const to = toMap(plan.ghost.x, plan.ghost.y);
    parts.push(
      `<path d="M${from.x.toFixed(1)} ${from.y.toFixed(1)} L${to.x.toFixed(1)} ${to.y.toFixed(1)}"
        stroke="var(--accent)" stroke-width="2.5" stroke-dasharray="9 7"
        marker-end="url(#mmarrow)" fill="none"></path>`,
      `<rect x="${(from.x - 8).toFixed(1)}" y="${(from.y - 8).toFixed(1)}" width="16" height="16"
        fill="none" stroke="${SEAT_COLOUR[plan.seat]}" stroke-width="2"
        transform="rotate(45 ${from.x.toFixed(1)} ${from.y.toFixed(1)})"></rect>`,
      `<circle cx="${to.x.toFixed(1)}" cy="${to.y.toFixed(1)}" r="17" fill="var(--accent-quiet)"
        stroke="var(--accent)" stroke-width="2.5" stroke-dasharray="6 5"></circle>`,
    );
  }

  /* The fighters go on last so a dot is never hidden under a path. */
  for (const [name, seat] of Object.entries(view.seats || {})) {
    if (!seat.position) continue;
    const at = toMap(seat.position.x, seat.position.y);
    parts.push(
      `<circle cx="${at.x.toFixed(1)}" cy="${at.y.toFixed(1)}" r="14"
        fill="${SEAT_COLOUR[name]}"></circle>`,
    );
  }

  group.innerHTML = parts.join('');
}
