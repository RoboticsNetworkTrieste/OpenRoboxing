/* OpenRoboxing player view — spec/protocol.md 0.5, spec/intent.md 1.1, spec/ui-design-guide/.
 *
 * The player's loop: pick a pose (the move's *last frame*), drive a shadow of your fighter to where
 * it should happen, commit. The generator walks you there and arrives in that pose. Up to five
 * commits stack up and run back to back; run out and the fighter stops.
 *
 * What the client owns
 * --------------------
 * The shadow, and only the shadow. It is drawn here so aiming costs no round trip, and the host
 * never sees it — the placement is transmitted once, on commit. Everything else is still the host's:
 * this file forwards keys and draws what arrives, and it never decides whether a commit is legal
 * (`can_commit` comes from the host, so when the rule changes this file does not).
 *
 * Two loops, deliberately different (spec/ui-design-guide/ §Interactions):
 *   - the ghost and the cost estimate redraw at 60 fps, locally, because they must feel like direct
 *     manipulation;
 *   - everything the host owns redraws on the 30 Hz `state` message, and nothing here animates as
 *     though it had 60 Hz truth.
 *
 * Hotseat: two seats, one page, one socket each. The host cannot tell this from two machines. A
 * seat this page cannot take — an agent is sitting in it — becomes a *remote* panel, showing only
 * what the host will show an opponent.
 */

'use strict';

import { Ring } from './ring.js';
import { drawArena, drawMap } from './overlay.js';

/* Each seat gets one region of the keyboard: a row picks the pose, a cluster drives the shadow, one
   big key commits. Movement keys are **held**, and they move the ghost in *screen* directions —
   the camera is fixed, so up-the-screen is a fixed world direction and nobody has to think about
   whose forward is whose. */
const SEATS = {
  red: {
    keys: ['1', '2', '3', '4', '5', '6'],
    keyRange: '1 — 6',
    commit: ' ',
    clear: 'q',
    drive: { w: [0, 1], s: [0, -1], a: [-1, 0], d: [1, 0] },
  },
  blue: {
    keys: ['u', 'i', 'o', 'j', 'k', 'l'],
    keyRange: 'U — L',
    commit: 'Enter',
    clear: 'p',
    drive: { ArrowUp: [0, 1], ArrowDown: [0, -1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] },
  },
};

/* How fast the ghost travels while a key is held, m/s. Fast enough to cross the ring in a couple of
   seconds, slow enough to place a punch. A UI number, not a physical one — it is how quickly you can
   *point*, not how quickly the fighter moves. */
const SHADOW_SPEED_M_S = 1.9;

/* Range bands, from spec/scoring.md's measured CONTACT_RANGE_M (0.80 m). A player cannot manage
   distance they cannot see, and "1.42 m" means nothing mid-fight — a word does. */
const RANGE_BANDS = [
  [0.80, 'IN RANGE', 'close'],
  [1.30, 'CLOSING', 'near'],
  [2.00, 'OUT OF RANGE', 'out'],
  [Infinity, 'MILES APART', 'miles'],
];

const TICK_HZ = 50;

/* How many queue cells are drawn. Presentational only: whether a commit is *accepted* is
   `can_commit`, which is the host's, and this number never gates anything. */
const QUEUE_CELLS = 5;

/* How long the host's own rejection sentence stays on screen. */
const ERROR_MS = 5000;

const state = {};                  // seat -> per-seat client state
const held = { red: new Set(), blue: new Set() };
const lastState = {};              // seat -> the last `state` message that seat's own socket saw

const canvas = document.getElementById('ring');
const overlay = document.getElementById('arena-overlay');
const banner = document.getElementById('banner');
const statusLine = document.getElementById('status');
const topbar = document.querySelector('.topbar');
const ring = new Ring(canvas);

let standHeight = 0.793;
let ringHalf = 2.45;
let getUpSeconds = 8;
let rounds = 3;

/* ---- the shadow, which is ours -------------------------------------------------------------------
 * Drawn at `anchor + offset`. The anchor is where the queue leaves the fighter, so it is stable
 * while a move plays out; the offset is what the player edits, and it resets on commit — which makes
 * an untouched shadow mean "do this where I will be".
 */
function shadowPosition(seat) {
  const entry = state[seat];
  const anchor = entry?.anchor;
  if (!anchor) return null;
  const x = Math.max(-ringHalf, Math.min(ringHalf, anchor.x + entry.offset.x));
  const y = Math.max(-ringHalf, Math.min(ringHalf, anchor.y + entry.offset.y));
  return { x, y };
}

/* Anywhere in the ring is reachable (spec/intent.md 1.1) — the fighter walks until it gets there.
   What a distant placement costs is *time*, and time is what a player cannot take back, so the ghost
   shows how long this commit will tie the queue up for: the walk at the measured approach speed,
   plus the pose's own length. Measured from the **anchor**, not from where the fighter is standing:
   the next move starts from the end of the queue. An estimate, and labelled as one. */
function commitCost(seat, at) {
  const entry = state[seat];
  if (!entry?.anchor || !entry.staged) return null;
  const pose = entry.poseSeconds?.[entry.staged];
  const speed = entry.approachSpeed;
  if (!pose || !speed) return null;
  const metres = Math.hypot(at.x - entry.anchor.x, at.y - entry.anchor.y);
  return { metres, walk: metres / speed, throw: pose, total: metres / speed + pose };
}

/* Heading is not a control: the ghost always faces the opponent from wherever it stands. One fewer
   key, and it is what a player wants every time bar none. */
function shadowHeading(seat, at) {
  const opponent = state[seat]?.opponentPosition;
  if (!opponent) return state[seat]?.anchor?.heading ?? 0;
  return Math.atan2(opponent.y - at.y, opponent.x - at.x);
}

/* One 60 fps pass: move every ghost, redraw its cost, and hand the annotation layer what it needs.
   Nothing here talks to the network. */
function driveShadows(dt) {
  const plans = [];

  for (const seat of Object.keys(SEATS)) {
    const entry = state[seat];
    if (!entry) continue;

    let dx = 0;
    let dy = 0;
    for (const key of held[seat]) {
      const step = SEATS[seat].drive[key];
      if (step) { dx += step[0]; dy += step[1]; }
    }
    if (dx || dy) {
      const scale = (SHADOW_SPEED_M_S * dt) / Math.hypot(dx, dy);
      entry.offset.x += dx * scale;
      entry.offset.y += dy * scale;
    }

    const at = shadowPosition(seat);
    const angles = entry.poses?.[entry.staged];
    if (!at || !entry.staged || !angles) { ring.hideShadow(seat); drawCost(seat, null); continue; }

    ring.showShadow(seat, at.x, at.y, shadowHeading(seat, at), angles, standHeight);
    const cost = commitCost(seat, at);
    drawCost(seat, cost);
    if (cost) plans.push({ seat, anchor: entry.anchor, ghost: at, walkM: cost.metres });
  }

  const view = { seats: visibleSeats(), plans };
  drawArena(overlay, ring, view);
  drawMap(document.getElementById('map-live'), view, ringHalf);
}

/* Every seat as this page is allowed to see it: its own from its own socket, anybody else's from
   whichever socket the host answered — which is already filtered, so there is nothing to hide here
   that the host did not hide first. */
function visibleSeats() {
  const seats = {};
  for (const seat of Object.keys(SEATS)) {
    const own = lastState[seat]?.seats?.[seat];
    const seen = Object.values(lastState).map((message) => message.seats?.[seat]).find(Boolean);
    const value = own || seen;
    if (value) seats[seat] = value;
  }
  return seats;
}

/* ---- the cost of this commit -----------------------------------------------------------------------
   The most important block on the page: it says what a placement will tie the fighter up for, while
   the player can still change their mind. */
function drawCost(seat, cost) {
  const block = document.getElementById(`cost-${seat}`);
  const value = document.getElementById(`cost-value-${seat}`);
  const sub = document.getElementById(`cost-sub-${seat}`);
  const bar = document.getElementById(`cost-bar-${seat}`);
  const split = document.getElementById(`cost-split-${seat}`);
  if (!block) return;

  if (!cost) {
    block.classList.add('idle');
    value.textContent = '—';
    sub.innerHTML = 'nothing<br>staged';
    bar.hidden = true;
    split.hidden = true;
    return;
  }

  block.classList.remove('idle');
  value.textContent = `~${cost.total.toFixed(1)} s`;
  sub.innerHTML = 'to walk<br>and throw';
  bar.hidden = false;
  split.hidden = false;

  const walkShare = (cost.walk / cost.total) * 100;
  document.getElementById(`cost-walk-${seat}`).style.width = `${walkShare.toFixed(1)}%`;
  document.getElementById(`cost-throw-${seat}`).style.width = `${(100 - walkShare).toFixed(1)}%`;
  document.getElementById(`cost-walk-text-${seat}`).textContent = `${cost.walk.toFixed(1)} s walk`;
  document.getElementById(`cost-throw-text-${seat}`).textContent = `${cost.throw.toFixed(1)} s throw`;
}

/* ---- the loadout bar ------------------------------------------------------------------------------ */
function buildLoadout(seat) {
  const bar = document.getElementById(`loadout-${seat}`);
  const config = SEATS[seat];
  const entry = state[seat];
  bar.innerHTML = '';

  entry.slots.forEach((slot, index) => {
    const cell = document.createElement('div');
    cell.className = 'slot';
    cell.id = `slot-${seat}-${slot}`;

    const key = document.createElement('span');
    key.className = 'key';
    key.textContent = config.keys[index].toUpperCase();

    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = entry.loadout[slot];

    /* The pose's own length — the strike, once the fighter has walked there. The walk is not in it,
       because the walk depends on where the ghost is standing; the cost block adds that live. */
    const duration = document.createElement('span');
    duration.className = 'dur';
    const seconds = entry.poseSeconds?.[slot];
    duration.textContent = seconds ? `${seconds.toFixed(2)} s` : '';

    cell.append(key, name, duration);
    bar.append(cell);
  });
}

/* A seat this page does not hold. The host sends no loadout for somebody else's fighter, and this
   draws that as withheld rather than as empty. */
function buildRemoteLoadout(seat) {
  const bar = document.getElementById(`loadout-${seat}`);
  if (bar.dataset.remote === 'yes') return;
  bar.dataset.remote = 'yes';
  bar.innerHTML = '';
  for (let i = 0; i < 6; i += 1) {
    const cell = document.createElement('div');
    cell.className = 'slot private';
    cell.innerHTML = '<span class="key">·</span><span class="name">private</span><span class="dur"></span>';
    bar.append(cell);
  }
  document.getElementById(`loadout-note-${seat}`).textContent = 'no loadout · no ghost';
}

/* ---- applying host state ----------------------------------------------------------------------------
   Composed per recipient by the host, so a seat's staging and its queued-but-unstarted commits never
   reach the other socket. This file re-reads that split rather than re-deriving it. */
function applyState(message, viewer) {
  lastState[viewer] = message;
  const entry = state[viewer];
  const own = message.seats?.[viewer];
  if (entry && own) {
    entry.anchor = own.anchor;
    const other = Object.keys(message.seats).find((name) => name !== viewer);
    if (other) entry.opponentPosition = message.seats[other].position;
  }
  render();
}

function render() {
  const primary = lastState.red || lastState.blue;
  if (!primary) return;

  const seconds = Math.max(0, Math.ceil(primary.clock_ticks / TICK_HZ));
  document.getElementById('clock').textContent =
    `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
  document.getElementById('round').textContent = `round ${primary.round} / ${rounds}`;
  topbar.classList.toggle('low', primary.clock_ticks < 10 * TICK_HZ);

  if (primary.separation_m !== null && primary.separation_m !== undefined) {
    applyRange(primary.separation_m);
  }
  if (primary.score) applyScore(primary.score);

  for (const seat of Object.keys(SEATS)) {
    /* Own socket first: it is the only view that carries the full queue. Falling back to another
       socket's view is not the same thing, and the two must not be confused — see `drawSeat`. */
    const full = Boolean(lastState[seat]);
    const source = full
      ? lastState[seat].seats?.[seat]
      : Object.values(lastState).map((m) => m.seats?.[seat]).find(Boolean);
    if (source) drawSeat(seat, source, { full, remote: Boolean(state[seat]?.remote) });
  }

  drawBanner(primary);
}

/* One seat panel.
 *
 * `full` says this came off that seat's *own* socket, which is the only view carrying the whole
 * queue. `remote` says the seat belongs to somebody else — an agent, or a second machine — so the
 * host will never send more than the executing commit for it.
 *
 * The two are not the same and must not be drawn the same. A seat whose first `state` has not
 * arrived yet is *not yet*, and drawing it as withheld would claim the host refused something it
 * has simply not sent — which is the same "no fabricated certainty" rule that keeps a null
 * `strike_at` off the screen.
 */
function drawSeat(seat, source, { full, remote }) {
  const panel = document.getElementById(`seat-${seat}`);
  const entry = state[seat];

  document.getElementById(`handle-${seat}`).textContent = source.handle;
  document.getElementById(`landed-${seat}`).textContent = source.hits_landed;
  panel.classList.toggle('down', source.down);
  panel.classList.toggle('remote', remote);
  panel.classList.toggle('dropped', Boolean(entry?.dropped));
  panel.classList.toggle('locked', !remote && !source.can_commit);

  if (remote) {
    buildRemoteLoadout(seat);
    drawQueue(seat, source.queue || [], { full: false, remote: true });
    document.getElementById(`queue-head-${seat}`).textContent = 'queue · private · no cancellation';
    const left = document.getElementById(`queue-left-${seat}`);
    left.textContent = 'withheld';
    left.className = 'torc-hud-label push';
    document.getElementById(`commit-note-${seat}`).textContent = 'not this keyboard';
    return;
  }

  document.getElementById(`commit-note-${seat}`).textContent =
    source.can_commit ? 'commit' : 'commit — queue full';
  document.getElementById(`loadout-note-${seat}`).textContent =
    source.can_commit ? SEATS[seat].keyRange : 'locked — queue full';

  if (entry) {
    entry.canCommit = source.can_commit;
    entry.slots.forEach((slot) => {
      const cell = document.getElementById(`slot-${seat}-${slot}`);
      if (cell) cell.classList.toggle('staged', entry.staged === slot);
    });
  }

  const queue = source.queue || [];
  const depth = full ? (source.queue_depth ?? queue.length) : queue.length;
  drawQueue(seat, queue, { full, remote: false });
  document.getElementById(`queue-head-${seat}`).textContent =
    `queue · ${depth} of ${QUEUE_CELLS} · no cancellation`;
  const left = document.getElementById(`queue-left-${seat}`);
  const remaining = QUEUE_CELLS - depth;
  left.textContent = remaining > 0 ? `${remaining} left` : 'full';
  left.className = `torc-hud-label push ${remaining > 0 ? 'queue-left-some' : 'queue-left-full'}`;
}

/* The queue is the new HUD: what you have paid for and cannot take back. Five rows, always drawn.
 *
 * `commit_at`, `strike_at` and `end_tick` are null until the move reaches each stage, and null means
 * *not yet* — never zero, never "already over". Only the walking row prints a timing, because it is
 * the only one where "not yet" says something: a fighter crossing the ring for four seconds must not
 * read as idle. */
function drawQueue(seat, queue, { full, remote }) {
  const list = document.getElementById(`queue-${seat}`);
  if (!list) return;

  list.innerHTML = '';
  for (let index = 0; index < QUEUE_CELLS; index += 1) {
    const commit = queue[index];
    const row = document.createElement('div');
    row.className = 'qrow';

    const label = document.createElement('span');
    label.className = 'state';

    if (commit) {
      const walking = commit.executing && commit.approaching;
      row.classList.add(walking ? 'walking' : commit.executing ? 'striking' : 'waiting');
      label.textContent = walking ? 'walking' : commit.executing ? 'striking' : 'waiting';

      const pose = document.createElement('span');
      pose.className = 'pose';
      pose.textContent = commit.pose;
      row.append(label, pose);

      if (walking) {
        const timing = document.createElement('span');
        timing.className = 'timing';
        timing.textContent = 'arrives —';
        row.append(timing);
      }
    } else if (remote) {
      /* Not ours, not executing: paid for or not, the host will not say — and neither will this. */
      row.classList.add('private');
      label.textContent = 'private';
      row.append(label);
    } else {
      /* Ours, or ours-to-be. An empty cell here is a real empty cell — from our own socket it is
         the host's own queue, and before that first message the queue is empty anyway. */
      label.textContent = full ? 'empty' : '—';
      row.append(label);
    }

    list.append(row);
  }
}

/* Range, in words. The bands come from the scorer's own contact range, not a second idea of "close". */
function applyRange(separation) {
  const [, word, band] = RANGE_BANDS.find(([limit]) => separation <= limit);
  document.querySelector('.sep-head').dataset.band = band;
  document.getElementById('sep-band').textContent = word;
  document.getElementById('sep-metres').textContent = `${separation.toFixed(2)} m`;
  document.querySelectorAll('.sep-strip span, .sep-legend .torc-hud-label').forEach((element) => {
    element.classList.toggle('on', element.dataset.band === band);
  });
}

/* The live score is spec/scoring.md's own number, run over the round so far — not a second
   scoreboard. It is provisional: damage is normalised *within* a round, so an early share swings
   hard on one landed punch. The chip says so rather than letting a player find out at the bell. */
function applyScore(score) {
  const red = score.share.red ?? 0.5;
  const blue = score.share.blue ?? 0.5;
  document.getElementById('share-red').textContent = `${Math.round(red * 100)}%`;
  document.getElementById('share-blue').textContent = `${Math.round(blue * 100)}%`;
  document.getElementById('share-fill-red').style.width = `${(red * 100).toFixed(1)}%`;
  document.getElementById('share-fill-blue').style.width = `${(blue * 100).toFixed(1)}%`;

  /* `leading` is null inside the scorer's own draw margin. Never guess a leader the scorer would
     call level. */
  const leading = document.getElementById('leading');
  leading.textContent = score.leading ? `leading · ${score.leading}` : 'leading · —';

  for (const dimension of ['damage', 'control', 'aggression']) {
    const r = score.dimensions?.red?.[dimension] ?? 0;
    const b = score.dimensions?.blue?.[dimension] ?? 0;
    const total = r + b;
    const shareRed = total > 0 ? (r / total) * 100 : 50;
    document.querySelector(`[data-dim="${dimension}"][data-seat="red"]`).style.width =
      `${shareRed.toFixed(1)}%`;
    document.querySelector(`[data-dim="${dimension}"][data-seat="blue"]`).style.width =
      `${(100 - shareRed).toFixed(1)}%`;
  }

  document.getElementById('rounds-won').textContent =
    `rounds ${score.rounds_won?.red ?? 0} — ${score.rounds_won?.blue ?? 0}`;
  document.getElementById('points').textContent =
    `points ${score.points?.red ?? 0} — ${score.points?.blue ?? 0}`;
}

/* ---- banners -----------------------------------------------------------------------------------------
 * They sit over the ring, never on the fighters. Precedence is worst-news-first: a dropped socket
 * outranks a knockdown, which outranks the bell. */
let lastEvent = null;

function showBanner(tone, kicker, headline, detail) {
  banner.dataset.tone = tone;
  banner.innerHTML =
    `<span class="kicker">${kicker}</span><span class="headline">${headline}</span>`
    + (detail ? `<span class="detail">${detail}</span>` : '');
  banner.classList.remove('hidden');
}

function drawBanner(primary) {
  const dropped = Object.keys(SEATS).filter((seat) => state[seat]?.dropped);
  if (dropped.length) {
    showBanner('warn', 'connection', 'Connection lost',
      `last state at tick ${primary.tick} · retrying · the queue on the server is unaffected`);
    return;
  }

  const seats = visibleSeats();
  const down = Object.entries(seats).find(([, seat]) => seat.down);
  if (down) {
    showBanner('warn', 'event · knockdown', `${down[1].handle} is down`,
      `get-up window · ${getUpSeconds} s · torso ${down[1].torso_height_m?.toFixed(2) ?? '—'} m`);
    return;
  }

  if (primary.phase === 'match_over') {
    showBanner('accent', 'event · match end', 'Match over', 'the record has been written');
    return;
  }
  if (primary.phase === 'round_over' && lastEvent?.event === 'round_end') {
    const how = lastEvent.knocked_out ? `${lastEvent.knocked_out} knocked out` : 'ended by bell';
    showBanner('accent', `round ${lastEvent.round} · ${how}`,
      `${primary.score?.points?.red ?? 0} — ${primary.score?.points?.blue ?? 0}`,
      'queues cleared · fighters reset to 1.20 m separation');
    return;
  }

  const waiting = Object.keys(SEATS).filter((seat) => !state[seat]?.joined && !state[seat]?.remote);
  if (waiting.length) {
    showBanner('accent', 'seats', `Waiting for the ${waiting[0]} seat`,
      'one of two seats taken · the clock does not start');
    return;
  }

  banner.classList.add('hidden');
}

function applyEvent(message) {
  lastEvent = message;
  render();
}

/* ---- one socket per seat ------------------------------------------------------------------------------ */
function connect(seat) {
  const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws?seat=${seat}`;
  const socket = new WebSocket(url);
  socket.binaryType = 'arraybuffer';

  state[seat] = {
    socket, slots: [], loadout: {}, poses: {},
    staged: null, anchor: null, opponentPosition: null,
    poseSeconds: {}, approachSpeed: null,
    offset: { x: 0, y: 0 }, canCommit: true,
    joined: false, remote: false, dropped: false,
  };

  socket.addEventListener('open', () => {
    socket.send(JSON.stringify({ type: 'join', handle: seat }));
    setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'ping', t: Date.now() }));
      }
    }, 2000);
  });

  socket.addEventListener('message', (raw) => {
    /* Only one seat's socket needs to drive the shared 3-D view; both would just do the work twice
       for the same bytes — the binary frame is the same for every viewer by design. */
    if (raw.data instanceof ArrayBuffer) {
      if (seat === firstOwned()) ring.applyFrame(raw.data);
      return;
    }

    const message = JSON.parse(raw.data);
    switch (message.type) {
      case 'welcome':
        state[seat].joined = true;
        state[seat].loadout = message.loadout;
        state[seat].poseSeconds = message.pose_seconds || {};
        state[seat].approachSpeed = message.approach_speed_m_s || null;
        state[seat].poses = message.poses || {};
        state[seat].slots = Object.keys(message.loadout).sort();
        rounds = message.format?.rounds ?? rounds;
        getUpSeconds = Math.round((message.format?.get_up_window_ticks ?? 400) / TICK_HZ);
        buildLoadout(seat);
        statusLine.innerHTML = `<span class="status-dot ok"></span>connected · ${message.match_id}`;
        break;
      case 'state':
        applyState(message, seat);
        break;
      case 'event':
        applyEvent(message);
        break;
      case 'pong':
        if (seat === firstOwned()) {
          document.getElementById('ping').textContent = `ping ${Date.now() - message.t} ms`;
        }
        break;
      case 'error':
        showError(seat, message);
        break;
    }
  });

  socket.addEventListener('close', () => {
    /* A seat we never got into is somebody else's — an agent, or a second machine. That is not a
       failure, it is the remote-opponent case, and the panel says so. A seat we *had* is a dropped
       socket, and the readouts hold their last value rather than faking a fresh one. */
    if (state[seat].joined) {
      state[seat].dropped = true;
      statusLine.innerHTML = `<span class="status-dot bad"></span>${seat} disconnected`;
    } else {
      state[seat].remote = true;
    }
    render();
  });
}

/* The seat whose socket speaks for the shared chrome: the ring frame and the ping. */
function firstOwned() {
  return Object.keys(SEATS).find((seat) => state[seat]?.joined && !state[seat]?.dropped);
}

/* The host's own sentence, verbatim — it is the one that knows why. It does not steal focus, does
   not block the fight, and is not a modal. */
function showError(seat, message) {
  const element = document.getElementById(`error-${seat}`);
  if (!element) return;
  element.textContent = message.message;
  element.hidden = false;
  clearTimeout(state[seat].errorTimer);
  state[seat].errorTimer = setTimeout(() => { element.hidden = true; }, ERROR_MS);
}

/* ---- committing ---------------------------------------------------------------------------------------
 * The one moment the host learns where the ghost was. Place then commit, in that order and in one
 * batch, so the host applies both on the same intent tick and the commit cannot land against a stale
 * placement.
 */
function commit(seat) {
  const entry = state[seat];
  const at = shadowPosition(seat);
  if (!entry || !entry.staged || !at) return;

  entry.socket.send(JSON.stringify({
    type: 'place', x: at.x, y: at.y, heading: shadowHeading(seat, at),
  }));
  entry.socket.send(JSON.stringify({ type: 'commit' }));
  entry.offset = { x: 0, y: 0 };   // the next move starts from the new anchor
}

/* ---- keyboard ------------------------------------------------------------------------------------------ */
/* One handler for both seats. A key belongs to exactly one seat, so hotseat needs no focus
   management and no click-to-activate — two people share the keyboard and nothing steals it. */
document.addEventListener('keydown', (raw) => {
  if (raw.repeat) return;
  const pressed = raw.key.length === 1 ? raw.key.toLowerCase() : raw.key;

  for (const [seat, config] of Object.entries(SEATS)) {
    const entry = state[seat];
    if (!entry || entry.socket.readyState !== WebSocket.OPEN) continue;

    if (config.drive[pressed]) {
      held[seat].add(pressed);
      raw.preventDefault();
      return;
    }

    const index = config.keys.indexOf(pressed);
    if (index >= 0 && index < entry.slots.length) {
      entry.staged = entry.slots[index];
      entry.socket.send(JSON.stringify({ type: 'stage', slot: entry.staged }));
      render();
      raw.preventDefault();
      return;
    }
    if (pressed === config.commit) {
      commit(seat);
      raw.preventDefault();
      return;
    }
    if (pressed === config.clear) {
      entry.staged = null;
      entry.offset = { x: 0, y: 0 };
      entry.socket.send(JSON.stringify({ type: 'clear' }));
      render();
      raw.preventDefault();
      return;
    }
  }
});

document.addEventListener('keyup', (raw) => {
  const released = raw.key.length === 1 ? raw.key.toLowerCase() : raw.key;
  for (const seat of Object.keys(SEATS)) {
    if (held[seat].delete(released)) { raw.preventDefault(); return; }
  }
});

/* A window that loses focus keeps no keys down. Without this a ghost drifts into the ropes for the
   rest of the round because a keyup was never delivered. */
window.addEventListener('blur', () => {
  Object.keys(SEATS).forEach((seat) => held[seat].clear());
});

/* ---- the loop -------------------------------------------------------------------------------------------- */
let previous = performance.now();
function loop(now) {
  const dt = Math.min(0.1, (now - previous) / 1000);
  previous = now;
  driveShadows(dt);
  ring.render();
  requestAnimationFrame(loop);
}

showBanner('accent', 'state 01', 'Connecting', 'no state received yet · the ring is drawn, empty');

ring.load().then((description) => {
  ringHalf = (description.arena?.ring_size ?? 4.9) / 2;
  standHeight = description.arena?.start_height ?? 0.793;
  /* The map is authored at 100 units to the metre for a 4.90 m ring. Ring size is a match parameter
     (`M4-T4` will change it), so the box is re-derived from the arena rather than assumed. */
  const side = ringHalf * 2 * 100;
  document.getElementById('map-size').textContent = `${(ringHalf * 2).toFixed(2)} m`;
  document.getElementById('map').setAttribute('viewBox', `-24 -24 ${side + 48} ${side + 48}`);
  for (const id of ['map-grid', 'map-bounds']) {
    document.getElementById(id).setAttribute('width', side);
    document.getElementById(id).setAttribute('height', side);
  }
  Object.keys(SEATS).forEach(connect);
  requestAnimationFrame(loop);
}).catch((error) => {
  statusLine.innerHTML = `<span class="status-dot bad"></span>could not build the ring: ${error.message}`;
  throw error;
});
