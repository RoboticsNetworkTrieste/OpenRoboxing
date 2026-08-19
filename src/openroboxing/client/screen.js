/* The fight-night screen (M5-T3), built to spec/ui-design-guide/ screen 02.
 *
 * Its acceptance is "driven live at a meetup without an operator touching a terminal", so this page
 * takes no input at all. It connects as a spectator, reconnects on its own when a match ends and the
 * next one starts, and shows no controls to touch.
 *
 * It is deliberately a *spectator*: the host sends it no loadouts, no staged slot, no ghost and no
 * queue — anybody in the room can read a projector, and none of that is public information.
 */

'use strict';

import { Ring } from './ring.js';
import { drawSpectatorArena } from './overlay.js';

const TICK_HZ = 50;
const RECONNECT_MS = 3000;
const FEED_LIMIT = 12;
/* The table is optional: it is served by the league, which may not be running. A screen with no
   table is still a screen; a screen that crashes because the league is down is not. */
const TABLE_URL = '/static/table.json';
const TABLE_REFRESH_MS = 30000;
const TABLE_ROWS = 8;

const canvas = document.getElementById('ring');
const overlay = document.getElementById('arena-overlay');
const banner = document.getElementById('banner');
const feed = document.getElementById('feed');
const statusLine = document.getElementById('status');
const scoreline = document.querySelector('.scoreline');

const ring = new Ring(canvas);

let seenHits = { red: 0, blue: 0 };
let seenDown = { red: false, blue: false };
let latest = null;
let rounds = 3;

function formatClock(ticks) {
  const seconds = Math.max(0, Math.ceil(ticks / TICK_HZ));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

function note(text, actor) {
  const item = document.createElement('li');
  if (actor) item.dataset.actor = actor;
  const at = document.createElement('span');
  at.className = 'at';
  at.textContent = latest ? formatClock(latest.clock_ticks) : '—';
  const mark = document.createElement('span');
  mark.className = 'mark';
  const body = document.createElement('span');
  body.textContent = text;
  item.append(at, mark, body);
  feed.prepend(item);
  while (feed.children.length > FEED_LIMIT) feed.lastChild.remove();
}

/* ---- state ---------------------------------------------------------------------------------------- */
function applyState(message) {
  latest = message;
  document.getElementById('clock').textContent = formatClock(message.clock_ticks);
  scoreline.classList.toggle('low', message.clock_ticks < 10 * TICK_HZ);
  document.getElementById('round').textContent = `round ${message.round} of ${rounds}`;

  const note_ = document.getElementById('arena-note');
  note_.textContent = message.separation_m === null || message.separation_m === undefined
    ? 'arena · unitree g1'
    : `arena · unitree g1 · separation ${message.separation_m.toFixed(2)} m`;

  for (const [seat, seatState] of Object.entries(message.seats)) {
    const name = document.getElementById(`name-${seat}`);
    const tally = document.getElementById(`tally-${seat}`);
    if (!name || !tally) continue;

    name.textContent = seatState.handle;
    tally.textContent = seatState.hits_landed;

    /* The feed is built from what changed, because the host sends state and not a log. */
    if (seatState.hits_landed > seenHits[seat]) {
      note(`${seatState.handle} lands (${seatState.hits_landed})`, seat);
    }
    seenHits[seat] = seatState.hits_landed;

    if (seatState.down && !seenDown[seat]) note(`Knockdown — ${seatState.handle} is down`, 'knockdown');
    seenDown[seat] = seatState.down;
  }

  if (message.score) applyScore(message.score);
  if (message.phase === 'fighting') banner.classList.add('hidden');
}

/* The official scorer's number over the round so far, not a second scoreboard. Provisional until
   the bell, and labelled that way — a projector that quietly changed its mind at the end would look
   like the game cheated. */
function applyScore(score) {
  const red = score.share.red ?? 0.5;
  const blue = score.share.blue ?? 0.5;
  document.getElementById('share-red').textContent = `${Math.round(red * 100)}%`;
  document.getElementById('share-blue').textContent = `${Math.round(blue * 100)}%`;
  document.getElementById('share-fill-red').style.width = `${(red * 100).toFixed(1)}%`;
  document.getElementById('share-fill-blue').style.width = `${(blue * 100).toFixed(1)}%`;
  document.getElementById('totals').textContent =
    `rounds ${score.rounds_won?.red ?? 0} — ${score.rounds_won?.blue ?? 0}`
    + ` · points ${score.points?.red ?? 0} — ${score.points?.blue ?? 0}`;
}

function showBanner(tone, kicker, headline, detail) {
  banner.dataset.tone = tone;
  banner.innerHTML =
    `<span class="kicker">${kicker}</span><span class="headline">${headline}</span>`
    + (detail ? `<span class="detail">${detail}</span>` : '');
  banner.classList.remove('hidden');
}

function applyEvent(message) {
  if (message.event === 'round_end') {
    const how = message.knocked_out ? `${message.knocked_out} knocked out` : 'ended by bell';
    showBanner('accent', `round ${message.round} · ${how}`,
      `${message.hits.red} — ${message.hits.blue} landed`, 'the round is scored at the bell');
    note(`Round ${message.round} — ${how}`, 'bell');
    seenHits = { red: 0, blue: 0 };
    seenDown = { red: false, blue: false };
  } else if (message.event === 'match_end') {
    showBanner('accent', 'event · match end', 'Match over', 'waiting for the next one');
    note('Match over', 'resolved');
  } else if (message.event === 'knockout') {
    showBanner('warn', 'event · knockout', `${message.knocked_out} is out`, '');
    note(`Knockout — ${message.knocked_out}`, 'knockdown');
  }
}

/* ---- the table -------------------------------------------------------------------------------------- */
function renderTable(season) {
  const rows = (season.table || []).slice(0, TABLE_ROWS);
  const card = document.getElementById('table-card');
  if (!rows.length) { card.classList.add('hidden'); return; }

  const table = document.getElementById('table');
  while (table.children.length > 4) table.lastChild.remove();   // keep the header cells

  for (const entry of rows) {
    const cells = [
      ['td', entry.handle],
      ['td num', Number(entry.rating).toFixed(1)],
      ['td num rd', Number(entry.rd).toFixed(1)],
      ['td num', `${entry.won}-${entry.drawn}-${entry.lost}`],
    ];
    for (const [className, text] of cells) {
      const cell = document.createElement('span');
      cell.className = className;
      cell.textContent = text;
      table.append(cell);
    }
  }
  card.classList.remove('hidden');
}

function pollTable() {
  fetch(TABLE_URL, { cache: 'no-store' })
    .then((response) => (response.ok ? response.json() : null))
    .then((season) => { if (season) renderTable(season); })
    .catch(() => {});   /* no league running; the screen is still a screen */
}

/* ---- the socket, which reconnects itself ---------------------------------------------------------- */
function connect() {
  const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws?seat=spectator`;
  const socket = new WebSocket(url);
  socket.binaryType = 'arraybuffer';

  socket.addEventListener('open', () => {
    statusLine.classList.remove('down');
    statusLine.innerHTML = '<span class="live-dot"></span>connected';
    seenHits = { red: 0, blue: 0 };
    seenDown = { red: false, blue: false };
  });

  socket.addEventListener('message', (raw) => {
    if (raw.data instanceof ArrayBuffer) { ring.applyFrame(raw.data); return; }
    const message = JSON.parse(raw.data);
    if (message.type === 'state') applyState(message);
    else if (message.type === 'event') applyEvent(message);
    else if (message.type === 'welcome') {
      rounds = message.format?.rounds ?? rounds;
      document.getElementById('venue').textContent =
        `OpenRoboxing · fight night · ${message.match_id}`;
      for (const [seat, handle] of Object.entries(message.handles || {})) {
        const name = document.getElementById(`name-${seat}`);
        if (name) name.textContent = handle;
      }
    }
  });

  /* No operator, so no manual reconnect. The next match brings the socket back by itself. */
  socket.addEventListener('close', () => {
    statusLine.classList.add('down');
    statusLine.innerHTML = '<span class="live-dot"></span>waiting for the next match';
    setTimeout(connect, RECONNECT_MS);
  });
  socket.addEventListener('error', () => socket.close());
}

function loop() {
  if (latest) drawSpectatorArena(overlay, ring, latest.seats, latest.separation_m);
  ring.render();
  requestAnimationFrame(loop);
}

/* The ring is built before the socket opens: a projector that connected first would spend its first
   seconds drawing nothing while it fetched 10 MB of geometry. */
ring.load().then((description) => {
  ring.frameRing(description.arena);
  requestAnimationFrame(loop);
  connect();
}).catch((error) => {
  statusLine.classList.add('down');
  statusLine.innerHTML = `<span class="live-dot"></span>could not build the ring: ${error.message}`;
});

pollTable();
setInterval(pollTable, TABLE_REFRESH_MS);
