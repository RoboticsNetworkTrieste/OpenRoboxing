/* The sparring bench client — spec/sparring_protocol.md 0.1.
 *
 * Three bodies in the ring and a strip of instruments under it:
 *   - the robot (and the sacco): streamed binary frames, exactly as a match;
 *   - the aim ghost: yours, drawn locally, transmitted only on commit (as the match client);
 *   - the plan ghost + trail: what MotionBricks generated and GEAR-SONIC is chasing, from the
 *     10 Hz `debug` message. Clay, so it can never be mistaken for the aim.
 *
 * Two view modes. LIVE follows the stream; SCRUB replays any recorded tick through
 * `GET /api/frame/{tick}` — same binary format, same draw path, so a scrubbed frame cannot lie.
 */

'use strict';

import * as THREE from './vendor/three.module.min.js';
import { Ring } from './ring.js';
import { ChartStack } from './sparring-charts.js';

const TICK_HZ = 50;
const SHADOW_SPEED_M_S = 1.9;         // how fast the aim ghost travels — a UI number (app.js)
const SERIES_POLL_MS = 1000;
const SERIES_MAX_POINTS = 1200;
const SCRUB_DEBOUNCE_MS = 30;

const DRIVE = { w: [0, 1], s: [0, -1], a: [-1, 0], d: [1, 0] };
const KNOB_ORDER = [
  'replan_dt', 'horizon_ticks', 'max_outstanding',
  'arrival_radius_m', 'approach_leg_m', 'approach_timeout_ticks', 'pose_dwell_ticks',
];
const KNOB_STEP = {
  replan_dt: 0.05, horizon_ticks: 5, max_outstanding: 1,
  arrival_radius_m: 0.05, approach_leg_m: 0.25, approach_timeout_ticks: 25, pose_dwell_ticks: 5,
};

const canvas = document.getElementById('ring');
const canvasBox = canvas.parentElement;
const overlay = document.getElementById('arena-overlay');
const ring = new Ring(canvas);
const charts = new ChartStack(document.getElementById('charts'));

const S = {
  socket: null,
  mode: 'live',              // 'live' | 'scrub'
  paused: false,
  loadout: {}, poses: {}, poseSeconds: {}, slots: [],
  staged: null, offset: { x: 0, y: 0 },
  anchor: null, canCommit: true, position: null,
  debug: null,               // last debug message (live)
  knobsBuilt: false,
  heatmap: false, showPlan: true, showTrail: true,
  picking: false,            // teleport-sacco click mode
  scrubTick: null,
  held: new Set(),
  script: { running: false, entries: [], index: 0 },
  ringHalf: 2.45, standHeight: 0.793,
  bluePelvis: -1,
  heatBodies: [],            // [{group, joints:[names]}] for the red fighter
  description: null,
};

/* ---- connection ---------------------------------------------------------------------------- */
function connect() {
  const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
  const socket = new WebSocket(url);
  socket.binaryType = 'arraybuffer';
  S.socket = socket;

  socket.addEventListener('open', () => {
    setStatus('ok', 'connected');
    setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) send({ type: 'ping', t: Date.now() });
    }, 2000);
  });

  socket.addEventListener('message', (raw) => {
    if (raw.data instanceof ArrayBuffer) {
      if (S.mode === 'live') ring.applyFrame(raw.data);
      return;
    }
    let message;
    try {
      message = JSON.parse(raw.data);
    } catch (error) {
      /* One unparseable message must not kill the stream silently. */
      setStatus('bad', `unreadable message from the bench: ${error.message}`);
      return;
    }
    switch (message.type) {
      case 'welcome': applyWelcome(message); break;
      case 'state': applyState(message); break;
      case 'debug': applyDebug(message); break;
      case 'error': showError(message.message); break;
      default: break;
    }
  });

  socket.addEventListener('close', () => setStatus('bad', 'disconnected — reload to rejoin'));
}

function send(message) {
  if (S.socket?.readyState === WebSocket.OPEN) S.socket.send(JSON.stringify(message));
}

function setStatus(tone, text) {
  document.getElementById('status').innerHTML =
    `<span class="status-dot ${tone}"></span>${text}`;
}

function showError(text) {
  const element = document.getElementById('error');
  element.textContent = text;
  element.hidden = false;
  clearTimeout(S.errorTimer);
  S.errorTimer = setTimeout(() => { element.hidden = true; }, 5000);
}

/* ---- welcome / loadout ----------------------------------------------------------------------- */
function applyWelcome(message) {
  S.loadout = message.loadout;
  S.poses = message.poses || {};
  S.poseSeconds = message.pose_seconds || {};
  S.slots = Object.keys(message.loadout).sort();
  buildLoadout();
}

function buildLoadout() {
  const bar = document.getElementById('loadout');
  bar.innerHTML = '';
  S.slots.forEach((slot, index) => {
    const cell = document.createElement('div');
    cell.className = 'slot';
    cell.id = `slot-${slot}`;
    cell.innerHTML =
      `<span class="key">${index + 1}</span><span class="name">${S.loadout[slot]}</span>`;
    bar.append(cell);
  });
}

/* ---- state / debug --------------------------------------------------------------------------- */
function applyState(message) {
  const seat = message.seats?.red;
  if (!seat) return;
  S.anchor = seat.anchor;
  S.canCommit = seat.can_commit;
  S.position = seat.position;
  S.paused = Boolean(message.paused);
  document.getElementById('btn-pause').textContent = S.paused ? 'resume' : 'pause';
  document.getElementById('btn-pause').dataset.on = S.paused ? 'yes' : 'no';
}

function applyDebug(message) {
  S.debug = message;
  /* The pilot's refusals arrive latched on the debug stream (the pilot applies messages a tick
     after the socket, so a direct reply is impossible). Same sentence twice = one event. */
  if (message.pilot_error && message.pilot_error.message !== S.lastPilotError) {
    S.lastPilotError = message.pilot_error.message;
    showError(`refused at tick ${message.pilot_error.tick}: ${message.pilot_error.message}`);
  }
  const rec = message.recording;
  const scrubber = document.getElementById('scrubber');
  scrubber.min = rec.start_tick;
  scrubber.max = rec.end_tick;
  if (S.mode === 'live') {
    scrubber.value = rec.end_tick;
    charts.setCursor(null);
    drawPanel(message);
  }
  document.getElementById('scrub-line').textContent =
    `${S.mode === 'live' ? message.tick : S.scrubTick} / ${rec.end_tick}`;
  if (!S.knobsBuilt && message.knobs) buildKnobs(message.knobs);
  else if (message.knobs) updateKnobs(message.knobs);
  runScriptStep(message.tick);
}

/* The one panel-update path: live `debug` messages and scrub payloads both land here, so the two
   modes cannot drift apart in what they show. */
function drawPanel(data) {
  document.getElementById('machine-chip').textContent = data.machine;
  document.getElementById('machine-chip').dataset.state = data.machine;
  document.getElementById('tick-line').textContent =
    `tick ${data.tick}${data.paused ? ' · paused' : ''}`;
  drawQueue(data.queue || []);

  const head = data.series_head;
  document.getElementById('head-line').textContent = head
    ? `err ${head.err_mean?.toFixed(3)} rad · dist body ${head.dist ?? '—'} m`
      + ` / plan ${head.dist_plan ?? '—'} m · step ${head.step_ms} ms`
    : '';
  charts.arrivalRadius = knobValue('arrival_radius_m');

  if (S.showPlan && data.ghost) {
    const plan = ring.shadowFor('plan');
    plan.material.color.set(0xbc7b4c);          // clay — never mistakable for the red aim ghost
    plan.material.opacity = 0.28;
    ring.showShadow('plan', data.ghost.x, data.ghost.y, data.ghost.heading,
      data.ghost.angles, data.ghost.z);
  } else {
    ring.hideShadow('plan');
  }
  drawOverlay(data);
  if (S.heatmap && head?.err_by_joint) paintHeatmap(head.err_by_joint);
}

function drawQueue(queue) {
  const list = document.getElementById('queue');
  list.innerHTML = '';
  document.getElementById('queue-note').textContent =
    `${queue.length} of ${knobValue('max_outstanding') ?? 10} · no cancellation`;

  for (const entry of queue) {
    const row = document.createElement('div');
    const cls = entry.approaching ? 'approach' : entry.executing ? 'dwell' : 'waiting';
    row.className = `qrow ${cls}`;
    const spans =
      `i${entry.issued_at} · c${entry.commit_at ?? '—'} · s${entry.strike_at ?? '—'}`
      + ` · e${entry.end_tick ?? '—'}${entry.completed_by ? ` (${entry.completed_by})` : ''}`;
    /* `arrived === false` is a move whose approach ran out of time and threw the pose where it
       stood. It is not a detail: it is the difference between a landed strike and a missed one. */
    const state = entry.approaching ? 'walking'
      : entry.executing ? (entry.arrived === false ? 'timed out' : 'striking')
        : 'waiting';
    row.innerHTML =
      `<span class="state">${state}</span>`
      + `<span class="pose">${entry.pose}</span><span class="spans">${spans}</span>`;
    list.append(row);
  }
  if (!queue.length) {
    const row = document.createElement('div');
    row.className = 'qrow';
    row.innerHTML = '<span class="state">empty</span><span class="pose">—</span>';
    list.append(row);
  }
}

/* ---- overlay: trail + arrival circle --------------------------------------------------------- */
function svgPoint(x, y, z = 0) {
  const p = ring.project(x, y, z);
  return p.behind ? null : p;
}

function drawOverlay(data) {
  const parts = [];
  if (S.showTrail && data.trail?.length > 1) {
    const points = data.trail
      .map(([x, y]) => svgPoint(x, y))
      .filter(Boolean)
      .map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`);
    if (points.length > 1) {
      parts.push(`<polyline points="${points.join(' ')}" fill="none" `
        + 'stroke="var(--phase-dwell)" stroke-width="1.5" stroke-opacity="0.7" '
        + 'stroke-dasharray="5 4"/>');
    }
  }

  /* The arrival circle, at the placement of the commit that is walking. The radius is the live
     knob, not the constant — turning the knob moves the circle, which is the point. */
  const walking = (data.queue || []).find((entry) => entry.approaching);
  const radius = knobValue('arrival_radius_m');
  if (walking?.placement && radius) {
    const { x, y } = walking.placement;
    const rim = [];
    for (let i = 0; i <= 24; i += 1) {
      const a = (i / 24) * Math.PI * 2;
      const p = svgPoint(x + radius * Math.cos(a), y + radius * Math.sin(a));
      if (p) rim.push(`${p.x.toFixed(1)},${p.y.toFixed(1)}`);
    }
    if (rim.length > 2) {
      parts.push(`<polygon points="${rim.join(' ')}" fill="none" `
        + 'stroke="var(--phase-approach)" stroke-width="1.5" stroke-opacity="0.8"/>');
    }
    const centre = svgPoint(x, y);
    if (centre) {
      parts.push(`<line x1="${centre.x - 5}" y1="${centre.y}" x2="${centre.x + 5}" y2="${centre.y}" stroke="var(--phase-approach)"/>`);
      parts.push(`<line x1="${centre.x}" y1="${centre.y - 5}" x2="${centre.x}" y2="${centre.y + 5}" stroke="var(--phase-approach)"/>`);
    }
  }
  overlay.innerHTML = parts.join('');
}

/* ---- heatmap --------------------------------------------------------------------------------- */
const HEAT_LOW = new THREE.Color(0x000000);
const HEAT_HIGH = new THREE.Color(0xbc7b4c);   // clay: the "off" end of the TORC scale
const HEAT_SCALE_RAD = 0.5;                    // fixed scale, per the design spec

function buildHeatBodies(description) {
  const kin = description.shadow_kinematics;
  S.heatBodies = [];
  description.shadow_bodies.forEach((short, i) => {
    const bodyIndex = description.bodies.indexOf(`red_${short}`);
    if (bodyIndex < 0) return;
    const joints = (kin.bodies[i]?.joints || []).map((j) => j.name);
    if (joints.length) S.heatBodies.push({ group: ring.bodies[bodyIndex], joints });
  });
}

function paintHeatmap(errByJoint) {
  for (const { group, joints } of S.heatBodies) {
    let worst = 0;
    for (const name of joints) worst = Math.max(worst, errByJoint[name] ?? 0);
    const t = Math.min(1, worst / HEAT_SCALE_RAD);
    for (const mesh of group.children) {
      if (mesh.material?.emissive) mesh.material.emissive.lerpColors(HEAT_LOW, HEAT_HIGH, t);
    }
  }
}

function clearHeatmap() {
  for (const { group } of S.heatBodies) {
    for (const mesh of group.children) mesh.material?.emissive?.set(0x000000);
  }
}

/* ---- the aim ghost (the match client's mechanism, red seat only) ------------------------------ */
function shadowPosition() {
  if (!S.anchor) return null;
  const clamp = (v) => Math.max(-S.ringHalf, Math.min(S.ringHalf, v));
  return { x: clamp(S.anchor.x + S.offset.x), y: clamp(S.anchor.y + S.offset.y) };
}

function shadowHeading(at) {
  const sacco = saccoPosition();
  if (!sacco) return S.anchor?.heading ?? 0;
  return Math.atan2(sacco.y - at.y, sacco.x - at.x);
}

function saccoPosition() {
  if (S.bluePelvis < 0) return null;
  const p = ring.bodies[S.bluePelvis]?.position;
  return p ? { x: p.x, y: p.y } : null;
}

function driveShadow(dt) {
  let dx = 0;
  let dy = 0;
  for (const key of S.held) {
    const step = DRIVE[key];
    if (step) { dx += step[0]; dy += step[1]; }
  }
  if (dx || dy) {
    const scale = (SHADOW_SPEED_M_S * dt) / Math.hypot(dx, dy);
    S.offset.x += dx * scale;
    S.offset.y += dy * scale;
  }
  const at = shadowPosition();
  const angles = S.poses[S.staged];
  if (!at || !S.staged || !angles) { ring.hideShadow('red'); return; }
  ring.showShadow('red', at.x, at.y, shadowHeading(at), angles, S.standHeight);
}

function commit() {
  /* Refusing silently is how the bench's first bug hid — say why nothing will happen. */
  if (!S.staged) { showError('nothing staged — pick a pose with 1–6 first'); return; }
  const at = shadowPosition();
  if (!at) { showError('no anchor yet — waiting for the first state message'); return; }
  send({ type: 'place', x: at.x, y: at.y, heading: shadowHeading(at) });
  send({ type: 'commit' });
  S.offset = { x: 0, y: 0 };
}

/* ---- knobs ----------------------------------------------------------------------------------- */
function knobValue(name) {
  return S.debug?.knobs?.[name]?.current ?? null;
}

function buildKnobs(knobs) {
  S.knobsBuilt = true;
  const box = document.getElementById('knobs');
  box.innerHTML = '';
  for (const name of KNOB_ORDER) {
    const entry = knobs[name];
    if (!entry) continue;
    const row = document.createElement('div');
    row.className = 'knob';
    row.id = `knob-${name}`;
    row.innerHTML =
      `<span>${name}</span>`
      + `<input type="number" step="${KNOB_STEP[name] ?? 1}" value="${entry.current}">`
      + `<button class="action canon" title="back to canonical">${entry.canonical}</button>`;
    const input = row.querySelector('input');
    input.addEventListener('change', () => postKnob(name, input.value));
    row.querySelector('.canon').addEventListener('click', () => postKnob(name, entry.canonical));
    box.append(row);
    if (name === 'pose_dwell_ticks') {
      const warning = document.createElement('div');
      warning.className = 'knob-warning';
      warning.textContent = '⚠ dwell also rewrites in-flight end_ticks (spec/sparring_protocol.md)';
      box.append(warning);
    }
  }
}

function updateKnobs(knobs) {
  for (const [name, entry] of Object.entries(knobs)) {
    const row = document.getElementById(`knob-${name}`);
    if (!row) continue;
    const input = row.querySelector('input');
    if (document.activeElement !== input) input.value = entry.current;
    row.classList.toggle('deviated', Math.abs(entry.current - entry.canonical) > 1e-9);
  }
}

async function postKnob(name, value) {
  const response = await fetch('/api/knobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ [name]: Number(value) }),
  });
  const body = await response.json();
  if (!response.ok) { showError(body.error || 'knob refused'); return; }
  if (S.debug) S.debug.knobs = body;
  updateKnobs(body);
}

/* ---- scrubbing ------------------------------------------------------------------------------- */
let scrubTimer = null;

function enterScrub(tick) {
  S.mode = 'scrub';
  S.scrubTick = tick;
  document.getElementById('mode-line').textContent = `SCRUB · t ${tick}`;
  document.getElementById('btn-live').dataset.on = 'no';
  charts.setCursor(tick);
  clearTimeout(scrubTimer);
  scrubTimer = setTimeout(async () => {
    const response = await fetch(`/api/frame/${tick}`);
    if (!response.ok) return;
    const payload = await response.json();
    if (payload.frame) {
      const bytes = Uint8Array.from(atob(payload.frame), (c) => c.charCodeAt(0));
      ring.applyFrame(bytes.buffer);
    }
    drawPanel({ ...payload, paused: true, queue: payload.queue ?? [], knobs: S.debug?.knobs });
    document.getElementById('scrub-line').textContent =
      `${tick} / ${payload.recording.end_tick}`;
  }, SCRUB_DEBOUNCE_MS);
}

function goLive() {
  S.mode = 'live';
  S.scrubTick = null;
  document.getElementById('mode-line').textContent = 'LIVE';
  document.getElementById('btn-live').dataset.on = 'yes';
  charts.setCursor(null);
}

/* ---- the traces ------------------------------------------------------------------------------ */
/* A swallowed failure here is invisible: the charts simply stay empty, which looks exactly like a
   bench that recorded nothing. That is how 0.1's blank strip hid a payload the browser could not
   parse at all (the server wrote bare `NaN`, `JSON.parse` refused it, and this `catch` ate the
   exception on every poll for the whole session). It says so now, once per new message. */
async function pollSeries() {
  const rec = S.debug?.recording;
  if (rec && rec.end_tick > rec.start_tick) {
    const span = rec.end_tick - rec.start_tick;
    const stride = Math.max(1, Math.ceil(span / SERIES_MAX_POINTS));
    try {
      const response = await fetch(
        `/api/series?from=${rec.start_tick}&to=${rec.end_tick}&stride=${stride}`);
      if (!response.ok) throw new Error(`/api/series answered ${response.status}`);
      charts.setData(await response.json());
      S.seriesFault = null;
    } catch (error) {
      const text = `traces: ${error.message}`;
      if (text !== S.seriesFault) { S.seriesFault = text; setStatus('bad', text); }
    }
  }
  setTimeout(pollSeries, SERIES_POLL_MS);
}

charts.onSeek = (tick) => {
  document.getElementById('scrubber').value = tick;
  enterScrub(tick);
};

/* ---- the script runner ----------------------------------------------------------------------- */
function startScript() {
  let entries;
  try {
    entries = JSON.parse(document.getElementById('script').value);
    if (!Array.isArray(entries)) throw new Error('the script must be a JSON array');
  } catch (error) {
    showError(`script: ${error.message}`);
    return;
  }
  S.script = { running: true, entries, index: 0 };
  document.getElementById('btn-script-run').disabled = true;
  document.getElementById('btn-script-stop').disabled = false;
  noteScript();
}

function stopScript() {
  S.script.running = false;
  document.getElementById('btn-script-run').disabled = false;
  document.getElementById('btn-script-stop').disabled = true;
  noteScript();
}

function noteScript() {
  const { running, entries, index } = S.script;
  document.getElementById('script-note').textContent = running
    ? `running · ${index} of ${entries.length} sent`
    : entries?.length ? `stopped · ${index} of ${entries.length} sent` : 'not running';
}

/* Called on every debug tick: fire the next entry when its conditions hold. Staging is sent just
   before the commit so a manual key press cannot interleave a different pose into a script step. */
function runScriptStep(tick) {
  const script = S.script;
  if (!script.running) return;
  if (script.index >= script.entries.length) { stopScript(); return; }
  const entry = script.entries[script.index];
  if (entry.at_tick !== undefined && tick < entry.at_tick) return;
  if (!S.canCommit) return;

  send({ type: 'stage', slot: String(entry.slot) });
  if (entry.x !== undefined) {
    send({ type: 'place', x: entry.x, y: entry.y, heading: entry.heading ?? 0 });
  }
  send({ type: 'commit' });
  script.index += 1;
  noteScript();
}

/* ---- teleport picking ------------------------------------------------------------------------ */
const raycaster = new THREE.Raycaster();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
const pickPoint = new THREE.Vector3();

canvas.addEventListener('click', (event) => {
  if (!S.picking) return;
  const rect = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );
  raycaster.setFromCamera(ndc, ring.camera);
  if (raycaster.ray.intersectPlane(groundPlane, pickPoint)) {
    const me = S.position;
    const heading = me ? Math.atan2(me.y - pickPoint.y, me.x - pickPoint.x) : 0;
    send({ type: 'teleport_sacco', x: pickPoint.x, y: pickPoint.y, heading });
  }
  S.picking = false;
  canvasBox.classList.remove('picking');
  document.getElementById('btn-teleport').dataset.on = 'no';
});

/* ---- controls -------------------------------------------------------------------------------- */
document.getElementById('btn-pause').addEventListener('click', () =>
  send({ type: S.paused ? 'resume' : 'pause' }));
document.getElementById('btn-reset').addEventListener('click', () =>
  send({ type: 'reset', seed: Number(document.getElementById('reset-seed').value) || undefined }));
document.getElementById('btn-live').addEventListener('click', goLive);
document.getElementById('btn-teleport').addEventListener('click', (event) => {
  S.picking = !S.picking;
  canvasBox.classList.toggle('picking', S.picking);
  event.currentTarget.dataset.on = S.picking ? 'yes' : 'no';
});
document.getElementById('tgl-fall').addEventListener('change', (event) =>
  send({ type: 'pause_on_fall', on: event.currentTarget.checked }));
document.getElementById('tgl-heat').addEventListener('change', (event) => {
  S.heatmap = event.currentTarget.checked;
  if (!S.heatmap) clearHeatmap();
});
document.getElementById('tgl-plan').addEventListener('change', (event) => {
  S.showPlan = event.currentTarget.checked;
  if (!S.showPlan) ring.hideShadow('plan');
});
document.getElementById('tgl-trail').addEventListener('change', (event) => {
  S.showTrail = event.currentTarget.checked;
});
document.getElementById('btn-script-run').addEventListener('click', startScript);
document.getElementById('btn-script-stop').addEventListener('click', stopScript);
document.getElementById('btn-script-save').addEventListener('click', () => {
  localStorage.setItem('sparring-script', document.getElementById('script').value);
  document.getElementById('script-note').textContent = 'saved to this browser';
});
document.getElementById('btn-script-load').addEventListener('click', () => {
  const saved = localStorage.getItem('sparring-script');
  if (saved) document.getElementById('script').value = saved;
});

const scrubber = document.getElementById('scrubber');
scrubber.addEventListener('input', () => enterScrub(Number(scrubber.value)));

document.addEventListener('keydown', (raw) => {
  if (raw.target.tagName === 'INPUT' || raw.target.tagName === 'TEXTAREA') return;
  if (raw.repeat) return;
  const key = raw.key.length === 1 ? raw.key.toLowerCase() : raw.key;

  if (DRIVE[key]) { S.held.add(key); raw.preventDefault(); return; }

  const index = Number(key) - 1;
  if (index >= 0 && index < S.slots.length && key >= '1' && key <= '6') {
    S.staged = S.slots[index];
    send({ type: 'stage', slot: S.staged });
    S.slots.forEach((slot) =>
      document.getElementById(`slot-${slot}`)?.classList.toggle('staged', slot === S.staged));
    raw.preventDefault();
    return;
  }
  switch (key) {
    case ' ': commit(); raw.preventDefault(); break;
    case 'q':
      S.staged = null;
      S.offset = { x: 0, y: 0 };
      send({ type: 'clear' });
      S.slots.forEach((slot) => document.getElementById(`slot-${slot}`)?.classList.remove('staged'));
      break;
    case 'p': send({ type: S.paused ? 'resume' : 'pause' }); break;
    case 'r': send({ type: 'reset', seed: Number(document.getElementById('reset-seed').value) || undefined }); break;
    case 'ArrowLeft':
    case 'ArrowRight': {
      const step = (key === 'ArrowLeft' ? -1 : 1) * (raw.shiftKey ? 50 : 1);
      const target = Math.max(Number(scrubber.min),
        Math.min(Number(scrubber.max), Number(scrubber.value) + step));
      scrubber.value = target;
      enterScrub(target);
      raw.preventDefault();
      break;
    }
    default: break;
  }
});

document.addEventListener('keyup', (raw) => {
  const key = raw.key.length === 1 ? raw.key.toLowerCase() : raw.key;
  S.held.delete(key);
});
window.addEventListener('blur', () => S.held.clear());

/* ---- boot ------------------------------------------------------------------------------------ */
/* A debug bench must not fail invisibly: an uncaught JS error becomes the status line, not a
   console entry nobody has open. */
window.addEventListener('error', (event) => {
  setStatus('bad', `client error: ${event.message} (${event.filename?.split('/').pop()}:${event.lineno})`);
});
window.addEventListener('unhandledrejection', (event) => {
  setStatus('bad', `client error: ${event.reason?.message ?? event.reason}`);
});

let previous = performance.now();
function loop(now) {
  const dt = Math.min(0.1, (now - previous) / 1000);
  previous = now;
  driveShadow(dt);
  ring.render();
  requestAnimationFrame(loop);
}

ring.load().then((description) => {
  S.description = description;
  S.ringHalf = (description.arena?.ring_size ?? 4.9) / 2;
  S.standHeight = description.arena?.start_height ?? 0.793;
  S.bluePelvis = description.bodies.indexOf('blue_pelvis');
  buildHeatBodies(description);
  goLive();
  connect();
  pollSeries();
  requestAnimationFrame(loop);
}).catch((error) => {
  setStatus('bad', `could not build the ring: ${error.message}`);
  throw error;
});
