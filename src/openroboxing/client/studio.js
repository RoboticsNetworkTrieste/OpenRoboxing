/* The Pose Studio (S-T1), built to spec/ui-design-guide/ screen 03.
 *
 * Every rule lives on the server — joint limits come from the model, validation from
 * studio/pose_record.py. This file draws sliders and shows what came back. That is deliberate: a
 * browser that decided for itself what a legal pose was would drift from what a match accepts.
 *
 * The one rule this file *does* hold is a sequencing rule, not a pose rule: **save is blocked until
 * check passes**, and any edit invalidates the check. It cannot approve anything the server would
 * refuse; it can only refuse to ask.
 */

'use strict';

/* Joints are grouped the way an author thinks about them, not the way MuJoCo orders them. The legs
   open collapsed: a boxing pose is authored from the waist up, and eight open sliders you never
   touch are eight sliders in the way. */
const GROUPS = [
  ['left arm', (n) => n.startsWith('left_shoulder') || n.startsWith('left_elbow') || n.startsWith('left_wrist'), true],
  ['right arm', (n) => n.startsWith('right_shoulder') || n.startsWith('right_elbow') || n.startsWith('right_wrist'), true],
  ['waist', (n) => n.startsWith('waist'), true],
  ['left leg', (n) => n.startsWith('left_hip') || n.startsWith('left_knee') || n.startsWith('left_ankle'), false],
  ['right leg', (n) => n.startsWith('right_hip') || n.startsWith('right_knee') || n.startsWith('right_ankle'), false],
];

/* From spec/rates.md: a token is 4 generator frames at 30 Hz. */
const SECONDS_PER_TOKEN = 4 / 30;
const MIN_TOKENS = 6;
const MAX_TOKENS = 16;

const state = { limits: {}, defaults: {}, angles: {}, horizon: 8, checked: false };
let renderPending = false;
let renderQueued = false;

const view = document.getElementById('view');
const plateEmpty = document.getElementById('plate-empty');
const statusLine = document.getElementById('status');
const saveButton = document.getElementById('save');
const unsaved = document.getElementById('unsaved');

function say(text, tone) {
  statusLine.textContent = text;
  statusLine.dataset.tone = tone || 'info';
}

/* Any edit invalidates the last check: the pose the server approved is not this one any more. */
function dirty() {
  state.checked = false;
  saveButton.disabled = true;
  unsaved.classList.remove('hidden');
}

/* ---- talking to the server ---------------------------------------------------------------------- */
async function post(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  return response.json();
}

function payload() {
  return {
    name: document.getElementById('name').value.trim() || 'untitled',
    joint_angles: state.angles,
    horizon_tokens: state.horizon,
  };
}

/* Renders are coalesced: dragging a slider fires many changes and the server renders in ~30 ms.
   Queueing one more rather than all of them keeps the preview live without a backlog. */
async function refresh() {
  if (renderPending) { renderQueued = true; return; }
  renderPending = true;

  try {
    const result = await post('/api/render', { joint_angles: state.angles });
    if (result.ok) {
      view.src = result.png;
      plateEmpty.classList.add('hidden');
      document.getElementById('plate-caption').textContent =
        `${payload().name} · /api/render · 640 × 640`;
    } else {
      say(result.error, 'danger');
    }
  } catch (error) {
    say(`render failed: ${error}`, 'danger');
  } finally {
    renderPending = false;
    if (renderQueued) { renderQueued = false; refresh(); }
  }

  for (const side of ['left', 'right']) {
    post('/api/reach', { joint_angles: state.angles, side }).then((reach) => {
      if (!reach.ok) return;
      document.getElementById(`reach-${side}`).textContent =
        `${reach.forward_m >= 0 ? '+' : ''}${reach.forward_m.toFixed(2)} m fwd · `
        + `${reach.up_m.toFixed(2)} m up`;
    }).catch(() => {});
  }
}

/* ---- the sliders --------------------------------------------------------------------------------- */
function paintRow(row, name) {
  const limit = state.limits[name];
  const angle = state.angles[name];
  const span = limit.high - limit.low || 1;
  const fill = Math.max(0, Math.min(1, (angle - limit.low) / span));
  row.style.setProperty('--fill', `${(fill * 100).toFixed(1)}%`);
  row.querySelector('.value').textContent = Number(angle).toFixed(2);
  /* The model's own limits, so this can only fire when a loaded angle really is outside them. */
  row.classList.toggle('out', angle < limit.low || angle > limit.high);
}

function buildSlider(name) {
  const limit = state.limits[name];
  const row = document.createElement('div');
  row.className = 'row';
  row.dataset.joint = name;

  const head = document.createElement('div');
  head.className = 'row-head';
  const label = document.createElement('span');
  label.textContent = name.replace(/_joint$/, '').replace(/^(left|right)_/, '').replace(/_/g, ' ');
  const value = document.createElement('span');
  value.className = 'value';
  head.append(label, value);

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = limit.low;
  slider.max = limit.high;
  slider.step = 0.01;
  slider.value = state.angles[name];
  slider.setAttribute('aria-label', name);

  slider.addEventListener('input', () => {
    state.angles[name] = Number(slider.value);
    paintRow(row, name);
    dirty();
    refresh();
  });

  row.append(head, slider);
  paintRow(row, name);
  return row;
}

function buildEditor() {
  const container = document.getElementById('groups');
  container.innerHTML = '';
  const names = Object.keys(state.limits);

  GROUPS.forEach(([title, matches, open]) => {
    const members = names.filter(matches);
    if (!members.length) return;

    const group = document.createElement('section');
    group.className = 'group';

    const head = document.createElement('button');
    head.className = 'group-head';
    head.type = 'button';
    head.setAttribute('aria-expanded', String(open));
    head.innerHTML = `<span>${title}</span><span class="torc-hud-label push">${members.length}`
      + `${open ? '' : ' · collapsed'}</span>`;

    const body = document.createElement('div');
    body.className = 'group-body';
    body.append(...members.map(buildSlider));

    head.addEventListener('click', () => {
      const nowOpen = head.getAttribute('aria-expanded') !== 'true';
      head.setAttribute('aria-expanded', String(nowOpen));
      head.querySelector('.torc-hud-label').textContent =
        `${members.length}${nowOpen ? '' : ' · collapsed'}`;
    });

    group.append(head, body);
    container.append(group);
  });
}

function setAngles(angles) {
  state.angles = { ...angles };
  document.querySelectorAll('.row').forEach((row) => {
    const name = row.dataset.joint;
    row.querySelector('input').value = state.angles[name];
    paintRow(row, name);
  });
  dirty();
  refresh();
}

/* ---- the horizon ------------------------------------------------------------------------------------
   The pose's own length, in tokens. Sixteen cells because that is the clip's mask; the first five
   are drawn but never selectable, because a pose shorter than MIN_TOKENS is one the generator will
   refuse (spec/intent.md §"The horizon reaches the generator"). */
function buildHorizon() {
  const strip = document.getElementById('horizon-strip');
  strip.innerHTML = '';
  for (let tokens = 1; tokens <= MAX_TOKENS; tokens += 1) {
    const cell = document.createElement('button');
    cell.type = 'button';
    cell.disabled = tokens < MIN_TOKENS;
    cell.title = `${tokens} tokens · ${(tokens * SECONDS_PER_TOKEN).toFixed(2)} s`;
    cell.addEventListener('click', () => { state.horizon = tokens; paintHorizon(); dirty(); });
    strip.append(cell);
  }
  paintHorizon();
}

function paintHorizon() {
  document.getElementById('horizon-value').textContent = state.horizon;
  document.getElementById('horizon-seconds').textContent =
    `${(state.horizon * SECONDS_PER_TOKEN).toFixed(2)} s`;
  document.querySelectorAll('#horizon-strip button').forEach((cell, index) => {
    cell.classList.toggle('on', index + 1 <= state.horizon);
  });
}

/* ---- actions ------------------------------------------------------------------------------------- */
document.getElementById('reset').addEventListener('click', () => {
  setAngles(state.defaults);
  say('back to the default stance', 'info');
});

document.getElementById('check').addEventListener('click', async () => {
  const result = await post('/api/check', payload());
  state.checked = result.ok;
  saveButton.disabled = !result.ok;
  say(result.ok ? `valid — ${result.name} (${result.admission})` : result.error,
      result.ok ? 'ok' : 'warn');
});

saveButton.addEventListener('click', async () => {
  const result = await post('/api/save', payload());
  if (result.ok) {
    unsaved.classList.add('hidden');
    say(`saved ${result.path} — ${result.note}`, 'ok');
  } else {
    say(result.error, 'danger');
  }
});

document.getElementById('name').addEventListener('input', dirty);

/* ---- start --------------------------------------------------------------------------------------- */
document.getElementById('origin').textContent = location.host;

fetch('/api/joints')
  .then((response) => response.json())
  .then((data) => {
    state.limits = data.limits;
    state.defaults = data.defaults;
    state.angles = { ...data.defaults };
    buildEditor();
    buildHorizon();
    document.getElementById('joint-count').textContent =
      `g1 · 1.3 m · ${Object.keys(state.limits).length} joints`;
    say(`${Object.keys(state.limits).length} joints, limits from the robot model`, 'info');
    refresh();
  })
  .catch((error) => say(`could not load joints: ${error}`, 'danger'));
