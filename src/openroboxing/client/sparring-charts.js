/* The sparring bench's trace charts: strip charts on one shared x-axis of session ticks.
 *
 * Five strips — phase bands, tracking error, distance to target, root heights, step cost — drawn on
 * one canvas so the shared cursor and the shared hover cannot drift between them. Data arrives from
 * `GET /api/series` (already downsampled server-side); this file only draws.
 *
 * Colour discipline (dataviz): series colours follow the entity (the red/blue seats keep the
 * product's seat colours; the five phase colours are validated against the ink surface and defined
 * once, in sparring.css). Text wears text tokens, never the series colour.
 */

'use strict';

const PHASE_NAMES = ['OPENING', 'WAITING', 'APPROACH', 'DWELL', 'HOLD'];

/* Read a CSS custom property off <body>, with a fallback so a missing token sheet degrades to
   something visible rather than to invisible black-on-black. */
function cssVar(name, fallback) {
  const value = getComputedStyle(document.body).getPropertyValue(name).trim();
  return value || fallback;
}

function palette() {
  return {
    phases: [
      cssVar('--phase-opening', '#93AC9E'),
      cssVar('--phase-waiting', '#3FA9F5'),
      cssVar('--phase-approach', '#BC7B4C'),
      cssVar('--phase-dwell', '#4FD1A0'),
      cssVar('--phase-hold', '#B7C9BE'),
    ],
    red: cssVar('--seat-red', '#ff6b6b'),
    blue: cssVar('--seat-blue', '#6bb6ff'),
    accent: cssVar('--accent', '#4FD1A0'),
    clay: cssVar('--secondary', '#DDAE86'),
    info: cssVar('--phase-waiting', '#3FA9F5'),
    muted: cssVar('--text-muted', '#6E8C7E'),
    grid: 'rgba(234, 242, 237, 0.07)',
    cursor: cssVar('--text-secondary', '#B7C9BE'),
  };
}

/* One strip's spec: which series it draws and how its y-domain is found. */
const STRIPS = [
  { id: 'phase', label: 'phase', height: 1 },
  { id: 'error', label: 'tracking err · rad', height: 3, series: ['err_mean', 'err_max'] },
  /* Body and plan against the same placement. The pair is the diagnosis of an approach: the
     kinematic plan arrives every time, the body under physics is what has to get there. */
  { id: 'dist', label: 'dist to target · m', height: 3, series: ['dist', 'dist_plan'] },
  { id: 'root', label: 'root height · m', height: 2, series: ['root_h_red', 'root_h_blue'] },
  { id: 'step', label: 'step · ms', height: 2, series: ['step_ms'] },
];

/* What a series is called when a strip draws more than one and needs a legend. */
const SERIES_NAMES = {
  err_mean: 'mean', err_max: 'max',
  dist: 'body', dist_plan: 'plan',
  root_h_red: 'red', root_h_blue: 'blue',
};

export class ChartStack {
  constructor(container, { budgetMs = 20 } = {}) {
    this.container = container;
    this.budgetMs = budgetMs;
    this.canvas = document.createElement('canvas');
    this.container.append(this.canvas);
    this.tip = document.createElement('div');
    this.tip.className = 'chart-tip';
    this.tip.hidden = true;
    this.container.append(this.tip);

    this.data = null;
    this.cursor = null;      // tick the vertical cursor sits on, or null
    this.onSeek = null;      // cb(tick) — a click scrubs there
    this.arrivalRadius = null;  // live knob: the threshold drawn across the distance strip

    this.canvas.addEventListener('mousemove', (event) => this._hover(event));
    this.canvas.addEventListener('mouseleave', () => { this.tip.hidden = true; this._hoverTick = null; this.draw(); });
    this.canvas.addEventListener('click', (event) => {
      const tick = this._tickAt(event);
      if (tick !== null && this.onSeek) this.onSeek(tick);
    });
    new ResizeObserver(() => this.draw()).observe(this.container);
  }

  setData(series) { this.data = series; this.draw(); }
  setCursor(tick) { this.cursor = tick; this.draw(); }

  _layout() {
    const width = this.container.clientWidth || 600;
    const height = this.container.clientHeight || 240;
    const left = 8;
    const right = 8;
    const labelW = 118;                       // strip labels live in the left gutter
    const units = STRIPS.reduce((sum, s) => sum + s.height, 0);
    const gap = 6;
    const usable = height - gap * (STRIPS.length - 1) - 8;
    let y = 4;
    const rows = STRIPS.map((strip) => {
      const h = Math.max(12, Math.floor((usable * strip.height) / units));
      const row = { ...strip, y, h };
      y += h + gap;
      return row;
    });
    return { width, height, left: left + labelW, right, plotW: width - left - labelW - right, rows, labelW };
  }

  _x(tick, layout) {
    const [t0, t1] = this._domain();
    if (t1 <= t0) return layout.left;
    return layout.left + ((tick - t0) / (t1 - t0)) * layout.plotW;
  }

  _domain() {
    const ticks = this.data?.tick;
    if (!ticks || ticks.length === 0) return [0, 1];
    return [ticks[0], ticks[ticks.length - 1]];
  }

  _tickAt(event) {
    if (!this.data?.tick?.length) return null;
    const layout = this._layout();
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const [t0, t1] = this._domain();
    const frac = Math.min(1, Math.max(0, (x - layout.left) / layout.plotW));
    return Math.round(t0 + frac * (t1 - t0));
  }

  _nearestIndex(tick) {
    const ticks = this.data.tick;
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < ticks.length; i += 1) {
      const d = Math.abs(ticks[i] - tick);
      if (d < bestDist) { best = i; bestDist = d; }
    }
    return best;
  }

  _hover(event) {
    const tick = this._tickAt(event);
    if (tick === null) return;
    this._hoverTick = tick;
    const i = this._nearestIndex(tick);
    const d = this.data;
    const val = (arr, digits = 3) =>
      arr && Number.isFinite(arr[i]) ? arr[i].toFixed(digits) : '—';
    this.tip.innerHTML =
      `<b>t ${d.tick[i]}</b> · ${PHASE_NAMES[d.machine?.[i]] ?? '—'}`
      + ` · err <b>${val(d.err_mean)}</b>/${val(d.err_max)}`
      + ` · dist body <b>${val(d.dist, 2)}</b> plan <b>${val(d.dist_plan, 2)}</b>`
      + ` · step <b>${val(d.step_ms, 1)}</b> ms`;
    this.tip.hidden = false;
    const rect = this.canvas.getBoundingClientRect();
    this.tip.style.left = `${event.clientX - rect.left}px`;
    this.tip.style.top = '2px';
    this.draw();
  }

  draw() {
    const layout = this._layout();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = layout.width * dpr;
    this.canvas.height = layout.height * dpr;
    const ctx = this.canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, layout.width, layout.height);
    const colors = palette();
    const d = this.data;

    ctx.font = '9px "Red Hat Mono", monospace';
    for (const row of layout.rows) {
      /* Gutter label + recessive frame. The grid is structure you sense, not see. */
      ctx.fillStyle = colors.muted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(row.label.toUpperCase(), 8, row.y + 1);
      ctx.strokeStyle = colors.grid;
      ctx.lineWidth = 1;
      ctx.strokeRect(layout.left + 0.5, row.y + 0.5, layout.plotW - 1, row.h - 1);
    }
    if (!d || !d.tick || d.tick.length < 2) {
      ctx.fillStyle = colors.muted;
      ctx.fillText('NO RECORDING YET', layout.left + 8, layout.rows[0].y + 2);
      return;
    }

    const xs = d.tick.map((t) => this._x(t, layout));

    for (const row of layout.rows) {
      if (row.id === 'phase') this._drawPhases(ctx, row, xs, d, colors, layout);
      else this._drawLines(ctx, row, xs, d, colors, layout);
    }

    /* Replan events: one recessive tick mark each, on the error strip's top edge. */
    const errRow = layout.rows.find((r) => r.id === 'error');
    ctx.strokeStyle = colors.muted;
    for (const [tick] of d.replans || []) {
      const x = this._x(tick, layout);
      if (x < layout.left || x > layout.left + layout.plotW) continue;
      ctx.beginPath();
      ctx.moveTo(x, errRow.y);
      ctx.lineTo(x, errRow.y + 5);
      ctx.stroke();
    }

    /* The shared cursor: scrub position first, hover as the lighter twin. */
    for (const [tick, alpha] of [[this.cursor, 0.9], [this._hoverTick, 0.35]]) {
      if (tick === null || tick === undefined) continue;
      const x = this._x(tick, layout);
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = colors.cursor;
      ctx.beginPath();
      ctx.moveTo(x, layout.rows[0].y);
      const last = layout.rows[layout.rows.length - 1];
      ctx.lineTo(x, last.y + last.h);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  _drawPhases(ctx, row, xs, d, colors) {
    for (let i = 0; i < xs.length - 1; i += 1) {
      const phase = d.machine[i];
      ctx.fillStyle = colors.phases[phase] ?? colors.muted;
      ctx.globalAlpha = 0.75;
      ctx.fillRect(xs[i], row.y + 2, Math.max(1, xs[i + 1] - xs[i]), row.h - 4);
    }
    ctx.globalAlpha = 1;
  }

  _drawLines(ctx, row, xs, d, colors, layout) {
    /* Colour follows the entity: the body wears the red seat's colour and the plan wears the clay
       of the plan ghost, exactly as they appear in the ring above. */
    const styles = {
      err_mean: { color: colors.accent, width: 2 },
      err_max: { color: colors.muted, width: 1 },
      dist: { color: colors.red, width: 2 },
      dist_plan: { color: colors.clay, width: 2 },
      root_h_red: { color: colors.red, width: 2 },
      root_h_blue: { color: colors.blue, width: 2 },
      step_ms: { color: colors.info, width: 2 },
    };

    /* One y-domain per strip, over its own series, zero-based where that is the honest baseline. */
    let hi = -Infinity;
    for (const name of row.series) {
      for (const v of d[name] || []) if (Number.isFinite(v)) hi = Math.max(hi, v);
    }
    if (!Number.isFinite(hi) || hi <= 0) hi = 1;
    if (row.id === 'step') hi = Math.max(hi, this.budgetMs * 1.1);
    if (row.id === 'dist' && this.arrivalRadius) hi = Math.max(hi, this.arrivalRadius * 1.4);
    const yOf = (v) => row.y + row.h - 3 - (v / hi) * (row.h - 8);

    /* Reference lines: the tick budget, and the arrival radius — the threshold that decides
       whether a commit lands or waits out its timeout, so "the body never got inside" has to be
       readable at a glance rather than inferred from the numbers. */
    const rule = row.id === 'step' ? this.budgetMs
      : row.id === 'dist' ? this.arrivalRadius : null;
    if (rule) {
      const y = yOf(rule);
      ctx.strokeStyle = colors.grid;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(layout.left, y);
      ctx.lineTo(layout.left + layout.plotW, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = colors.muted;
      const label = row.id === 'step' ? `${this.budgetMs}` : `arrival ${rule.toFixed(2)}`;
      ctx.fillText(label, layout.left + layout.plotW - 60, y - 10);
    }

    for (const name of row.series) {
      const values = d[name];
      if (!values) continue;
      const style = styles[name] || { color: colors.accent, width: 2 };
      ctx.strokeStyle = style.color;
      ctx.lineWidth = style.width;
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < xs.length; i += 1) {
        const v = values[i];
        if (!Number.isFinite(v)) { pen = false; continue; }   // null = no target: a real gap
        const y = yOf(v);
        if (pen) ctx.lineTo(xs[i], y);
        else { ctx.moveTo(xs[i], y); pen = true; }
      }
      ctx.stroke();
    }
    ctx.lineWidth = 1;

    /* A strip with two lines names them: a coloured swatch carries the identity, the word stays in
       text ink, so the pair is never colour-alone. */
    if (row.series.length > 1) {
      let x = layout.left + 6;
      for (const name of row.series) {
        if (!d[name]) continue;
        const style = styles[name] || { color: colors.accent };
        ctx.fillStyle = style.color;
        ctx.fillRect(x, row.y + 6, 8, 2);
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        const text = SERIES_NAMES[name] ?? name;
        ctx.fillText(text, x + 12, row.y + 7);
        x += 20 + ctx.measureText(text).width;
      }
      ctx.textBaseline = 'top';
    }
  }
}
