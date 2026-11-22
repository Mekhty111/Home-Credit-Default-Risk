import {
  t, api, apiCached, history, fmtNum, fmtPct, relTime, escapeHtml, countUp, onReveal, reducedMotion,
  BANDS, SCORE_MIN, SCORE_MAX, bandByApi, bandByScore, bandColor, hydrateBands, decisionGroup,
} from '../core.js';
import { PRESETS } from '../presets.js';

let stats = null;
let hero = null;
let heroResult = null;

// ── Hero: live warm-up score drives the floating cards and the 3D card ─────
async function initHero() {
  const canvas = document.getElementById('hero-canvas');
  const heroModule = import('../hero3d.js').catch(() => null);

  let result = null;
  try {
    const body = JSON.stringify(PRESETS.prime);
    await api('/score', { method: 'POST', body }); // warm-up: first call pays for lazy init
    const t0 = performance.now();
    result = await api('/score', { method: 'POST', body });
    document.getElementById('fc-latency').textContent = `${Math.max(1, Math.round(performance.now() - t0))} ms`;
  } catch { /* hero still renders with defaults */ }

  const band = result ? bandByApi(result.risk_band) : BANDS[4];
  const score = result?.score ?? 583;
  heroResult = { score, band };
  renderHeroCards();

  const mod = await heroModule;
  if (mod) hero = await mod.mountHero(canvas, { score, bandKey: band.key, bandLabel: t(`band_${band.key}_short`) });
}

function renderHeroCards() {
  if (!heroResult) return;
  const { score, band } = heroResult;
  const decision = document.getElementById('fc-decision');
  decision.dataset.i18n = band.actionKey;
  decision.textContent = t(band.actionKey);
  document.getElementById('fc-pd').textContent = fmtPct(band.dr);
  document.querySelector('.float-card--b .float-card__label').textContent = t('band_dr_short');
  const ring = document.getElementById('fc-ring');
  ring.style.stroke = bandColor(band);
  ring.style.strokeDashoffset = String(100.5 * (1 - (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)));
}

// ── Count-ups & benchmark bars ─────────────────────────────────────────────
function initCounters() {
  document.querySelectorAll('[data-count]').forEach((el) => {
    const host = el.closest('[data-reveal]') || el;
    onReveal(host, () => countUp(el, Number(el.dataset.count), { digits: Number(el.dataset.digits || 0), duration: 1600 }));
  });
  const kpis = document.querySelector('.kpis');
  onReveal(kpis, () => kpis.querySelectorAll('.kpi__bench-fill').forEach((el, i) => {
    setTimeout(() => (el.style.width = `${el.dataset.w}%`), 120 * i);
  }));
}

// ── Score distribution chart ───────────────────────────────────────────────
function niceMax(v) {
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  return Math.ceil(v / p / 2) * 2 * p;
}

function renderChart() {
  const box = document.getElementById('dist-chart');
  const hist = stats.histogram;
  const W = 720, H = 300, L = 44, R = 46, T = 14, B = 30;
  const iw = W - L - R, ih = H - T - B;
  const x = (s) => L + ((s - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)) * iw;
  const yMax = niceMax(Math.max(...hist.map((h) => h.count)));
  const drMax = 0.6;
  const y = (c) => T + ih - (c / yMax) * ih;
  const yr = (d) => T + ih - (d / drMax) * ih;

  const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const yy = T + ih - f * ih;
    return `<line class="gridline" x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}"/>
      <text x="${L - 8}" y="${yy + 4}" text-anchor="end">${fmtNum((yMax * f) / 1000, 0)}k</text>
      <text x="${W - R + 8}" y="${yy + 4}">${Math.round(drMax * f * 100)}%</text>`;
  }).join('');
  const xTicks = [440, 480, 500, 520, 540, 560, 600, 620].map((s) =>
    `<text x="${x(s)}" y="${H - 8}" text-anchor="middle">${s}</text>`).join('');
  const cuts = [500, 520, 540, 560].map((s) =>
    `<line class="cut-line" x1="${x(s)}" x2="${x(s)}" y1="${T}" y2="${T + ih}"/>`).join('');

  const bars = hist.map((h, i) => {
    const band = bandByScore(h.from);
    const bx = x(h.from) + 1.5, bw = x(h.to) - x(h.from) - 3;
    const by = y(h.count);
    return `<rect class="bar-rect" data-i="${i}" x="${bx}" y="${by}" width="${bw}" height="${T + ih - by}" rx="3"
      fill="${bandColor(band)}" style="transform:scaleY(0);transition:transform .9s var(--ease-out) ${i * 28}ms, opacity .2s"/>`;
  }).join('');

  const pts = hist.filter((h) => h.default_rate !== null).map((h) => [x((h.from + h.to) / 2), yr(h.default_rate)]);
  const line = pts.map(([px, py], i) => `${i ? 'L' : 'M'}${px.toFixed(1)} ${py.toFixed(1)}`).join(' ');
  const dots = pts.map(([px, py], i) => `<circle class="dr-dot" data-i="${i}" cx="${px}" cy="${py}" r="3" style="opacity:0;transition:opacity .3s ${600 + i * 40}ms"/>`).join('');
  const hits = hist.map((h, i) =>
    `<rect data-hit="${i}" x="${x(h.from)}" y="${T}" width="${x(h.to) - x(h.from)}" height="${ih}" fill="transparent"/>`).join('');

  box.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHtml(t('dist_title'))}">
      <g class="axis">${grid}${xTicks}</g>
      ${cuts}
      <g>${bars}</g>
      <path class="dr-line" d="${line}" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"
            style="transition:stroke-dashoffset 1.6s var(--ease-out) .4s"/>
      <g>${dots}</g>
      <g>${hits}</g>
    </svg>
    <div class="chart-tip" id="chart-tip"></div>`;

  const svg = box.querySelector('svg');
  const animateIn = () => {
    svg.querySelectorAll('.bar-rect').forEach((r) => (r.style.transform = 'scaleY(1)'));
    svg.querySelector('.dr-line').style.strokeDashoffset = '0';
    svg.querySelectorAll('.dr-dot').forEach((d) => (d.style.opacity = 1));
  };
  if (reducedMotion) animateIn();
  else onReveal(box.closest('[data-reveal]'), () => requestAnimationFrame(animateIn));

  const tip = box.querySelector('#chart-tip');
  const total = stats.n;
  svg.querySelectorAll('[data-hit]').forEach((hit) => {
    hit.addEventListener('pointerenter', () => {
      const i = Number(hit.dataset.hit);
      const h = hist[i];
      const band = bandByScore(h.from);
      box.classList.add('is-hovering');
      svg.querySelectorAll('.bar-rect').forEach((r) => r.classList.toggle('is-hover', Number(r.dataset.i) === i));
      tip.innerHTML = `
        <div class="chart-tip__title">${h.from}–${h.to - 1}</div>
        <div class="chart-tip__row"><span>${t('tt_applicants')}</span><span class="mono">${fmtNum(h.count)}</span></div>
        <div class="chart-tip__row"><span>${t('tt_share')}</span><span class="mono">${fmtPct(h.count / total)}</span></div>
        <div class="chart-tip__row"><span>${t('tt_dr')}</span><span class="mono">${fmtPct(h.default_rate)}</span></div>
        <div class="chart-tip__row" style="margin-top:6px;opacity:1"><span style="display:inline-flex;align-items:center;gap:6px"><span class="band-dot" style="background:${bandColor(band)}"></span>${t(`band_${band.key}_short`)}</span></div>`;
      const br = box.getBoundingClientRect();
      const hr = hit.getBoundingClientRect();
      let left = hr.left - br.left + hr.width / 2 + 12;
      if (left + 170 > br.width) left = hr.left - br.left - 170;
      tip.style.left = `${left}px`;
      tip.style.top = `${Math.max(0, (hr.top - br.top) + 10)}px`;
      tip.classList.add('is-on');
    });
  });
  svg.addEventListener('pointerleave', () => {
    box.classList.remove('is-hovering');
    tip.classList.remove('is-on');
  });

  document.getElementById('dist-sub').textContent = t('dist_sub', { n: fmtNum(stats.n) });
}

// ── Band ladder ────────────────────────────────────────────────────────────
function renderBands() {
  const el = document.getElementById('bands');
  el.innerHTML = [...BANDS].reverse().map((b) => `
    <div class="band-row">
      <span class="band-row__swatch" style="background:${bandColor(b)}"></span>
      <div>
        <div class="band-row__name">${t(`band_${b.key}`)}</div>
        <div class="band-row__meta"><span class="mono">${b.from === SCORE_MIN ? '< 500' : b.to === SCORE_MAX ? '≥ 560' : `${b.from}–${b.to - 1}`}</span> · ${t(b.actionKey)}</div>
      </div>
      <div class="band-row__stat">
        <div class="band-row__dr mono" style="color:${bandColor(b)}">${fmtPct(b.dr)}</div>
        <div class="band-row__share">${t('band_of_applicants', { x: fmtPct(b.share, 0) })}</div>
      </div>
    </div>`).join('');
}

// ── Session ────────────────────────────────────────────────────────────────
function renderSession() {
  const list = history.all();
  const total = document.getElementById('st-total');
  total.textContent = fmtNum(list.length);
  if (list.length) {
    const approved = list.filter((e) => decisionGroup(e.result.action) === 'approved').length;
    document.getElementById('st-approval').textContent = fmtPct(approved / list.length, 0);
    document.getElementById('st-avg').textContent = fmtNum(list.reduce((s, e) => s + e.result.score, 0) / list.length);
  }

  const recent = document.getElementById('recent');
  if (!list.length) {
    recent.innerHTML = `
      <div class="empty" style="padding:28px 12px 8px">
        <svg class="empty__art" viewBox="0 0 120 90" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="18" y="14" width="84" height="62" rx="10"/>
          <path d="M32 34h36M32 46h56M32 58h24" stroke-linecap="round"/>
          <circle cx="92" cy="22" r="12" fill="var(--surface)"/><path d="M92 16v12M86 22h12" stroke-linecap="round"/>
        </svg>
        <div class="empty__title">${t('recent_empty_title')}</div>
        <div class="empty__text">${t('recent_empty_text')}</div>
      </div>`;
    return;
  }
  recent.innerHTML = list.slice(0, 4).map((e) => {
    const band = bandByApi(e.result.risk_band);
    return `
      <a class="recent__item" href="/ui/history?id=${encodeURIComponent(e.id)}">
        <span class="recent__score" style="--band-color:${bandColor(band)}">${e.result.score}</span>
        <span>
          <div class="recent__title">${t(band.actionKey)}</div>
          <div class="recent__meta"><span class="mono">${escapeHtml(e.id)}</span> · ${relTime(e.ts)}</div>
        </span>
        <span class="badge badge--${band.tone}">${t(`band_${band.key}_short`)}</span>
      </a>`;
  }).join('');
}

async function init() {
  initCounters();
  initHero();
  renderSession();
  try {
    stats = await apiCached('/stats');
    hydrateBands(stats);
    document.getElementById('meta-validated').dataset.count = stats.n;
  } catch {
    document.getElementById('dist-chart').innerHTML = `<div class="empty"><div class="empty__title">${t('status_offline')}</div></div>`;
  }
  renderBands();
  if (stats) renderChart();
}

document.addEventListener('langchange', () => {
  renderBands();
  renderSession();
  renderHeroCards();
  if (stats) renderChart();
  if (hero && heroResult) hero.setCard({ score: heroResult.score, bandKey: heroResult.band.key, bandLabel: t(`band_${heroResult.band.key}_short`) });
});
document.addEventListener('themechange', () => {
  if (hero && heroResult) hero.setCard({ score: heroResult.score, bandKey: heroResult.band.key, bandLabel: t(`band_${heroResult.band.key}_short`) });
});

init();
