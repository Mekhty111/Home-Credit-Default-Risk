import { t, apiCached, hydrateBands, fmtNum, fmtPct, escapeHtml, featureLabel, featureGroup, onReveal, BANDS, SCORE_MIN, SCORE_MAX, bandColor } from '../core.js';

const $ = (sel) => document.querySelector(sel);
const GROUP_COLORS = { app: 'var(--accent)', bureau: '#e98a5a', prev: '#a46ee8', inst: 'var(--band-h)', card: '#2bb3c0', pos: 'var(--band-m)', prop: 'var(--ink-3)' };
const TOP = 15;

let features = [];
let expanded = false;
let query = '';

const strength = (iv) => (iv >= 0.3 ? 'iv_strong' : iv >= 0.1 ? 'iv_medium' : 'iv_weak');

function renderPolicy() {
  $('#policy').innerHTML = [...BANDS].reverse().map((b) => `
    <tr>
      <td><span style="display:inline-flex;align-items:center;gap:8px"><span class="band-dot" style="background:${bandColor(b)}"></span><span class="mono">${b.from === SCORE_MIN ? '< 500' : b.to === SCORE_MAX ? '≥ 560' : `${b.from}–${b.to - 1}`}</span></span></td>
      <td>${t(b.actionKey)}</td>
      <td class="num" style="color:${bandColor(b)}">${fmtPct(b.dr)}</td>
      <td class="num">${fmtPct(b.share, 0)}</td>
    </tr>`).join('');
}

function renderLegend() {
  const used = [...new Set(features.map((f) => featureGroup(f.feature)))];
  $('#iv-legend').innerHTML = used.map((g) =>
    `<span class="legend__item"><span class="band-dot" style="background:${GROUP_COLORS[g]}"></span>${t(`g_${g}`)}</span>`).join('');
}

function renderIv({ animate = true } = {}) {
  const list = $('#iv-list');
  const filtered = features.filter((f) => !query || `${f.feature} ${featureLabel(f.feature)}`.toLowerCase().includes(query));
  const rows = query || expanded ? filtered : filtered.slice(0, TOP);
  const max = Math.max(...features.map((f) => f.iv));

  list.innerHTML = rows.map((f) => `
    <div class="iv-row">
      <span class="iv-row__rank">${String(f.rank).padStart(2, '0')}</span>
      <div class="iv-row__name">
        <div class="iv-row__label">${escapeHtml(featureLabel(f.feature))}</div>
        <div class="iv-row__code">${escapeHtml(f.feature)} · ${t(strength(f.iv))}</div>
      </div>
      <div class="iv-row__bar"><div class="iv-row__fill" data-w="${((f.iv / max) * 100).toFixed(1)}" style="background:${GROUP_COLORS[featureGroup(f.feature)]}"></div></div>
      <span class="iv-row__val">${fmtNum(f.iv, 3)}</span>
    </div>`).join('') || `<div class="empty" style="padding:28px"><div class="empty__text">—</div></div>`;

  const grow = () => list.querySelectorAll('.iv-row__fill').forEach((el, i) => {
    el.style.transitionDelay = animate ? `${Math.min(i, 20) * 25}ms` : '0ms';
    el.style.width = `${el.dataset.w}%`;
  });
  requestAnimationFrame(() => requestAnimationFrame(grow));

  const more = $('#iv-more');
  more.hidden = Boolean(query) || features.length <= TOP;
  more.innerHTML = `${expanded ? t('show_top_features') : t('show_all_features', { n: features.length })}
    <svg class="icon" style="transform:rotate(${expanded ? 180 : 0}deg)"><use href="#i-chevron-down"/></svg>`;
}

async function init() {
  renderPolicy();
  try {
    const [feat, stats] = await Promise.all([apiCached('/features'), apiCached('/stats')]);
    hydrateBands(stats);
    renderPolicy();
    features = feat.iv
      .filter((f) => f.iv !== null)
      .sort((a, b) => b.iv - a.iv)
      .map((f, i) => ({ ...f, rank: i + 1 }));
    renderLegend();
    onReveal($('#iv-list').closest('[data-reveal]'), () => renderIv());
  } catch {
    $('#iv-list').innerHTML = `<div class="empty"><div class="empty__title">${t('status_offline')}</div></div>`;
  }

  $('#iv-more').addEventListener('click', () => { expanded = !expanded; renderIv({ animate: expanded }); });
  let timer;
  $('#iv-search').addEventListener('input', (e) => {
    clearTimeout(timer);
    timer = setTimeout(() => { query = e.target.value.trim().toLowerCase(); renderIv({ animate: false }); }, 120);
  });
  document.addEventListener('langchange', () => { renderPolicy(); renderLegend(); if (features.length) renderIv({ animate: false }); });
}

init();
