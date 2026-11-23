import {
  t, history, apiCached, hydrateBands, toast, copyText, fmtNum, fmtPct, fmtDateTime, relTime, escapeHtml,
  bandByApi, bandColor, decisionGroup, featureLabel, SCORE_MIN, SCORE_MAX,
} from '../core.js';
import { renderDecision, formatFeatureValue } from '../ui.js';

const $ = (sel) => document.querySelector(sel);
let filter = 'all';
let query = '';
let openId = null;
let lastFocus = null;

function matches(e) {
  if (filter !== 'all' && decisionGroup(e.result.action) !== filter) return false;
  if (!query) return true;
  const band = bandByApi(e.result.risk_band);
  const hay = [e.id, e.result.score, t(`band_${band.key}`), t(band.actionKey), e.result.action].join(' ').toLowerCase();
  return hay.includes(query);
}

function renderSummary(list) {
  $('#sum-total').textContent = fmtNum(list.length);
  const approved = list.filter((e) => decisionGroup(e.result.action) === 'approved').length;
  const rejected = list.filter((e) => decisionGroup(e.result.action) === 'rejected').length;
  $('#sum-approval').textContent = list.length ? fmtPct(approved / list.length, 0) : '—';
  $('#sum-rejected').textContent = fmtNum(rejected);
  $('#sum-avg').textContent = list.length ? fmtNum(list.reduce((s, e) => s + e.result.score, 0) / list.length) : '—';
}

function render() {
  const all = history.all();
  renderSummary(all);
  const rows = all.filter(matches);
  const empty = $('#empty');

  $('#rows').innerHTML = rows.map((e, i) => {
    const r = e.result;
    const band = bandByApi(r.risk_band);
    const pct = ((Math.min(SCORE_MAX, Math.max(SCORE_MIN, r.score)) - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)) * 100;
    return `
      <tr class="is-clickable" data-id="${escapeHtml(e.id)}" tabindex="0" style="animation:rise .4s var(--ease-out) ${Math.min(i, 15) * 25}ms both">
        <td class="id-cell">${escapeHtml(e.id)}</td>
        <td class="time-cell" title="${escapeHtml(fmtDateTime(e.ts))}">${relTime(e.ts)}</td>
        <td>
          <div class="score-cell">
            <span class="mono" style="font-weight:500;min-width:30px">${r.score}</span>
            <span class="score-cell__bar"><span class="score-cell__fill" style="width:${pct}%;background:${bandColor(band)}"></span></span>
          </div>
        </td>
        <td><span class="badge badge--${band.tone}">${t(`band_${band.key}_short`)}</span></td>
        <td style="font-weight:500">${t(band.actionKey)}</td>
        <td class="num mono">${fmtPct(r.lr_probability)}</td>
        <td class="num mono">${fmtPct(r.lgbm_probability)}</td>
        <td><svg class="icon row-chevron"><use href="#i-chevron-right"/></svg></td>
      </tr>`;
  }).join('');

  if (!all.length) {
    empty.innerHTML = `
      <div class="empty">
        <svg class="empty__art" viewBox="0 0 120 90" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="14" y="10" width="92" height="70" rx="10"/>
          <path d="M14 28h92M30 44h20M30 58h44M62 44h28" stroke-linecap="round"/>
        </svg>
        <div class="empty__title">${t('hi_empty_title')}</div>
        <div class="empty__text">${t('hi_empty_text')}</div>
        <a href="/ui/score" class="btn btn--primary"><span>${t('hero_cta_primary')}</span><svg class="icon arrow"><use href="#i-arrow-right"/></svg></a>
      </div>`;
  } else if (!rows.length) {
    empty.innerHTML = `<div class="empty" style="padding:40px"><div class="empty__text">${t('hi_noresults')}</div></div>`;
  } else {
    empty.innerHTML = '';
  }
  $('#export-btn').disabled = $('#clear-btn').disabled = !all.length;
}

// ── Drawer ─────────────────────────────────────────────────────────────────
function openDrawer(id, { animate = true } = {}) {
  const e = history.get(id);
  if (!e) return;
  openId = id;
  lastFocus = document.activeElement;
  $('#drawer-title').textContent = e.id;
  $('#drawer-time').textContent = `${fmtDateTime(e.ts)}${e.ms ? ` · ${e.ms} ms` : ''}`;
  renderDecision($('#drawer-decision'), e, { animate });

  const inputs = Object.entries(e.payload || {});
  $('#drawer-inputs').innerHTML = inputs.length
    ? inputs.map(([k, v]) => `<dt title="${escapeHtml(k)}">${escapeHtml(featureLabel(k))}</dt><dd>${escapeHtml(formatFeatureValue(k, v))}</dd>`).join('')
    : `<dt>${t('dr_no_inputs')}</dt><dd></dd>`;
  $('#drawer-open').href = `/ui/score?from=${encodeURIComponent(e.id)}`;
  $('#drawer-open').hidden = !inputs.length;

  $('#overlay').classList.add('is-open');
  $('#drawer').classList.add('is-open');
  document.body.style.overflow = 'hidden';
  setTimeout(() => $('#drawer-close').focus(), 50);
  const url = new URL(location.href);
  url.searchParams.set('id', id);
  window.history.replaceState(null, '', url);
}

function closeDrawer() {
  if (!openId) return;
  openId = null;
  $('#overlay').classList.remove('is-open');
  $('#drawer').classList.remove('is-open');
  document.body.style.overflow = '';
  window.history.replaceState(null, '', location.pathname);
  lastFocus?.focus?.();
}

// ── Modal ──────────────────────────────────────────────────────────────────
function openModal() {
  $('#modal-text').textContent = t('modal_clear_text', { n: history.all().length });
  $('#overlay').classList.add('is-open');
  $('#modal').classList.add('is-open');
  setTimeout(() => $('#modal-cancel').focus(), 50);
}
function closeModal() {
  $('#modal').classList.remove('is-open');
  if (!openId) $('#overlay').classList.remove('is-open');
}

// ── Export ─────────────────────────────────────────────────────────────────
function exportCsv() {
  const list = history.all();
  const head = ['id', 'timestamp', 'score', 'risk_band', 'action', 'lr_probability', 'lgbm_probability'];
  const lines = list.map((e) => [e.id, new Date(e.ts).toISOString(), e.result.score, e.result.risk_band, e.result.action, e.result.lr_probability, e.result.lgbm_probability]
    .map((v) => `"${String(v).replace(/"/g, '""')}"`).join(','));
  const blob = new Blob([[head.join(','), ...lines].join('\n')], { type: 'text/csv;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `credence-decisions-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  toast(t('toast_exported'));
}

function wire() {
  $('#filter').addEventListener('change', (e) => { filter = e.target.value; render(); });
  let timer;
  $('#search').addEventListener('input', (e) => {
    clearTimeout(timer);
    timer = setTimeout(() => { query = e.target.value.trim().toLowerCase(); render(); }, 120);
  });

  $('#rows').addEventListener('click', (e) => { const tr = e.target.closest('tr[data-id]'); if (tr) openDrawer(tr.dataset.id); });
  $('#rows').addEventListener('keydown', (e) => {
    const tr = e.target.closest('tr[data-id]');
    if (tr && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); openDrawer(tr.dataset.id); }
  });

  $('#drawer-close').addEventListener('click', closeDrawer);
  $('#overlay').addEventListener('click', () => { closeModal(); closeDrawer(); });
  $('#drawer-copy').addEventListener('click', () => {
    const e = history.get(openId);
    if (e) copyText(JSON.stringify({ id: e.id, input: e.payload, result: e.result }, null, 2));
  });

  $('#export-btn').addEventListener('click', exportCsv);
  $('#clear-btn').addEventListener('click', openModal);
  $('#modal-cancel').addEventListener('click', closeModal);
  $('#modal-confirm').addEventListener('click', () => {
    history.clear();
    closeModal();
    render();
    toast(t('toast_cleared'));
  });

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if ($('#modal').classList.contains('is-open')) closeModal();
    else closeDrawer();
  });

  document.addEventListener('langchange', () => {
    render();
    if (openId) openDrawer(openId, { animate: false });
  });
}

async function init() {
  wire();
  render();
  const id = new URLSearchParams(location.search).get('id');
  if (id) setTimeout(() => openDrawer(id), 250);
  try { hydrateBands(await apiCached('/stats')); } catch {}
}

init();
