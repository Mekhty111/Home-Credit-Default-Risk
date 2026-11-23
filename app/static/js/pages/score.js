import {
  t, api, apiCached, history, draft, newId, toast, copyText, fmtNum, fmtPct, optionLabel, hydrateBands, reducedMotion,
} from '../core.js';
import { OPTIONS, ORGANIZATIONS } from '../i18n.js';
import { gaugeMarkup, renderDecision, resetGauge } from '../ui.js';
import { PRESETS } from '../presets.js';

const $ = (sel) => document.querySelector(sel);
const form = $('#score-form');
const DAYS = 365.25;
const NOT_EMPLOYED = 365243; // sentinel used in the source data

// ── Field schema: form control ⇄ model feature ─────────────────────────────
const yearsAgo = { to: (v) => -Math.round(v * DAYS), from: (d) => Math.round((Math.abs(d) / DAYS) * 10) / 10 };
const pct = { to: (v) => v / 100, from: (r) => Math.round(r * 1000) / 10 };
const same = { to: (v) => v, from: (v) => v };

const NUMBERS = [
  { id: 'age',         feature: 'DAYS_BIRTH',             map: { to: (v) => -Math.round(v * DAYS), from: (d) => Math.floor(Math.abs(d) / DAYS) }, min: 18, max: 100, digits: 0 },
  { id: 'credit',      feature: 'AMT_CREDIT',             map: same, min: 0, max: 1e8, amount: true },
  { id: 'annuity',     feature: 'AMT_ANNUITY',            map: same, min: 0, max: 1e7, amount: true },
  { id: 'goods',       feature: 'AMT_GOODS_PRICE',        map: same, min: 0, max: 1e8, amount: true },
  { id: 'income',      feature: 'AMT_INCOME_TOTAL',       map: same, min: 0, max: 1e9, amount: true },
  { id: 'b_active',    feature: 'bureau_active_count',    map: same, min: 0, max: 100, digits: 0 },
  { id: 'b_closed',    feature: 'bureau_closed_count',    map: same, min: 0, max: 200, digits: 0 },
  { id: 'b_debt',      feature: 'bureau_debt_sum',        map: same, min: 0, max: 1e9, amount: true },
  { id: 'b_overdue',   feature: 'bureau_max_overdue',     map: same, min: 0, max: 1e9, amount: true },
  { id: 'p_refused',   feature: 'prev_refused_count',     map: same, min: 0, max: 100, digits: 0 },
  { id: 'p_approval',  feature: 'prev_approval_rate',     map: pct, min: 0, max: 100 },
  { id: 'i_late',      feature: 'inst_late_payment_rate', map: pct, min: 0, max: 100 },
  { id: 'i_days_late', feature: 'inst_days_late_max',     map: same, min: -60, max: 3000, digits: 0 },
  { id: 'id_years',    feature: 'DAYS_ID_PUBLISH',        map: yearsAgo, min: 0, max: 80 },
  { id: 'reg_years',   feature: 'DAYS_REGISTRATION',      map: yearsAgo, min: 0, max: 80 },
  { id: 'phone_years', feature: 'DAYS_LAST_PHONE_CHANGE', map: yearsAgo, min: 0, max: 30 },
  { id: 'region_pop',  feature: 'REGION_POPULATION_RELATIVE', map: same, min: 0, max: 0.1 },
  { id: 'car_age',     feature: 'OWN_CAR_AGE',            map: same, min: 0, max: 80, digits: 0 },
];
const SELECTS = [
  { id: 'education',    feature: 'NAME_EDUCATION_TYPE', values: Object.keys(OPTIONS.NAME_EDUCATION_TYPE) },
  { id: 'income_type',  feature: 'NAME_INCOME_TYPE',    values: Object.keys(OPTIONS.NAME_INCOME_TYPE) },
  { id: 'occupation',   feature: 'OCCUPATION_TYPE',     values: Object.keys(OPTIONS.OCCUPATION_TYPE) },
  { id: 'organization', feature: 'ORGANIZATION_TYPE',   values: ORGANIZATIONS.filter((o) => o !== 'XNA') },
];
const SWITCHES = [
  { id: 'work_outside', feature: 'REG_CITY_NOT_WORK_CITY', fallback: false },
  { id: 'live_outside', feature: 'REG_CITY_NOT_LIVE_CITY', fallback: false },
  { id: 'emp_phone',    feature: 'FLAG_EMP_PHONE',         fallback: true },
  { id: 'doc3',         feature: 'FLAG_DOCUMENT_3',        fallback: true },
];
const RADIOS = [
  { name: 'gender', feature: 'CODE_GENDER' },
  { name: 'region', feature: 'REGION_RATING_CLIENT_W_CITY', numeric: true },
];

let modelFeatures = [];
let ivByFeature = {};
let mode = 'form';
let current = null;      // last decision entry
let activePreset = null;

// ── Number parsing/formatting ──────────────────────────────────────────────
const parseNum = (raw) => {
  if (raw === null || raw === undefined) return null;
  const s = String(raw).replace(/[\s  ]/g, '').replace(/,(?=\d{1,2}$)/, '.').replace(/,/g, '');
  if (s === '' || s === '-') return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : NaN;
};
const display = (f, n) => (n === null || n === undefined ? '' : f.amount ? fmtNum(n) : fmtNum(n, f.digits ?? (Number.isInteger(n) ? 0 : String(n).split('.')[1]?.length ?? 0)).replace(/[  ]/g, ' '));

// ── Build select options (localized labels, canonical values) ──────────────
function buildSelects() {
  SELECTS.forEach((s) => {
    const el = $(`#${s.id}`);
    const value = el.value;
    el.innerHTML = `<option value="">${t('select_placeholder')}</option>` +
      s.values.map((v) => `<option value="${v}">${optionLabel(s.feature, v)}</option>`).join('');
    el.value = value;
  });
}

// ── Form ⇄ payload ─────────────────────────────────────────────────────────
function collect() {
  const p = {};
  for (const n of [1, 2, 3]) {
    if ($(`#ext${n}-on`).checked) p[`EXT_SOURCE_${n}`] = Number($(`#ext${n}`).value);
  }
  NUMBERS.forEach((f) => {
    const v = parseNum($(`#${f.id}`).value);
    if (v !== null && !Number.isNaN(v)) p[f.feature] = f.map.to(v);
  });
  if ($('#not_employed').checked) p.DAYS_EMPLOYED = NOT_EMPLOYED;
  else {
    const v = parseNum($('#employed').value);
    if (v !== null && !Number.isNaN(v)) p.DAYS_EMPLOYED = -Math.round(v * DAYS);
  }
  SELECTS.forEach((s) => { const v = $(`#${s.id}`).value; if (v) p[s.feature] = v; });
  RADIOS.forEach((r) => {
    const el = form.querySelector(`input[name="${r.name}"]:checked`);
    if (el) p[r.feature] = r.numeric ? Number(el.value) : el.value;
  });
  SWITCHES.forEach((s) => { p[s.feature] = $(`#${s.id}`).checked ? 1 : 0; });
  return p;
}

function applyPayload(p = {}) {
  for (const n of [1, 2, 3]) {
    const v = p[`EXT_SOURCE_${n}`];
    $(`#ext${n}-on`).checked = v !== undefined && v !== null;
    $(`#ext${n}`).value = v ?? 0.5;
  }
  NUMBERS.forEach((f) => {
    const v = p[f.feature];
    $(`#${f.id}`).value = v === undefined || v === null ? '' : display(f, f.map.from(v));
  });
  const emp = p.DAYS_EMPLOYED;
  $('#not_employed').checked = emp !== undefined && emp > 0;
  $('#employed').value = emp === undefined || emp === null || emp > 0 ? '' : fmtNum(Math.round((Math.abs(emp) / DAYS) * 10) / 10, 1).replace(/[,.]0$/, '');
  SELECTS.forEach((s) => { $(`#${s.id}`).value = s.values.includes(p[s.feature]) ? p[s.feature] : ''; });
  RADIOS.forEach((r) => {
    form.querySelectorAll(`input[name="${r.name}"]`).forEach((el) => { el.checked = String(p[r.feature]) === el.value; });
  });
  SWITCHES.forEach((s) => { $(`#${s.id}`).checked = p[s.feature] === undefined ? s.fallback : Boolean(Number(p[s.feature])); });
  form.querySelectorAll('.is-invalid').forEach((el) => el.classList.remove('is-invalid'));
  form.querySelectorAll('[data-error-for]').forEach((el) => (el.textContent = ''));
  syncAll();
}

// ── Validation ─────────────────────────────────────────────────────────────
function setError(id, msg) {
  const input = $(`#${id}`);
  input?.classList.toggle('is-invalid', Boolean(msg));
  const box = form.querySelector(`[data-error-for="${id}"]`);
  if (box) box.textContent = msg || '';
}

function validateField(f) {
  const v = parseNum($(`#${f.id}`).value);
  if (v === null) return setError(f.id, ''), true;
  if (Number.isNaN(v) || v < f.min || v > f.max) {
    setError(f.id, t('err_range', { a: fmtNum(f.min), b: fmtNum(f.max) }));
    return false;
  }
  setError(f.id, '');
  return true;
}

function validateEmployment() {
  if ($('#not_employed').checked) return setError('employed', ''), true;
  const v = parseNum($('#employed').value);
  const age = parseNum($('#age').value);
  if (v === null) return setError('employed', ''), true;
  if (Number.isNaN(v) || v < 0 || v > 60) return setError('employed', t('err_range', { a: 0, b: 60 })), false;
  if (age && v > age - 14) return setError('employed', t('err_employment')), false;
  setError('employed', '');
  return true;
}

function validateAll() {
  let ok = true;
  NUMBERS.forEach((f) => { ok = validateField(f) && ok; });
  ok = validateEmployment() && ok;
  return ok;
}

// ── Live UI: sliders, ratios, coverage ─────────────────────────────────────
function syncSlider(n) {
  const input = $(`#ext${n}`);
  const on = $(`#ext${n}-on`).checked;
  const v = Number(input.value);
  input.disabled = !on;
  input.closest('.bureau-score').classList.toggle('is-off', !on);
  $(`#ext${n}-val`).textContent = on ? fmtNum(v, 2) : '—';
  const range = input.closest('.range');
  range.style.setProperty('--pct', `${v * 100}%`);
  range.style.setProperty('--range-color', v < 0.3 ? 'var(--band-vh)' : v < 0.45 ? 'var(--band-h)' : v < 0.6 ? 'var(--band-m)' : 'var(--band-vl)');
}

function syncRatios() {
  const val = (id) => parseNum($(`#${id}`).value);
  const set = (id, ratio, warn, bad, asPct) => {
    const el = $(`#${id}`);
    const out = el.querySelector('.ratio__value');
    if (ratio === null || !Number.isFinite(ratio)) { out.textContent = '—'; el.removeAttribute('data-level'); return; }
    out.textContent = asPct ? fmtPct(ratio, 0) : `${fmtNum(ratio, 1)}×`;
    el.dataset.level = ratio >= bad ? 'bad' : ratio >= warn ? 'warn' : 'good';
  };
  const income = val('income'), annuity = val('annuity'), credit = val('credit'), goods = val('goods');
  set('r-pti', income && annuity !== null ? annuity / income : null, 0.25, 0.35, true);
  set('r-cti', income && credit !== null ? credit / income : null, 4, 6, false);
  set('r-ltv', goods && credit !== null ? credit / goods : null, 1.1, 1.3, true);
}

function currentPayload() {
  if (mode === 'json') {
    try { return JSON.parse($('#json-input').value || '{}'); } catch { return null; }
  }
  return collect();
}

function syncCoverage() {
  const p = currentPayload();
  if (!p || !modelFeatures.length) return;
  const provided = modelFeatures.filter((f) => p[f] !== undefined && p[f] !== null && p[f] !== '');
  const total = modelFeatures.reduce((s, f) => s + (ivByFeature[f] || 0), 0) || 1;
  const covered = provided.reduce((s, f) => s + (ivByFeature[f] || 0), 0);
  const share = covered / total;
  $('#coverage-ring').style.strokeDashoffset = String(100.53 * (1 - share));
  $('#coverage-pct').textContent = fmtPct(share, 0);
  $('#coverage-sub').textContent = t('coverage_sub', { n: provided.length, m: modelFeatures.length });
}

let draftTimer;
function syncAll() {
  [1, 2, 3].forEach(syncSlider);
  $('#employed').disabled = $('#not_employed').checked;
  syncRatios();
  syncCoverage();
  clearTimeout(draftTimer);
  draftTimer = setTimeout(() => draft.set(mode === 'json' ? currentPayload() : collect()), 400);
}

function setPreset(name) {
  activePreset = name;
  document.querySelectorAll('.preset').forEach((b) => b.classList.toggle('is-active', b.dataset.preset === name));
}

// ── Mode switch ────────────────────────────────────────────────────────────
function setMode(next) {
  if (next === mode) return;
  if (next === 'json') {
    $('#json-input').value = JSON.stringify(collect(), null, 2);
  } else {
    let parsed;
    try { parsed = JSON.parse($('#json-input').value || '{}'); }
    catch {
      toast(t('toast_invalid_json'), 'bad');
      $('#mode-json').checked = true;
      return;
    }
    applyPayload(parsed);
  }
  mode = next;
  $('#form-mode').hidden = next === 'json';
  $('#json-mode').hidden = next !== 'json';
  syncCoverage();
}

// ── Scoring ────────────────────────────────────────────────────────────────
const flip = $('#flip');
const steps = [...document.querySelectorAll('#steps .step')];
const wait = (ms) => new Promise((r) => setTimeout(r, reducedMotion ? 0 : ms));

function resetSteps() {
  steps.forEach((s) => { s.classList.remove('is-active', 'is-done'); s.querySelector('.step__time').textContent = ''; });
}

async function runSteps(request) {
  for (let i = 0; i < steps.length; i++) {
    steps[i].classList.add('is-active');
    await wait(i === steps.length - 1 ? 120 : 210);
    if (i === steps.length - 1) {
      try { await request; } catch { steps[i].classList.remove('is-active'); return; }
    }
    steps[i].classList.remove('is-active');
    steps[i].classList.add('is-done');
  }
}

let busy = false;
async function submit() {
  if (busy) return;
  const payload = currentPayload();
  if (!payload) return toast(t('toast_invalid_json'), 'bad');
  if (mode === 'form' && !validateAll()) {
    toast(t('toast_fix_errors'), 'bad');
    form.querySelector('.is-invalid')?.focus();
    return;
  }

  busy = true;
  const btn = $('#score-btn');
  btn.disabled = true;

  if (flip.classList.contains('is-flipped')) {
    flip.classList.remove('is-flipped');
    await wait(450);
  }
  resetSteps();
  if (matchMedia('(max-width: 960px)').matches) flip.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'center' });

  const t0 = performance.now();
  const request = api('/score', { method: 'POST', body: JSON.stringify(payload) });
  const latency = request.then(() => Math.round(performance.now() - t0)).catch(() => null);
  await runSteps(request);

  try {
    const result = await request;
    const ms = await latency;
    steps[steps.length - 1].querySelector('.step__time').textContent = `${ms} ms`;
    current = { id: newId(), ts: Date.now(), ms, payload, result };
    history.add(current);
    await wait(250);
    renderDecision($('#decision-body'), current);
    flip.classList.add('is-flipped');
    toast(`${t('toast_scored')} · ${current.id}`);
  } catch (e) {
    resetSteps();
    toast(`${t('toast_api_error')}: ${e.message}`, 'bad');
  } finally {
    busy = false;
    btn.disabled = false;
  }
}

function newApplication() {
  flip.classList.remove('is-flipped');
  resetSteps();
  setTimeout(() => { $('#decision-body').innerHTML = ''; }, 900);
  window.scrollTo({ top: 0, behavior: reducedMotion ? 'auto' : 'smooth' });
}

// ── Wiring ─────────────────────────────────────────────────────────────────
function wire() {
  form.addEventListener('submit', (e) => { e.preventDefault(); submit(); });
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); submit(); }
  });
  if (/Mac|iPhone|iPad/.test(navigator.platform)) $('#kbd-hint').textContent = '⌘ ↵';

  form.addEventListener('input', (e) => {
    if (e.target.name === 'mode') return;
    if (e.target.id !== 'json-input') setPreset(null);
    syncAll();
  });
  form.addEventListener('change', (e) => {
    if (e.target.name === 'mode') return setMode(e.target.value);
    syncAll();
  });

  NUMBERS.forEach((f) => {
    const el = $(`#${f.id}`);
    el.addEventListener('focus', () => {
      const v = parseNum(el.value);
      if (v !== null && !Number.isNaN(v)) el.value = String(v);
      requestAnimationFrame(() => el.select());
    });
    el.addEventListener('blur', () => {
      const v = parseNum(el.value);
      if (v !== null && !Number.isNaN(v)) el.value = display(f, v);
      validateField(f);
      if (f.id === 'age') validateEmployment();
    });
  });
  $('#employed').addEventListener('blur', validateEmployment);

  document.querySelectorAll('.preset').forEach((b) => b.addEventListener('click', () => {
    if (mode === 'json') $('#json-input').value = JSON.stringify(PRESETS[b.dataset.preset], null, 2);
    applyPayload(PRESETS[b.dataset.preset]);
    setPreset(b.dataset.preset);
    toast(t('toast_preset'));
  }));
  $('#reset-btn').addEventListener('click', () => {
    applyPayload({});
    if (mode === 'json') $('#json-input').value = '{}';
    setPreset(null);
    draft.clear();
  });

  const more = $('#more');
  more.querySelector('.collapsible__toggle').addEventListener('click', (e) => {
    const open = !more.classList.contains('is-open');
    more.classList.toggle('is-open', open);
    e.currentTarget.setAttribute('aria-expanded', String(open));
  });

  $('#copy-btn').addEventListener('click', () => current && copyText(JSON.stringify({ id: current.id, input: current.payload, result: current.result }, null, 2)));
  $('#new-btn').addEventListener('click', newApplication);

  document.addEventListener('langchange', () => {
    buildSelects();
    syncAll();
    if (current && flip.classList.contains('is-flipped')) renderDecision($('#decision-body'), current, { animate: false });
  });
}

// Sticky aside that pins by its bottom edge when taller than the viewport
function initStickyAside() {
  const aside = document.querySelector('.scorer__aside');
  const place = () => {
    const top = 80;
    const fit = window.innerHeight - aside.offsetHeight - 16;
    aside.style.top = `${Math.min(top, fit)}px`;
  };
  new ResizeObserver(place).observe(aside);
  window.addEventListener('resize', place);
}

async function init() {
  initStickyAside();
  $('#pending-gauge').insertAdjacentHTML('afterbegin', gaugeMarkup());
  resetGauge($('#pending-gauge'));
  buildSelects();
  wire();

  const params = new URLSearchParams(location.search);
  const from = params.get('from') && history.get(params.get('from'));
  const preset = params.get('preset');
  if (from?.payload) applyPayload(from.payload);
  else if (PRESETS[preset]) { applyPayload(PRESETS[preset]); setPreset(preset); }
  else if (draft.get()) applyPayload(draft.get());
  else { applyPayload(PRESETS.average); setPreset('average'); }
  if (params.toString()) window.history.replaceState(null, '', location.pathname);

  try {
    const [features, stats] = await Promise.all([apiCached('/features'), apiCached('/stats')]);
    modelFeatures = features.features;
    ivByFeature = Object.fromEntries(features.iv.map((r) => [r.feature, r.iv || 0]));
    hydrateBands(stats);
    syncCoverage();
  } catch { /* coverage stays empty when the API is down */ }
}

init();
