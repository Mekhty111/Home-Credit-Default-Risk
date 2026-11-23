import { STRINGS, FEATURES, OPTIONS } from './i18n.js';

const LANGS = ['en', 'ru', 'az'];
const LANG_INDEX = { en: 0, ru: 1, az: 2 };
const KEYS = { lang: 'credence.lang', theme: 'credence.theme', history: 'credence.history.v2', draft: 'credence.draft' };

export const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

// ── Safe storage ───────────────────────────────────────────────────────────
export const store = {
  get(key, fallback = null) {
    try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
  },
  remove(key) { try { localStorage.removeItem(key); } catch {} },
};

// ── i18n ───────────────────────────────────────────────────────────────────
let lang = (() => {
  const saved = store.get(KEYS.lang);
  if (LANGS.includes(saved)) return saved;
  const nav = (navigator.language || 'en').slice(0, 2);
  return LANGS.includes(nav) ? nav : 'en';
})();

export const getLang = () => lang;
export const locale = () => ({ en: 'en-US', ru: 'ru-RU', az: 'az-Latn-AZ' }[lang]);

export function t(key, vars) {
  let s = STRINGS[lang]?.[key] ?? STRINGS.en[key] ?? key;
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, v);
  return s;
}

export function featureLabel(name) {
  const f = FEATURES[name];
  return f ? f[1 + LANG_INDEX[lang]] : name;
}
export const featureGroup = (name) => FEATURES[name]?.[0] ?? 'app';

export function optionLabel(field, value) {
  const o = OPTIONS[field]?.[value];
  return o ? o[LANG_INDEX[lang]] : value;
}

export function applyI18n(root = document) {
  root.querySelectorAll('[data-i18n]').forEach((el) => {
    el.textContent = t(el.dataset.i18n, el.dataset.i18nVars ? JSON.parse(el.dataset.i18nVars) : undefined);
  });
  root.querySelectorAll('[data-i18n-attr]').forEach((el) => {
    el.dataset.i18nAttr.split(';').forEach((pair) => {
      const [attr, key] = pair.split(':').map((s) => s.trim());
      if (attr && key) el.setAttribute(attr, t(key));
    });
  });
  document.documentElement.lang = lang;
}

export function setLang(next) {
  if (!LANGS.includes(next) || next === lang) return;
  lang = next;
  store.set(KEYS.lang, next);
  applyI18n();
  document.dispatchEvent(new CustomEvent('langchange', { detail: next }));
}

// ── Theme ──────────────────────────────────────────────────────────────────
export const getTheme = () => document.documentElement.getAttribute('data-theme') || 'light';

export function setTheme(next, { animate = true } = {}) {
  const apply = () => {
    document.documentElement.setAttribute('data-theme', next);
    store.set(KEYS.theme, next);
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', next === 'dark' ? '#0b0c10' : '#f4f3ef');
    document.dispatchEvent(new CustomEvent('themechange', { detail: next }));
  };
  if (animate && document.startViewTransition && !reducedMotion) document.startViewTransition(apply);
  else apply();
}

export const cssVar = (name, el = document.documentElement) => getComputedStyle(el).getPropertyValue(name).trim();

// ── Formatting ─────────────────────────────────────────────────────────────
export const fmtNum = (n, digits = 0) =>
  n === null || n === undefined || Number.isNaN(n) ? '—'
    : new Intl.NumberFormat(locale(), { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(n);

export const fmtPct = (p, digits = 1) =>
  p === null || p === undefined || Number.isNaN(p) ? '—' : `${fmtNum(p * 100, digits)}%`;

export const fmtCompact = (n) => new Intl.NumberFormat(locale(), { notation: 'compact', maximumFractionDigits: 1 }).format(n);

export function relTime(ts) {
  const diff = Math.max(0, Date.now() - ts) / 1000;
  if (diff < 45) return t('just_now');
  if (diff < 3600) return t('min_ago', { n: Math.round(diff / 60) });
  if (diff < 86400) return t('h_ago', { n: Math.round(diff / 3600) });
  if (diff < 86400 * 7) return t('d_ago', { n: Math.round(diff / 86400) });
  return new Date(ts).toLocaleDateString(locale(), { day: 'numeric', month: 'short' });
}

export const fmtDateTime = (ts) =>
  new Date(ts).toLocaleString(locale(), { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });

export const escapeHtml = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// ── Risk bands (mirror app/predict.py RISK_BANDS) ──────────────────────────
export const SCORE_MIN = 440;
export const SCORE_MAX = 620;

export const BANDS = [
  { key: 'very_high', api: 'Very High', action: 'REJECT',              actionKey: 'act_reject',       tone: 'bad',  color: '--band-vh', from: SCORE_MIN, to: 500, dr: 0.2228, share: 0.1768 },
  { key: 'high',      api: 'High',      action: 'MANUAL REVIEW',       actionKey: 'act_review',       tone: 'warn', color: '--band-h',  from: 500, to: 520, dr: 0.0991, share: 0.2120 },
  { key: 'medium',    api: 'Medium',    action: 'CONDITIONAL APPROVE', actionKey: 'act_conditional',  tone: 'warn', color: '--band-m',  from: 520, to: 540, dr: 0.0489, share: 0.2636 },
  { key: 'low',       api: 'Low',       action: 'APPROVE',             actionKey: 'act_approve',      tone: 'good', color: '--band-l',  from: 540, to: 560, dr: 0.0258, share: 0.2134 },
  { key: 'very_low',  api: 'Very Low',  action: 'APPROVE + BEST RATE', actionKey: 'act_approve_best', tone: 'good', color: '--band-vl', from: 560, to: SCORE_MAX, dr: 0.0142, share: 0.1341 },
];

export const bandByApi = (name) => BANDS.find((b) => b.api === name) || BANDS[0];
export const bandByScore = (score) => BANDS.find((b) => score < b.to) || BANDS[BANDS.length - 1];
export const bandColor = (band) => `var(${band.color})`;
export const decisionGroup = (action) =>
  action?.startsWith('APPROVE') || action === 'CONDITIONAL APPROVE' ? 'approved' : action === 'MANUAL REVIEW' ? 'review' : 'rejected';
export const actionIcon = (tone) => (tone === 'good' ? 'i-check' : tone === 'warn' ? 'i-eye' : 'i-x');

export function hydrateBands(stats) {
  stats?.bands?.forEach((s) => {
    const b = BANDS.find((x) => x.api === s.band);
    if (b) { b.dr = s.default_rate; b.share = s.share; }
  });
}

// ── API ────────────────────────────────────────────────────────────────────
const cache = new Map();
export async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = Array.isArray(body.detail) ? body.detail.map((d) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join('; ') : body.detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}
export function apiCached(path) {
  if (!cache.has(path)) cache.set(path, api(path).catch((e) => { cache.delete(path); throw e; }));
  return cache.get(path);
}

// ── Decision history ───────────────────────────────────────────────────────
export const history = {
  all: () => store.get(KEYS.history, []),
  add(entry) {
    const list = history.all();
    list.unshift(entry);
    store.set(KEYS.history, list.slice(0, 100));
    updateNavCount();
  },
  get: (id) => history.all().find((e) => e.id === id),
  clear() { store.remove(KEYS.history); updateNavCount(); },
};
export const draft = {
  get: () => store.get(KEYS.draft),
  set: (v) => store.set(KEYS.draft, v),
  clear: () => store.remove(KEYS.draft),
};

export function newId() {
  const s = (Date.now().toString(36) + Math.random().toString(36).slice(2, 6)).toUpperCase();
  return `APP-${s.slice(-6)}`;
}

function updateNavCount() {
  const el = document.getElementById('nav-history-count');
  if (!el) return;
  const n = history.all().length;
  el.textContent = n > 99 ? '99+' : n;
  el.hidden = n === 0;
}

// ── Toasts ─────────────────────────────────────────────────────────────────
export function toast(message, tone = 'good') {
  const box = document.getElementById('toasts');
  if (!box) return;
  const el = document.createElement('div');
  el.className = `toast toast--${tone}`;
  el.setAttribute('role', tone === 'bad' ? 'alert' : 'status');
  el.innerHTML = `<span class="toast__icon"><svg class="icon"><use href="#${tone === 'bad' ? 'i-alert' : 'i-check'}"/></svg></span><span></span>`;
  el.lastElementChild.textContent = message;
  box.appendChild(el);
  setTimeout(() => {
    el.classList.add('is-leaving');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, tone === 'bad' ? 5200 : 3200);
}

export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast(t('toast_copied')); }
  catch { toast(t('toast_api_error'), 'bad'); }
}

// ── Motion helpers ─────────────────────────────────────────────────────────
export function countUp(el, to, { duration = 1200, digits = 0, from = 0, format } = {}) {
  if (!el) return;
  const render = (v) => { el.textContent = format ? format(v) : fmtNum(v, digits); };
  if (reducedMotion) return render(to);
  const start = performance.now();
  const tick = (now) => {
    const p = Math.min(1, (now - start) / duration);
    const e = 1 - Math.pow(1 - p, 4);
    render(from + (to - from) * e);
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

let revealObserver;
export function observeReveals(root = document) {
  const els = root.querySelectorAll('[data-reveal]:not(.is-in)');
  if (!('IntersectionObserver' in window) || reducedMotion) { els.forEach((el) => el.classList.add('is-in')); return; }
  revealObserver ??= new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-in');
      entry.target.dispatchEvent(new CustomEvent('reveal'));
      revealObserver.unobserve(entry.target);
    });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
  els.forEach((el) => revealObserver.observe(el));
  // Safety net: background tabs may throttle IntersectionObserver callbacks
  setTimeout(() => els.forEach((el) => {
    if (!el.classList.contains('is-in') && el.getBoundingClientRect().top < innerHeight) {
      el.classList.add('is-in');
      el.dispatchEvent(new CustomEvent('reveal'));
      revealObserver.unobserve(el);
    }
  }), 1200);
}

export function onReveal(el, fn) {
  if (!el) return;
  if (el.classList.contains('is-in') || !el.hasAttribute('data-reveal')) fn();
  else el.addEventListener('reveal', fn, { once: true });
}

// ── Chrome: nav indicator, menus, theme, API status ────────────────────────
function initNav() {
  const nav = document.getElementById('nav');
  if (!nav) return;
  const indicator = nav.querySelector('.nav__indicator');
  const hover = nav.querySelector('.nav__hover');
  const active = nav.querySelector('.nav__link.is-active');
  const place = (el, link, instant) => {
    if (instant) el.style.transition = 'none';
    el.style.width = `${link.offsetWidth}px`;
    el.style.transform = `translateX(${link.offsetLeft}px)`;
    if (instant) { el.offsetWidth; el.style.transition = ''; }
  };
  const placeActive = () => active && place(indicator, active, true);
  placeActive();
  nav.classList.add('is-ready');
  let hovering = false;
  nav.querySelectorAll('.nav__link').forEach((link) => {
    link.addEventListener('mouseenter', () => {
      if (link === active) { hover.style.opacity = 0; return; }
      place(hover, link, !hovering);
      hover.style.opacity = 1;
      hovering = true;
    });
  });
  nav.addEventListener('mouseleave', () => { hover.style.opacity = 0; hovering = false; });
  document.addEventListener('langchange', () => requestAnimationFrame(placeActive));
  new ResizeObserver(placeActive).observe(nav);
}

function initMenus() {
  document.querySelectorAll('.menu').forEach((menu) => {
    const btn = menu.querySelector('button[aria-haspopup]');
    const close = () => { menu.classList.remove('is-open'); btn.setAttribute('aria-expanded', 'false'); };
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const open = !menu.classList.contains('is-open');
      document.querySelectorAll('.menu.is-open').forEach((m) => m.classList.remove('is-open'));
      menu.classList.toggle('is-open', open);
      btn.setAttribute('aria-expanded', String(open));
      if (open) menu.querySelector('[aria-checked="true"]')?.focus();
    });
    menu.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { close(); btn.focus(); }
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        const items = [...menu.querySelectorAll('[role^="menuitem"]')];
        const i = items.indexOf(document.activeElement);
        items[(i + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length]?.focus();
        e.preventDefault();
      }
    });
    document.addEventListener('click', close);
  });

  const langMenu = document.getElementById('lang-menu');
  const syncLang = () => {
    document.getElementById('lang-current').textContent = lang.toUpperCase();
    langMenu.querySelectorAll('[data-lang]').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.lang === lang)));
  };
  langMenu.querySelectorAll('[data-lang]').forEach((b) => b.addEventListener('click', () => { setLang(b.dataset.lang); syncLang(); }));
  syncLang();
}

function initTheme() {
  document.getElementById('theme-btn')?.addEventListener('click', () => setTheme(getTheme() === 'dark' ? 'light' : 'dark'));
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!store.get(KEYS.theme)) setTheme(e.matches ? 'dark' : 'light', { animate: false });
  });
}

function initTopbar() {
  const bar = document.getElementById('topbar');
  const onScroll = () => bar.classList.toggle('is-scrolled', window.scrollY > 8);
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
}

async function checkApi() {
  const el = document.getElementById('api-status');
  const label = el.querySelector('.status__label');
  const latency = document.getElementById('api-latency');
  const set = (state, key, ms) => {
    el.dataset.state = state;
    label.dataset.i18n = key;
    label.textContent = t(key);
    latency.textContent = ms ? `${ms} ms` : '';
  };
  try {
    const t0 = performance.now();
    await api('/health');
    set('online', 'status_online', Math.max(1, Math.round(performance.now() - t0)));
  } catch {
    set('offline', 'status_offline');
  }
}

export function bootChrome() {
  applyI18n();
  initNav();
  initMenus();
  initTheme();
  initTopbar();
  updateNavCount();
  observeReveals();
  checkApi();
  setInterval(checkApi, 30000);
}
