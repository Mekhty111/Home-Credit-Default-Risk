import {
  t, fmtNum, fmtPct, fmtCompact, featureLabel, optionLabel, escapeHtml, countUp, reducedMotion,
  BANDS, SCORE_MIN, SCORE_MAX, bandByApi, bandColor, actionIcon,
} from './core.js';

// ── Gauge ──────────────────────────────────────────────────────────────────
const CX = 160, CY = 160, R = 130;
const clampScore = (s) => Math.min(SCORE_MAX, Math.max(SCORE_MIN, s));
const angleOf = (s) => Math.PI - ((clampScore(s) - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)) * Math.PI;
const point = (s, r = R) => [CX + r * Math.cos(angleOf(s)), CY - r * Math.sin(angleOf(s))];
const arc = (a, b, r = R) => {
  const [x1, y1] = point(a, r), [x2, y2] = point(b, r);
  return `M${x1.toFixed(2)} ${y1.toFixed(2)} A${r} ${r} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
};

export function gaugeMarkup() {
  const gap = 0.9;
  const segs = BANDS.map((b, i) => {
    const a = b.from + (i ? gap : 0), z = b.to - (i < BANDS.length - 1 ? gap : 0);
    return `<path class="gauge-seg" data-band="${b.key}" d="${arc(a, z)}" stroke="${bandColor(b)}" stroke-width="16"/>`;
  }).join('');
  const ticks = [SCORE_MIN, 500, 520, 540, 560, SCORE_MAX].map((s) => {
    const [x, y] = point(s, R - 30);
    return `<text x="${x.toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="middle">${s}</text>`;
  }).join('');
  const [nx, ny] = point(SCORE_MIN);
  return `
    <svg viewBox="0 0 320 176" role="img" aria-hidden="true">
      <path class="gauge-track" d="${arc(SCORE_MIN, SCORE_MAX)}" stroke-width="16"/>
      ${segs}
      <path class="gauge-progress" d="${arc(SCORE_MIN, SCORE_MAX, R - 16)}" pathLength="100" stroke-width="3"
            stroke-dasharray="100" stroke-dashoffset="100" style="transition: stroke-dashoffset 1.4s var(--ease-out), stroke .4s"/>
      <g class="gauge-tick">${ticks}</g>
      <g class="gauge-needle" style="transform: rotate(0deg)">
        <circle cx="${nx}" cy="${ny}" r="13" fill="var(--surface)" stroke="var(--line-2)" stroke-width="1"/>
        <circle class="gauge-needle__dot" cx="${nx}" cy="${ny}" r="6.5" fill="var(--ink)"/>
      </g>
    </svg>`;
}

export function setGauge(root, score, { animate = true } = {}) {
  const band = BANDS.find((b) => clampScore(score) < b.to) || BANDS[BANDS.length - 1];
  const pct = ((clampScore(score) - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)) * 100;
  const needle = root.querySelector('.gauge-needle');
  const progress = root.querySelector('.gauge-progress');
  const apply = () => {
    needle.style.transform = `rotate(${(pct / 100) * 180}deg)`;
    progress.style.strokeDashoffset = String(100 - pct);
    progress.style.stroke = bandColor(band);
    root.querySelector('.gauge-needle__dot').style.fill = bandColor(band);
    root.querySelectorAll('.gauge-seg').forEach((s) => s.classList.toggle('is-active', s.dataset.band === band.key));
  };
  if (!animate || reducedMotion) {
    [needle, progress].forEach((el) => (el.style.transition = 'none'));
    apply();
    requestAnimationFrame(() => [needle, progress].forEach((el) => (el.style.transition = '')));
  } else {
    requestAnimationFrame(() => requestAnimationFrame(apply));
  }
}

export function resetGauge(root) {
  root.querySelector('.gauge-needle').style.transform = 'rotate(0deg)';
  root.querySelector('.gauge-progress').style.strokeDashoffset = '100';
  root.querySelector('.gauge-needle__dot').style.fill = '';
  root.querySelectorAll('.gauge-seg').forEach((s) => s.classList.remove('is-active'));
}

// ── Feature value formatting ───────────────────────────────────────────────
const RATE_FEATURES = new Set(['inst_late_payment_rate', 'prev_approval_rate', 'cc_util_mean', 'cc_util_max', 'prev_credit_util_mean']);
const FLAG_FEATURES = new Set(['FLAG_EMP_PHONE', 'FLAG_DOCUMENT_3', 'REG_CITY_NOT_WORK_CITY', 'REG_CITY_NOT_LIVE_CITY']);

export function formatFeatureValue(name, value) {
  if (value === null || value === undefined || value === '') return t('not_provided');
  if (typeof value === 'string') return optionLabel(name, value);
  if (name === 'DAYS_EMPLOYED' && value > 0) return t('f_not_employed').toLowerCase();
  if (name.startsWith('DAYS_')) return `${fmtNum(Math.abs(value) / 365.25, 1)} ${t('unit_years')}`;
  if (name.startsWith('EXT_SOURCE')) return fmtNum(value, 2);
  if (RATE_FEATURES.has(name)) return fmtPct(value, 0);
  if (FLAG_FEATURES.has(name)) return value ? '✓' : '—';
  if (Math.abs(value) >= 10000) return fmtCompact(value);
  return fmtNum(value, Number.isInteger(value) ? 0 : 2);
}

// ── Factors ────────────────────────────────────────────────────────────────
export function renderFactors(container, contributions = [], { limit = 8 } = {}) {
  const ranked = [...contributions].filter((c) => Math.abs(c.impact) >= 0.05).sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
  const max = Math.max(1, ...ranked.map((c) => Math.abs(c.impact)));
  let expanded = false;

  const draw = () => {
    const rows = expanded ? ranked : ranked.slice(0, limit);
    container.innerHTML = `
      <div class="factors">
        ${rows.map((c, i) => {
          const w = (Math.abs(c.impact) / max) * 50;
          const pos = c.impact > 0;
          return `
            <div class="factor" style="animation-delay:${Math.min(i, 12) * 40}ms" title="${escapeHtml(c.feature)}">
              <div class="factor__name">
                ${escapeHtml(featureLabel(c.feature))}
                <div class="factor__value">${escapeHtml(formatFeatureValue(c.feature, c.value))}</div>
              </div>
              <div class="factor__bar"><span class="factor__fill factor__fill--${pos ? 'pos' : 'neg'}" data-w="${w.toFixed(2)}"></span></div>
              <div class="factor__pts ${pos ? 'c-good' : 'c-bad'}">${pos ? '+' : '−'}${fmtNum(Math.abs(c.impact), 1)}</div>
            </div>`;
        }).join('')}
      </div>
      ${ranked.length > limit ? `
        <button class="btn btn--ghost btn--sm factors__more" type="button">
          ${expanded ? t('show_less') : t('show_all_factors', { n: ranked.length })}
          <svg class="icon" style="transform:rotate(${expanded ? 180 : 0}deg)"><use href="#i-chevron-down"/></svg>
        </button>` : ''}`;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      container.querySelectorAll('.factor__fill').forEach((el) => (el.style.width = `${el.dataset.w}%`));
    }));
    container.querySelector('.factors__more')?.addEventListener('click', () => { expanded = !expanded; draw(); });
  };
  draw();
}

// ── Decision card ──────────────────────────────────────────────────────────
export function renderDecision(container, entry, { animate = true } = {}) {
  const r = entry.result;
  const band = bandByApi(r.risk_band);
  const color = bandColor(band);
  const diverge = Math.abs(r.lr_probability - r.lgbm_probability);

  container.innerHTML = `
    <div class="decision__head">
      <span class="badge badge--${band.tone}">${t(`band_${band.key}`)}</span>
      <span class="decision__ref">${escapeHtml(entry.id)}</span>
    </div>
    <div class="decision__gauge">
      ${gaugeMarkup()}
      <div class="gauge-center">
        <div class="gauge-center__score" data-score>${animate ? SCORE_MIN : r.score}</div>
        <div class="gauge-center__label">${t('score_label')}</div>
      </div>
    </div>
    <div class="decision__verdict">
      <span class="decision__action" style="color:${color};background:color-mix(in srgb, ${color} 13%, transparent)">
        <svg class="icon"><use href="#${actionIcon(band.tone)}"/></svg>${t(band.actionKey)}
      </span>
      <p class="decision__interp">${t(`interp_${band.key}`)}</p>
    </div>
    <div class="decision__metrics">
      ${[['m_lr', r.lr_probability], ['m_lgbm', r.lgbm_probability]].map(([k, p]) => `
        <div class="decision__metric">
          <div class="decision__metric-label">${t(k)}</div>
          <div class="decision__metric-value" data-pd="${p}">${fmtPct(animate ? 0 : p)}</div>
          <div class="bar"><span class="bar__fill" data-w="${(p * 100).toFixed(1)}" style="background:${color}"></span></div>
        </div>`).join('')}
    </div>
    <div class="decision__section">
      <div class="decision__section-title">
        <span>${t('band_dr_label')}</span>
        <span class="mono" style="color:${color}">${fmtPct(band.dr)}</span>
      </div>
      <div class="agree">
        <svg class="icon ${diverge > 0.15 ? 'c-warn' : 'c-good'}"><use href="#${diverge > 0.15 ? 'i-alert' : 'i-check'}"/></svg>
        <span>${diverge > 0.15 ? t('models_diverge', { x: fmtNum(diverge * 100, 0) }) : t('models_agree')}</span>
      </div>
    </div>
    ${r.contributions?.length ? `
      <div class="decision__section">
        <div class="decision__section-title"><span>${t('factors_title')}</span><span class="muted">${t('factors_sub')}</span></div>
        <div data-factors></div>
      </div>` : ''}`;

  const gauge = container.querySelector('.decision__gauge');
  setGauge(gauge, r.score, { animate });
  if (animate) {
    countUp(container.querySelector('[data-score]'), r.score, { duration: 1400, from: SCORE_MIN });
    container.querySelectorAll('[data-pd]').forEach((el) => countUp(el, Number(el.dataset.pd), { duration: 1100, format: (v) => fmtPct(v) }));
  }
  requestAnimationFrame(() => requestAnimationFrame(() => {
    container.querySelectorAll('.bar__fill[data-w]').forEach((el) => (el.style.width = `${el.dataset.w}%`));
  }));
  const factors = container.querySelector('[data-factors]');
  if (factors) renderFactors(factors, r.contributions);
}
