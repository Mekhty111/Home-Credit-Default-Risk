import os
import json

import numpy as np
import pandas as pd
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')


def _model_path(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


LR              = joblib.load(_model_path('logreg_woe.pkl'))
LGBM            = joblib.load(_model_path('lgbm_model.pkl'))
BINNING_PROCESS = joblib.load(_model_path('binning_process.pkl'))

with open(_model_path('feature_names.json')) as f:
    FEATURES = json.load(f)

# Category vocabularies used at training time: pandas `.cat.codes` on the full
# training column, i.e. the sorted unique values (with NaN -> 'Missing').
with open(_model_path('category_codes.json'), encoding='utf-8') as f:
    CATEGORY_CODES = {col: {v: float(i) for i, v in enumerate(vals)}
                      for col, vals in json.load(f).items()}

# Все переменные которые знает BinningProcess
BP_VARS = BINNING_PROCESS.variable_names

PDO        = 20
BASE_SCORE = 600
BASE_ODDS  = 19
FACTOR     = PDO / np.log(2)
OFFSET     = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

# Score = OFFSET + FACTOR * ln(good/bad odds) = OFFSET - FACTOR * logit(PD),
# so a higher score always means a safer applicant.
N_FEATURES     = len(FEATURES)
POINTS_OFFSET  = (OFFSET - FACTOR * LR.intercept_[0]) / N_FEATURES
COEFS          = dict(zip(FEATURES, LR.coef_[0]))

# Cut-offs calibrated on the out-of-time sample (observed default rate):
#   <500: 22.3% · 500–519: 9.9% · 520–539: 4.9% · 540–559: 2.6% · 560+: 1.4%
RISK_BANDS = [
    (500,    'Very High', 'REJECT',              'High default risk — application declined'),
    (520,    'High',      'MANUAL REVIEW',       'Elevated risk — requires manual underwriting'),
    (540,    'Medium',    'CONDITIONAL APPROVE', 'Moderate risk — approve with conditions'),
    (560,    'Low',       'APPROVE',             'Low risk — approve standard terms'),
    (np.inf, 'Very Low',  'APPROVE + BEST RATE', 'Minimal risk — approve with best interest rate'),
]


def feature_points(feature: str, woe: float) -> float:
    """Scorecard points a single feature contributes for a given WoE value."""
    return -FACTOR * COEFS[feature] * woe + POINTS_OFFSET


def get_risk_band(score: int) -> tuple:
    for upper, band, action, interpretation in RISK_BANDS:
        if score < upper:
            return band, action, interpretation


def preprocess(input_dict: dict) -> pd.DataFrame:
    """Encode categoricals and apply WoE transformation."""
    df = pd.DataFrame([input_dict])

    # Добавляем ВСЕ переменные binning_process — отсутствующие → NaN
    for col in BP_VARS:
        if col not in df.columns:
            df[col] = np.nan

    # Encode categoricals with the training-time vocabulary; unseen -> NaN (missing bin)
    for col, codes in CATEGORY_CODES.items():
        value = df.at[0, col]
        value = 'Missing' if value is None or (isinstance(value, float) and np.isnan(value)) else value
        df[col] = codes.get(value, np.nan)

    # Передаём ВСЕ bp_vars, потом выбираем только FEATURES
    X = df[BP_VARS].apply(pd.to_numeric, errors='coerce').astype(float)
    woe_all = BINNING_PROCESS.transform(X, metric='woe')

    # Возвращаем только финальные фичи модели
    return woe_all[FEATURES]


def predict(input_dict: dict) -> dict:
    """Full scoring pipeline for one applicant."""
    woe = preprocess(input_dict)

    lr_prob   = float(LR.predict_proba(woe)[0][1])
    lgbm_prob = float(LGBM.predict_proba(woe)[0][1])

    contributions = []
    for feat in FEATURES:
        w = float(woe[feat].values[0])
        raw = input_dict.get(feat)
        contributions.append({
            'feature' : feat,
            'value'   : raw if raw is None or isinstance(raw, (str, int, float)) else str(raw),
            'provided': raw is not None,
            'woe'     : round(w, 4),
            # Points relative to a neutral (WoE = 0) applicant: >0 lowers risk
            'impact'  : round(float(-FACTOR * COEFS[feat] * w), 2),
        })

    score = int(round(sum(feature_points(f, float(woe[f].values[0])) for f in FEATURES)))
    band, action, interpretation = get_risk_band(score)

    return {
        'lr_probability'  : round(lr_prob, 4),
        'lgbm_probability': round(lgbm_prob, 4),
        'score'           : score,
        'risk_band'       : band,
        'action'          : action,
        'interpretation'  : interpretation,
        'contributions'   : sorted(contributions, key=lambda c: c['impact']),
    }
