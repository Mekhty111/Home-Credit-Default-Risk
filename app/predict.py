import numpy as np
import pandas as pd
import joblib
import json

# Load artifacts once at startup
LR              = joblib.load('../models/logreg_woe.pkl')
LGBM            = joblib.load('../models/lgbm_model.pkl')
BINNING_PROCESS = joblib.load('../models/binning_process.pkl')

with open('../models/feature_names.json') as f:
    FEATURES = json.load(f)

# Scorecard scaling
PDO        = 20
BASE_SCORE = 600
BASE_ODDS  = 19
FACTOR     = PDO / np.log(2)
OFFSET     = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def get_risk_band(score: int) -> tuple:
    if score < 520:
        return 'Very High',  'REJECT',              'High default risk — application declined'
    elif score < 560:
        return 'High',       'MANUAL REVIEW',        'Elevated risk — requires manual underwriting'
    elif score < 600:
        return 'Medium',     'CONDITIONAL APPROVE',  'Moderate risk — approve with conditions'
    elif score < 640:
        return 'Low',        'APPROVE',              'Low risk — approve standard terms'
    else:
        return 'Very Low',   'APPROVE + BEST RATE',  'Minimal risk — approve with best interest rate'


def preprocess(input_dict: dict) -> pd.DataFrame:
    """Encode categoricals and apply WoE transformation."""
    df = pd.DataFrame([input_dict])

    # Fill missing features with NaN
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # Encode categoricals
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        df[col] = (df[col]
                   .fillna('Missing')
                   .astype('category')
                   .cat.codes
                   .astype(float))

    return BINNING_PROCESS.transform(df[FEATURES], metric='woe')


def predict(input_dict: dict) -> dict:
    """Full scoring pipeline for one applicant."""
    woe = preprocess(input_dict)

    lr_prob   = float(LR.predict_proba(woe)[0][1])
    lgbm_prob = float(LGBM.predict_proba(woe)[0][1])

    # Scorecard
    n      = len(FEATURES)
    offset = (OFFSET + FACTOR * LR.intercept_[0]) / n
    score  = sum(
        FACTOR * LR.coef_[0][i] * woe[feat].values[0] + offset
        for i, feat in enumerate(FEATURES)
    )
    score = int(round(score))

    band, action, interpretation = get_risk_band(score)

    return {
        'lr_probability'  : round(lr_prob, 4),
        'lgbm_probability': round(lgbm_prob, 4),
        'score'           : score,
        'risk_band'       : band,
        'action'          : action,
        'interpretation'  : interpretation,
    }