# Home Credit Default Risk
**Probability of Default (PD) model** built on the [Home Credit Kaggle dataset](https://www.kaggle.com/c/home-credit-default-risk), following retail banking methodology end-to-end — from raw multi-table data to a deployable scorecard API.


[![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Jinja2](https://img.shields.io/badge/Jinja2-3.x-B41717?style=flat-square&logo=jinja)](https://jinja.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?style=flat-square&logo=scikitlearn)](https://scikit-learn.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-latest-blue?style=flat-square)](https://lightgbm.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-gray?style=flat-square)](LICENSE)

---

## Table of Contents

- [Results](#results)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Quickstart](#quickstart)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Notebooks](#notebooks)
- [Methodology](#methodology)
- [Dataset](#dataset)

---

## Results

| Model | AUC Train | AUC OOT | KS OOT | Gini OOT | PSI |
|---|---|---|---|---|---|
| DummyClassifier | 0.500 | 0.502 | 0.003 | 0.003 | — |
| **Logistic Regression (WoE Scorecard)** | **0.763** | **0.762** | **0.399** | **0.524** | **0.0001 ✅** |
| LightGBM | 0.825 | 0.772 | 0.409 | 0.543 | — |

**Logistic Regression is selected for the scorecard** — competitive performance, full interpretability, and regulatory compliance. LightGBM serves as a performance benchmark.

### Industry Benchmark Comparison

| Metric | Minimum | This Model | Status |
|---|---|---|---|
| AUC | 0.65 | 0.762 | ✅ Good |
| KS  | 0.20 | 0.399 | ✅ Good |
| Gini| 0.30 | 0.524 | ✅ Good |
| PSI | < 0.10 | 0.0001 | ✅ Stable |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Jinja2 Frontend                          │
│         HTML · CSS · Vanilla JS · Server-Side Rendering      │
│    Dashboard · Score · History · Model Info                  │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP (fetch / form submit)
┌────────────────────────▼────────────────────────────────────┐
│                       FastAPI Backend                        │
│   GET / · POST /score · GET /scorecard · GET /features       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      ML Artifacts                           │
│   logreg_woe.pkl · lgbm_model.pkl · binning_process.pkl     │
│   feature_names.json · scorecard_table.csv                  │
└─────────────────────────────────────────────────────────────┘
```

**Single server:** FastAPI serves Jinja2 templates directly — no build step, no separate frontend server.

---

## Project Structure

```
home-credit-default-risk/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                        # Kaggle CSV files (not tracked)
│   └── processed/                  # Parquet intermediates (not tracked)
│       ├── master_table.parquet
│       ├── train_woe.parquet
│       ├── oot_woe.parquet
│       ├── train_scores.parquet
│       └── oot_scores.parquet
│
├── notebooks/
│   ├── 01_data_overview.ipynb      # Table schemas, join keys, size
│   ├── 02_eda_main.ipynb           # EDA, statistical tests, visualizations
│   ├── 03_feature_engineering.ipynb # Join 7 tables, aggregate features
│   ├── 04_woe_iv_selection.ipynb   # IV filtering, WoE encoding, correlation
│   ├── 05_baseline_models.ipynb    # DummyClassifier → Logistic Regression
│   ├── 06_lgbm_model.ipynb         # LightGBM, SHAP, CV tuning
│   ├── 07_scorecard.ipynb          # PDO scaling, scorecard table
│   ├── 08_validation.ipynb         # KS, Gini, PSI, calibration, OOT
│   └── 09_final_summary.ipynb      # Dashboard, model comparison, conclusions
│
├── src/                            # Reusable Python modules
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── woe_encoding.py
│   ├── metrics.py                  # KS, Gini, PSI, AUC
│   ├── scorecard.py                # LR → scorecard points conversion
│   └── visualization.py
│
├── models/                         # Serialized artifacts (not tracked)
│   ├── logreg_woe.pkl
│   ├── lgbm_model.pkl
│   ├── binning_process.pkl
│   ├── feature_names.json
│   └── scorecard_table.csv         # ← tracked: key deliverable
│
├── app/                            # FastAPI + Jinja2 UI
│   ├── main.py                     # Routes: GET / · POST /score · GET /scorecard
│   ├── predict.py                  # ML inference pipeline
│   ├── schemas.py                  # Pydantic input/output models
│   └── templates/                  # Jinja2 HTML templates
│       ├── base.html               # Layout, navigation, shared CSS
│       ├── dashboard.html          # Model metrics + session overview
│       ├── score.html              # Applicant form + live scoring result
│       ├── history.html            # Scoring log + CSV export
│       └── model_info.html         # Scorecard table + benchmarks + features
```

---

## ML Pipeline

```
Raw Data (7 tables)
        │
        ▼
  EDA + Cleaning              notebook 02
  ├── Missing value analysis
  ├── Mann-Whitney U test (numerical features)
  └── Chi-Square test (categorical features)
        │
        ▼
  Feature Engineering         notebook 03
  ├── bureau + bureau_balance → aggregated delinquency history
  ├── previous_application    → approval rates, credit utilization
  ├── installments_payments   → late payment rate, underpaid amounts
  ├── credit_card_balance     → utilization, DPD history
  └── POS_CASH_balance        → contract completion, DPD months
        │
        ▼
  Feature Selection           notebook 04
  ├── IV ≥ 0.02  (Information Value filtering)
  ├── WoE encoding via OptimalBinning
  └── Correlation filter < 0.85 → 56 final features
        │
        ├──────────────────────────────┐
        ▼                              ▼
  Logistic Regression          LightGBM              notebooks 05–06
  (interpretable, scorecard)   (performance benchmark)
        │                              │
        ▼                              ▼
  Scorecard (PDO=20)           SHAP values            notebook 07
  Points-based credit scoring
        │
        ▼
  OOT Validation               notebook 08
  ├── KS, AUC, Gini
  ├── PSI (score stability)
  └── Calibration curves
```

### Key Engineered Features

| Feature | Source | IV | Description |
|---|---|---|---|
| `EXT_SOURCE_3` | application | 0.335 | Third-party credit bureau score |
| `EXT_SOURCE_2` | application | 0.319 | Third-party credit bureau score |
| `bureau_bb_max_dpd` | bureau_balance | — | Worst-ever delinquency in bureau history |
| `inst_late_payment_rate` | installments | — | Fraction of late installment payments |
| `DAYS_EMPLOYED` | application | 0.116 | Employment duration |
| `prev_approval_rate` | previous_app | — | Prior application approval ratio |

---

## Quickstart

### Prerequisites

- Python 3.13
- Kaggle account (for dataset download)

### 1 · Clone and install

```bash
git clone https://github.com/Mekhty111/Home-Credit-Default-Risk.git
cd Home-Credit-Default-Risk

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 · Download dataset

```bash
# Via Kaggle CLI
kaggle competitions download -c home-credit-default-risk -p data/raw/
cd data/raw && unzip home-credit-default-risk.zip && cd ../..
```

Or download manually from [kaggle.com/c/home-credit-default-risk/data](https://www.kaggle.com/c/home-credit-default-risk/data) into `data/raw/`.

### 3 · Run notebooks in order

```bash
jupyter notebook
```

Execute `notebooks/01_data_overview.ipynb` through `notebooks/09_final_summary.ipynb` sequentially. Each notebook saves artifacts consumed by the next.

### 4 · Start the application

```bash
cd app
uvicorn main:app --reload --port 8000
```

App: `http://localhost:8000`
Swagger UI: `http://localhost:8000/docs`

No build step required — Jinja2 templates are rendered server-side on every request.

---

## API Reference

### `POST /score`

Score a single loan applicant.

**Request body** (all fields optional — missing values handled as separate WoE bins):

```json
{
  "EXT_SOURCE_1":           0.51,
  "EXT_SOURCE_2":           0.62,
  "EXT_SOURCE_3":           0.72,
  "DAYS_BIRTH":             -12000,
  "DAYS_EMPLOYED":          -2000,
  "AMT_INCOME_TOTAL":       180000,
  "AMT_CREDIT":             450000,
  "AMT_ANNUITY":            22500,
  "AMT_GOODS_PRICE":        400000,
  "CODE_GENDER":            "M",
  "NAME_EDUCATION_TYPE":    "Higher education",
  "NAME_INCOME_TYPE":       "Working",
  "REGION_RATING_CLIENT":   2,
  "FLAG_OWN_CAR":           "N",
  "FLAG_OWN_REALTY":        "Y"
}
```

**Response:**

```json
{
  "lr_probability":    0.1243,
  "lgbm_probability":  0.1187,
  "score":             612,
  "risk_band":         "Low",
  "action":            "APPROVE",
  "interpretation":    "Low risk — approve standard terms"
}
```

### `GET /scorecard`

Returns the full scorecard table as a JSON array (Feature, Bin, WoE, Coefficient, Points).

### `GET /features`

Returns the list of 56 features used by the model.

### `GET /health`

Health check endpoint. Returns `{"status": "healthy"}`.

### Risk Bands

| Score | Risk Band | Recommended Action |
|---|---|---|
| < 520 | Very High | Reject |
| 520 – 560 | High | Manual Review |
| 560 – 600 | Medium | Conditional Approve |
| 600 – 640 | Low | Approve |
| > 640 | Very Low | Approve + Best Rate |

---

## Frontend

A server-rendered terminal UI built with **FastAPI · Jinja2 · HTML · CSS · Vanilla JS**.

No frontend framework, no build step — templates are rendered server-side and served directly by FastAPI.

**Pages:**

| Route | Template | Description |
|---|---|---|
| `GET /` | `dashboard.html` | Model metrics, session stats, pipeline overview |
| `GET /score` | `score.html` | Applicant input form + live scoring result + gauge |
| `GET /history` | `history.html` | Scoring log with CSV export |
| `GET /model` | `model_info.html` | Scorecard table, benchmarks, feature list |

**Design:** Dark terminal aesthetic, JetBrains Mono, green primary accent. Score gauge and results rendered via Vanilla JS after form submission to `POST /score`.

---

## Notebooks

| # | Notebook | Key Output |
|---|---|---|
| 01 | Data Overview | Table schemas, join diagram |
| 02 | EDA | Statistical tests, missing value analysis |
| 03 | Feature Engineering | `master_table.parquet` (307k rows) |
| 04 | WoE / IV Selection | `train_woe.parquet`, `binning_process.pkl` |
| 05 | Baseline Models | `logreg_woe.pkl`, baseline AUC |
| 06 | LightGBM + SHAP | `lgbm_model.pkl`, SHAP importance |
| 07 | Scorecard | `scorecard_table.csv` |
| 08 | Validation | KS/Gini/PSI/Calibration plots |
| 09 | Final Summary | Dashboard, full model comparison |

---

## Methodology

### WoE / IV (Weight of Evidence / Information Value)

Each feature is binned using **OptimalBinning** and transformed to WoE scores, which represent the log-odds ratio of good to bad applicants in each bin. Features with IV < 0.02 are dropped as non-predictive.

```
WoE_i = ln(Distribution_Good_i / Distribution_Bad_i)
IV    = Σ (Distribution_Good_i − Distribution_Bad_i) × WoE_i
```

| IV Range | Interpretation |
|---|---|
| < 0.02 | Useless — dropped |
| 0.02 – 0.10 | Weak |
| 0.10 – 0.30 | Medium |
| > 0.30 | Strong |

### Scorecard Scaling

The Logistic Regression model is converted to a points-based scorecard using standard PDO scaling:

```
Score  = OFFSET + FACTOR × log(odds)
FACTOR = PDO / ln(2)
OFFSET = Base Score − FACTOR × ln(Base Odds)
```

With `PDO=20`, `Base Score=600`, `Base Odds=19` (5% default rate):
every 20 points on the scorecard correspond to a halving of the default odds.

### OOT Validation

An **out-of-time (OOT)** holdout split (80/20 stratified) is used throughout — the OOT set is never touched during feature selection, WoE fitting, or model training. This mirrors production deployment where the model is validated on future applicants.

**PSI (Population Stability Index)** confirms the score distribution is stable between train and OOT:

```
PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)
PSI = 0.0001  →  No population shift detected
```

---

## Dataset

**Source:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle)

| Table | Rows | Description |
|---|---|---|
| `application_train.csv` | 307,511 | Main applicant data + target |
| `bureau.csv` | 1,716,428 | Credit bureau history |
| `bureau_balance.csv` | 27,299,925 | Monthly bureau statuses |
| `previous_application.csv` | 1,670,214 | Prior Home Credit applications |
| `installments_payments.csv` | 13,605,401 | Installment payment history |
| `credit_card_balance.csv` | 3,840,312 | Credit card monthly balances |
| `POS_CASH_balance.csv` | 10,001,358 | POS and cash loan monthly data |

**Target:** `TARGET = 1` indicates loan default (8% positive rate — highly imbalanced).

---

## Requirements

```
fastapi
uvicorn[standard]
pandas
numpy
scikit-learn==1.5.2
lightgbm
optbinning==0.20.0
shap
joblib
jinja2
python-multipart
pyarrow
matplotlib
seaborn
scipy
statsmodels
```

---

*Built with Python · FastAPI · Jinja2 · scikit-learn · LightGBM · OptimalBinning*
