# notebooks/

This directory contains the full ML pipeline as a sequence of Jupyter notebooks.  
Each notebook produces artifacts consumed by the next — run them **in order**.

---

## Execution Order

| # | Notebook | Description | Key Output |
|---|---|---|---|
| 01 | `01_data_overview.ipynb` | Load all 7 tables, inspect schemas, dtypes, missing rates, join keys | — |
| 02 | `02_eda_main.ipynb` | EDA on `application_train`: distributions, class imbalance, Mann-Whitney U, Chi-Square | — |
| 03 | `03_feature_engineering.ipynb` | Aggregate bureau, previous_application, installments, credit_card, POS → join to master table | `data/processed/master_table.parquet` |
| 04 | `04_woe_iv_selection.ipynb` | IV filtering (≥ 0.02), WoE encoding via OptimalBinning, correlation filter → 56 features | `data/processed/train_woe.parquet` `data/processed/oot_woe.parquet` `models/binning_process.pkl` `models/feature_names.json` |
| 05 | `05_baseline_models.ipynb` | DummyClassifier → Logistic Regression on WoE features, ROC curves, coefficients | `models/logreg_woe.pkl` |
| 06 | `06_lgbm_model.ipynb` | LightGBM with CV tuning, SHAP feature importance, comparison vs LogReg | `models/lgbm_model.pkl` |
| 07 | `07_scorecard.ipynb` | Convert LogReg coefficients to points-based scorecard (PDO=20, Base=600), risk bands | `models/scorecard_table.csv` `data/processed/train_scores.parquet` `data/processed/oot_scores.parquet` |
| 08 | `08_validation.ipynb` | KS plot, PSI, Gini/Lorenz curve, calibration — full OOT validation | — |
| 09 | `09_final_summary.ipynb` | Model comparison dashboard, single-client scoring example, conclusions | `models/final_dashboard.png` |

---

## Data Flow

```
01_data_overview
       │
       ▼
02_eda_main
       │
       ▼
03_feature_engineering  →  master_table.parquet
       │
       ▼
04_woe_iv_selection     →  train_woe.parquet · oot_woe.parquet
       │                   binning_process.pkl · feature_names.json
       ├──────────────────────────┐
       ▼                          ▼
05_baseline_models        06_lgbm_model
       │                          │
       ▼                          ▼
  logreg_woe.pkl            lgbm_model.pkl
       │
       ▼
07_scorecard            →  scorecard_table.csv · train_scores · oot_scores
       │
       ▼
08_validation
       │
       ▼
09_final_summary
```

---

## Notes

- `data/raw/` and `data/processed/` are excluded from version control — see root `.gitignore`.  
  Download the dataset from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place CSV files in `data/raw/` before running.
