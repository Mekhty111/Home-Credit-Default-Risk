## Credit Risk Scoring API

### Install dependencies
```bash
pip install fastapi uvicorn
```

### Run
```bash
uvicorn main:app --app-dir app --reload
```

Web UI: http://localhost:8000/ui

API runs at: http://localhost:8000
Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc

### Example request
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "EXT_SOURCE_1": 0.51,
    "EXT_SOURCE_2": 0.62,
    "EXT_SOURCE_3": 0.72,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -2000,
    "AMT_CREDIT": 450000,
    "AMT_ANNUITY": 22500,
    "AMT_GOODS_PRICE": 400000,
    "CODE_GENDER": "M",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_INCOME_TYPE": "Working",
    "bureau_active_count": 1,
    "inst_late_payment_rate": 0.02
  }'
```

### Example response
```json
{
  "lr_probability":   0.2609,
  "lgbm_probability": 0.2469,
  "score":            545,
  "risk_band":        "Low",
  "action":           "APPROVE",
  "interpretation":   "Low risk — approve standard terms",
  "contributions":    [ "... 56 per-feature reason codes ..." ]
}
```

### Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | / | Health check |
| GET | /health | Service status |
| POST | /score | Score one applicant |
| GET | /scorecard | Full scorecard table |
| GET | /features | Feature list with IV |
| GET | /stats | Validation metrics, OOT score distribution, band calibration |