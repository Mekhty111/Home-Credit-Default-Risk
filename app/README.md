## Credit Risk Scoring API

### Install dependencies
```bash
pip install fastapi uvicorn
```

### Run
```bash
cd app
uvicorn main:app --reload
```

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
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 450000,
    "AMT_ANNUITY": 22500,
    "CODE_GENDER": "M",
    "NAME_EDUCATION_TYPE": "Higher education",
    "REGION_RATING_CLIENT": 2
  }'
```

### Example response
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

### Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | / | Health check |
| GET | /health | Service status |
| POST | /score | Score one applicant |
| GET | /scorecard | Full scorecard table |
| GET | /features | Feature list |