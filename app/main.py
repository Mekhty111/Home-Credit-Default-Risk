from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from schemas import ApplicantInput, ScoringResult
from predict import predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    print("Loading models...")
    import predict  # triggers model loading at module level
    print("Models loaded successfully")
    yield
    print("Shutting down")


app = FastAPI(
    title       = "Credit Risk Scoring API",
    description = """
## Home Credit Default Risk — Scoring API

End-to-end ML pipeline for retail credit risk assessment.

### Models
- **Logistic Regression** on WoE-encoded features → Scorecard
- **LightGBM** → probability benchmark

### Scoring
Submit applicant features → receive:
- Default probability (LR + LightGBM)
- Scorecard points (PDO=20, Base=600)
- Risk band + recommended action
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


@app.get("/", tags=["Health"])
def root():
    return {
        "status" : "ok",
        "service": "Credit Risk Scoring API",
        "version": "1.0.0",
        "docs"   : "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/score", response_model=ScoringResult, tags=["Scoring"])
def score_applicant(applicant: ApplicantInput):
    """
    Score a single loan applicant.

    Returns default probability, scorecard points, and risk band.
    Missing features are handled automatically via WoE binning.
    """
    try:
        result = predict(applicant.model_dump())
        return ScoringResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scorecard", tags=["Model Info"])
def get_scorecard():
    """Return the full scorecard table."""
    import pandas as pd
    sc = pd.read_csv('../models/scorecard_table.csv')
    return sc.to_dict(orient='records')


@app.get("/features", tags=["Model Info"])
def get_features():
    """Return list of features used by the model."""
    import json
    with open('../models/feature_names.json') as f:
        features = json.load(f)
    return {"features": features, "count": len(features)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10, reload=True)