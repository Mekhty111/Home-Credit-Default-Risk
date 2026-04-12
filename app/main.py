from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
import uvicorn
import os

from schemas import ApplicantInput, ScoringResult
from predict import predict

templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    import predict
    print("Models loaded successfully")
    yield
    print("Shutting down")


app = FastAPI(
    title="Credit Risk Scoring API",
    description="""
## Home Credit Default Risk — Scoring API

End-to-end ML pipeline for retail credit risk assessment.

### Models
- **Logistic Regression** on WoE-encoded features → Scorecard
- **LightGBM** → probability benchmark

### UI
Open **/ui** for the web interface.
""",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── UI routes ────────────────────────────────────────────────────────────────

@app.get("/ui", include_in_schema=False)
async def ui_dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")

@app.get("/ui/score", include_in_schema=False)
async def ui_score(request: Request):
    return templates.TemplateResponse(request, "score.html")

@app.get("/ui/history", include_in_schema=False)
async def ui_history(request: Request):
    return templates.TemplateResponse(request, "history.html")

@app.get("/ui/model", include_in_schema=False)
async def ui_model(request: Request):
    return templates.TemplateResponse(request, "model_info.html")


# ── API routes ───────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return RedirectResponse(url="/ui")

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
        result = predict(applicant.model_dump(exclude_none=False))
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)