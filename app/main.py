from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from functools import lru_cache
import uvicorn
import json
import os

from schemas import ApplicantInput, ScoringResult
from predict import predict, FACTOR, POINTS_OFFSET, RISK_BANDS, MODELS_DIR

APP_DIR = os.path.dirname(os.path.abspath(__file__))
APP_VERSION = "1.1.0"

templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))
templates.env.globals["app_version"] = APP_VERSION


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
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")


@app.middleware("http")
async def revalidate_static(request: Request, call_next):
    # ES modules import each other without a version query, so make browsers
    # revalidate (cheap 304 via ETag) instead of serving stale code after updates.
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# ── UI routes ────────────────────────────────────────────────────────────────

def render(request: Request, template: str, page: str):
    return templates.TemplateResponse(request, template, {"page": page})

@app.get("/ui", include_in_schema=False)
async def ui_dashboard(request: Request):
    return render(request, "dashboard.html", "overview")

@app.get("/ui/score", include_in_schema=False)
async def ui_score(request: Request):
    return render(request, "score.html", "score")

@app.get("/ui/history", include_in_schema=False)
async def ui_history(request: Request):
    return render(request, "history.html", "history")

@app.get("/ui/model", include_in_schema=False)
async def ui_model(request: Request):
    return render(request, "model_info.html", "model")


# ── API routes ───────────────────────────────────────────────────────────────

@lru_cache
def load_json(name: str):
    with open(os.path.join(MODELS_DIR, name), encoding="utf-8") as f:
        return json.load(f)

@app.get("/", tags=["Health"])
def root():
    return RedirectResponse(url="/ui")

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "version": APP_VERSION}

@app.post("/score", response_model=ScoringResult, tags=["Scoring"])
def score_applicant(applicant: ApplicantInput):
    """
    Score a single loan applicant.

    Returns default probability, scorecard points, risk band and the
    per-feature contributions (reason codes) behind the score.
    Missing features are handled automatically via WoE binning.
    """
    try:
        result = predict(applicant.model_dump(exclude_none=False))
        return ScoringResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scorecard", tags=["Model Info"])
def get_scorecard():
    """Return the full scorecard table (points per WoE bin; higher = safer)."""
    import pandas as pd
    sc = pd.read_csv(os.path.join(MODELS_DIR, "scorecard_table.csv"))
    sc["Points"] = (-FACTOR * sc["Coefficient"] * sc["WoE"] + POINTS_OFFSET).round().astype(int)
    return sc.to_dict(orient="records")

@app.get("/features", tags=["Model Info"])
def get_features():
    """Return list of features used by the model, with Information Value."""
    features = load_json("feature_names.json")
    iv = {row["feature"]: row["iv"] for row in load_json("portfolio_stats.json")["iv"]}
    return {
        "features": features,
        "count": len(features),
        "iv": [{"feature": f, "iv": iv.get(f)} for f in features],
    }

@app.get("/stats", tags=["Model Info"])
def get_stats():
    """Validation metrics, out-of-time score distribution and risk band calibration."""
    stats = load_json("portfolio_stats.json")
    bands = [
        {"band": band, "action": action, "max_score": None if upper == float("inf") else int(upper)}
        for upper, band, action, _ in RISK_BANDS
    ]
    return {**stats, "band_config": bands}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
