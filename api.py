import os, sys, traceback, joblib, numpy as np
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from db import init_db, log_prediction

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "./models/model.joblib")
EXPLAINER_PATH = os.getenv("EXPLAINER_PATH", "./models/explainer.joblib")

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024:
    raise RuntimeError(
        f"Model artifact at {MODEL_PATH} is missing or too small. "
        f"Run `python train.py` to regenerate."
    )

art = joblib.load(MODEL_PATH)
try:
    explainer = joblib.load(EXPLAINER_PATH)
except Exception:
    explainer = None

MODEL_VERSION: str = art["model_version"]
FEATURES: List[str] = art["features"]
pre = art["preproc"]
model = art["model"]
cal = art["calibrator"]

app = FastAPI(title="AI Medical Diagnosis API", version=MODEL_VERSION)

class PredictIn(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature name -> value")

class PredictOut(BaseModel):
    prob: float
    label: int
    model_version: str
    top_factors: List[Dict[str, float]]

RANGES = {
    "sex": (0, 1), "cp": (0, 3), "fbs": (0, 1), "restecg": (0, 2),
    "exang": (0, 1), "slope": (0, 2), "ca": (0, 3), "thal": (0, 3)
}

@app.on_event("startup")
def _startup():
    init_db()

@app.get("/")
def root():
    return {"status": "ok", "version": MODEL_VERSION}

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    # basic range checks for categorical features
    for k, (lo, hi) in RANGES.items():
        v = inp.features.get(k, None)
        if v is None:
            raise HTTPException(status_code=400, detail=f"Missing required feature: {k}")
        try:
            fv = float(v)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Feature {k} is not numeric: {v}")
        if not (lo <= fv <= hi):
            raise HTTPException(status_code=400, detail=f"Feature {k} must be in [{lo},{hi}] but got {v}")

    # order features and predict
    try:
        x = np.array([[float(inp.features.get(f, np.nan)) for f in FEATURES]])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature types: {e}")

    try:
        Xp = pre.transform(x)
        if np.isnan(Xp).any():
            raise ValueError("NaNs present after preprocessing")
        p = float(cal.predict_proba(Xp)[:, 1][0])
    except Exception as e:
        print("=== INFERENCE ERROR ===", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {type(e).__name__}: {e}")

    label = int(p >= 0.5)

    # SHAP local explanation (optional)
    top: List[Dict[str, float]] = []
    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(Xp)
            if isinstance(shap_vals, list):
                sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            else:
                sv = shap_vals
            impacts = np.array(sv)[0]
            idx = np.argsort(np.abs(impacts))[::-1][:5]
            top = [{"feature": FEATURES[i], "impact": float(impacts[i])} for i in idx]
        except Exception:
            top = []

    try:
        log_prediction({k: float(inp.features.get(k, float("nan"))) for k in FEATURES}, p, label, MODEL_VERSION)
    except Exception:
        pass

    return {"prob": p, "label": label, "model_version": MODEL_VERSION, "top_factors": top}
