"""
ml_api.py  —  LAYOFF-NET Python ML API (FastAPI)
Serves the trained ICECB model via REST endpoints.
Run: uvicorn ml_api:app --host 0.0.0.0 --port 8000 --reload
"""
import os, pickle, sys
import subprocess

# Auto-install deps
for pkg in ['fastapi','uvicorn','pandas','numpy','scikit-learn','openpyxl']:
    try: __import__(pkg.replace('-','_'))
    except ImportError:
        subprocess.check_call([sys.executable,'-m','pip','install',pkg,'-q'])

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

# ── Load model ────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'layoff_model.pkl')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run final_analysis.py first.")

with open(MODEL_PATH, 'rb') as f:
    arts = pickle.load(f)

MODEL        = arts['model']
ENC          = arts['enc']
STRAT_MODELS = arts['strat_models']
RARE_INDS    = set(arts['rare_inds'])
ALPHA        = arts['alpha']
TAU          = arts['tau']
FEAT_COLS    = arts['feat_cols']
CAT_COLS     = arts['cat_cols']
NUM_COLS     = arts['num_cols']
IND_CLASSES  = arts['ind_classes']
CTY_CLASSES  = arts['cty_classes']

print(f"✓ Model loaded  |  τ*={TAU:.3f}  |  α={ALPHA}")

# ── FastAPI ───────────────────────────────────────────────────
app = FastAPI(
    title="LAYOFF-NET ML API",
    description="Industry-Conditional Ensemble with Calibrated Blending (ICECB)",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Schemas ───────────────────────────────────────────────────
class CompanyInput(BaseModel):
    industry:       str   = Field(..., example="Tech")
    country:        str   = Field(..., example="USA")
    funding_amount: float = Field(0.0,  example=500000000)
    employee_count: float = Field(0.0,  example=5000)
    growth_rate:    float = Field(...,  example=45.0)
    valuation:      float = Field(0.0,  example=2000000000)

class BatchInput(BaseModel):
    companies: List[CompanyInput]

# ── Feature prep ─────────────────────────────────────────────
def prepare_row(data: dict) -> np.ndarray:
    df = pd.DataFrame([data])
    for col in CAT_COLS:
        le = ENC[col]
        df[col] = df[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in set(le.classes_) else -1)
    df[NUM_COLS] = ENC['imp'].transform(df[NUM_COLS])
    for col in ['funding_amount','employee_count','valuation']:
        df[col] = np.log1p(df[col].clip(lower=0))
    df[NUM_COLS] = ENC['sc'].transform(df[NUM_COLS])
    return df[FEAT_COLS].values

def blend_predict(row_data: dict) -> dict:
    X = prepare_row(row_data)
    g_prob = MODEL.predict_proba(X)[0,1]
    industry = row_data['industry']
    stratum = 'Other' if industry in RARE_INDS else industry
    if stratum in STRAT_MODELS:
        m2, enc2 = STRAT_MODELS[stratum]
        df2 = pd.DataFrame([row_data])
        for col in CAT_COLS:
            le = enc2[col]
            df2[col] = df2[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in set(le.classes_) else -1)
        df2[NUM_COLS] = enc2['imp'].transform(df2[NUM_COLS])
        for col in ['funding_amount','employee_count','valuation']:
            df2[col] = np.log1p(df2[col].clip(lower=0))
        df2[NUM_COLS] = enc2['sc'].transform(df2[NUM_COLS])
        s_prob = m2.predict_proba(df2[FEAT_COLS].values)[0,1]
        prob = ALPHA * s_prob + (1 - ALPHA) * g_prob
    else:
        prob = g_prob
    pred = int(prob >= TAU)
    risk = 'Low' if prob < 0.35 else ('Medium' if prob < 0.60 else 'High')
    return {
        'probability': round(float(prob), 4),
        'prediction': pred,
        'risk_level': risk,
        'global_probability': round(float(g_prob), 4),
        'threshold': TAU
    }

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "LAYOFF-NET ML API", "status": "running", "version": "1.0.0"}

@app.get("/info")
def model_info():
    return {
        "model": "ICECB (Industry-Conditional Ensemble + Calibrated Blending)",
        "base_classifier": "Random Forest (500 trees, depth=12)",
        "strata": list(STRAT_MODELS.keys()),
        "alpha": ALPHA,
        "tau": TAU,
        "auc": 0.8787,
        "f1": 0.7214,
        "features": FEAT_COLS,
        "industries": IND_CLASSES,
        "countries": CTY_CLASSES
    }

@app.post("/predict")
def predict(data: CompanyInput):
    try:
        return blend_predict(data.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(data: BatchInput):
    results = []
    for comp in data.companies:
        try:
            results.append(blend_predict(comp.dict()))
        except Exception as e:
            results.append({"error": str(e)})
    high = sum(1 for r in results if r.get('risk_level') == 'High')
    return {
        "results": results,
        "summary": {
            "total": len(results),
            "high_risk": high,
            "medium_risk": sum(1 for r in results if r.get('risk_level') == 'Medium'),
            "low_risk": sum(1 for r in results if r.get('risk_level') == 'Low'),
        }
    }

if __name__ == "__main__":
    uvicorn.run("ml_api:app", host="0.0.0.0", port=8000, reload=True)
