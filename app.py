from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def find_latest_model(base_dir: str = "outputs") -> Path:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"'{base_dir}' does not exist")

    run_dirs = [p for p in base.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in '{base_dir}'")

    latest_run = sorted(run_dirs)[-1]
    model_path = latest_run / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"model.joblib not found in {latest_run}")

    return model_path


MODEL_PATH = find_latest_model("outputs")
model = joblib.load(MODEL_PATH)

# On récupère le nombre de features attendu (scikit-learn >=1.0)
N_FEATURES = getattr(model, "n_features_in_", None)

app = FastAPI(title="LogReg API", version="0.2.0")


class PredictRequest(BaseModel):
    x: List[float]


class PredictResponse(BaseModel):
    prediction: int
    proba_class_1: Optional[float] = None
    model_path: str
    n_features_expected: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_path": str(MODEL_PATH), "n_features_expected": N_FEATURES}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = np.array(req.x, dtype=float)

    if N_FEATURES is not None and x.shape[0] != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {N_FEATURES} features, got {x.shape[0]}",
        )

    X = x.reshape(1, -1)

    try:
        pred = int(model.predict(X)[0])

        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0, 1])

        return PredictResponse(
            prediction=pred,
            proba_class_1=proba,
            model_path=str(MODEL_PATH),
            n_features_expected=N_FEATURES,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
