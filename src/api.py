import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


# ------------------- Paths ------------------- #
MODEL_PATH = "models/exoplanet_model.pkl"

# ------------------- Load full model ------------------- #
print("Loading full ensemble model...")
full_model = joblib.load(MODEL_PATH)

pipeline = full_model["pipeline"]
base_models = full_model["base_models"]
meta_model = full_model["meta_model"]
le = full_model["label_encoder"]

print("Model loaded successfully.")

# ------------------- FastAPI App ------------------- #
app = FastAPI(title="Exoplanet Prediction API")

# ------------------- Input Schema ------------------- #
class ExoplanetInput(BaseModel):
    koi_period: float
    koi_duration: float
    koi_depth: float
    koi_prad: float
    fpp_prob: float
    fpp_prob_heb: float
    fpp_prob_ueb: float
    fpp_prob_beb: float

# ------------------- Prediction Endpoint ------------------- #
@app.post("/predict")
def predict(inputs: List[ExoplanetInput]):
    df = pd.DataFrame([i.dict() for i in inputs])
    X_processed = pipeline.transform(df)

    # Generate meta-features for stacking
    base_probs = [m.predict_proba(X_processed) for m in base_models.values()]
    X_meta = np.hstack(base_probs)

    # Predict final label
    y_pred = meta_model.predict(X_meta)
    y_labels = le.inverse_transform(y_pred)
    return {"predictions": y_labels.tolist()}


# ------------------- Health Check ------------------- #
@app.get("/health")
def health():
    return {"status": "OK"}
