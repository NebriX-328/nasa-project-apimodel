import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ------------------- Paths ------------------- #
PIPELINE_PATH = "models/preprocess_pipeline.pkl"
BASE_MODEL_PATHS = {
    "rf": "models/rf_final.pkl",
    "et": "models/et_final.pkl",
    "xgb": "models/xgb_final.pkl",
    "lgb": "models/lgb_final.pkl",
    "cat": "models/cat_final.pkl"
}
META_MODEL_PATH = "models/meta_model_stacking.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
FULL_MODEL_PATH = "models/exoplanet_model.pkl"

# ------------------- Load components ------------------- #
print("Loading preprocessing pipeline...")
pipeline = joblib.load(PIPELINE_PATH)

print("Loading base models...")
base_models = {name: joblib.load(path) for name, path in BASE_MODEL_PATHS.items()}

print("Loading stacking meta-model...")
meta_model = joblib.load(META_MODEL_PATH)

print("Loading label encoder...")
le = joblib.load(LABEL_ENCODER_PATH)

# ------------------- Bundle all together ------------------- #
full_model = {
    "pipeline": pipeline,
    "base_models": base_models,
    "meta_model": meta_model,
    "label_encoder": le
}

joblib.dump(full_model, FULL_MODEL_PATH)
print(f"Saved full ensemble model as {FULL_MODEL_PATH}")

# ------------------- Test Prediction ------------------- #
sample_input = pd.DataFrame([{
    "koi_period": 10.5,
    "koi_duration": 2.3,
    "koi_depth": 0.1,
    "koi_prad": 1.2,
    "fpp_prob": 0.05,
    "fpp_prob_heb": 0.01,
    "fpp_prob_ueb": 0.02,
    "fpp_prob_beb": 0.03
}])

# Preprocess
X_processed = pipeline.transform(sample_input)

# Generate meta-features for stacking
base_probs = [m.predict_proba(X_processed) for m in base_models.values()]
X_meta = np.hstack(base_probs)

# Predict
y_pred = meta_model.predict(X_meta)
y_label = le.inverse_transform(y_pred)
print("Test prediction output:", y_label.tolist())
