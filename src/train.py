import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from src.preprocessing import OutlierLogTransformer

# ------------------- Paths ------------------- #
RAW_DATA_PATH = "data/merged_exoplanet_dataset.csv"
PIPELINE_PATH = "models/preprocess_pipeline.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
BASE_MODEL_PATHS = {
    "rf": "models/rf_final.pkl",
    "et": "models/et_final.pkl",
    "xgb": "models/xgb_final.pkl",
    "lgb": "models/lgb_final.pkl",
    "cat": "models/cat_final.pkl"
}
META_MODEL_PATH = "models/meta_model_stacking.pkl"

os.makedirs("models", exist_ok=True)

# ------------------- Load Data ------------------- #
df = pd.read_csv(RAW_DATA_PATH)

features = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
    'fpp_prob', 'fpp_prob_heb', 'fpp_prob_ueb', 'fpp_prob_beb'
]
log_features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad']
target = 'koi_disposition'

# Encode labels
df[target] = df[target].astype(str).str.strip().str.upper()
le = LabelEncoder()
df[target] = le.fit_transform(df[target])
joblib.dump(le, LABEL_ENCODER_PATH)
print("Saved label encoder with classes:", list(le.classes_))

X = df[features]  # keep as DataFrame
y = df[target].values

# Preprocessing pipeline
pipeline = Pipeline([
    ('outlier_log', OutlierLogTransformer(log_features=log_features)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

X_processed = pipeline.fit_transform(X)
joblib.dump(pipeline, PIPELINE_PATH)
print("Saved preprocessing pipeline")

# ------------------- Base Models ------------------- #
base_models = {
    "rf": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"),
    "et": ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"),
    "xgb": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", max_depth=6, n_estimators=200, random_state=42),
    "lgb": LGBMClassifier(max_depth=6, n_estimators=200, random_state=42),
    "cat": CatBoostClassifier(depth=6, iterations=200, verbose=0, random_state=42)
}

# ------------------- Stratified CV Training ------------------- #
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = {name: np.zeros((len(y), len(np.unique(y)))) for name in base_models}

for name, model in base_models.items():
    print(f"\nTraining {name.upper()} with Stratified CV...")
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed, y), 1):
        X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        f1 = f1_score(y_val, val_pred, average="weighted")
        fold_scores.append(f1)
        print(f"  Fold {fold}: F1 = {f1:.4f}")

        # Save out-of-fold probabilities for stacking
        oof_preds[name][val_idx] = model.predict_proba(X_val)

    print(f"Mean F1 for {name.upper()}: {np.mean(fold_scores):.4f}")
    joblib.dump(model, BASE_MODEL_PATHS[name])
    print(f"Saved model: {BASE_MODEL_PATHS[name]}")

# ------------------- Train Stacking Meta-Model ------------------- #
print("\nTraining Stacking Meta-Model...")
X_meta = np.hstack([oof_preds[name] for name in base_models])
meta_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
meta_model.fit(X_meta, y)
joblib.dump(meta_model, META_MODEL_PATH)
print(f"Saved stacking meta-model: {META_MODEL_PATH}")

# ------------------- Evaluate on Full Dataset ------------------- #
X_meta_full = np.hstack([base_models[name].predict_proba(X_processed) for name in base_models])
y_pred = meta_model.predict(X_meta_full)
f1_full = f1_score(y, y_pred, average="weighted")
print(f"\nStacking Ensemble Full F1 Score: {f1_full:.4f}")
