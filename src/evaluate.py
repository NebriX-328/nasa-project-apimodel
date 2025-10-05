import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.special import softmax
import shap
from src.preprocessing import OutlierLogTransformer

# ------------------- Paths ------------------- #
RAW_DATA_PATH = "data/merged_exoplanet_dataset.csv"
PIPELINE_PATH = "models/preprocess_pipeline.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

BASE_MODEL_PATHS = {
    "rf": "models/rf_final.pkl",
    "et": "models/et_final.pkl",
    "xgb": "models/xgb_final.pkl",
    "lgb": "models/lgb_final.pkl",
    "cat": "models/cat_final.pkl"
}
META_MODEL_PATH = "models/meta_model_stacking.pkl"

# ------------------- Load Data ------------------- #
print("\nLoading raw dataset...")
df = pd.read_csv(RAW_DATA_PATH)
pipeline = joblib.load(PIPELINE_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

features = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
    'fpp_prob', 'fpp_prob_heb', 'fpp_prob_ueb', 'fpp_prob_beb'
]
target = 'koi_disposition'

df[target] = df[target].astype(str).str.strip().str.upper()
known_mask = df[target].isin(le.classes_)
df_known = df[known_mask]

X = df_known[features]  # keep as DataFrame
y_true = le.transform(df_known[target])
X_processed = pipeline.transform(X)

# ------------------- Load Models ------------------- #
base_models = {}
for name, path in BASE_MODEL_PATHS.items():
    if os.path.exists(path):
        base_models[name] = joblib.load(path)
        print(f"Loaded {name.upper()} model.")

meta_model = None
if os.path.exists(META_MODEL_PATH):
    meta_model = joblib.load(META_MODEL_PATH)
    print("Loaded Stacking Meta-Model")

# ------------------- Predictions ------------------- #
def ensemble_predict(base_models, meta_model=None, X_data=None):
    if meta_model:
        base_probs = [model.predict_proba(X_data) for model in base_models.values()]
        X_meta = np.hstack(base_probs)
        return meta_model.predict(X_meta)
    else:
        ensemble_probs = [model.predict_proba(X_data) for model in base_models.values()]
        return np.argmax(np.mean(ensemble_probs, axis=0), axis=1)

# Stacking
if meta_model:
    y_pred_stack = ensemble_predict(base_models, meta_model, X_processed)
    acc_stack = accuracy_score(y_true, y_pred_stack)
    f1_stack = f1_score(y_true, y_pred_stack, average="weighted")
    print(f"\nStacking Ensemble Accuracy: {acc_stack:.4f}")
    print(f"Stacking Ensemble F1 Score: {f1_stack:.4f}")
else:
    y_pred_stack = None

# Soft-voting
y_pred_soft = ensemble_predict(base_models, meta_model=None, X_data=X_processed)
acc_soft = accuracy_score(y_true, y_pred_soft)
f1_soft = f1_score(y_true, y_pred_soft, average="weighted")
print(f"\nSoft-Voting Ensemble Accuracy: {acc_soft:.4f}")
print(f"Soft-Voting Ensemble F1 Score: {f1_soft:.4f}")

# ------------------- Confusion Matrices ------------------- #
def plot_cm(y_true, y_pred, title, filename):
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()
    print(f"Saved: {filename}")

if y_pred_stack is not None:
    plot_cm(y_true, y_pred_stack, "Confusion Matrix (Stacking)", "confusion_matrix_stacking.png")
plot_cm(y_true, y_pred_soft, "Confusion Matrix (Soft-Voting)", "confusion_matrix_soft.png")

# ------------------- SHAP ------------------- #
model_for_shap = base_models.get("xgb", list(base_models.values())[0])
X_sample = X.sample(n=min(500, len(X)), random_state=42)
explainer = shap.Explainer(model_for_shap)
shap_values = explainer(X_sample)

shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Feature Importance Summary")
plt.tight_layout()
plt.savefig("plots/shap_summary.png", dpi=300)
plt.close()

shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title("Mean Absolute SHAP Value")
plt.tight_layout()
plt.savefig("plots/shap_bar.png", dpi=300)
plt.close()

print("Saved SHAP plots")
