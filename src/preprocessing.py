import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import os

# ------------------- Custom Transformer ------------------- #
class OutlierLogTransformer(BaseEstimator, TransformerMixin):
    """Clip outliers and apply log transform to specific features."""
    def __init__(self, lower=0.01, upper=0.99, log_features=None, epsilon=1e-6):
        self.lower = lower
        self.upper = upper
        self.log_features = log_features
        self.epsilon = epsilon

    def fit(self, X, y=None):
        # Convert ndarray to DataFrame
        if isinstance(X, np.ndarray):
            # Use all column indices as temporary names
            X = pd.DataFrame(X, columns=[f"col{i}" for i in range(X.shape[1])])
        self.columns_ = X.columns
        self.percentiles_ = {
            col: (np.percentile(X[col], self.lower * 100),
                  np.percentile(X[col], self.upper * 100))
            for col in X.columns
        }
        return self

    def transform(self, X):
        # Convert to DataFrame if ndarray
        is_array = False
        if isinstance(X, np.ndarray):
            is_array = True
            X = pd.DataFrame(X, columns=self.columns_)

        X_trans = X.copy()
        for col in X_trans.columns:
            lower, upper = self.percentiles_[col]
            X_trans[col] = np.clip(X_trans[col], lower, upper)
            if self.log_features and col in self.log_features:
                X_trans[col] = np.log1p(np.clip(X_trans[col], self.epsilon, None))

        return X_trans.values if is_array else X_trans

# ------------------- Main Preprocessing ------------------- #
def main():
    print("\n Starting preprocessing...")

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/merged_exoplanet_dataset.csv")

    features = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
        'fpp_prob', 'fpp_prob_heb', 'fpp_prob_ueb', 'fpp_prob_beb'
    ]
    log_features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad']
    target = 'koi_disposition'

    # Clean and encode labels
    df[target] = df[target].astype(str).str.strip().str.upper()
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    joblib.dump(le, "models/label_encoder.pkl")
    print(" Label classes:", list(le.classes_))

    # Preprocessing pipeline
    pipeline = Pipeline([
        ('outlier_log', OutlierLogTransformer(log_features=log_features)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    X = df[features]
    y = df[target]
    X_trans = pipeline.fit_transform(X)

    df_preprocessed = pd.DataFrame(X_trans, columns=features)
    df_preprocessed[target] = y

    df_preprocessed.to_csv("data/preprocessed_data.csv", index=False)
    joblib.dump(pipeline, "models/preprocess_pipeline.pkl")

    print("\n Preprocessing complete.")
    print("→ Saved preprocessed data: data/preprocessed_data.csv")
    print("→ Saved pipeline: models/preprocess_pipeline.pkl")

if __name__ == "__main__":
    main()
