import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ---- 1. Create dummy dataset ----
data = {
    "orbital_period": [365.25, 88, 10, 1000, 5, 300],
    "stellar_temp": [5778, 4500, 3800, 6000, 3200, 5500],
    "planet_radius": [1.0, 0.4, 0.2, 2.0, 0.1, 0.9],
    "is_exoplanet": [1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# ---- 2. Train a simple classifier ----
X = df[["orbital_period", "stellar_temp", "planet_radius"]]
y = df["is_exoplanet"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ---- 3. Save model to pkl ----
os.makedirs("model", exist_ok=True)  # Create folder if not exists
joblib.dump(model, "model/exoplanet_model.pkl")

print("✅ Model trained and saved as model/exoplanet_model.pkl")
