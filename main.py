from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model
model = joblib.load("model/exoplanet_model.pkl")

class ExoplanetInput(BaseModel):
    orbital_period: float
    stellar_temp: float
    planet_radius: float

@app.get("/")
def home():
    return {"message": "🚀 NebriX API is running!"}

@app.post("/predict")
def predict_exoplanet(data: ExoplanetInput):
    X = [[data.orbital_period, data.stellar_temp, data.planet_radius]]
    prediction = model.predict(X)[0]
    return {
        "is_exoplanet": bool(prediction),
        "message": "✅ Exoplanet detected" if prediction else "❌ False positive"
    }

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or "http://127.0.0.1:5500" if using Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
