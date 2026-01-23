from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Wine Quality Prediction API")

model = None

# 🔹 USE ONLY THE FEATURES USED DURING TRAINING
class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    chlorides: float
    total_sulfur_dioxide: float
    density: float
    sulphates: float
    alcohol: float

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: WineInput):
    features = np.array([[ 
        data.fixed_acidity,
        data.volatile_acidity,
        data.chlorides,
        data.total_sulfur_dioxide,
        data.density,
        data.sulphates,
        data.alcohol
    ]])

    prediction = model.predict(features)[0]
    return {"predicted_wine_quality": float(prediction)}
