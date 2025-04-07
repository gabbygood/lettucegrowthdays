# lettuce_growth_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Load model and label encoder
model = joblib.load("lettuce_growth_classifier.pkl")
label_encoder = joblib.load("growth_stage_label_encoder.pkl")

# FastAPI app
app = FastAPI(title="Lettuce Growth Stage Classifier API")

class InputData(BaseModel):
    temperature: float
    humidity: float
    tds: float
    ph: float

@app.post("/predict")
async def predict_growth_stage(data: InputData):
    features = [[data.temperature, data.humidity, data.tds, data.ph]]
    prediction = model.predict(features)[0]  # numpy.int64
    growth_stage = label_encoder.inverse_transform([prediction])[0]  # string
    return {"growth_stage": str(growth_stage)}

# Run the app (for local dev)
if __name__ == "__main__":
    uvicorn.run("lettuce_growth_api:app", host="192.168.0.193", port=8000, reload=True)
