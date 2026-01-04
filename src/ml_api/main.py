from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ml_api.schemas import PredictRequest, PredictResponse
from ml_api.predict import predict_from_features
from ml_api.model import load_model

app = FastAPI(title="ML Prediction API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    try:
        model = load_model()
        return {"model_type": type(model).__name__}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        pred, proba = predict_from_features(req.features)
        return PredictResponse(prediction=pred, probability_class_1=proba)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
