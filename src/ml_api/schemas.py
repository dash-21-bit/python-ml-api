from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict


class PredictRequest(BaseModel):
    """
    Request body for predictions.
    A dictionary mapping feature_name -> value.
    """
    features: Dict[str, float] = Field(..., description="Feature name to float value")


class PredictResponse(BaseModel):
    prediction: int
    probability_class_1: float
