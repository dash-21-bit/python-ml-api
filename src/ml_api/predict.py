from __future__ import annotations

from typing import Dict
import pandas as pd

from ml_api.model import load_model


def predict_from_features(features: Dict[str, float]) -> tuple[int, float]:
    """
    Convert dict features -> DataFrame row -> model prediction.
    Returns (predicted_class, probability_of_class_1).
    """
    model = load_model()

    # Convert dict into a single-row DataFrame
    X = pd.DataFrame([features])

    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    return pred, proba
