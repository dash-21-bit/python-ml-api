from __future__ import annotations

from pathlib import Path
import joblib

MODEL_PATH = Path("models/logreg_pipeline.joblib")

_model = None

def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Train it first: python scripts/train_model.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model
