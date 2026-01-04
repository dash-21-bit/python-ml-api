from __future__ import annotations

from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA = Path("data/raw/breast_cancer.csv")
MODEL_PATH = Path("models/logreg_pipeline.joblib")

def main() -> None:
    df = pd.read_csv(DATA)

    # target is the label column in sklearn dataset frame
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline ensures the exact preprocessing used in training
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print("✅ Training complete")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("Feature count:", X.shape[1])

if __name__ == "__main__":
    main()
