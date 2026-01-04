from __future__ import annotations

from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score

DATA = Path("data/raw/breast_cancer.csv")
MODEL_PATH = Path("models/logreg_pipeline.joblib")

METRICS_OUT = Path("reports/metrics.csv")
CM_PNG = Path("reports/figures/confusion_matrix.png")
ROC_PNG = Path("reports/figures/roc_curve.png")

def save_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_roc(fpr, tpr, out_path: Path, auc: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve (AUC={auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("ERROR: Model not found. Run: python scripts/train_model.py")

    df = pd.read_csv(DATA)
    X = df.drop(columns=["target"])
    y = df["target"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    # Save CSV metrics
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", acc])
        writer.writerow(["roc_auc", auc])

    # Save images
    save_confusion_matrix(cm, CM_PNG)
    save_roc(fpr, tpr, ROC_PNG, auc)

    print("✅ Evaluation complete")
    print(f"Metrics CSV: {METRICS_OUT}")
    print(f"Confusion matrix image: {CM_PNG}")
    print(f"ROC curve image: {ROC_PNG}")
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
