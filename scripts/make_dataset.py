from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

OUT = Path("data/raw/breast_cancer.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()

df.to_csv(OUT, index=False)
print(f"Saved dataset to: {OUT}")
print(f"Shape: {df.shape}")
print("Columns:", list(df.columns)[:5], "...")  # preview
