from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "heart_disease.csv"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run: python src/download_data.py")
    df = pd.read_csv(DATA_PATH, na_values=["?", "NA", "nan", ""])
    df.columns = [c.strip().lower() for c in df.columns]
    if "num" not in df.columns:
        raise ValueError("Expected target column 'num' in dataset.")
    # Numeric conversion also converts UCI '?' markers to NaN.
    for col in FEATURES + ["num"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["num"]).copy()
    # UCI defines 0 as absence and 1-4 as presence.
    y = (df["num"] > 0).astype(int)
    X = df[FEATURES].copy()
    return X, y
