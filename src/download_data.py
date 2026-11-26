"""Download the Cleveland Heart Disease dataset from the UCI repository."""
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "heart_disease.csv"


def main():
    print("Fetching UCI Heart Disease dataset (ID 45)...")
    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    # UCI target column is named 'num'.
    if "num" not in y.columns:
        y.columns = ["num"]
    df = pd.concat([X, y], axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {df.shape[0]} rows and {df.shape[1]} columns to {OUT}")
    print("Target distribution (original 0-4 labels):")
    print(df["num"].value_counts(dropna=False).sort_index())

if __name__ == "__main__":
    main()
