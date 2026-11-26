"""Train and compare five supervised classification algorithms."""
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             RocCurveDisplay)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from common import ROOT, NUMERIC, CATEGORICAL, load_dataset

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])


def main():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE),
        "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.01, learning_rate_init=0.001, max_iter=1500, early_stopping=True, random_state=RANDOM_STATE)
    }
    out = ROOT / "outputs"
    model_dir = ROOT / "models"
    out.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    rows, fitted = [], {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba) if proba is not None else np.nan,
            "5-fold CV F1": cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()
        }
        rows.append(row)
        fitted[name] = pipe
        safe = name.lower().replace(" ", "_")
        joblib.dump(pipe, model_dir / f"{safe}.joblib")
        with open(out / f"{safe}_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, pred, target_names=["No disease", "Disease present"], zero_division=0))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No disease", "Disease"], yticklabels=["No disease", "Disease"], ax=ax)
        ax.set_title(f"{name} - Confusion Matrix")
        ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label")
        fig.tight_layout(); fig.savefig(out / f"{safe}_confusion_matrix.png", dpi=160); plt.close(fig)
        print(f"Trained {name}: F1={row['F1-score']:.3f}, Recall={row['Recall']:.3f}")

    results = pd.DataFrame(rows).sort_values(["5-fold CV F1", "F1-score"], ascending=False)
    results.to_csv(out / "model_comparison.csv", index=False)
    best_name = results.iloc[0]["Model"]
    best_pipe = fitted[best_name]
    joblib.dump(best_pipe, model_dir / "best_model.joblib")
    with open(model_dir / "model_metadata.json", "w") as f:
        json.dump({"best_model": best_name, "features": list(X.columns), "positive_class": 1,
                   "target_definition": "num > 0 means disease present", "random_state": RANDOM_STATE}, f, indent=2)

    # ROC curves for all models with probability estimates.
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in fitted.items():
        if hasattr(pipe, "predict_proba"):
            RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC Curve Comparison")
    fig.tight_layout(); fig.savefig(out / "roc_curve_comparison.png", dpi=160); plt.close(fig)

    # Save the held-out test set for reproducible interpretation and reporting.
    test_export = X_test.copy()
    test_export["actual_target"] = y_test
    test_export.to_csv(out / "test_split.csv", index=False)
    print("\nModel comparison:\n", results.to_string(index=False))
    print(f"\nSelected model by training-set 5-fold CV F1 (test F1 shown for comparison): {best_name}")
    print(f"Results saved in: {out}")

if __name__ == "__main__":
    main()
