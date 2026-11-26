"""Permutation importance and optional SHAP explanation for the best model."""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from common import ROOT, load_dataset


def main():
    X, y = load_dataset()
    test_path = ROOT / "outputs" / "test_split.csv"
    if not test_path.exists():
        raise FileNotFoundError("Run python src/train.py first.")
    test = pd.read_csv(test_path)
    X_test = test[X.columns]
    y_test = test["actual_target"]
    model = joblib.load(ROOT / "models" / "best_model.joblib")
    result = permutation_importance(model, X_test, y_test, n_repeats=30,
                                    random_state=42, scoring="f1", n_jobs=-1)
    imp = pd.DataFrame({"Feature": X.columns, "Importance": result.importances_mean,
                        "Std": result.importances_std}).sort_values("Importance", ascending=False)
    imp.to_csv(ROOT / "outputs" / "permutation_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot = imp.sort_values("Importance")
    ax.barh(plot["Feature"], plot["Importance"], xerr=plot["Std"])
    ax.set_title("Permutation Feature Importance (F1 decrease)")
    ax.set_xlabel("Mean decrease in F1 after feature shuffling")
    fig.tight_layout(); fig.savefig(ROOT / "outputs" / "feature_importance.png", dpi=160); plt.close(fig)

    # SHAP is optional: KernelExplainer works model-agnostically but can be slow on small datasets.
    try:
        import shap
        background = X.sample(min(40, len(X)), random_state=42)
        sample = X_test.sample(min(30, len(X_test)), random_state=42)
        explainer = shap.KernelExplainer(lambda data: model.predict_proba(pd.DataFrame(data, columns=X.columns))[:, 1], background)
        values = explainer.shap_values(sample, nsamples=100)
        if isinstance(values, list):
            values = values[-1]
        shap.summary_plot(values, sample, show=False)
        plt.tight_layout(); plt.savefig(ROOT / "outputs" / "shap_summary.png", dpi=160, bbox_inches="tight"); plt.close()
        print("SHAP summary saved.")
    except Exception as exc:
        print("SHAP step was skipped; permutation importance is available.")
        print("Reason:", exc)
    print("Feature importance saved to outputs/.")

if __name__ == "__main__":
    main()
