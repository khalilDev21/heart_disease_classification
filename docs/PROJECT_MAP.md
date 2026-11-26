# Project map

Use this page to find the main files and common workflows.

## Main folders

| Path | What it contains |
|---|---|
| `app/` | Streamlit interface and its deployment dependencies |
| `data/raw/` | Cleveland Heart Disease CSV used for training |
| `models/` | Fitted model pipelines; `best_model.joblib` is loaded by the app |
| `outputs/` | Model comparison, classification reports, plots and test split |
| `reports/` | Assignment report |
| `src/` | Dataset download, shared helpers, model training and explainability scripts |
| `docs/` | Project navigation and deployment instructions |

## Start the app

The saved model is included. From the project root, run:

```powershell
streamlit run app/streamlit_app.py
```

## Rebuild the model and outputs

Run these from the project root when you want to reproduce the full workflow:

```powershell
python src/download_data.py
python src/train.py
python src/explain.py
```

The dataset is saved to `data/raw/heart_disease.csv`, fitted pipelines to `models/`, and evaluation files to `outputs/`.

## Where to find results

- `outputs/model_comparison.csv` — comparison metrics for all five classifiers.
- `outputs/roc_curve_comparison.png` — ROC curve plot.
- `outputs/*_confusion_matrix.png` — confusion matrix for each classifier.
- `outputs/*_classification_report.txt` — precision, recall and F1 details.
- `models/model_metadata.json` — selected classifier and target definition.
- `reports/project_report.md` — report template and methodology.

For the high-level overview, setup steps and deployment notes, see the root [README](../README.md). For Community Cloud deployment, see [DEPLOYMENT.md](DEPLOYMENT.md).
