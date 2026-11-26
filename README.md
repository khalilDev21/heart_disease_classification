# Heart Disease Classification Using Supervised Machine Learning

**Course:** Artificial Intelligence  
**Project type:** Part 2 — Team Project (Classification)  
**Student roll number:** 5  
**Dataset:** UCI Heart Disease — Cleveland subset (303 records, 13 input features, 1 target)

## Project overview
This project compares five supervised classification algorithms for the binary classification formulation of the UCI Cleveland Heart Disease dataset. The original target `num` has values 0–4. Following the UCI task definition, `num == 0` is mapped to class 0 (absence) and `num > 0` to class 1 (presence).

**Important:** This is an educational machine-learning exercise. It is not a clinical diagnostic system and must not be used to make healthcare decisions.

## Algorithms
1. Decision Tree Classifier
2. K-Nearest Neighbours (KNN)
3. Gaussian Naive Bayes
4. Random Forest Classifier
5. Multi-Layer Perceptron (MLP) Neural Network

The assignment requests comparison of at least two or three algorithms. This implementation includes all five algorithms listed in the assignment's classification section.

## Project structure
```text
heart_disease_classification/
├── app/streamlit_app.py
├── data/raw/                  # downloaded dataset is saved here
├── models/                    # trained model files are saved here
├── outputs/                   # metrics, reports and plots are saved here
├── reports/project_report.md
├── src/
│   ├── common.py
│   ├── download_data.py
│   ├── train.py
│   └── explain.py
├── requirements.txt
└── README.md
```

## Setup in VS Code (Windows PowerShell)
Open this folder in VS Code, then run these commands in the terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use the current terminal session command:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the project
Run commands from the project root (the folder containing README.md):

```powershell
python src/download_data.py
python src/train.py
python src/explain.py
streamlit run app/streamlit_app.py
```

## Deploy on Streamlit Community Cloud

The app entrypoint is `app/streamlit_app.py`. The deployment dependencies are in `app/requirements.txt`; these versions match the saved model in `models/best_model.joblib`.

1. Push this project to a GitHub repository. Keep `models/best_model.joblib` in the repository because the app loads it at startup.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and create an app.
3. Choose your repository and branch, and set the app file path to `app/streamlit_app.py`.
4. In **Advanced settings**, choose Python **3.14** to match the environment used to create the saved model.
5. Choose whether the app should be public or private, then deploy.

Community Cloud reads the app-specific `app/requirements.txt` for this entrypoint. The root `requirements.txt` remains available for the training and explainability scripts. See the [Community Cloud deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) for details.

The first command downloads UCI dataset ID 45 using the `ucimlrepo` package. An internet connection is required for this one-time download. The data is saved as `data/raw/heart_disease.csv`.

## Methodology
- Dataset: UCI Cleveland Heart Disease subset.
- Input variables: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.
- Target transformation: `num == 0` → 0; `num > 0` → 1.
- Split: 80% training and 20% test, stratified, random state 42.
- Preprocessing: median imputation and standardization for numeric variables; most-frequent imputation and one-hot encoding for categorical variables. All preprocessing is fitted within a pipeline on training data to reduce leakage.
- Validation: 5-fold stratified cross-validation on the training partition.
- Metrics: accuracy, precision, recall, F1-score and ROC-AUC. Confusion matrices and ROC curves are saved.
- Model selection: highest mean 5-fold cross-validation F1-score on the training partition, with held-out F1 shown for comparison. The test set is not used to select the model.
- Explainability: permutation feature importance; optional model-agnostic SHAP KernelExplainer summary.

## Output files
After training, check `outputs/` for:
- `model_comparison.csv`
- `roc_curve_comparison.png`
- one confusion matrix PNG and classification report TXT per algorithm
- `test_split.csv`
- feature importance output after running `src/explain.py`

Trained pipelines are stored in `models/`; `best_model.joblib` is loaded by the Streamlit application.

## Dataset citation
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Heart Disease dataset. https://doi.org/10.24432/C52P4X. Original dataset creators: Andras Janosi, William Steinbrunn, Matthias Pfisterer and Robert Detrano. Dataset license: CC BY 4.0. Cite the UCI dataset and retain attribution when redistributing the data.

## Limitations
- The Cleveland subset is small (303 cases) and historically collected; it may not represent current or diverse populations.
- A model's output is affected by sampling, preprocessing, class balance and the train/test split.
- Model probability is not a calibrated individual clinical risk estimate.
- No algorithm here has been clinically validated for diagnosis or screening.
