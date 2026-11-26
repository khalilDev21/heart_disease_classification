# Artificial Intelligence — Classification Project Report

## Heart Disease Classification Using Supervised Machine Learning

**Student roll number:** 5  
**Project category:** Part 2 — Classification  
**Dataset:** UCI Heart Disease (Cleveland subset)  
**Development environment:** Python, VS Code, Scikit-learn and Streamlit

## 1. Abstract
This project implements a supervised machine-learning workflow to classify records in the Cleveland subset of the UCI Heart Disease dataset into two classes: absence or presence of heart disease according to the dataset target. Five algorithms are compared: Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes, Random Forest and Multi-Layer Perceptron. Data preparation is performed inside Scikit-learn pipelines. Models are evaluated using accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices. Permutation importance and an optional SHAP explanation are included to investigate influential input features. A Streamlit interface provides an interactive demonstration. This is an educational project, not a clinical diagnostic tool.

## 2. Problem statement
The objective is to use labelled historical records to train classification algorithms that learn patterns associated with the dataset's heart-disease target. The task is binary classification: target value 0 represents absence; target values 1, 2, 3 and 4 are grouped as presence.

## 3. Dataset description
The Cleveland subset contains 303 records and 14 columns: 13 input attributes and one target. Features include age, sex, chest-pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, slope, number of major vessels and thalassemia result. The dataset contains missing entries represented by `?`; these are treated as missing values and imputed within the model pipeline.

## 4. Algorithms
- **Decision Tree:** learns a sequence of feature-based decision rules.
- **K-Nearest Neighbours:** predicts using the labels of nearby training observations; feature scaling is important.
- **Gaussian Naive Bayes:** applies Bayes' theorem with a conditional-independence assumption and Gaussian likelihoods.
- **Random Forest:** combines predictions from multiple decision trees trained with randomization.
- **MLP Neural Network:** learns nonlinear relationships using hidden layers and backpropagation.

## 5. Methodology
1. Download the Cleveland subset from UCI.
2. Convert the original target to binary: `num == 0` → 0 and `num > 0` → 1.
3. Split data into 80% training and 20% test data using stratification and random state 42.
4. Impute numeric missing values with the training median and standardize numeric features.
5. Impute categorical missing values with the most frequent training value and one-hot encode categorical features.
6. Fit five classifiers in Scikit-learn pipelines.
7. Use 5-fold stratified cross-validation on training data and evaluate on the held-out test partition.
8. Compare metrics and confusion matrices; save the selected model.
9. Run permutation importance and optional SHAP interpretation.
10. Load the saved pipeline in the Streamlit application.

## 6. Evaluation metrics
- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** proportion of predicted positive cases that are truly positive.
- **Recall:** proportion of actual positive cases correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** ability to rank positive examples above negative examples across thresholds.
- **Confusion matrix:** counts of true negatives, false positives, false negatives and true positives.

For this dataset, recall and false-negative counts should be discussed alongside accuracy. No single metric establishes clinical usefulness.

## 7. Experimental results
**Important: populate this section only after running the code on the downloaded dataset. Do not invent or estimate scores.**

The program generates `outputs/model_comparison.csv`, which contains the actual measured scores. Insert that table here and include the five confusion-matrix figures and ROC plot from the outputs folder.

| Algorithm | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | Run experiment | Run experiment | Run experiment | Run experiment | Run experiment |
| KNN | Run experiment | Run experiment | Run experiment | Run experiment | Run experiment |
| Gaussian Naive Bayes | Run experiment | Run experiment | Run experiment | Run experiment | Run experiment |
| Random Forest | Run experiment | Run experiment | Run experiment | Run experiment | Run experiment |
| MLP Neural Network | Run experiment | Run experiment | Run experiment | Run experiment | Run experiment |

**Selected model:** The training script selects the model with the highest mean 5-fold cross-validation F1-score on the training partition. The held-out test metrics are reported for final comparison. Report the actual selected model from `models/model_metadata.json`.

## 8. Explainability
`src/explain.py` computes permutation feature importance by measuring the decrease in F1-score after shuffling each feature. It also attempts to generate a SHAP summary plot. The resulting figures should be interpreted as model explanations for this dataset, not causal relationships or medical advice.

## 9. Application
The Streamlit application accepts the 13 dataset features and displays the selected model's binary class and estimated positive-class probability. The interface includes a clear educational-use disclaimer.

## 10. Limitations and ethical considerations
The dataset is small, historical and limited to a particular cohort. It may contain sampling bias and may not generalize to other populations or current clinical practice. The classifier is not clinically validated, and its probability output must not be interpreted as a person's medical risk. Anyone with health concerns should consult a qualified healthcare professional.

## 11. Conclusion
This project demonstrates a complete supervised-classification workflow: data acquisition, preprocessing, model training, algorithm comparison, evaluation, feature interpretation and deployment of an interactive demonstration. The final conclusion about which algorithm performed best must be based on the measured results generated by the supplied code.

## References
1. UCI Machine Learning Repository, Heart Disease dataset, https://doi.org/10.24432/C52P4X.
2. Scikit-learn documentation, https://scikit-learn.org/stable/.
3. SHAP documentation, https://shap.readthedocs.io/.
4. Streamlit documentation, https://docs.streamlit.io/.
