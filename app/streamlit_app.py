from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Heart Disease Classification", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Classification")
st.caption("Educational machine-learning demonstration using the UCI Cleveland Heart Disease dataset")
st.warning("This application is for education and research only. It is not a medical device, diagnosis, screening test, or substitute for professional medical advice.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. From the project root, run: python src/download_data.py then python src/train.py")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
with st.form("prediction_form"):
    st.subheader("Enter the 13 dataset features")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", 18, 100, 55)
        sex = st.selectbox("Sex (1 = male, 0 = female)", [1, 0])
        cp = st.selectbox("Chest pain type (cp)", [1, 2, 3, 4], help="UCI coded category")
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 250, 130)
        chol = st.number_input("Serum cholesterol (mg/dl)", 80, 700, 240)
    with c2:
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        thalach = st.number_input("Maximum heart rate achieved (thalach)", 50, 250, 150)
        exang = st.selectbox("Exercise-induced angina (exang)", [0, 1])
        oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0, 0.1)
    with c3:
        slope = st.selectbox("Slope of peak exercise ST segment", [1, 2, 3])
        ca = st.selectbox("Major vessels (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", [3, 6, 7])
    submitted = st.form_submit_button("Run classification", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])
    pred = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    st.subheader("Model output")
    if pred == 1:
        st.error("Model class: Disease present (positive class)")
    else:
        st.success("Model class: No disease (negative class)")
    st.metric("Model-estimated positive-class probability", f"{probability:.1%}")
    st.caption("This probability reflects the model trained on a small historical dataset. It is not an individual's clinical risk estimate.")
    st.dataframe(row, use_container_width=True)
