import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(
    page_title="Heart disease classification",
    page_icon="❤️",
    layout="wide",
)


def yes_no_label(value: int) -> str:
    return "Yes (1)" if value == 1 else "No (0)"


def sex_label(value: int) -> str:
    return "Male (1)" if value == 1 else "Female (0)"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


st.title("Heart disease classification")
st.caption(
    "An educational machine-learning demonstration using the UCI Cleveland dataset."
)
st.warning(
    "For education and research only. This model is not a diagnosis, screening test, "
    "or substitute for professional medical advice.",
    title="Educational demo only",
    icon=":material/info:",
)

if not MODEL_PATH.exists():
    st.error(
        "The trained model file is missing. Add `models/best_model.joblib` to the project.",
        title="Model file not found",
    )
    st.stop()

model = load_model()
metadata = load_metadata()

with st.sidebar:
    st.subheader("About this demo")
    st.markdown("**Dataset**  \nUCI Cleveland Heart Disease")
    st.markdown("**Inputs**  \n13 coded dataset features")
    if metadata.get("best_model"):
        st.markdown(f"**Selected model**  \n{metadata['best_model']}")
    st.divider()
    st.caption(
        "The positive class means the dataset target was greater than zero. "
        "Model outputs are for learning only."
    )

st.subheader("Enter patient features")
st.caption(
    "Use the dataset-coded values where shown. Submit the form to view the model output."
)

with st.form("prediction_form", border=False):
    profile, symptoms, exercise = st.columns(3, gap="medium")

    with profile:
        with st.container(border=True):
            st.markdown("#### Profile and resting measures")
            age = st.number_input("Age (years)", 18, 100, 55)
            sex = st.selectbox(
                "Sex",
                [1, 0],
                format_func=sex_label,
                help="Dataset coding: 1 = male, 0 = female.",
            )
            trestbps = st.number_input(
                "Resting blood pressure (mm Hg)", 80, 250, 130
            )
            chol = st.number_input("Serum cholesterol (mg/dL)", 80, 700, 240)

    with symptoms:
        with st.container(border=True):
            st.markdown("#### Symptoms and ECG")
            cp = st.selectbox(
                "Chest pain type (cp)",
                [1, 2, 3, 4],
                help="Choose the UCI dataset code.",
            )
            fbs = st.selectbox(
                "Fasting blood sugar above 120 mg/dL",
                [0, 1],
                format_func=yes_no_label,
            )
            restecg = st.selectbox(
                "Resting ECG (restecg)",
                [0, 1, 2],
                help="Choose the UCI dataset code.",
            )
            thalach = st.number_input(
                "Maximum heart rate achieved", 50, 250, 150
            )
            exang = st.selectbox(
                "Exercise-induced angina",
                [0, 1],
                format_func=yes_no_label,
            )

    with exercise:
        with st.container(border=True):
            st.markdown("#### Exercise and vessel markers")
            oldpeak = st.number_input(
                "ST depression (oldpeak)", 0.0, 10.0, 1.0, 0.1
            )
            slope = st.selectbox(
                "Peak exercise ST slope",
                [1, 2, 3],
                help="Choose the UCI dataset code.",
            )
            ca = st.selectbox(
                "Major vessels (ca)",
                [0, 1, 2, 3, 4],
                help="Choose the UCI dataset code.",
            )
            thal = st.selectbox(
                "Thalassemia (thal)",
                [3, 6, 7],
                help="Choose the UCI dataset code.",
            )

    submitted = st.form_submit_button(
        "Run classification",
        type="primary",
        width="stretch",
    )

if submitted:
    row = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }
        ]
    )
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])

    st.divider()
    st.subheader("Model output")
    result, probability_card = st.columns(2, gap="medium")

    with result:
        with st.container(border=True):
            if prediction == 1:
                st.warning(
                    "Positive class: disease present",
                    title="Model prediction",
                    icon=":material/info:",
                )
            else:
                st.success(
                    "Negative class: no disease",
                    title="Model prediction",
                    icon=":material/check_circle:",
                )

    with probability_card:
        with st.container(border=True):
            st.metric(
                "Model-estimated positive-class probability",
                f"{probability:.1%}",
                help="This is the model's output, not an individual's clinical risk estimate.",
            )

    st.caption(
        "This probability reflects a model trained on a small historical dataset. "
        "It is not clinically validated and should not guide healthcare decisions."
    )

    with st.expander("Review submitted values"):
        submitted_values = pd.DataFrame(
            {"Feature": row.columns, "Value": row.iloc[0].to_list()}
        )
        st.dataframe(submitted_values, hide_index=True, width="stretch")
