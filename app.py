from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("tuned_model.pkl")
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .hero {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 24px 28px;
            margin-bottom: 22px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        .hero h1 {
            color: #0F172A;
            font-size: 2.2rem;
            line-height: 1.15;
            margin: 0 0 6px 0;
        }

        .hero p {
            color: #475569;
            font-size: 1.05rem;
            margin: 0;
        }

        .section-title {
            color: #0F172A;
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.02rem;
            text-transform: uppercase;
            border-bottom: 1px solid #E5E7EB;
            padding-bottom: 8px;
            margin: 22px 0 12px 0;
        }

        .result-card,
        .placeholder-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 24px;
            margin-top: 8px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        .risk-label {
            color: #475569;
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: 0.05rem;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .risk-score {
            font-size: 3.8rem;
            line-height: 1;
            font-weight: 900;
            margin: 0 0 8px 0;
        }

        .risk-classification {
            color: #0F172A;
            font-size: 1.35rem;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .interpretation {
            color: #0F172A;
            font-size: 1rem;
            line-height: 1.55;
            margin-bottom: 16px;
        }

        .insight-list {
            margin: 8px 0 0 0;
            padding-left: 18px;
        }

        .insight-list li {
            color: #0F172A;
            margin-bottom: 8px;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 18px;
        }

        .mini-metric {
            background: #F8FAFC;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 12px;
            color: #0F172A;
        }

        .mini-metric strong {
            color: #2563EB;
            display: block;
            font-size: 0.82rem;
            margin-bottom: 3px;
        }

        .footer {
            color: #475569;
            font-size: 0.86rem;
            text-align: center;
            padding: 22px 0 6px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "tuned_model.pkl was not found. Run `python train.py` before launching the app."
        )
    return joblib.load(MODEL_PATH)


def bool_label_to_value(label: str) -> bool:
    return label == "Yes"


def build_patient_dataframe(patient_inputs: dict) -> pd.DataFrame:
    patient_df = pd.DataFrame([patient_inputs], columns=FEATURE_COLUMNS)
    return patient_df


def predict_risk(model_package: dict, patient_df: pd.DataFrame) -> tuple[float, int]:
    model = model_package["model"]
    probability = float(model.predict_proba(patient_df)[0, 1])
    prediction = int(probability >= 0.50)
    return probability, prediction


def risk_style(probability: float) -> tuple[str, str]:
    if probability < 0.40:
        return "#16A34A", "Low Risk"
    if probability <= 0.60:
        return "#F59E0B", "Moderate Risk"
    return "#DC2626", "High Risk"


def build_feature_insights(patient_inputs: dict) -> list[str]:
    insights = []

    if patient_inputs["exang"]:
        insights.append("Exercise-induced angina is present, a strong model risk signal.")
    else:
        insights.append("No exercise-induced angina is present, which lowers model-estimated risk.")

    if patient_inputs["oldpeak"] >= 2.0:
        insights.append("ST depression is elevated, which tends to increase predicted risk.")
    elif patient_inputs["oldpeak"] > 0:
        insights.append("ST depression is present but not severely elevated.")
    else:
        insights.append("ST depression is minimal, which is generally a lower-risk signal.")

    if patient_inputs["chol"] >= 240:
        insights.append("Cholesterol is high, contributing to a higher risk estimate.")
    elif patient_inputs["thalch"] < 120:
        insights.append("Maximum heart rate is relatively low, which can increase model risk.")
    elif patient_inputs["cp"] == "asymptomatic":
        insights.append("Asymptomatic chest pain category is an important model driver.")
    else:
        insights.append("Chest pain and vital patterns are reviewed together by the model.")

    return insights[:3]


def validate_inputs(patient_inputs: dict) -> list[str]:
    issues = []
    if patient_inputs["trestbps"] <= 0:
        issues.append("Resting blood pressure must be greater than 0.")
    if patient_inputs["chol"] <= 0:
        issues.append("Cholesterol must be greater than 0.")
    if patient_inputs["thalch"] <= 0:
        issues.append("Maximum heart rate must be greater than 0.")
    return issues


def section_header(label: str) -> None:
    st.markdown(f'<div class="section-title">{label}</div>', unsafe_allow_html=True)


def render_input_form() -> tuple[bool, dict]:
    st.subheader("Patient Inputs")

    section_header("Demographics")
    age = st.slider("Age", min_value=28, max_value=90, value=54, step=1)
    sex = st.selectbox("Sex", options=["Male", "Female"])

    section_header("Symptoms")
    chest_pain = st.selectbox(
        "Chest pain type",
        options=["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
        index=3,
    )
    exercise_angina_label = st.selectbox("Exercise-induced angina", options=["No", "Yes"])

    section_header("Vital Measurements")
    resting_bp = st.slider("Resting blood pressure", min_value=80, max_value=220, value=130, step=1)
    cholesterol = st.slider("Cholesterol", min_value=100, max_value=620, value=240, step=1)
    max_heart_rate = st.slider("Max heart rate", min_value=60, max_value=220, value=150, step=1)
    oldpeak = st.slider("Oldpeak (ST depression)", min_value=0.0, max_value=6.5, value=1.0, step=0.1)

    section_header("Clinical Indicators")
    fasting_blood_sugar_label = st.selectbox("Fasting blood sugar > 120 mg/dl", options=["No", "Yes"])
    resting_ecg = st.selectbox(
        "Resting ECG",
        options=["normal", "st-t abnormality", "lv hypertrophy"],
    )
    slope = st.selectbox("Slope", options=["upsloping", "flat", "downsloping"], index=1)
    ca = st.slider("Number of vessels (ca)", min_value=0, max_value=3, value=0, step=1)
    thal = st.selectbox("Thal", options=["normal", "fixed defect", "reversable defect"])

    patient_inputs = {
        "age": age,
        "sex": sex,
        "cp": chest_pain,
        "trestbps": float(resting_bp),
        "chol": float(cholesterol),
        "fbs": bool_label_to_value(fasting_blood_sugar_label),
        "restecg": resting_ecg,
        "thalch": float(max_heart_rate),
        "exang": bool_label_to_value(exercise_angina_label),
        "oldpeak": float(oldpeak),
        "slope": slope,
        "ca": float(ca),
        "thal": thal,
    }

    submitted = st.button("Predict Risk", type="primary")
    return submitted, patient_inputs


def render_prediction(probability: float, prediction: int, patient_inputs: dict) -> None:
    color, risk_label = risk_style(probability)
    classification = "High Risk" if prediction == 1 else "Low Risk"
    if risk_label == "Moderate Risk":
        classification = "High Risk" if probability >= 0.50 else "Low Risk"

    if prediction == 1:
        interpretation = "Elevated likelihood of heart disease detected. Clinical follow-up recommended."
    else:
        interpretation = "Lower likelihood detected, but this is not a diagnosis."

    insights = build_feature_insights(patient_inputs)
    insight_items = "".join(f"<li>{item}</li>" for item in insights)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="risk-label">Estimated Risk Score</div>
            <div class="risk-score" style="color: {color};">{probability:.0%}</div>
            <div class="risk-classification">{classification}</div>
            <div class="interpretation">{interpretation}</div>
            <hr style="border: none; border-top: 1px solid #E5E7EB; margin: 1rem 0;">
            <div class="section-title" style="margin-top: 0;">Feature Insights</div>
            <ul class="insight-list">{insight_items}</ul>
            <div class="metric-strip">
                <div class="mini-metric"><strong>Model</strong>Tuned Random Forest</div>
                <div class="mini-metric"><strong>Threshold</strong>50%</div>
                <div class="mini-metric"><strong>Output</strong>Risk estimate</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="placeholder-card">
            <h3 style="margin-top: 0; color: #0F172A;">Prediction Output</h3>
            <p>Enter patient information on the left and click <strong>Predict Risk</strong> to view the estimated risk score, classification, and feature-level interpretation.</p>
            <p style="margin-bottom: 0;">The model is intended for educational risk estimation only.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Heart Disease Risk Predictor",
        page_icon="H",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <h1>Heart Disease Risk Predictor</h1>
            <p>Interpretable ML-based risk estimation using clinical features</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_column, right_column = st.columns([1.05, 1], gap="large")

    with left_column:
        submitted, patient_inputs = render_input_form()

    with right_column:
        st.subheader("Clinical Risk Summary")

        if not submitted:
            render_empty_state()
        else:
            issues = validate_inputs(patient_inputs)
            if issues:
                for issue in issues:
                    st.error(issue)
            else:
                try:
                    model_package = load_model()
                    patient_df = build_patient_dataframe(patient_inputs)
                    probability, prediction = predict_risk(model_package, patient_df)
                    render_prediction(probability, prediction, patient_inputs)
                except Exception as exc:
                    st.error(
                        "Something went wrong while generating the prediction. "
                        "Please confirm the model file exists and the inputs are valid."
                    )
                    st.caption(f"Technical detail: {exc}")

    st.markdown(
        '<div class="footer">Educational tool. Not for medical diagnosis.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
