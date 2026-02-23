"""
Streamlit frontend for the Student Performance Predictor.
Includes embedded prediction logic for Streamlit Cloud deployment.
"""

import os
import sys
import joblib
import numpy as np
import streamlit as st

# Add parent directory to path to import model modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Load ML artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    """Load model, scaler, and label encoder from disk. Cached for performance."""
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error("❌ Model artifacts not found. Please train the model first by running `python model/train.py`")
        st.stop()
        return None, None, None

model, scaler, label_encoder = load_model_artifacts()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("🎓 Student Performance Predictor")
st.markdown(
    "Enter the student details below and click **Predict** to get the predicted final grade."
)
st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input(
        "📚 Study Hours (per day)",
        min_value=0.0,
        max_value=12.0,
        value=6.0,
        step=0.5,
        help="Average number of hours spent studying each day (0–12).",
    )
    sleep_hours = st.number_input(
        "😴 Sleep Hours (per night)",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5,
        help="Average number of hours of sleep per night (0–24).",
    )

with col2:
    attendance = st.slider(
        "📅 Attendance (%)",
        min_value=0,
        max_value=100,
        value=80,
        step=1,
        help="Class attendance percentage (0–100).",
    )
    midterm_score = st.number_input(
        "📝 Midterm Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0,
        help="Score obtained in the midterm exam (0–100).",
    )

st.divider()

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
GRADE_COLORS = {
    "A": "🟢",
    "B": "🔵",
    "C": "🟡",
    "D": "🟠",
    "F": "🔴",
}


def predict_grade(study_hours: float, attendance: float, sleep_hours: float, midterm_score: float) -> dict:
    """Make prediction using the loaded model."""
    try:
        features = np.array([[study_hours, attendance, sleep_hours, midterm_score]])
        features_scaled = scaler.transform(features)

        prediction_idx = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]

        grade = label_encoder.inverse_transform([prediction_idx])[0]
        confidence = float(proba[prediction_idx])

        probabilities = {
            cls: round(float(p), 4)
            for cls, p in zip(label_encoder.classes_, proba)
        }

        return {
            "predicted_grade": grade,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
        }
    except Exception as exc:
        st.error(f"❌ Prediction error: {str(exc)}")
        st.stop()
        return {}


if st.button("🔮 Predict Grade", use_container_width=True, type="primary"):
    with st.spinner("Predicting…"):
        result = predict_grade(
            study_hours=study_hours,
            attendance=float(attendance),
            sleep_hours=sleep_hours,
            midterm_score=midterm_score,
        )

    grade = result["predicted_grade"]
    confidence = result["confidence"]
    probabilities = result.get("probabilities", {})

    icon = GRADE_COLORS.get(grade, "⚪")

    st.success(f"### {icon} Predicted Grade: **{grade}**")
    st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")

    # Probability bar chart
    if probabilities:
        st.divider()
        st.subheader("📊 Grade Probabilities")
        import pandas as pd

        prob_df = (
            pd.DataFrame.from_dict(probabilities, orient="index", columns=["Probability"])
            .sort_index()
        )
        st.bar_chart(prob_df)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Built with Scikit-learn · Streamlit")
