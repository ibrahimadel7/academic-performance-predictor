"""
Streamlit frontend for the Student Performance Predictor.
Sends prediction requests to the FastAPI backend and displays results.
"""

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
)

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


def call_predict_api(payload: dict) -> dict:
    """Send a POST request to the prediction endpoint and return JSON response."""
    response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


if st.button("🔮 Predict Grade", use_container_width=True, type="primary"):
    payload = {
        "study_hours": study_hours,
        "attendance": float(attendance),
        "sleep_hours": sleep_hours,
        "midterm_score": midterm_score,
    }

    with st.spinner("Predicting…"):
        try:
            result = call_predict_api(payload)
        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot connect to the backend. "
                "Make sure the FastAPI server is running: "
                "`uvicorn backend.main:app --reload`"
            )
            st.stop()
        except requests.exceptions.HTTPError as exc:
            st.error(f"❌ API error: {exc}")
            st.stop()

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
st.caption("Built with FastAPI · Scikit-learn · Streamlit")
