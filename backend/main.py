"""
FastAPI backend for the Student Performance Predictor.

Endpoints:
  GET  /          → health check
  POST /predict   → predict final grade from student input features
"""

import os
import sys

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Allow importing model artefacts from sibling directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

app = FastAPI(title="Student Performance Predictor", version="1.0.0")


# ---------------------------------------------------------------------------
# Load ML artefacts at startup
# ---------------------------------------------------------------------------

def _load_artifacts():
    """Load model, scaler, and label encoder from disk."""
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        return model, scaler, label_encoder
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Model artefacts not found. Run model/train.py first."
        ) from exc


_model, _scaler, _label_encoder = _load_artifacts()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class StudentInput(BaseModel):
    study_hours: float = Field(..., ge=0, le=12, description="Daily study hours (0–12)")
    attendance: float = Field(..., ge=0, le=100, description="Attendance percentage (0–100)")
    sleep_hours: float = Field(..., ge=0, le=24, description="Daily sleep hours (0–24)")
    midterm_score: float = Field(..., ge=0, le=100, description="Midterm exam score (0–100)")


class PredictionResponse(BaseModel):
    predicted_grade: str
    confidence: float
    probabilities: dict[str, float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def health_check():
    """Return a simple health-check response."""
    return {"status": "ok", "message": "Student Performance Predictor API is running."}


@app.post("/predict", response_model=PredictionResponse, summary="Predict final grade")
def predict(student: StudentInput):
    """
    Accept student features and return the predicted final grade with confidence.
    """
    try:
        features = np.array(
            [[student.study_hours, student.attendance, student.sleep_hours, student.midterm_score]]
        )
        features_scaled = _scaler.transform(features)

        prediction_idx = _model.predict(features_scaled)[0]
        proba = _model.predict_proba(features_scaled)[0]

        grade = _label_encoder.inverse_transform([prediction_idx])[0]
        confidence = float(proba[prediction_idx])

        probabilities = {
            cls: round(float(p), 4)
            for cls, p in zip(_label_encoder.classes_, proba)
        }

        return PredictionResponse(
            predicted_grade=grade,
            confidence=round(confidence, 4),
            probabilities=probabilities,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
