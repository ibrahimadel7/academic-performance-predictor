"""
Preprocessing module: loads data, encodes target, and scales features.
Saves scaler.pkl and label_encoder.pkl to the model directory.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

MODEL_DIR = os.path.dirname(__file__)
FEATURE_COLS = ["study_hours", "attendance", "sleep_hours", "midterm_score"]
TARGET_COL = "final_grade"


def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(csv_path)


def preprocess(
    df: pd.DataFrame,
    fit: bool = True,
    scaler: StandardScaler | None = None,
    label_encoder: LabelEncoder | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Preprocess the dataset.

    Args:
        df: Raw DataFrame.
        fit: If True, fit scaler and encoder on the data; otherwise transform only.
        scaler: Pre-fitted scaler (required when fit=False).
        label_encoder: Pre-fitted encoder (required when fit=False).

    Returns:
        X_scaled, y_encoded, scaler, label_encoder
    """
    X = df[FEATURE_COLS].values
    y_raw = df[TARGET_COL].values

    if fit:
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        X_scaled = scaler.fit_transform(X)
        y_encoded = label_encoder.fit_transform(y_raw)

        # Persist artefacts
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
        joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    else:
        X_scaled = scaler.transform(X)
        y_encoded = label_encoder.transform(y_raw)

    return X_scaled, y_encoded, scaler, label_encoder


def load_artifacts() -> tuple[StandardScaler, LabelEncoder]:
    """Load pre-fitted scaler and label encoder from disk."""
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return scaler, label_encoder
