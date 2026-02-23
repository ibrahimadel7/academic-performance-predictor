"""
Training script: trains a LogisticRegression model and evaluates it.
Saves model.pkl to the model directory.
"""

import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from preprocess import MODEL_DIR, load_data, preprocess

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")


def train() -> None:
    """Load data, train model, evaluate, and save artefacts."""
    df = load_data(DATA_PATH)

    X_scaled, y_encoded, scaler, label_encoder = preprocess(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Model: LogisticRegression")
    print(f"Accuracy: {acc:.4f}")
    print(f"\nClassification Report:\n")
    print(
        classification_report(
            y_test, y_pred, target_names=label_encoder.classes_
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"{'='*50}\n")

    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
