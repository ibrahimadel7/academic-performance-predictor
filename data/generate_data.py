"""
Script to generate synthetic student performance dataset.
"""

import numpy as np
import pandas as pd


def generate_student_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic student performance data with logical grade assignments."""
    rng = np.random.default_rng(random_state)

    study_hours = rng.uniform(0, 12, n_samples)
    attendance = rng.uniform(50, 100, n_samples)
    sleep_hours = rng.uniform(3, 10, n_samples)
    midterm_score = rng.uniform(0, 100, n_samples)

    # Compute a weighted performance score (0–100 scale)
    score = (
        (study_hours / 12) * 30
        + (attendance / 100) * 25
        + (sleep_hours / 10) * 10
        + (midterm_score / 100) * 35
    )

    # Add noise for realism
    noise = rng.normal(0, 5, n_samples)
    score = np.clip(score + noise, 0, 100)

    # Map score to grade
    def score_to_grade(s: float) -> str:
        if s >= 80:
            return "A"
        elif s >= 65:
            return "B"
        elif s >= 50:
            return "C"
        elif s >= 35:
            return "D"
        else:
            return "F"

    final_grade = [score_to_grade(s) for s in score]

    df = pd.DataFrame(
        {
            "study_hours": study_hours,
            "attendance": attendance,
            "sleep_hours": sleep_hours,
            "midterm_score": midterm_score,
            "final_grade": final_grade,
        }
    )
    return df


if __name__ == "__main__":
    import os

    output_path = os.path.join(os.path.dirname(__file__), "student_data.csv")
    df = generate_student_data()
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(df["final_grade"].value_counts())
