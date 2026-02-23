# Student Performance Predictor 🎓

An end-to-end Machine Learning project that predicts a student's **final grade** (A, B, C, D, F) based on study habits and academic performance features.

**✨ Ready for Streamlit Cloud Deployment!**

## Tech Stack

| Layer      | Technology                         |
|------------|------------------------------------|
| ML         | scikit-learn (LogisticRegression)  |
| Frontend   | Streamlit                          |
| Backend    | FastAPI + Uvicorn (legacy)         |
| Language   | Python 3.10+                       |

## Project Structure

```
student-performance-ml/
│
├── data/
│   ├── generate_data.py      # Synthetic dataset generator
│   └── student_data.csv      # Generated dataset (1,000 samples)
│
├── model/
│   ├── preprocess.py         # Feature scaling & label encoding
│   ├── train.py              # Model training & evaluation
│   ├── model.pkl             # Trained LogisticRegression model
│   ├── scaler.pkl            # Fitted StandardScaler
│   └── label_encoder.pkl     # Fitted LabelEncoder
│
├── backend/
│   └── main.py               # FastAPI application
│
├── frontend/
│   └── app.py                # Streamlit GUI
│
├── requirements.txt
└── README.md
```

## Quickstart

### Local Development

#### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

#### 2 — Generate dataset & train model

```bash
# Generate synthetic data
python data/generate_data.py

# Train model (also saves scaler.pkl and label_encoder.pkl)
python model/train.py
```

#### 3 — Run the Streamlit app

**Option A: Direct run (recommended)**
```bash
streamlit run frontend/app.py
```

**Option B: Using main entry point**
```bash
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

### ☁️ Streamlit Cloud Deployment

The app is optimized for one-click deployment to Streamlit Cloud!

1. Push this repository to GitHub (including the `.pkl` model files)
2. Visit [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub repository
4. Set main file to: `frontend/app.py`
5. Deploy!

📖 **See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.**

### Legacy: FastAPI Backend (Optional)

<details>
<summary>Click to expand FastAPI backend instructions</summary>

The project originally used a separate FastAPI backend. This is still available but not needed for Streamlit Cloud deployment.

#### Start the FastAPI backend

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

</details>

## API Reference

### `GET /`
Health check — returns `{"status": "ok", ...}`.

### `POST /predict`

**Request body:**

```json
{
  "study_hours": 7.0,
  "attendance": 85.0,
  "sleep_hours": 7.5,
  "midterm_score": 78.0
}
```

**Response:**

```json
{
  "predicted_grade": "B",
  "confidence": 0.64,
  "probabilities": {
    "A": 0.08,
    "B": 0.64,
    "C": 0.22,
    "D": 0.05,
    "F": 0.01
  }
}
```

## Input Features

| Feature         | Range    | Description                          |
|-----------------|----------|--------------------------------------|
| `study_hours`   | 0 – 12   | Average daily study hours            |
| `attendance`    | 0 – 100  | Class attendance percentage          |
| `sleep_hours`   | 0 – 24   | Average nightly sleep hours          |
| `midterm_score` | 0 – 100  | Midterm exam score                   |

## Grade Mapping

| Score Range | Grade |
|-------------|-------|
| ≥ 80        | A     |
| 65 – 79     | B     |
| 50 – 64     | C     |
| 35 – 49     | D     |
| < 35        | F     |