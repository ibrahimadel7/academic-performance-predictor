# Streamlit Cloud Deployment Guide

## Prerequisites

Before deploying to Streamlit Cloud, ensure you have:
1. A GitHub account
2. This repository pushed to GitHub
3. Model artifacts generated (model.pkl, scaler.pkl, label_encoder.pkl)

## Step 1: Generate Model Artifacts

If you haven't generated the model files yet, run:

```bash
# Activate your virtual environment (if using one)
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Generate training data (if not already done)
python data/generate_data.py

# Train the model
python model/train.py
```

This will create the following files in the `model/` directory:
- `model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`

**IMPORTANT:** Commit and push these model files to your GitHub repository:

```bash
git add model/*.pkl
git commit -m "Add trained model artifacts"
git push
```

## Step 2: Deploy to Streamlit Cloud

### Option A: Deploy from Streamlit Cloud Dashboard

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with your GitHub account
3. Click "New app"
4. Configure your app:
   - **Repository:** `ibrahimadel7/academic-performance-predictor`
   - **Branch:** `copilot/add-student-performance-predictor` (or `main`)
   - **Main file path:** `frontend/app.py` or `streamlit_app.py`
5. Click "Deploy!"

### Option B: Deploy via URL

Visit this URL (replace with your GitHub username if needed):
```
https://share.streamlit.io/ibrahimadel7/academic-performance-predictor/copilot/add-student-performance-predictor/frontend/app.py
```

## Step 3: Verify Deployment

Once deployed, your app should be accessible at:
```
https://[your-app-name].streamlit.app
```

The app will:
1. Load the trained model artifacts
2. Display the prediction interface
3. Allow users to input student data and get grade predictions

## Local Testing

To test locally before deploying:

```bash
# Run from the project root
streamlit run frontend/app.py
```

Or use the main entry point:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Troubleshooting

### Model artifacts not found
- Make sure you've run `python model/train.py` to generate the model files
- Verify the .pkl files exist in the `model/` directory
- Ensure the model files are committed to Git and pushed to GitHub

### Import errors
- Check that `requirements.txt` is in the root directory
- Verify all dependencies are listed in `requirements.txt`

### App won't start on Streamlit Cloud
- Check the logs in the Streamlit Cloud dashboard
- Verify the file path in the deployment settings
- Ensure all model artifacts are in the repository

## Architecture

The app has been optimized for Streamlit Cloud:
- **Previous setup:** Separate FastAPI backend + Streamlit frontend
- **Current setup:** Unified Streamlit app with embedded ML inference
- **Benefits:** Single-service deployment, no CORS issues, simplified architecture

## Features

- 📚 Input student study hours, attendance, sleep hours, and midterm scores
- 🔮 Get predicted final grade (A, B, C, D, or F)
- 📊 View confidence scores and probability distribution
- 🎨 Beautiful UI with grade-specific colors and icons

## Tech Stack

- **Frontend:** Streamlit
- **ML Framework:** Scikit-learn
- **Model:** Logistic Regression
- **Data Processing:** Pandas, NumPy

---

**Note:** The old FastAPI backend (`backend/main.py`) is kept for reference but is no longer used in the Streamlit Cloud deployment.
