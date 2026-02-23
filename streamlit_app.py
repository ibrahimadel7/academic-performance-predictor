"""
Main entry point for the Student Performance Predictor Streamlit app.
This file is used by Streamlit Cloud for deployment.
"""

import sys
import os

# Run the frontend app
if __name__ == "__main__":
    # Import and run the main app from frontend
    sys.path.insert(0, os.path.dirname(__file__))
    exec(open("frontend/app.py").read())
