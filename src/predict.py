import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import json
import pandas as pd
import numpy as np
from src.preprocessing import FEATURE_COLS, preprocess_features

MODEL_PATH = os.path.join("models", "best_model.joblib")
SCALER_PATH = os.path.join("models", "scaler.joblib")
METRICS_PATH = os.path.join("models", "model_metrics.json")

def load_model():
    """Loads the trained machine learning model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run src/train.py first.")
    return joblib.load(MODEL_PATH)

def load_scaler():
    """Loads the fitted StandardScaler for Time and Amount."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}. Please run src/train.py first.")
    return joblib.load(SCALER_PATH)

def load_metrics():
    """Loads saved model evaluation metrics."""
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"Metrics file not found at {METRICS_PATH}. Please run src/train.py first.")
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

def get_risk_level(prob: float) -> str:
    """Categorizes fraud risk into Low Risk or High Risk."""
    if prob >= 0.50:
        return "High Risk"
    else:
        return "Low Risk"

def predict_single_transaction(input_data: dict, model=None, scaler=None):
    """
    Predicts fraud for a single transaction input dictionary.
    """
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()
        
    df_single = pd.DataFrame([input_data])
    X_proc, _ = preprocess_features(df_single, scaler=scaler, fit_scaler=False)
    
    prob = float(model.predict_proba(X_proc)[0, 1])
    pred = int(model.predict(X_proc)[0])
    
    prediction_label = "Fraudulent" if pred == 1 else "Legitimate"
    risk_level = get_risk_level(prob)
    
    return {
        "fraud_probability": prob,
        "prediction_code": pred,
        "prediction": prediction_label,
        "risk_level": risk_level
    }

def predict_batch(df: pd.DataFrame, model=None, scaler=None):
    """
    Predicts fraud for a dataframe of uploaded transactions.
    Returns the dataframe augmented with Fraud_Probability, Prediction, and Risk_Level.
    """
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()
        
    df_out = df.copy()
    X_proc, _ = preprocess_features(df_out, scaler=scaler, fit_scaler=False)
    
    probs = model.predict_proba(X_proc)[:, 1]
    preds = model.predict(X_proc)
    
    df_out['Fraud_Probability'] = np.round(probs * 100, 2)
    df_out['Prediction'] = ['Fraudulent' if p == 1 else 'Legitimate' for p in preds]
    df_out['Risk_Level'] = [get_risk_level(p) for p in probs]
    
    return df_out
