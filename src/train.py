import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocessing import prepare_train_test_data
from src.evaluation import evaluate_predictions

def train_all_models(data_path="data/creditcard.csv"):
    print("=" * 60)
    print("STARTING ML PIPELINE: CREDIT CARD FRAUD DETECTION")
    print("=" * 60)
    
    # 1. Load, split, scale, and balance (SMOTE ONLY ON TRAIN)
    X_train_res, y_train_res, X_test_scaled, y_test, scaler, feature_cols = prepare_train_test_data(
        data_path=data_path, test_size=0.2, random_state=42
    )
    
    # 2. Define Models
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
    }
    
    metrics_summary = {}
    trained_models = {}
    
    print("\nTraining and evaluating models...")
    for name, model in classifiers.items():
        print(f"\n---> Training {name}...")
        model.fit(X_train_res, y_train_res)
        
        # Predict on test set (un-smoted real test set)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Compute evaluation metrics
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics_summary[name] = metrics
        trained_models[name] = model
        
        print(f"[{name}] Results:")
        print(f"  - Precision : {metrics['precision']:.4f}")
        print(f"  - Recall    : {metrics['recall']:.4f}")
        print(f"  - F1-Score  : {metrics['f1']:.4f}")
        print(f"  - ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"  - PR-AUC    : {metrics['pr_auc']:.4f}")
        
    # Select best model based on PR-AUC / F1-Score
    best_name = max(metrics_summary, key=lambda k: metrics_summary[k]['pr_auc'])
    best_model = trained_models[best_name]
    
    print("\n" + "=" * 60)
    print(f"BEST MODEL SELECTED: {best_name}")
    print(f"PR-AUC: {metrics_summary[best_name]['pr_auc']:.4f} | F1-Score: {metrics_summary[best_name]['f1']:.4f}")
    print("=" * 60)
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    
    metadata = {
        'best_model_name': best_name,
        'features': feature_cols,
        'metrics': metrics_summary,
        'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("models/model_metrics.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\n[SUCCESS] Model pipeline saved to models/best_model.joblib")
    print("[SUCCESS] Scaler saved to models/scaler.joblib")
    print("[SUCCESS] Empirical metrics saved to models/model_metrics.json")

if __name__ == "__main__":
    train_all_models()
