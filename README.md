# 💳 Credit Card Fraud Detection - Machine Learning Pipeline

An end-to-end Machine Learning educational & resume project demonstrating data preprocessing, class imbalance handling with SMOTE, comparative model evaluation, and an interactive Streamlit analytics dashboard.

---

## 📌 Project Overview
Credit card fraud detection is a classic highly imbalanced classification problem. In real-world financial datasets, fraudulent transactions account for less than 0.2% of all transactions. 

This project demonstrates the complete machine learning workflow:
```text
Dataset → Preprocessing → Class Imbalance Handling (SMOTE) → Model Training → Evaluation → Deployment (Streamlit)
```

---

## 🎯 Problem Statement
To build a robust Machine Learning model capable of identifying fraudulent credit card transactions while minimizing false positives and avoiding data leakage.

Key challenges addressed:
1. **Extreme Class Imbalance**: Fraud instances represent ~0.172% of total transactions.
2. **Preventing Data Leakage**: SMOTE oversampling and feature scaling must be applied **strictly to the training set**, never to the test evaluation set.
3. **Evaluating Beyond Accuracy**: Standard accuracy is misleading (a naive model predicting all transactions as legitimate achieves 99.83% accuracy). Models must be evaluated using **Precision, Recall, F1-Score, ROC-AUC, and PR-AUC**.

---

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Machine Learning**: `scikit-learn`, `xgboost`, `imbalanced-learn`
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Frontend / UI**: `streamlit`
- **Model Persistence**: `joblib`

---

## 📊 Dataset Information
- **Source**: Kaggle Credit Card Fraud Detection Dataset (284,807 transactions)
- **Features**:
  - `Time`: Seconds elapsed since the first transaction in the dataset.
  - `Amount`: Transaction amount in USD.
  - `V1` – `V28`: Principal Component Analysis (PCA) features engineered for privacy protection.
  - `Class`: Target binary label (`1` = Fraudulent, `0` = Legitimate).

---

## 🔄 Machine Learning Pipeline Workflow

1. **Data Ingestion & Validation**:
   - Validates feature columns (`Time`, `V1`..`V28`, `Amount`, `Class`).
2. **Train/Test Splitting**:
   - 80/20 Stratified Train/Test split (`X_train`, `X_test`, `y_train`, `y_test`).
3. **Standard Scaling**:
   - `StandardScaler` is fitted on `X_train[['Time', 'Amount']]` and applied to `X_test`.
   - `V1` through `V28` are preserved as they are already PCA-transformed.
4. **Class Imbalance Handling**:
   - `SMOTE` (Synthetic Minority Over-sampling Technique) is applied **ONLY** to `X_train`, oversampling minority fraud instances to a 30% ratio.
   - The test set (`X_test`) remains untouched to reflect true real-world distribution.
5. **Model Training & Comparison**:
   - **Logistic Regression**
   - **Random Forest Classifier**
   - **XGBoost Classifier**
6. **Empirical Model Evaluation**:
   - Models are evaluated on the untouched test set using Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
7. **Model Selection & Artifact Storage**:
   - The best performing model pipeline and scaler are exported to `models/best_model.joblib` and `models/scaler.joblib`.

---

## ⚖️ Class Imbalance Handling & Data Leakage Prevention

> [!IMPORTANT]
> **Why SMOTE is applied ONLY to the Training Split:**
> Applying oversampling before splitting the dataset causes **data leakage**. Synthetic samples generated from test-set observations leak into the training process, causing overly optimistic, invalid evaluation metrics. In this project:
> 1. Data is split into 80% train and 20% test FIRST.
> 2. `StandardScaler` is fit on `X_train` ONLY.
> 3. `SMOTE` is fit on `X_train` ONLY.
> 4. `X_test` remains raw and un-oversampled to measure true model generalization.

---

## 📈 Empirical Model Evaluation Results

All metrics below are derived directly from model training on the Kaggle credit card dataset (`data/creditcard.csv`):

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Selected Best |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.1603 | **0.8980** | 0.2720 | 0.9679 | 0.7441 | ❌ |
| **Random Forest** | **0.6589** | 0.8673 | **0.7489** | **0.9784** | **0.8506** | **🏆 YES** |
| **XGBoost Classifier** | 0.5425 | 0.8469 | 0.6614 | 0.9770 | 0.8501 | ❌ |

---

## 💡 Why Accuracy is Insufficient for Fraud Detection

In highly imbalanced datasets (0.172% fraud):
- A dummy classifier that predicts `0` (Legitimate) for every single transaction will achieve **99.83% Accuracy**, but will catch **0% of fraud cases** (0% Recall).
- **Precision** measures: *Of all transactions predicted as fraud, how many were actually fraud?*
- **Recall** measures: *Of all actual fraudulent transactions, how many did the model catch?*
- **PR-AUC (Precision-Recall Area Under Curve)** is the gold-standard metric for imbalanced evaluation because it ignores true negatives (which dominate imbalanced datasets) and focuses on minority class trade-offs.

---

## 🚀 How to Run the Application

### 1. Prerequisites & Dependencies
Ensure Python 3.10+ is installed. Install requirements:
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Place `creditcard.csv` in the `data/` directory:
```text
data/
└── creditcard.csv
```

### 3. Run Training Pipeline
To train models and generate saved artifacts:
```bash
python src/train.py
```

### 4. Launch Streamlit Web Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📁 Project Directory Structure

```text
Credic_Card_Fraud_prediction/
│
├── app.py                     # Streamlit frontend (Dashboard, Single & Batch Prediction)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation & interview guide
├── .gitignore                 # Git ignore rules
│
├── data/
│   ├── README.md              # Dataset instructions
│   └── creditcard.csv         # Kaggle Credit Card dataset
│
├── models/
│   ├── best_model.joblib      # Saved best model (Random Forest)
│   ├── scaler.joblib          # Saved StandardScaler
│   └── model_metrics.json     # Empirical evaluation metrics JSON
│
├── src/
│   ├── preprocessing.py       # Validation, Scaling, Train/Test Split, SMOTE
│   ├── train.py               # Model training script
│   ├── predict.py             # Single and batch prediction logic
│   └── evaluation.py          # Metric calculations & Matplotlib plotting
│
└── notebooks/
    └── model_training.ipynb   # Jupyter notebook demonstrating the ML pipeline
```

---

## 🎯 Technical Interview Cheatsheet

Here are key technical concepts to discuss when presenting this project in an interview:

1. **Why SMOTE?**:
   - Class imbalance causes models to bias towards the majority class. SMOTE creates synthetic samples along line segments connecting k-nearest neighbors in minority class feature space.
2. **Preventing Data Leakage**:
   - Preprocessing and resampling must strictly be performed inside cross-validation or after train/test splitting.
3. **Choice of Model**:
   - **Random Forest** achieved the highest PR-AUC (0.8506) and F1-Score (0.7489) due to its ensemble tree structure handling non-linear PCA features effectively without overfitting.
4. **Precision vs. Recall Tradeoff**:
   - In banking, higher Recall ensures fraud is caught, but lower Precision means more legitimate customers experience false decline alerts. Fine-tuning probability thresholds balances customer experience with loss prevention.
