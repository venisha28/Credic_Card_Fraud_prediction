# Dataset Instructions

This project uses the Kaggle **Credit Card Fraud Detection** dataset.

## Dataset Overview
- **File Name**: `creditcard.csv`
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172% fraud rate)
- **Features**:
  - `Time`: Number of seconds elapsed between this transaction and the first transaction in the dataset.
  - `V1` through `V28`: Numerical features resulting from a PCA transformation (to protect privacy).
  - `Amount`: Transaction amount in USD.
  - `Class`: Target variable (`1` for fraud, `0` for legitimate).

## Where to Place the Dataset
Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it directly inside this directory:

```text
data/
└── creditcard.csv
```

Once placed, run the training pipeline:
```bash
python src/train.py
```
