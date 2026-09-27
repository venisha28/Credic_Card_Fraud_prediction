import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

FEATURE_COLS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

def validate_dataset(df: pd.DataFrame):
    """
    Validates that the uploaded dataframe matches the expected Credit Card schema.
    Returns: (is_valid, error_message)
    """
    if df.empty:
        return False, "The uploaded dataset is empty."
    
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing[:5])}"
        
    return True, ""

def preprocess_features(df: pd.DataFrame, scaler=None, fit_scaler=False):
    """
    Preprocesses transaction features:
    - Extracts Time, V1-V28, Amount columns.
    - Scales Time and Amount using StandardScaler (fitted on train set, applied on test/inference).
    - Preserves V1-V28 (already PCA transformed).
    """
    df_clean = df.copy()
    
    # Impute any missing numerical values with 0
    for col in FEATURE_COLS:
        if col not in df_clean.columns:
            df_clean[col] = 0.0
        else:
            df_clean[col] = df_clean[col].fillna(0.0)
            
    X_num = df_clean[['Time', 'Amount']]
    
    if fit_scaler:
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
    else:
        if scaler is not None:
            X_num_scaled = scaler.transform(X_num)
        else:
            X_num_scaled = X_num.values
            
    v_cols = [f'V{i}' for i in range(1, 29)]
    X_v = df_clean[v_cols].values
    
    # Combined feature matrix: Scaled Time, V1..V28, Scaled Amount
    X_processed = np.column_stack([X_num_scaled[:, 0:1], X_v, X_num_scaled[:, 1:2]])
    
    return X_processed, scaler

def prepare_train_test_data(data_path="data/creditcard.csv", test_size=0.2, random_state=42):
    """
    Loads dataset, performs train/test split FIRST, scales Time/Amount,
    and applies SMOTE ONLY on the training split to prevent data leakage.
    
    Returns:
    X_train_res, y_train_res, X_test_scaled, y_test, scaler, feature_cols
    """
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    is_valid, msg = validate_dataset(df)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {msg}")
        
    X = df[FEATURE_COLS]
    y = df['Class']
    
    # 1. Train/Test split BEFORE SMOTE or scaling to prevent leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set size: {len(X_train)} (Fraud: {sum(y_train)}), Test set size: {len(X_test)} (Fraud: {sum(y_test)})")
    
    # 2. Scale Time & Amount using scaler fit ONLY on training set
    scaler = StandardScaler()
    
    # Extract numerical cols to scale
    train_num = scaler.fit_transform(X_train[['Time', 'Amount']])
    test_num = scaler.transform(X_test[['Time', 'Amount']])
    
    v_cols = [f'V{i}' for i in range(1, 29)]
    
    X_train_scaled = np.column_stack([train_num[:, 0:1], X_train[v_cols].values, train_num[:, 1:2]])
    X_test_scaled = np.column_stack([test_num[:, 0:1], X_test[v_cols].values, test_num[:, 1:2]])
    
    # 3. Apply SMOTE strictly on training data
    print("Applying SMOTE on training split...")
    smote = SMOTE(sampling_strategy=0.3, random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Resampled Train set size: {len(X_train_res)} (Fraud: {sum(y_train_res)})")
    
    return X_train_res, y_train_res, X_test_scaled, y_test.values, scaler, FEATURE_COLS
