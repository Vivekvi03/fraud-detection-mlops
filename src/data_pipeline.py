"""
Data Pipeline Module for Fraud Detection
Handles data generation/loading, preprocessing, and stratified splitting.

EDUCATIONAL NOTE FOR INTERVIEWS:
1. Why RobustScaler? 
   StandardScaler assumes normal distribution and is heavily influenced by outliers. 
   Transaction amounts have extreme skewness and outliers (e.g., $1 micro-charges vs $10,000 fraud spikes).
   RobustScaler uses the median and interquartile range (IQR), making it immune to extreme values.

2. Why Stratified Sampling?
   When only 0.17% of data is fraud, a standard random split might put 90% of frauds in train 
   and almost none in test. StratifiedKFold or Stratified train_test_split ensures the exact 
   same fraud-to-normal ratio across all splits.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib
import os


def generate_synthetic_credit_card_data(n_samples: int = 100000, fraud_ratio: float = 0.0017, random_state: int = 42) -> pd.DataFrame:
    """
    Generates realistic credit card transaction data matching the Kaggle European Cardholders schema:
    - Time: Seconds elapsed since first transaction
    - V1 to V28: PCA transformed numerical features
    - Amount: Transaction amount
    - Class: 0 (Normal), 1 (Fraud)
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # Generate legitimate transactions
    normal_v = np.random.normal(loc=0.0, scale=1.0, size=(n_normal, 28))
    normal_time = np.sort(np.random.uniform(0, 172800, size=n_normal)) # 48 hours
    normal_amount = np.random.exponential(scale=50.0, size=n_normal) + 1.0 # mostly small amounts

    # Generate fraudulent transactions (shifted distributions on key PCA components like V4, V11, V14)
    fraud_v = np.random.normal(loc=0.0, scale=1.5, size=(n_fraud, 28))
    # In actual fraud studies, certain latent features (V4, V11, V14) show significant divergence
    fraud_v[:, 3] += np.random.normal(loc=4.0, scale=1.0, size=n_fraud)   # V4 higher in fraud
    fraud_v[:, 10] += np.random.normal(loc=3.5, scale=1.0, size=n_fraud)  # V11 higher in fraud
    fraud_v[:, 13] -= np.random.normal(loc=5.0, scale=1.0, size=n_fraud)  # V14 lower in fraud
    
    fraud_time = np.random.uniform(0, 172800, size=n_fraud)
    fraud_amount = np.random.choice([np.random.uniform(1.0, 5.0), np.random.uniform(300.0, 2500.0)], size=n_fraud)

    # Combine into DataFrames
    normal_df = pd.DataFrame(normal_v, columns=[f"V{i}" for i in range(1, 29)])
    normal_df["Time"] = normal_time
    normal_df["Amount"] = normal_amount
    normal_df["Class"] = 0

    fraud_df = pd.DataFrame(fraud_v, columns=[f"V{i}" for i in range(1, 29)])
    fraud_df["Time"] = fraud_time
    fraud_df["Amount"] = fraud_amount
    fraud_df["Class"] = 1

    df = pd.concat([normal_df, fraud_df], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def prepare_data(data_path: str = None, save_scaler_path: str = "models/scaler.joblib"):
    """
    Loads raw data (or generates realistic benchmark data), applies RobustScaler,
    and returns stratified train, validation, and test splits.
    """
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("Real data path not found. Generating realistic benchmark transaction data...")
        df = generate_synthetic_credit_card_data()

    # Features and Target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 1. Stratified Train-Test split (80% train+val, 20% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2. Stratified Train-Validation split (from the 80%, 25% val -> 60% train, 20% val, 20% test total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
    )

    # 3. Fit scaler ONLY on train data to prevent Data Leakage!
    scaler = RobustScaler()
    cols_to_scale = ["Time", "Amount"]
    
    # Copy to avoid SettingWithCopy warnings
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # Save scaler artifact
    os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
    joblib.dump(scaler, save_scaler_path)
    print(f"Scaler saved to {save_scaler_path}")

    print(f"Dataset summary:")
    print(f"  Train set: {len(X_train)} samples ({y_train.sum()} frauds, {y_train.mean():.4%})")
    print(f"  Val set:   {len(X_val)} samples ({y_val.sum()} frauds, {y_val.mean():.4%})")
    print(f"  Test set:  {len(X_test)} samples ({y_test.sum()} frauds, {y_test.mean():.4%})")

    return X_train, X_val, X_test, y_train, y_val, y_test
