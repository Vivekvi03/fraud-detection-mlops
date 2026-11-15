"""
Model Training Module with XGBoost & Threshold Optimization.

EDUCATIONAL NOTE FOR INTERVIEWS:
1. Why scale_pos_weight instead of SMOTE?
   SMOTE generates synthetic samples in feature space, which can create noisy/unrealistic 
   transactions in high-dimensional PCA spaces and distorts calibrated probabilities.
   `scale_pos_weight = (count(negative) / count(positive))` instructs XGBoost to penalize 
   False Negatives proportionally to the class imbalance, keeping the true data distribution intact.

2. Why NOT use default 0.5 threshold?
   In fraud detection, default 0.5 probability is arbitrary. If a transaction has a 20% (0.20) 
   chance of being fraud, the bank should already flag it for step-up verification (OTP/fraud check).
   We evaluate the Precision-Recall curve to find the optimal operating threshold (tau).
"""

import json
import os
import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.data_pipeline import prepare_data


def train_and_evaluate(
    data_path: str = None,
    model_save_path: str = "models/fraud_model.joblib",
    meta_save_path: str = "models/model_metadata.json",
):
    print("=== Step 1: Preparing Data ===")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(data_path)

    # Calculate scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"\nCalculated scale_pos_weight: {scale_pos_weight:.2f}")

    print("\n=== Step 2: Training XGBoost Model ===")
    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric=["logloss", "aucpr"],
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print("\n=== Step 3: Threshold Tuning on Validation Set ===")
    # Predict probabilities on validation set
    y_val_probs = model.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_probs)
    
    # Calculate F1 score for each threshold (avoid divide-by-zero)
    f1_scores = (2 * precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
    best_idx = np.argmax(f1_scores)
    
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])
    
    print(f"Optimal Decision Threshold determined: {optimal_threshold:.4f} (Max Val F1: {best_f1:.4f})")

    print("\n=== Step 4: Final Evaluation on Unseen Test Set ===")
    y_test_probs = model.predict_proba(X_test)[:, 1]
    y_test_preds = (y_test_probs >= optimal_threshold).astype(int)

    pr_auc = float(average_precision_score(y_test, y_test_probs))
    roc_auc = float(roc_auc_score(y_test, y_test_probs))
    cm = confusion_matrix(y_test, y_test_preds)

    print(f"Test PR-AUC Score:  {pr_auc:.4f}")
    print(f"Test ROC-AUC Score: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives (Legit classified as Legit):   {cm[0][0]}")
    print(f"  False Positives (Legit flagged as Fraud):     {cm[0][1]}")
    print(f"  False Negatives (Fraud missed - CRITICAL!):   {cm[1][0]}")
    print(f"  True Positives (Fraud caught):                {cm[1][1]}")

    print("\nDetailed Classification Report:")
    report = classification_report(y_test, y_test_preds, output_dict=True)
    print(classification_report(y_test, y_test_preds))

    # Save Model Artifacts
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Trained model saved to: {model_save_path}")

    # Save Metadata for Production Serving & Interviews
    metadata = {
        "model_type": "XGBClassifier",
        "optimal_threshold": optimal_threshold,
        "metrics": {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "fraud_recall": report["1"]["recall"],
            "fraud_precision": report["1"]["precision"],
            "fraud_f1": report["1"]["f1-score"],
        },
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1]),
        },
        "features": list(X_train.columns),
    }

    with open(meta_save_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata & metrics saved to: {meta_save_path}")

    return model, metadata


if __name__ == "__main__":
    train_and_evaluate()
