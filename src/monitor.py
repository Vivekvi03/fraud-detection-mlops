"""
Model & Data Drift Monitoring Engine.

EDUCATIONAL NOTE FOR INTERVIEWS:
1. What is Data Drift (Covariate Shift)?
   Over time, real-world customer behaviors or fraud patterns change. 
   For example, during holiday sales, transaction amounts spike legitimately, 
   or fraudsters shift from micro-charges ($1) to high-value gift card purchases ($800).
   This means the distribution of inputs P(X) has drifted from what the model was trained on!

2. How do we catch it without waiting for chargebacks?
   Chargebacks take 30-90 days to register. If you wait for labels (P(Y|X)), you lose millions.
   We monitor statistical divergence between Reference Data (Training) and Current Production Data:
   - Kolmogorov-Smirnov (KS) test or Wasserstein Distance for numerical features.
   - P-value threshold (typically p < 0.05 indicates significant drift).
"""

import json
import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    p_value_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes Kolmogorov-Smirnov test for all numerical features between 
    reference (training) distribution and current production traffic.
    """
    drifted_features = []
    feature_metrics = {}

    common_columns = [
        c for c in reference_df.columns 
        if c in current_df.columns and c not in ["Class", "is_fraud"]
    ]

    for col in common_columns:
        stat, p_value = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
        is_drifted = bool(p_value < p_value_threshold)

        if is_drifted:
            drifted_features.append(col)

        feature_metrics[col] = {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "is_drifted": is_drifted,
            "ref_mean": round(float(reference_df[col].mean()), 4),
            "current_mean": round(float(current_df[col].mean()), 4),
        }

    dataset_drift_ratio = len(drifted_features) / max(len(common_columns), 1)
    dataset_drift_detected = dataset_drift_ratio >= 0.20  # If >20% features drift, raise alert!

    return {
        "dataset_drift_detected": dataset_drift_detected,
        "drift_share": round(dataset_drift_ratio, 4),
        "total_features_evaluated": len(common_columns),
        "number_of_drifted_features": len(drifted_features),
        "drifted_features": drifted_features,
        "feature_details": feature_metrics,
    }


def generate_drift_simulation(reference_df: pd.DataFrame, drift_scenario: str = "high_amount") -> pd.DataFrame:
    """
    Simulates production traffic scenarios to demonstrate drift detection in the dashboard.
    """
    simulated = reference_df.sample(n=min(len(reference_df), 1000), random_state=123).copy()

    if drift_scenario == "high_amount":
        # Simulating holiday shopping season or massive spike in high-ticket transactions
        simulated["Amount"] = simulated["Amount"] * np.random.uniform(4.0, 10.0, size=len(simulated))
        simulated["V4"] = simulated["V4"] + np.random.normal(loc=3.0, scale=1.0, size=len(simulated))
    elif drift_scenario == "botnet_attack":
        # Micro-transactions rapid fire
        simulated["Amount"] = np.random.uniform(0.1, 2.0, size=len(simulated))
        simulated["V11"] = simulated["V11"] + np.random.normal(loc=4.0, scale=0.5, size=len(simulated))
        simulated["V14"] = simulated["V14"] - np.random.normal(loc=4.5, scale=0.5, size=len(simulated))
    
    return simulated
