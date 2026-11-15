"""
Unit & Integration Tests for ML Pipeline.
Run via: pytest tests/
"""

import os
import pytest
import numpy as np
from src.data_pipeline import generate_synthetic_credit_card_data
from src.monitor import detect_feature_drift


def test_data_generation_schema():
    df = generate_synthetic_credit_card_data(n_samples=1000)
    assert "Class" in df.columns
    assert "Time" in df.columns
    assert "Amount" in df.columns
    assert len(df) == 1000
    assert df["Class"].isin([0, 1]).all()


def test_data_imbalance_ratio():
    df = generate_synthetic_credit_card_data(n_samples=10000, fraud_ratio=0.002)
    fraud_count = df["Class"].sum()
    assert fraud_count > 0
    assert fraud_count < 100  # Imbalanced sanity check


def test_drift_detection_no_drift():
    df1 = generate_synthetic_credit_card_data(n_samples=500, random_state=1)
    df2 = generate_synthetic_credit_card_data(n_samples=500, random_state=2)
    report = detect_feature_drift(df1, df2)
    assert "dataset_drift_detected" in report
    assert isinstance(report["drift_share"], float)
