"""
Low-Latency Inference Engine for Fraud Scoring.

EDUCATIONAL NOTE FOR INTERVIEWS:
1. Warm Loading:
   In production, models must be loaded into memory once during service initialization,
   never re-read from disk on every HTTP request. This keeps inference latency under 20-30ms.

2. Risk Stratification:
   Binary outputs (0 or 1) aren't enough for business operations.
   We stratify transactions into risk tiers:
   - LOW (< 0.20): Auto-approved
   - MEDIUM (0.20 to Threshold): Soft flag / 2FA OTP prompt
   - HIGH (>= Threshold): Hard block or manual analyst review
"""

import json
import os
import time
from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd


class FraudInferenceEngine:
    def __init__(
        self,
        model_path: str = "models/fraud_model.joblib",
        scaler_path: str = "models/scaler.joblib",
        meta_path: str = "models/model_metadata.json",
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.meta_path = meta_path
        self.model = None
        self.scaler = None
        self.threshold = 0.5
        self.features = []
        self._load_artifacts()

    def _load_artifacts(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}. Train model first!")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler artifact not found at {self.scaler_path}. Train model first!")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                meta = json.load(f)
                self.threshold = meta.get("optimal_threshold", 0.5)
                self.features = meta.get("features", [])
        else:
            self.threshold = 0.5
            self.features = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

    def predict_single(self, transaction: Dict[str, float]) -> Dict[str, Any]:
        """
        Processes a single transaction dict and returns the prediction result with latency metrics.
        """
        start_time = time.perf_counter()

        # Convert to DataFrame
        df = pd.DataFrame([transaction])

        # Ensure all expected columns are present
        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.features]

        # Apply RobustScaler to Time and Amount
        df[["Time", "Amount"]] = self.scaler.transform(df[["Time", "Amount"]])

        # Model Inference
        prob = float(self.model.predict_proba(df)[0, 1])
        is_fraud = bool(prob >= self.threshold)

        # Risk Stratification
        if prob >= self.threshold:
            risk_tier = "HIGH"
            action = "DECLINE_AND_NOTIFY"
        elif prob >= (self.threshold * 0.6):
            risk_tier = "MEDIUM"
            action = "TRIGGER_STEP_UP_2FA"
        else:
            risk_tier = "LOW"
            action = "APPROVE"

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "is_fraud": is_fraud,
            "fraud_probability": round(prob, 4),
            "threshold_used": round(self.threshold, 4),
            "risk_tier": risk_tier,
            "recommended_action": action,
            "latency_ms": elapsed_ms,
        }

    def predict_batch(self, transactions: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        High-throughput batch prediction.
        """
        start_time = time.perf_counter()
        df = pd.DataFrame(transactions)
        
        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.features]

        df[["Time", "Amount"]] = self.scaler.transform(df[["Time", "Amount"]])
        probs = self.model.predict_proba(df)[:, 1]

        results = []
        for prob in probs:
            p = float(prob)
            if p >= self.threshold:
                tier = "HIGH"
                act = "DECLINE_AND_NOTIFY"
            elif p >= (self.threshold * 0.6):
                tier = "MEDIUM"
                act = "TRIGGER_STEP_UP_2FA"
            else:
                tier = "LOW"
                act = "APPROVE"

            results.append({
                "is_fraud": bool(p >= self.threshold),
                "fraud_probability": round(p, 4),
                "risk_tier": tier,
                "recommended_action": act,
            })
        return results
