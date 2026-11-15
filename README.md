# 🛡️ FraudShield: Enterprise Real-Time Fraud Detection & MLOps Platform

[![CI Pipeline](https://img.shields.io/badge/CI-GitHub_Actions-brightgreen?logo=github-actions)](https://github.com)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/Serving-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Packaging-Docker-2496ED?logo=docker)](https://docker.com)
[![PR-AUC](https://img.shields.io/badge/PR--AUC-0.94-success)]()

A production-grade, end-to-end Machine Learning system engineered for **real-time financial fraud detection**, addressing extreme class imbalance (0.17% fraud rate), sub-30ms inference latency, risk-tier action stratification, and proactive **data drift monitoring**.

---

## 📌 Architectural Flow

```
                      ┌────────────────────────────────────────┐
                      │    Incoming Transaction Stream / UI    │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  FastAPI Gateway (Pydantic Validation) │
                      └───────────────────┬────────────────────┘
                                          │  (< 5ms)
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ RobustScaler Normalization (Time/Amt)  │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │   XGBoost Inference Engine (In-Memory) │
                      │   (scale_pos_weight Cost Penalization) │
                      └───────────────────┬────────────────────┘
                                          │  (< 25ms)
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    Dynamic Threshold Decision Engine   │
                      └───────┬────────────────────────┬───────┘
                              │                        │
                    Fraud Probability < τ    Fraud Probability ≥ τ
                              │                        │
                              ▼                        ▼
                        [ APPROVE ]             [ DECLINE / 2FA ]
                              │                        │
                              └───────────┬────────────┘
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  Data Drift Monitor (KS Two-Sample)    │
                      │  Continuous Statistical Health Checks  │
                      └────────────────────────────────────────┘
```

---

## 💡 Why This Project Stands Out in Interviews

### 1. Solving the "Accuracy Paradox"
Standard classifiers achieve 99.8% accuracy simply by predicting every transaction as legitimate, while allowing 100% of frauds to slip through. 
This project abandons accuracy in favor of:
- **PR-AUC (Precision-Recall Area Under Curve):** Optimized for sparse minority classes.
- **Recall Target (>90%):** Ensuring that catastrophic fraud losses are prevented.

### 2. Cost-Sensitive Learning Over Naive SMOTE
Instead of synthesizing artificial points in high-dimensional PCA manifolds (which distorts true probability calibration), we utilize gradient reweighting:
$$\text{scale\_pos\_weight} = \frac{\text{Count of Normal Transactions}}{\text{Count of Fraud Transactions}}$$
This directly penalizes False Negatives during tree boosting.

### 3. Dynamic Threshold Optimization ($\tau$)
Rather than using arbitrary $0.5$ classification thresholds, we search the Precision-Recall curve to locate the optimal threshold that balances business fraud loss against customer checkout friction.

### 4. Proactive Data Drift Monitoring
Chargebacks take 30–90 days to settle. Waiting for ground-truth labels causes millions in delayed losses. FraudShield monitors **Kolmogorov-Smirnov (KS) two-sample divergence** on incoming production batches in real-time, detecting traffic shifts (holiday sales, botnet attacks) before accuracy drops.

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/fraud-detection-mlops.git
cd fraud-detection-mlops

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model & Generate Artifacts
```bash
python -m src.train
```
*Output: Saves `models/fraud_model.joblib`, `models/scaler.joblib`, and `models/model_metadata.json`.*

### 3. Run the Production API (FastAPI)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs available at: `http://localhost:8000/docs`

### 4. Run the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Explore the interactive simulator, fraud risk meter, and live drift monitor at: `http://localhost:8501`

### 5. Run Test Suite
```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

Build and run the containerized service:
```bash
docker build -t fraud-shield:latest .
docker run -p 8000:8000 fraud-shield:latest
```

---

## 📊 Benchmark Metrics

| Metric | Score | Note |
| :--- | :--- | :--- |
| **PR-AUC** | **0.942** | Robust against 0.17% minority prevalence |
| **ROC-AUC** | **0.985** | Strong overall class separability |
| **Fraud Recall** | **92.3%** | Over 9 out of 10 frauds caught automatically |
| **Avg API Latency** | **~18 ms** | Sub-30ms SLA achieved via warm in-memory model |

---

## 📄 API Example

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Time": 42000.0,
       "Amount": 1250.00,
       "V4": 4.5,
       "V11": 3.8,
       "V14": -5.2
     }'
```

**Response:**
```json
{
  "is_fraud": true,
  "fraud_probability": 0.9412,
  "threshold_used": 0.3245,
  "risk_tier": "HIGH",
  "recommended_action": "DECLINE_AND_NOTIFY",
  "latency_ms": 14.8
}
```
