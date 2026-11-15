"""
FastAPI High-Performance Production API for Fraud Detection.

EDUCATIONAL NOTE FOR INTERVIEWS:
1. Lifespan / Startup Event:
   Instead of loading the heavy model and scaler on every incoming request,
   we load it once into global application state during startup.
   This drops per-request latency from 400ms+ down to ~5-15ms!

2. Pydantic Type Enforcement:
   Any invalid types (e.g., negative amount, string instead of float) are rejected 
   at the API gateway layer before wasting compute on ML inference.
"""

from contextlib import asynccontextmanager
import json
import os
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import BatchTransactionPayload, HealthResponse, PredictionResponse, TransactionPayload
from src.inference import FraudInferenceEngine


engine: FraudInferenceEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Starting Fraud Detection Service... Loading ML artifacts.")
    try:
        engine = FraudInferenceEngine()
        print(f"Engine loaded successfully! Active Decision Threshold: {engine.threshold}")
    except Exception as e:
        print(f"Warning: Artifacts not found ({e}). Make sure to run src/train.py before serving.")
    yield
    print("Shutting down Fraud Detection Service.")


app = FastAPI(
    title="Production Real-Time Fraud Detection API",
    description="Low-Latency ML Serving with Risk Stratification and Drift Monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health & Readiness probe for Kubernetes / cloud hosts."""
    if engine is None or engine.model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model engine not loaded")

    metrics = {}
    if os.path.exists("models/model_metadata.json"):
        with open("models/model_metadata.json", "r") as f:
            meta = json.load(f)
            metrics = meta.get("metrics", {})

    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold": engine.threshold,
        "metrics": metrics,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(payload: TransactionPayload):
    """
    Real-time single transaction scoring endpoint.
    Returns fraud likelihood, risk tier, and recommended operational action.
    """
    if engine is None or engine.model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not loaded. Run training first.")

    tx_dict = payload.model_dump()
    result = engine.predict_single(tx_dict)
    return result


@app.post("/predict/batch")
def predict_transactions_batch(payload: BatchTransactionPayload):
    """
    High-throughput batch transaction scoring.
    """
    if engine is None or engine.model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not loaded.")

    tx_list = [tx.model_dump() for tx in payload.transactions]
    results = engine.predict_batch(tx_list)
    return {"total_transactions": len(results), "results": results}
