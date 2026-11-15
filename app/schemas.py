"""
Pydantic Schemas for Request & Response Data Validation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TransactionPayload(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since the reference start", example=120.0)
    Amount: float = Field(..., description="Transaction amount in currency", example=149.99)
    # PCA components V1 to V28
    V1: float = Field(0.0, example=-1.359807)
    V2: float = Field(0.0, example=-0.072781)
    V3: float = Field(0.0, example=2.536347)
    V4: float = Field(0.0, example=1.378155)
    V5: float = Field(0.0, example=-0.338321)
    V6: float = Field(0.0, example=0.462388)
    V7: float = Field(0.0, example=0.239599)
    V8: float = Field(0.0, example=0.098698)
    V9: float = Field(0.0, example=0.363787)
    V10: float = Field(0.0, example=0.090794)
    V11: float = Field(0.0, example=-0.551600)
    V12: float = Field(0.0, example=-0.617801)
    V13: float = Field(0.0, example=-0.991390)
    V14: float = Field(0.0, example=-0.311169)
    V15: float = Field(0.0, example=1.468177)
    V16: float = Field(0.0, example=-0.470401)
    V17: float = Field(0.0, example=0.207971)
    V18: float = Field(0.0, example=0.025791)
    V19: float = Field(0.0, example=0.403993)
    V20: float = Field(0.0, example=0.251412)
    V21: float = Field(0.0, example=-0.018307)
    V22: float = Field(0.0, example=0.277838)
    V23: float = Field(0.0, example=-0.110474)
    V24: float = Field(0.0, example=0.066928)
    V25: float = Field(0.0, example=0.128539)
    V26: float = Field(0.0, example=-0.189115)
    V27: float = Field(0.0, example=0.133558)
    V28: float = Field(0.0, example=-0.021053)

    class Config:
        extra = "allow"


class BatchTransactionPayload(BaseModel):
    transactions: List[TransactionPayload]


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold_used: float
    risk_tier: str
    recommended_action: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold: float
    metrics: Dict[str, float]
