import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
try:
    from api_core.predictor import AIOpsPredictor
    from api_core.rca import analyze_logs_with_ollama
except ModuleNotFoundError:
    from predictor import AIOpsPredictor
    from rca import analyze_logs_with_ollama

app = FastAPI(
    title="AIOps Inference & NLP Core Service",
    description="Exposes predictive build failure metrics (XGBoost + BiLSTM) and NLP-driven Root Cause Analysis (Llama 3B via Ollama).",
    version="1.0.0"
)

# Enable CORS for external dashboard consumption
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global predictor
predictor = AIOpsPredictor()

# --- Pydantic Schemas ---
class PredictRequest(BaseModel):
    current: Dict[str, Any] = Field(..., description="Metadata dictionary for the commit to evaluate.")
    history: List[Dict[str, Any]] = Field(default=[], description="List of dictionaries representing past build runs of this repository (newest first or chronological).")

class PredictResponse(BaseModel):
    xgboost_probability: float
    lstm_probability: float
    ensemble_risk_score: float
    risk_category: str

class RcaRequest(BaseModel):
    log_text: str = Field(..., description="Raw console logs from the failed build execution.")

class RcaResponse(BaseModel):
    summary: str
    root_cause: str
    recommendations: List[str]
    source: str

# --- Endpoints ---
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "xgboost": predictor.xgb_model is not None,
            "lstm": predictor.lstm_model is not None,
            "scaler": predictor.scaler is not None
        }
    }

@app.post("/predict", response_model=PredictResponse)
def predict_risk(payload: PredictRequest):
    try:
        results = predictor.predict(payload.current, payload.history)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/rca", response_model=RcaResponse)
async def log_rca(payload: RcaRequest):
    try:
        rca_output = await analyze_logs_with_ollama(payload.log_text)
        return rca_output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Log analysis error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
