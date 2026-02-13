from fastapi import FastAPI
from pydantic import BaseModel
from typing import Tuple
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Insurance Pricing API")

ensemble = None


# -----------------------------
# Request / Response Models
# -----------------------------
class InsuranceRequest(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str


class InsuranceResponse(BaseModel):
    final_price: float
    confidence_interval: Tuple[float, float]
    xgb_price: float
    frontier_price: float
    frontier_explanation: str
    ensemble_explanation: str


# -----------------------------
# Startup: Load Models
# -----------------------------
@app.on_event("startup")
async def load_model():
    global ensemble

    from rag.retriever_xgboost import InsuranceRAG_XGB
    from agents.frontier_agent import FrontierAgent
    from agents.ensemble_agent_light import EnsemblePredictor

    # Load RAG retriever
    rag = InsuranceRAG_XGB(
        model_path="./rag/xgb_model.json",
        db_path="./chroma_db",
        collection_name="insurance_cases_xgb",
    )

    # FrontierAgent using Llama 3.2 (free)
    frontier = FrontierAgent(
        model="meta-llama/llama-3.2-3b-instruct",
        rag=rag,
    )

    # Ensemble predictor
    ensemble = EnsemblePredictor(
        frontier_agent=frontier,
        xgb_model_path="models/xgb_predictor.json",
    )


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict", response_model=InsuranceResponse)
def predict(req: InsuranceRequest):
    result = ensemble.predict(req.dict())
    return result
