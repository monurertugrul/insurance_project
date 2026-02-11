from fastapi import FastAPI
from pydantic import BaseModel
from typing import Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Insurance Pricing API")

ensemble = None


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


@app.on_event("startup")
async def load_model():
    global ensemble
    from rag.retriever_xgboost import InsuranceRAG_XGB
    from agents.frontier_agent import FrontierAgent
    from agents.ensemble_agent_light import EnsemblePredictor

    rag = InsuranceRAG_XGB(
        model_path="./rag/xgb_model.json",
        db_path="./chroma_db",
        collection_name="insurance_cases_xgb",
    )

    frontier = FrontierAgent(
        model="gpt-4o-mini",
        rag=rag,
    )

    ensemble = EnsemblePredictor(
        frontier_agent=frontier,
        xgb_model_path="models/xgb_predictor.json",
    )


@app.post("/predict", response_model=InsuranceResponse)
def predict(req: InsuranceRequest):
    result = ensemble.predict(req.dict())
    return result
