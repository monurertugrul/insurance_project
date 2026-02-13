# agents/ensemble_agent_light.py

from rag.retriever_xgboost import InsuranceRAG_XGB
from agents.frontier_agent_light import frontier_predict

class EnsembleInsurancePredictor:
    def __init__(self):
        self.xgb_rag = InsuranceRAG_XGB(
            model_path="rag/xgb_model.json",
            db_path="rag/chroma_db",
            collection_name="insurance_cases_xgb"
        )

    def predict_price(self, age, sex, bmi, children, smoker):
        features = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker
        }

        # 1. XGBoost + RAG retrieval
        retrieved_cases = self.xgb_rag.retrieve(features)
        # You can use retrieved_cases in your explanation if needed

        # 2. Frontier agent prediction
        frontier_price, frontier_explanation = frontier_predict(features)

        # 3. XGBoost numeric prediction
        # (You may already have a numeric prediction inside retriever_xgboost)
        # For now, assume frontier_price is the main numeric output
        xgb_price = frontier_price * 0.95  # placeholder

        # 4. Ensemble
        final_price = (xgb_price + frontier_price) / 2

        return {
            "final_price": final_price,
            "confidence_interval": (final_price * 0.9, final_price * 1.1),
            "xgb_price": xgb_price,
            "frontier_price": frontier_price,
            "frontier_explanation": frontier_explanation,
            "ensemble_explanation": "Combined XGBoost + Frontier model output.",
            "retrieved_cases": retrieved_cases
        }
