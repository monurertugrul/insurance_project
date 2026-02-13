import os
import numpy as np
from rag.retriever_xgboost import InsuranceRAG_XGB
from agents.frontier_agent_light import frontier_predict

class InsurancePredictor:
    def __init__(self):
        base_dir = "/root/app"
        rag_dir = os.path.join(base_dir, "rag")
        model_path = os.path.join(rag_dir, "xgb_model.json")
        db_path = os.path.join(rag_dir, "chroma_db")

        self.xgb_rag = InsuranceRAG_XGB(
            model_path=model_path,
            db_path=db_path,
            collection_name="insurance_cases"
        )

    def predict(self, age, sex, bmi, children, smoker):
        features = {
            "age": age, 
            "sex": sex, 
            "bmi": bmi, 
            "children": children, 
            "smoker": smoker
        }
        
        # 1. RAG Retrieval
        try:
            retrieved_cases = self.xgb_rag.retrieve(features)
        except Exception as e:
            print(f"⚠️ RAG Retrieval failed: {e}")
            retrieved_cases = []

        # 2. XGBoost Numeric Prediction
        X_raw = self.xgb_rag._encode_features(features)
        
        # Ensure 2D (1 sample, N features)
        X = np.array(X_raw).reshape(1, -1).astype(float)
        
        
        
        try:
            # XGBoost predict returns an array, take the first element
            xgb_price = float(self.xgb_rag.model.predict(X)[0])
        except Exception as e:
            print(f"❌ XGBoost Prediction failed: {e}")
            xgb_price = 0.0

        # 3. Gemini LLM Prediction
        try:
            f_price, f_explanation = frontier_predict(features, retrieved_cases)
        except Exception as e:
            print(f"❌ Gemini Agent failed: {e}")
            f_price, f_explanation = None, str(e)

        # 4. Ensemble Logic
        if f_price is None or f_price <= 0:
            final_f_price = xgb_price
            f_explanation = f_explanation or "LLM Error."
            ensemble_note = "Fallback: 100% XGBoost."
        else:
            final_f_price = f_price
            ensemble_note = "Hybrid: 0.6 XGBoost + 0.4 Gemini 2.0 Flash."

        final_price = (0.6 * xgb_price) + (0.4 * final_f_price)

        return {
            "final_price": final_price,
            "confidence_interval": (final_price * 0.9, final_price * 1.1),
            "xgb_price": xgb_price,
            "frontier_price": final_f_price,
            "frontier_explanation": f_explanation,
            "ensemble_explanation": ensemble_note,
            "retrieved_cases": retrieved_cases,
        }