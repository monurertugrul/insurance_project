# agents/ensemble_agent_light.py

import numpy as np
import xgboost as xgb


class EnsemblePredictor:
    def __init__(self, frontier_agent, xgb_model_path):
        self.frontier = frontier_agent

        self.xgb = xgb.Booster()
        self.xgb.load_model(xgb_model_path)

    def _prepare_xgb_features(self, features):
        return np.array([[
            features["age"],
            features["bmi"],
            features["children"],
            1 if features["sex"] == "male" else 0,
            1 if features["smoker"] == "yes" else 0,
        ]])

    def predict(self, features):
        # Prepare XGB input
        xgb_input = self._prepare_xgb_features(features)

        # Convert to DMatrix
        dmat = xgb.DMatrix(xgb_input)

        # XGB prediction
        xgb_price = float(self.xgb.predict(dmat)[0])

        # FrontierAgent prediction
        frontier_out = self.frontier.price(features)
        frontier_price = frontier_out.get("predicted_charges_usd")
        frontier_explanation = frontier_out.get("explanation", "")

        # Ensemble
        final_price = 0.5 * xgb_price + 0.5 * frontier_price
        ci_low = final_price * 0.9
        ci_high = final_price * 1.1

        return {
            "final_price": final_price,
            "confidence_interval": (ci_low, ci_high),
            "xgb_price": xgb_price,
            "frontier_price": frontier_price,
            "frontier_explanation": frontier_explanation,
            "ensemble_explanation": "Weighted blend of XGB=0.5 and Frontier=0.5."
        }

