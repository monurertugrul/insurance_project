import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.xgb_predictor import XGBPredictor
from agents.frontier_agent import FrontierAgent
from agents.gemini_frontier_agent import GeminiFrontierAgent


class ModelComparisonEvaluator:
    """
    Compare:
    - XGBoost
    - FrontierAgent (OpenAI)
    - GeminiFrontierAgent (Google Gemini)
    """

    def __init__(
        self,
        xgb_model_path,
        frontier_agent: FrontierAgent,
        gemini_agent: GeminiFrontierAgent,
    ):
        self.xgb = XGBPredictor()
        self.xgb.load(xgb_model_path)

        self.frontier = frontier_agent
        self.gemini = gemini_agent

    def evaluate(self, df_test: pd.DataFrame):
        df = df_test.drop(columns=["region"], errors="ignore")
        y_true = df["charges"].values

        xgb_preds = []
        frontier_preds = []
        gemini_preds = []

        for _, row in df.iterrows():
            features = row.to_dict()

            # XGBoost
            xgb_price = self.xgb.predict(features)
            xgb_preds.append(xgb_price)

            # FrontierAgent (OpenAI)
            frontier_out = self.frontier.price(features)
            frontier_price = frontier_out.get("predicted_charges_usd", xgb_price)
            frontier_preds.append(frontier_price)

            # Gemini
            gemini_out = self.gemini.price(features)
            gemini_price = gemini_out.get("predicted_charges_usd", xgb_price)
            gemini_preds.append(gemini_price)

        xgb_preds = np.array(xgb_preds)
        frontier_preds = np.array(frontier_preds)
        gemini_preds = np.array(gemini_preds)

        metrics = {
            "XGBoost": {
                "mae": mean_absolute_error(y_true, xgb_preds),
                "rmse": mean_squared_error(y_true, xgb_preds) ** 0.5,
                "r2": r2_score(y_true, xgb_preds),
            },
            "FrontierAgent (OpenAI)": {
                "mae": mean_absolute_error(y_true, frontier_preds),
                "rmse": mean_squared_error(y_true, frontier_preds) ** 0.5,
                "r2": r2_score(y_true, frontier_preds),
            },
            "GeminiFrontierAgent": {
                "mae": mean_absolute_error(y_true, gemini_preds),
                "rmse": mean_squared_error(y_true, gemini_preds) ** 0.5,
                "r2": r2_score(y_true, gemini_preds),
            },
        }

        return metrics
