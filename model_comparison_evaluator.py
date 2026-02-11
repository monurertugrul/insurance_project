# model_comparison_evaluator.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from agents.ensemble_agent import EnsemblePredictor
from agents.frontier_agent import FrontierAgent


class ModelComparisonEvaluator:
    def __init__(
        self,
        frontier_agent: FrontierAgent,
        dnn_model_path,
        dnn_encoder_path,
        dnn_scaler_path,
        xgb_model_path,
        qlora_model_path="adapter_tinyllama",
        qlora_base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="mps",
    ):
        # The ensemble loads ALL models internally
        self.ensemble = EnsemblePredictor(
            frontier_agent=frontier_agent,
            dnn_model_path=dnn_model_path,
            dnn_encoder_path=dnn_encoder_path,
            dnn_scaler_path=dnn_scaler_path,
            xgb_model_path=xgb_model_path,
            qlora_model_path=qlora_model_path,
            qlora_base_model=qlora_base_model,
            device=device,
        )

    def evaluate(self, df_test: pd.DataFrame):
        df = df_test.drop(columns=["region"], errors="ignore")
        y_true = df["charges"].values

        xgb_preds = []
        dnn_preds = []
        qlora_preds = []
        frontier_preds = []
        ensemble_preds = []

        for _, row in df.iterrows():
            out = self.ensemble.predict(row.to_dict())

            xgb_preds.append(out["xgb_price"])
            dnn_preds.append(out["dnn_price"])
            qlora_preds.append(out["qlora_price"])
            frontier_preds.append(out["frontier_price"])
            ensemble_preds.append(out["final_price"])

        xgb_preds = np.array(xgb_preds)
        dnn_preds = np.array(dnn_preds)
        qlora_preds = np.array(qlora_preds)
        frontier_preds = np.array(frontier_preds)
        ensemble_preds = np.array(ensemble_preds)

        metrics = {
            "XGBoost": {
                "mae": mean_absolute_error(y_true, xgb_preds),
                "rmse": mean_squared_error(y_true, xgb_preds) ** 0.5,
                "r2": r2_score(y_true, xgb_preds),
            },
            "DNN": {
                "mae": mean_absolute_error(y_true, dnn_preds),
                "rmse": mean_squared_error(y_true, dnn_preds) ** 0.5,
                "r2": r2_score(y_true, dnn_preds),
            },
            "QLoRA": {
                "mae": mean_absolute_error(y_true, qlora_preds),
                "rmse": mean_squared_error(y_true, qlora_preds) ** 0.5,
                "r2": r2_score(y_true, qlora_preds),
            },
            "FrontierAgent": {
                "mae": mean_absolute_error(y_true, frontier_preds),
                "rmse": mean_squared_error(y_true, frontier_preds) ** 0.5,
                "r2": r2_score(y_true, frontier_preds),
            },
            "Ensemble": {
                "mae": mean_absolute_error(y_true, ensemble_preds),
                "rmse": mean_squared_error(y_true, ensemble_preds) ** 0.5,
                "r2": r2_score(y_true, ensemble_preds),
            },
        }

        return metrics
