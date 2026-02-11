# agents/ensemble_agent.py

import numpy as np
import pandas as pd
import logging

from dnn_evaluator import DNNEvaluator
from models.xgb_predictor import XGBPredictor
from agents.qlora_agent import QLoraAgent
from agents.frontier_agent import FrontierAgent


class EnsemblePredictor:
    """
    Unified prediction interface for:
    - XGBoost
    - DNN
    - QLoRA (TinyLlama)
    - FrontierAgent (OpenAI)
    - Weighted ensemble of all models
    """

    def __init__(
        self,
        xgb_model_path="models/xgb_predictor.json",

        # FIXED DNN PATHS
        dnn_model_path="models/deep_neural_network.pth",
        dnn_encoder_path="models/encoder.pkl",
        dnn_scaler_path="models/target_scaler.pkl",

        qlora_model_path="adapter_tinyllama",
        qlora_base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="mps",
        frontier_agent=None,
        w_xgb=0.25,
        w_dnn=0.25,
        w_qlora=0.25,
        w_frontier=0.25,
    ):

        # -----------------------------
        # Load XGBoost
        # -----------------------------
        self.xgb = XGBPredictor()
        self.xgb.load(xgb_model_path)

        # -----------------------------
        # Load DNN
        # -----------------------------
        self.dnn = DNNEvaluator(
            model_path=dnn_model_path,
            encoder_path=dnn_encoder_path,
            scaler_path=dnn_scaler_path,
            device=device,
        )

        # -----------------------------
        # Load QLoRA
        # -----------------------------
        self.qlora = QLoraAgent(
            model_path=qlora_model_path,
            base_model_name=qlora_base_model,
            device=device,
            rag=None,
            xgb_predictor=self.xgb,
        )

        # -----------------------------
        # FrontierAgent (optional)
        # -----------------------------
        self.frontier = frontier_agent

        # -----------------------------
        # Normalize weights
        # -----------------------------
        weights = np.array([w_xgb, w_dnn, w_qlora, w_frontier], dtype=float)
        weights = weights / weights.sum()
        self.w_xgb, self.w_dnn, self.w_qlora, self.w_frontier = weights

    # Helper: single-row DNN prediction
    def _predict_dnn_single(self, features: dict) -> float:
        return self.dnn.predict_single(features)


    def predict(self, features: dict) -> dict:
        """
        Returns predictions from all models + weighted ensemble.
        """

        # --- XGBoost ---
        xgb_price = self.xgb.predict(features)

        # --- DNN ---
        dnn_price = self._predict_dnn_single(features)

        # --- QLoRA ---
        qlora_out = self.qlora.price(features)
        qlora_price = qlora_out.get("predicted_charges")
        qlora_explanation = qlora_out.get("explanation", "")

        if qlora_price is None:
            qlora_price = xgb_price
            qlora_explanation = "QLoRA returned invalid JSON; fallback to XGBoost."

        # --- FrontierAgent ---
        if self.frontier:
            frontier_out = self.frontier.price(features)
            frontier_price = frontier_out.get("predicted_charges_usd")
            frontier_explanation = frontier_out.get("explanation", "")
        else:
            frontier_price = dnn_price
            frontier_explanation = "FrontierAgent disabled; using DNN fallback."

        if frontier_price is None:
            frontier_price = dnn_price
            frontier_explanation = "FrontierAgent failed; fallback to DNN."

        # --- Weighted ensemble ---
        prices = np.array([xgb_price, dnn_price, qlora_price, frontier_price])
        weights = np.array([self.w_xgb, self.w_dnn, self.w_qlora, self.w_frontier])
        final_price = float(np.sum(prices * weights))

        # --- Confidence interval ---
        std_dev = float(np.std(prices))
        ci_low = final_price - 1.28 * std_dev
        ci_high = final_price + 1.28 * std_dev

        return {
            "final_price": final_price,
            "confidence_interval": (ci_low, ci_high),

            "xgb_price": xgb_price,
            "dnn_price": dnn_price,
            "qlora_price": qlora_price,
            "frontier_price": frontier_price,

            "qlora_explanation": qlora_explanation,
            "frontier_explanation": frontier_explanation,

            "ensemble_explanation": (
                f"Weighted blend of XGB={self.w_xgb:.2f}, "
                f"DNN={self.w_dnn:.2f}, QLoRA={self.w_qlora:.2f}, "
                f"Frontier={self.w_frontier:.2f}."
            ),
        }
