# models/xgb_predictor.py

import numpy as np


class XGBPredictor:
    def __init__(self):
        """
        Do NOT import xgboost here.
        Modal will import it inside the container.
        Your local machine should NOT import xgboost.
        """
        self.model = None

    def train(self, save_path="models/xgb_predictor.json"):
        """
        Your training code goes here.
        This method is optional for Modal training.
        """
        raise NotImplementedError("Training is handled elsewhere.")

    def load(self, path="models/xgb_predictor.json"):
        """
        Load the trained XGBoost model.
        Import xgboost INSIDE the method so it works on Modal.
        """
        import xgboost as xgb
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)

    def _encode_features(self, features: dict):
        """
        Must match training order:
        [ age, bmi, children, sex_female, sex_male, smoker_no, smoker_yes ]
        """

        age = float(features["age"])
        bmi = float(features["bmi"])
        children = float(features["children"])

        # One-hot: sex
        if features["sex"].lower() == "female":
            sex_female, sex_male = 1.0, 0.0
        else:
            sex_female, sex_male = 0.0, 1.0

        # One-hot: smoker
        if features["smoker"].lower() == "no":
            smoker_no, smoker_yes = 1.0, 0.0
        else:
            smoker_no, smoker_yes = 0.0, 1.0

        X = np.array([
            age,
            bmi,
            children,
            sex_female,
            sex_male,
            smoker_no,
            smoker_yes
        ]).reshape(1, -1)

        return X

    def predict(self, features: dict):
        """
        Predict using the loaded XGBoost model.
        """
        X = self._encode_features(features)
        return float(self.model.predict(X)[0])
