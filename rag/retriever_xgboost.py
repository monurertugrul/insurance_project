import chromadb
import xgboost as xgb
import numpy as np

class InsuranceRAG_XGB:
    def __init__(self, model_path: str, db_path: str, collection_name: str):
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

    def _encode_features(self, features: dict) -> np.ndarray:
        # Create a flat list of numeric features
        # Assuming your model was trained with: age, bmi, children, sex_female, sex_male, smoker_no, smoker_yes
        age = float(features["age"])
        bmi = float(features["bmi"])
        children = float(features["children"])
        
        # One-hot encoding for Sex
        sex_female, sex_male = (0.0, 1.0) if features["sex"] == "male" else (1.0, 0.0)
        
        # One-hot encoding for Smoker
        smoker_no, smoker_yes = (0.0, 1.0) if features["smoker"] == "yes" else (1.0, 0.0)
        
        # Return a simple 1D array
        return np.array([age, bmi, children, sex_female, sex_male, smoker_no, smoker_yes])

    def retrieve(self, features: dict, k: int = 3):
        X = self._encode_features(features).reshape(1, -1) # Ensure 2D for XGBoost
        # apply() returns leaf indices for the forest
        leaf_emb = self.model.apply(X).astype(float).tolist()
        results = self.collection.query(query_embeddings=leaf_emb, n_results=k)
        return results["documents"][0]