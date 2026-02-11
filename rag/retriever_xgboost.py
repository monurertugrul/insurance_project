# rag/retriever_xgboost.py

import chromadb
import xgboost as xgb
import numpy as np


class InsuranceRAG_XGB:
    """
    Retrieves similar insurance cases using:
    - XGBoost leaf embeddings
    - ChromaDB vector search
    """

    def __init__(self, model_path, db_path, collection_name):
        # Load XGBoost model (same one used to build the RAG index)
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

        # Load ChromaDB collection
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

    def _encode_features(self, features):
        """
        Must match EXACT feature order used in build_rag_index_xgb:
        [age, bmi, children, sex_female, sex_male, smoker_no, smoker_yes]
        """
        age = np.array([[float(features["age"])]])
        bmi = np.array([[float(features["bmi"])]])
        children = np.array([[float(features["children"])]])
        
        # One-hot encode sex
        if features["sex"] == "male":
            sex = np.array([[0.0, 1.0]])   # female, male
        else:
            sex = np.array([[1.0, 0.0]])

        # One-hot encode smoker
        if features["smoker"] == "yes":
            smoker = np.array([[0.0, 1.0]])  # no, yes
        else:
            smoker = np.array([[1.0, 0.0]])

        X = np.concatenate([age, bmi, children, sex, smoker], axis=1)
        return X

    def retrieve(self, features: dict, k=3):
        """
        Returns list of retrieved case texts.
        """
        X = self._encode_features(features)

        # Leaf embeddings (same as build_rag_index_xgb)
        leaf_emb = self.model.apply(X).astype(float).tolist()

        results = self.collection.query(
            query_embeddings=leaf_emb,
            n_results=k
        )

        return results["documents"][0]
