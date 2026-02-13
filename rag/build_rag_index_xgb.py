import os
import chromadb
import xgboost as xgb
import pandas as pd
import numpy as np
from datasets import load_dataset


def encode_features(row):
    """Match EXACT encoding used in predictor + retriever_xgboost."""
    age = float(row["age"])
    bmi = float(row["bmi"])
    children = float(row["children"])

    sex_female = 1.0 if row["sex"] == "female" else 0.0
    sex_male = 1.0 if row["sex"] == "male" else 0.0

    smoker_no = 1.0 if row["smoker"] == "no" else 0.0
    smoker_yes = 1.0 if row["smoker"] == "yes" else 0.0

    return np.array([[age, bmi, children, sex_female, sex_male, smoker_no, smoker_yes]])


def build_rag_index_xgb():
    BASE_DIR = os.path.dirname(__file__)

    # Paths
    MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.json")
    DB_PATH = os.path.join(BASE_DIR, "chroma_db")
    COLLECTION_NAME = "insurance_cases"

    print("📦 Loading dataset from HuggingFace...")
    dataset = load_dataset("onurfbwd/medical-insurance-cost-prediction")
    dfs = [dataset[split].to_pandas() for split in dataset.keys()]
    df = pd.concat(dfs, ignore_index=True)

    print("📘 Loading existing XGBoost model...")
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    print("🧬 Encoding features and generating leaf embeddings...")
    embeddings = []
    documents = []
    ids = []

    for i, row in df.iterrows():
        X = encode_features(row)
        leaf_emb = model.apply(X).astype(float).flatten().tolist()
        embeddings.append(leaf_emb)

        doc = (
            f"age: {row['age']}, sex: {row['sex']}, bmi: {row['bmi']}, "
            f"children: {row['children']}, smoker: {row['smoker']}, "
            f"charges: {row['charges']}"
        )
        documents.append(doc)
        ids.append(str(i))

    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Total documents: {len(documents)}")

    print("🗂️ Rebuilding ChromaDB collection...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # Delete old collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "l2"}
    )

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )

    print("✅ ChromaDB RAG index rebuilt successfully!")
    print(f"📁 Stored at: {DB_PATH}")
    print(f"📚 Collection name: {COLLECTION_NAME}")


if __name__ == "__main__":
    build_rag_index_xgb()
