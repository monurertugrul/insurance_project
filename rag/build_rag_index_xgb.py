import chromadb
import xgboost as xgb
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def build_rag_index_xgb(
    hf_path="onurfbwd/medical-insurance-cost-prediction",
    db_path="./chroma_db",
    collection_name="insurance_cases_xgb",
    xgb_model_path="./xgb_model.json"
):
    """
    1. Loads dataset
    2. Splits into train/test
    3. Trains XGBoost on training set
    4. Evaluates on test set
    5. Extracts leaf embeddings for ALL samples (train + test)
    6. Builds ChromaDB RAG index
    """

    print(f"Loading dataset from Hugging Face: {hf_path}")
    dataset = load_dataset(hf_path)

    # Combine all splits
    dfs = [dataset[split].to_pandas() for split in dataset.keys()]
    df = pd.concat(dfs, ignore_index=True)

    # -----------------------------
    # 1. Prepare features
    # -----------------------------
    feature_cols = ["age", "bmi", "children"]
    X_num = df[feature_cols].values

    # One-hot encode categorical features
    X_cat = pd.get_dummies(df[["sex", "smoker"]]).values

    X = np.hstack([X_num, X_cat])
    y = df["charges"].values

    # -----------------------------
    # 2. Train/Test Split
    # -----------------------------
    print("Splitting dataset into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 3. Train XGBoost model
    # -----------------------------
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # -----------------------------
    # 4. Evaluate model
    # -----------------------------
    print("Evaluating model on test set...")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== XGBoost Model Performance ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.4f}")
    print("=================================\n")

    # Save model for inference
    model.get_booster().save_model(xgb_model_path)
    print(f"Saved XGBoost model to: {xgb_model_path}")

    # -----------------------------
    # 5. Extract leaf embeddings for ALL samples
    # -----------------------------
    print("Extracting leaf embeddings for RAG index...")
    leaf_embeddings = model.apply(X).astype(float).tolist()

    # -----------------------------
    # 6. Build ChromaDB index
    # -----------------------------
    print("Building ChromaDB index...")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "l2"}
    )

    docs = []
    ids = []
    for i, row in df.iterrows():
        text = (
            f"age: {row['age']}, sex: {row['sex']}, bmi: {row['bmi']}, "
            f"children: {row['children']}, smoker: {row['smoker']}, "
            f"charges: {row['charges']}"
        )
        docs.append(text)
        ids.append(str(i))

    collection.add(
        documents=docs,
        embeddings=leaf_embeddings,
        ids=ids
    )

    print(f"RAG index built successfully with {len(docs)} documents.")
    print(f"Stored in: {db_path}, collection: {collection_name}")


if __name__ == "__main__":
    build_rag_index_xgb()

