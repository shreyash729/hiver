import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import config

DATA_PATH = "data/amazonhelp_cleaned_with_intents.csv"
MODEL_NAME = config.EMBEDDING_MODEL
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Only manually labelled examples are used for supervised training.
    df = df[df["Intent"].notna() & (df["Intent"].str.strip() != "")].copy()

    X = df["cleaned_customer_message"].astype(str)
    y = df["Intent"].astype(str)

    return X, y


def train_model():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")

    # BGE sentence embeddings
    embedding_model = SentenceTransformer(MODEL_NAME)

    X_train_embeddings = embedding_model.encode(
        X_train.tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )

    X_test_embeddings = embedding_model.encode(
        X_test.tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Final supervised classifier
    intent_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    intent_model.fit(X_train_embeddings, y_train)

    y_pred = intent_model.predict(X_test_embeddings)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nBGE + Logistic Regression Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    return (
        embedding_model,
        intent_model,
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":
    train_model()