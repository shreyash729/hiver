import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import config

DATA_PATH = "data/amazonhelp_cleaned_with_intents.csv"

MODEL_NAME = config.EMBEDDING_MODEL
RANDOM_STATE = config.RANDOM_STATE


def load_data():
    data = pd.read_csv(DATA_PATH)

    labeled_data = data[
        data["Intent"].notna()
    ].copy()

    return labeled_data


def train_model():

    labeled_data = load_data()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        labeled_data["cleaned_customer_message"].tolist(),
        labeled_data["Intent"].tolist(),
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labeled_data["Intent"]
    )

    print(f"Training examples: {len(X_train_text)}")
    print(f"Test examples: {len(X_test_text)}")

    print("\nLoading BGE model...")
    embedding_model = SentenceTransformer(MODEL_NAME)

    print("\nGenerating training embeddings...")
    X_train = embedding_model.encode(
        X_train_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("\nGenerating test embeddings...")
    X_test = embedding_model.encode(
        X_test_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("\nTraining Logistic Regression...")

    intent_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    intent_model.fit(X_train, y_train)

    y_pred = intent_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    return embedding_model, intent_model


if __name__ == "__main__":
    train_model()