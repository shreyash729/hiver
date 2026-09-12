import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import src.config as config

DATA_PATH = "data/amazonhelp_cleaned_with_intents.csv"
MODEL_NAME = config.EMBEDDING_MODEL

RANDOM_STATE = config.RANDOM_STATE
TEST_SIZE = config.TEST_SIZE


def load_data():

    data = pd.read_csv(DATA_PATH)

    labeled_data = data[
        data["Intent"].notna()
    ].copy()

    historical_data = data[
        data["Intent"].isna()
    ].copy()

    return labeled_data, historical_data


def train_intent_classifier(
    labeled_data,
    model
):

    train_data, test_data = train_test_split(
        labeled_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labeled_data["Intent"]
    )

    train_embeddings = model.encode(
        train_data["cleaned_customer_message"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    test_embeddings = model.encode(
        test_data["cleaned_customer_message"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    classifier.fit(
        train_embeddings,
        train_data["Intent"]
    )

    return classifier, test_data, test_embeddings


def evaluate_retrieval():

    labeled_data, historical_data = load_data()

    print(f"Labelled examples: {len(labeled_data)}")
    print(f"Historical cases: {len(historical_data)}")

    print("\nLoading BGE model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    classifier, test_data, test_embeddings = (
        train_intent_classifier(
            labeled_data,
            model
        )
    )

    print("\nGenerating historical embeddings...")

    historical_embeddings = model.encode(
        historical_data[
            "cleaned_customer_message"
        ].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("\nAssigning pseudo-intents to historical cases...")

    historical_intents = classifier.predict(
        historical_embeddings
    )

    results = {}

    for k in [1, 3, 5, 10]:

        hits = 0

        for query_embedding, true_intent in zip(
            test_embeddings,
            test_data["Intent"]
        ):

            scores = (
                historical_embeddings @ query_embedding
            )

            top_idx = np.argsort(scores)[-k:][::-1]

            retrieved_intents = historical_intents[
                top_idx
            ]

            if true_intent in retrieved_intents:
                hits += 1

        hit_rate = hits / len(test_data)

        results[k] = hit_rate

        print(
            f"Hit@{k}: {hit_rate:.4f}"
        )

    return results


if __name__ == "__main__":

    results = evaluate_retrieval()

    print("\n===== RETRIEVAL RESULTS =====")

    for k, score in results.items():
        print(
            f"Hit@{k}: {score:.2%}"
        )