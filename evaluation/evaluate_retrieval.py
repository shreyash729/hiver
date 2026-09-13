import sys
import os

# Add repository root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

import src.config as config


DATA_PATH = "data/amazonhelp_cleaned_with_intents.csv"

MODEL_NAME = config.EMBEDDING_MODEL

RANDOM_STATE = config.RANDOM_STATE
TEST_SIZE = config.TEST_SIZE

CLASSIFIER_PATH = "artifacts/intent_model.joblib"
EMBEDDINGS_PATH = "artifacts/historical_embeddings.npy"


def load_data():

    data = pd.read_csv(DATA_PATH)

    labeled_data = data[
        data["Intent"].notna()
        & (data["Intent"].astype(str).str.strip() != "")
    ].copy()

    historical_data = data[
        data["Intent"].isna()
        | (data["Intent"].astype(str).str.strip() == "")
    ].copy()

    # Same safety cleaning used by the retriever
    historical_data = historical_data[
        historical_data["cleaned_customer_message"].notna()
    ].copy()

    historical_data["cleaned_customer_message"] = (
        historical_data["cleaned_customer_message"]
        .astype(str)
        .str.strip()
    )

    historical_data = historical_data[
        historical_data["cleaned_customer_message"] != ""
    ].copy()

    historical_data.reset_index(drop=True, inplace=True)

    return labeled_data, historical_data


def train_intent_classifier(labeled_data, model):

    train_data, test_data = train_test_split(
        labeled_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labeled_data["Intent"]
    )

    print("\nGenerating labelled training embeddings...")

    train_embeddings = model.encode(
        train_data["cleaned_customer_message"].astype(str).tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("\nGenerating labelled test embeddings...")

    test_embeddings = model.encode(
        test_data["cleaned_customer_message"].astype(str).tolist(),
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


def load_or_create_classifier(labeled_data, model):

    if os.path.exists(CLASSIFIER_PATH):

        print(
            "\nLoading cached intent classifier..."
        )

        classifier = joblib.load(
            CLASSIFIER_PATH
        )

        # We still need the test split and its embeddings
        # for the retrieval evaluation queries.
        _, test_data = train_test_split(
            labeled_data,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labeled_data["Intent"]
        )

        test_embeddings = model.encode(
            test_data[
                "cleaned_customer_message"
            ].astype(str).tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return classifier, test_data, test_embeddings

    print(
        "\nCached classifier not found."
    )

    print(
        "Training intent classifier..."
    )

    return train_intent_classifier(
        labeled_data,
        model
    )


def load_or_create_historical_embeddings(
    historical_data,
    model
):

    if os.path.exists(EMBEDDINGS_PATH):

        print(
            "\nLoading cached historical embeddings..."
        )

        historical_embeddings = np.load(
            EMBEDDINGS_PATH
        )

        if len(historical_embeddings) != len(
            historical_data
        ):

            raise ValueError(
                "Cached historical embeddings do not "
                "match the current historical dataset size.\n"
                f"Embeddings: {len(historical_embeddings)}\n"
                f"Historical rows: {len(historical_data)}\n"
                "Delete artifacts/historical_embeddings.npy "
                "and regenerate the cache."
            )

        print(
            f"Loaded {len(historical_embeddings)} "
            "historical embeddings."
        )

        return historical_embeddings

    print(
        "\nCached historical embeddings not found."
    )

    print(
        "Generating historical embeddings..."
    )

    historical_embeddings = model.encode(
        historical_data[
            "cleaned_customer_message"
        ].astype(str).tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    os.makedirs(
        os.path.dirname(EMBEDDINGS_PATH),
        exist_ok=True
    )

    np.save(
        EMBEDDINGS_PATH,
        historical_embeddings
    )

    print(
        f"Historical embeddings saved to "
        f"{EMBEDDINGS_PATH}"
    )

    return historical_embeddings


def evaluate_retrieval():

    labeled_data, historical_data = load_data()

    print(
        f"Labelled examples: "
        f"{len(labeled_data)}"
    )

    print(
        f"Historical cases: "
        f"{len(historical_data)}"
    )

    print(
        "\nLoading BGE model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # ---------------------------------------------------------
    # Load cached classifier when available
    # ---------------------------------------------------------

    classifier, test_data, test_embeddings = (
        load_or_create_classifier(
            labeled_data,
            model
        )
    )

    # ---------------------------------------------------------
    # Load cached historical embeddings
    # ---------------------------------------------------------

    historical_embeddings = (
        load_or_create_historical_embeddings(
            historical_data,
            model
        )
    )

    # ---------------------------------------------------------
    # Assign pseudo-intents to historical cases
    # ---------------------------------------------------------

    print(
        "\nAssigning pseudo-intents to "
        "historical cases..."
    )

    historical_intents = classifier.predict(
        historical_embeddings
    )

    # ---------------------------------------------------------
    # Calculate retrieval Hit@K
    # ---------------------------------------------------------

    results = {}

    for k in [1, 3, 5, 10]:

        hits = 0

        for query_embedding, true_intent in zip(
            test_embeddings,
            test_data["Intent"]
        ):

            scores = (
                historical_embeddings
                @ query_embedding
            )

            top_idx = np.argsort(
                scores
            )[-k:][::-1]

            retrieved_intents = (
                historical_intents[top_idx]
            )

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

    print(
        "\n===== RETRIEVAL RESULTS ====="
    )

    for k, score in results.items():

        print(
            f"Hit@{k}: {score:.2%}"
        )