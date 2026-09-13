import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import src.config as config

MODEL_NAME = config.EMBEDDING_MODEL

ARTIFACT_DIR = "artifacts"
EMBEDDINGS_PATH = os.path.join(
    ARTIFACT_DIR,
    "historical_embeddings.npy"
)


class HistoricalRetriever:

    def __init__(self, historical_data):

        # Make a clean copy
        self.data = historical_data.copy()

        # Remove rows where customer message is missing
        self.data = self.data[
            self.data["cleaned_customer_message"].notna()
        ].copy()

        # Convert messages to strings
        self.data["cleaned_customer_message"] = (
            self.data["cleaned_customer_message"]
            .astype(str)
            .str.strip()
        )

        # Remove empty messages
        self.data = self.data[
            self.data["cleaned_customer_message"] != ""
        ].copy()

        self.data.reset_index(drop=True, inplace=True)

        print(
            f"Valid historical retrieval cases: {len(self.data)}"
        )

        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        self.embedding_model = SentenceTransformer(MODEL_NAME)

        # Try loading cached embeddings
        if os.path.exists(EMBEDDINGS_PATH):

            print("Loading saved historical embeddings...")

            self.embeddings = np.load(
                EMBEDDINGS_PATH
            )

            # Make sure embeddings correspond to current dataset
            if len(self.embeddings) != len(self.data):

                print(
                    "Saved embeddings do not match "
                    "the current historical dataset."
                )

                print("Regenerating embeddings...")

                self._create_embeddings()

            else:

                print(
                    f"Loaded {len(self.embeddings)} "
                    "historical embeddings."
                )

        else:

            print(
                "No saved historical embeddings found."
            )

            self._create_embeddings()


    def _create_embeddings(self):

        print(
            f"Generating BGE embeddings for "
            f"{len(self.data)} historical cases..."
        )

        messages = (
            self.data["cleaned_customer_message"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        # Safety check
        if any(not isinstance(x, str) for x in messages):

            raise ValueError(
                "Historical customer messages contain "
                "non-string values."
            )

        self.embeddings = self.embedding_model.encode(
            messages,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        np.save(
            EMBEDDINGS_PATH,
            self.embeddings
        )

        print(
            f"Historical embeddings saved to: "
            f"{EMBEDDINGS_PATH}"
        )


    def retrieve_cases(self, query, top_k=5):

        query = str(query)

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        # Because both embeddings are normalized,
        # dot product = cosine similarity
        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = self.data.iloc[top_indices].copy()

        results["similarity"] = scores[top_indices]

        return results