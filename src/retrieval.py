import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
import config

MODEL_NAME = config.EMBEDDING_MODEL


ARTIFACT_DIR = "artifacts"
EMBEDDINGS_PATH = os.path.join(
    ARTIFACT_DIR,
    "historical_embeddings.npy"
)


class HistoricalRetriever:

    def __init__(self, historical_data):
        """
        historical_data:
            DataFrame containing historical AmazonHelp conversations.
            Required column:
                cleaned_customer_message
        """

        self.data = historical_data.reset_index(drop=True).copy()

        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        # BGE is needed for encoding new customer queries.
        self.embedding_model = SentenceTransformer(MODEL_NAME)

        # ---------------------------------------------------------
        # Load cached historical embeddings if available
        # ---------------------------------------------------------
        if os.path.exists(EMBEDDINGS_PATH):

            print("Loading saved historical embeddings...")

            self.embeddings = np.load(
                EMBEDDINGS_PATH
            )

            # Safety check: make sure embeddings match the dataset.
            if len(self.embeddings) != len(self.data):
                print(
                    "Saved embeddings do not match the current "
                    "historical dataset."
                )
                print("Regenerating embeddings...")

                self._create_embeddings()

            else:
                print(
                    f"Loaded {len(self.embeddings)} historical embeddings."
                )

        else:
            print("No saved historical embeddings found.")
            self._create_embeddings()

    def _create_embeddings(self):

        print(
            f"Generating BGE embeddings for "
            f"{len(self.data)} historical cases..."
        )

        self.embeddings = self.embedding_model.encode(
            self.data["cleaned_customer_message"]
            .astype(str)
            .tolist(),

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
        """
        Retrieve the top-k most similar historical customer messages.
        """

        query_embedding = self.embedding_model.encode(
            [str(query)],
            normalize_embeddings=True
        )[0]

        # Since embeddings are normalized,
        # dot product is equivalent to cosine similarity.
        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = self.data.iloc[top_indices].copy()

        results["similarity"] = scores[top_indices]

        return results