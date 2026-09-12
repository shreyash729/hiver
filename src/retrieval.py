import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
import config

MODEL_NAME = config.EMBEDDING_MODEL


class HistoricalRetriever:

    def __init__(self, historical_data):
        """
        historical_data:
            DataFrame containing historical AmazonHelp conversations.
            Required column:
                cleaned_customer_message
        """

        self.data = historical_data.reset_index(drop=True).copy()

        self.embedding_model = SentenceTransformer(MODEL_NAME)

        self.embeddings = self.embedding_model.encode(
            self.data["cleaned_customer_message"].astype(str).tolist(),
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def retrieve_cases(self, query, top_k=5):
        """
        Retrieve the top-k most similar historical customer messages.
        """

        query_embedding = self.embedding_model.encode(
            [str(query)],
            normalize_embeddings=True
        )[0]

        # Since embeddings are normalized, dot product = cosine similarity.
        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = self.data.iloc[top_indices].copy()
        results["similarity"] = scores[top_indices]

        return results