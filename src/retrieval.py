import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import config

MODEL_NAME = config.EMBEDDING_MODEL


class HistoricalRetriever:

    def __init__(self, historical_data):
        self.historical_data = historical_data.reset_index(drop=True)

        print("Loading BGE model...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Generating historical embeddings...")
        self.embeddings = self.model.encode(
            self.historical_data["cleaned_customer_message"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def retrieve_cases(self, query, top_k=5):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        # Because both embeddings are normalized,
        # dot product is cosine similarity.
        scores = self.embeddings @ query_embedding

        top_idx = np.argsort(scores)[-top_k:][::-1]

        results = self.historical_data.iloc[top_idx].copy()
        results["similarity"] = scores[top_idx]

        return results.reset_index(drop=True)