import pandas as pd

from sentence_transformers import SentenceTransformer
import config

MODEL_NAME = config.EMBEDDING_MODEL


class AmazonSupportAgent:

    def __init__(
        self,
        labeled_data_path,
        historical_data_path
    ):
        # Load data
        data = pd.read_csv(labeled_data_path)

        self.historical_data = pd.read_csv(
            historical_data_path
        ).reset_index(drop=True)

        # Load BGE model
        self.embedding_model = SentenceTransformer(
            MODEL_NAME
        )

        # The trained classifier will be attached after initialization.
        self.intent_model = None

        # Generate embeddings for historical retrieval
        self.historical_embeddings = self.embedding_model.encode(
            self.historical_data[
                "cleaned_customer_message"
            ].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def set_intent_model(self, intent_model):
        self.intent_model = intent_model

    def retrieve_cases(self, query, top_k=5):

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        scores = (
            self.historical_embeddings @ query_embedding
        )

        top_idx = scores.argsort()[-top_k:][::-1]

        results = self.historical_data.iloc[
            top_idx
        ].copy()

        results["similarity"] = scores[top_idx]

        return results.reset_index(drop=True)

    def predict_intent(self, query):

        if self.intent_model is None:
            raise ValueError(
                "Intent model has not been attached."
            )

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        return self.intent_model.predict(
            query_embedding.reshape(1, -1)
        )[0]

    def run(self, query, top_k=5):

        predicted_intent = self.predict_intent(query)

        retrieved_cases = self.retrieve_cases(
            query,
            top_k=top_k
        )

        return {
            "query": query,
            "intent": predicted_intent,
            "retrieved_cases": retrieved_cases
        }