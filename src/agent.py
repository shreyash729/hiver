import pandas as pd

from src.train_intent import train_model
from src.retrieval import HistoricalRetriever
from src.response_generator import generate_response
from src.escalation import decide_escalation


DATA_PATH = "data/amazonhelp_cleaned_with_intents.csv"


class AmazonSupportAgent:

    def __init__(self):
        print("Loading and training intent classifier...")

        (
            self.embedding_model,
            self.intent_model,
            _,
            _,
            _,
            _
        ) = train_model()

        print("\nLoading historical support cases...")

        df = pd.read_csv(DATA_PATH)

        # Historical retrieval cases are the rows without manual labels.
        historical_data = df[
            df["Intent"].isna() | (df["Intent"].str.strip() == "")
        ].copy()

        self.retriever = HistoricalRetriever(historical_data)

        print("\nAgent ready.")

    def predict_intent(self, customer_message):
        """
        Predict the intent and confidence for a customer message.
        """

        embedding = self.embedding_model.encode(
            [str(customer_message)],
            normalize_embeddings=True
        )

        intent = self.intent_model.predict(embedding)[0]

        probabilities = self.intent_model.predict_proba(embedding)[0]
        confidence = probabilities.max()

        return intent, float(confidence)

    def run(self, customer_message, top_k=5):
        """
        Run the complete Amazon support agent pipeline.
        """

        # 1. Intent classification
        intent, confidence = self.predict_intent(customer_message)

        # 2. Historical retrieval
        historical_cases = self.retriever.retrieve_cases(
            customer_message,
            top_k=top_k
        )

        # 3. Grounded response generation
        generated_response = generate_response(
            customer_message,
            historical_cases
        )

        # 4. Escalation decision
        escalation = decide_escalation(
            customer_message,
            generated_response
        )

        return {
            "customer_message": customer_message,
            "intent": intent,
            "intent_confidence": confidence,
            "retrieved_cases": historical_cases,
            "response": generated_response,
            "decision": escalation["decision"],
            "escalation_reason": escalation["reason"]
        }