import os
import pandas as pd

from src.train_intent import train_model
from src.retrieval import HistoricalRetriever
from src.response_generator import generate_response
from src.escalation import decide_escalation


DATA_PATH = "data/amazonhelp_cleaned_with_intents.csv"


class AmazonSupportAgent:

    def __init__(self):
        print("Initializing Amazon Support Agent...\n")

        # Load saved classifier, or train it if it doesn't exist.
        self.embedding_model, self.intent_model = train_model()

        # Load the complete dataset.
        df = pd.read_csv(DATA_PATH)

        # Only unlabelled historical cases are used for retrieval.
        historical_data = df[
            (
                df["Intent"].isna()
                | (df["Intent"].str.strip() == "")
            )
            & df["cleaned_customer_message"].notna()
        ].copy()

        print(
            f"\nHistorical retrieval cases: "
            f"{len(historical_data)}"
        )

        # HistoricalRetriever loads cached embeddings if available.
        self.retriever = HistoricalRetriever(
            historical_data
        )

        print("\nAmazon Support Agent ready.")


    def predict_intent(self, customer_message):
        """
        Predict intent and classifier confidence.
        """

        embedding = self.embedding_model.encode(
            [str(customer_message)],
            normalize_embeddings=True
        )

        intent = self.intent_model.predict(
            embedding
        )[0]

        probabilities = self.intent_model.predict_proba(
            embedding
        )[0]

        confidence = probabilities.max()

        return intent, float(confidence)


    def run(self, customer_message, top_k=5):
        """
        Run the complete support-agent pipeline.
        """

        # -----------------------------------------------------
        # 1. Intent classification
        # -----------------------------------------------------

        intent, confidence = self.predict_intent(
            customer_message
        )

        # -----------------------------------------------------
        # 2. Historical retrieval
        # -----------------------------------------------------

        historical_cases = self.retriever.retrieve_cases(
            customer_message,
            top_k=top_k
        )

        # -----------------------------------------------------
        # 3. Grounded response generation
        # -----------------------------------------------------

        generated_response = generate_response(
            customer_message,
            historical_cases
        )

        # -----------------------------------------------------
        # 4. Escalation decision
        # -----------------------------------------------------

        escalation = decide_escalation(
            customer_message,
            generated_response
        )

        # -----------------------------------------------------
        # 5. Return structured result
        # -----------------------------------------------------

        return {
            "customer_message": customer_message,
            "intent": intent,
            "intent_confidence": confidence,
            "retrieved_cases": historical_cases,
            "response": generated_response,
            "decision": escalation["decision"],
            "escalation_reason": escalation["reason"]
        }