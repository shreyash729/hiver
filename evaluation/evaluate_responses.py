import json
import pandas as pd

from src.agent import AmazonSupportAgent
from src.response_generator import create_llm, generate_response
from src.escalation import decide_escalation
from src.judge import evaluate_response


DATA_PATH = "data/amazonhelp_cleaned_with_intents.csv"


def build_historical_examples(retrieved_cases):
    return "\n\n".join(
        [
            f"Customer: {row['cleaned_customer_message']}\n"
            f"Amazon response: {row['cleaned_amazon_response']}"
            for _, row in retrieved_cases.iterrows()
        ]
    )


def evaluate_example(agent, llm, query):

    predicted_intent = agent.predict_intent(query)

    retrieved_cases = agent.retrieve_cases(
        query,
        top_k=5
    )

    response = generate_response(
        llm,
        query,
        retrieved_cases
    )

    escalation = decide_escalation(
        llm,
        query,
        predicted_intent,
        retrieved_cases
    )

    judge_scores = evaluate_response(
        llm,
        query,
        retrieved_cases,
        response
    )

    return {
        "query": query,
        "predicted_intent": predicted_intent,
        "response": response,
        "escalation": escalation,
        "judge_scores": judge_scores
    }


def main():

    # This script expects a trained intent model to be supplied.
    # The exact model-loading step will be added when we finalize
    # the model artifact/reproduction workflow.

    print(
        "Response evaluation module created."
    )

    print(
        "Use evaluate_example() with a trained "
        "AmazonSupportAgent."
    )


if __name__ == "__main__":
    main()