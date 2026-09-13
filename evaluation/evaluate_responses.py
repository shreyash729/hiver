import os
import pandas as pd
import sys
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
from sklearn.model_selection import train_test_split

from src.agent import AmazonSupportAgent
from src.judge import judge_response
import src.config as config

DATA_PATH = "data/amazonhelp_cleaned_with_intents.csv"

RANDOM_STATE = config.RANDOM_STATE
TEST_SIZE = config.TEST_SIZE

# Number of response-generation examples to evaluate.
NUM_EVAL_EXAMPLES = 10


def load_test_examples():

    df = pd.read_csv(DATA_PATH)

    # Keep only the 397 manually labelled examples.
    labelled_data = df[
        df["Intent"].notna()
        & (df["Intent"].astype(str).str.strip() != "")
    ].copy()

    # Reproduce the same stratified split used during classifier evaluation.
    _, test_data = train_test_split(
        labelled_data,
        test_size=TEST_SIZE,
        stratify=labelled_data["Intent"],
        random_state=RANDOM_STATE
    )

    # Use a fixed sample so the evaluation is reproducible.
    test_data = test_data.sort_index()

    if len(test_data) > NUM_EVAL_EXAMPLES:
        test_data = test_data.iloc[:NUM_EVAL_EXAMPLES]

    return test_data.reset_index(drop=True)


def evaluate_example(agent, customer_message):

    # Run the complete production agent.
    result = agent.run(
        customer_message,
        top_k=5
    )

    # Evaluate the generated response using the LLM judge.
    judge_result = judge_response(
        customer_message=customer_message,
        historical_cases=result["retrieved_cases"],
        generated_response=result["response"]
    )

    return {
        "customer_message": customer_message,
        "intent": result["intent"],
        "intent_confidence": result["intent_confidence"],
        "response": result["response"],
        "decision": result["decision"],
        "escalation_reason": result["escalation_reason"],
        "resolution_relevance": judge_result["resolution_relevance"],
        "grounding": judge_result["grounding"],
        "helpfulness": judge_result["helpfulness"],
        "unsupported_claims": judge_result["unsupported_claims"],
        "overall": judge_result["overall"],
        "judge_reason": judge_result["reason"]
    }


def main():

    # ---------------------------------------------------------
    # Check API key
    # ---------------------------------------------------------

    if not os.getenv("GROQ_API_KEY"):

        try:
            api_key = input(
                "Enter your GROQ_API_KEY: "
            ).strip().strip("'\"")

            while not api_key:

                print(
                    "GROQ_API_KEY cannot be empty. "
                    "Please enter a valid key."
                )

                api_key = input(
                    "Enter your GROQ_API_KEY: "
                ).strip().strip("'\"")

            os.environ["GROQ_API_KEY"] = api_key

        except (KeyboardInterrupt, EOFError):

            print("\nOperation cancelled.")
            return

    # ---------------------------------------------------------
    # Load evaluation examples
    # ---------------------------------------------------------

    test_data = load_test_examples()

    print(
        f"Evaluating {len(test_data)} held-out response examples..."
    )

    # ---------------------------------------------------------
    # Initialize agent once
    # ---------------------------------------------------------

    agent = AmazonSupportAgent()

    # ---------------------------------------------------------
    # Evaluate examples
    # ---------------------------------------------------------

    results = []

    for i, row in test_data.iterrows():

        customer_message = str(
            row["cleaned_customer_message"]
        )

        print(
            f"\nEvaluating example "
            f"{i + 1}/{len(test_data)}..."
        )

        result = evaluate_example(
            agent,
            customer_message
        )

        results.append(result)

        print(
            f"Intent: {result['intent']} "
            f"(confidence: "
            f"{result['intent_confidence']:.4f})"
        )

        print(
            f"Escalation: {result['decision']}"
        )

        print(
            f"Judge overall: "
            f"{result['overall']}/5"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    output_path = "results/response_evaluation_generated.csv"

    os.makedirs("results", exist_ok=True)

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("RESPONSE EVALUATION SUMMARY")
    print("=" * 70)

    valid_results = results_df[
        results_df["overall"].notna()
    ]

    if len(valid_results) > 0:

        print(
            f"\nExamples evaluated: "
            f"{len(valid_results)}"
        )

        print(
            f"Average resolution relevance: "
            f"{valid_results['resolution_relevance'].mean():.2f}/5"
        )

        print(
            f"Average grounding: "
            f"{valid_results['grounding'].mean():.2f}/5"
        )

        print(
            f"Average helpfulness: "
            f"{valid_results['helpfulness'].mean():.2f}/5"
        )

        print(
            f"Average unsupported claims safety: "
            f"{valid_results['unsupported_claims'].mean():.2f}/5"
        )

        print(
            f"Average overall: "
            f"{valid_results['overall'].mean():.2f}/5"
        )

    print(
        f"\nDetailed results saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()