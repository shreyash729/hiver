from src.agent import AmazonSupportAgent
from src.judge import judge_response
import os
import sys

def print_separator():
    print("\n" + "=" * 80 + "\n")


def interactive_mode():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        try:
            groq_api_key = input("Enter your GROQ_API_KEY: ").strip().strip("'\"")
            while not groq_api_key:
                print("GROQ_API_KEY cannot be empty. Please enter a valid key.")
                groq_api_key = input("Enter your GROQ_API_KEY: ").strip().strip("'\"")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(1)

        os.environ["GROQ_API_KEY"] = groq_api_key
    print("=" * 80)
    print("Amazon Customer Support AI Agent")
    print("=" * 80)
    print("Enter a customer message.")
    print("Enter -1 to exit.")
    print_separator()

    # Initialize agent once.
    # This loads the cached classifier and historical embeddings.
    agent = AmazonSupportAgent()

    while True:

        print_separator()

        customer_message = input("Customer message:\n> ").strip()

        # Exit condition
        if customer_message == "-1":
            print("\nExiting agent. Goodbye!")
            break

        if not customer_message:
            print("Please enter a valid customer message.")
            continue

        print("\nProcessing...")

        # Run the complete agent pipeline
        result = agent.run(
            customer_message,
            top_k=5
        )

        # --------------------------------------------------
        # 1. Intent
        # --------------------------------------------------

        print("\nIntent:")
        print(
            f"{result['intent']} "
            f"(confidence: {result['intent_confidence']:.4f})"
        )

        # --------------------------------------------------
        # 2. Top 5 historical cases
        # --------------------------------------------------

        historical_cases = result["retrieved_cases"]

        print("\nTop 5 Historical Cases:")

        for i, (_, row) in enumerate(
            historical_cases.iterrows(),
            start=1
        ):

            print(f"\n--- Case {i} ---")

            print(
                f"Customer: "
                f"{row['cleaned_customer_message']}"
            )

            print(
                f"Amazon response: "
                f"{row['cleaned_amazon_response']}"
            )

            print(
                f"Similarity: "
                f"{row['similarity']:.4f}"
            )

        # --------------------------------------------------
        # 3. Generated response
        # --------------------------------------------------

        print("\nBot_response:")
        print(result["response"])

        # --------------------------------------------------
        # 4. Escalation decision
        # --------------------------------------------------

        print("\nEscalation:")
        print(result["decision"])

        print("\nEscalation reason:")
        print(result["escalation_reason"])

        # --------------------------------------------------
        # 5. LLM Judge
        # --------------------------------------------------

        print("\nRunning LLM judge...")

        judge_result = judge_response(
            customer_message=customer_message,
            historical_cases=historical_cases,
            generated_response=result["response"]
        )

        print("\nLLM judge result:")

        if judge_result["overall"] is None:

            print("Judge failed to return a valid result.")
            print(
                f"Reason: {judge_result['reason']}"
            )

        else:

            print(
                f"Resolution relevance: "
                f"{judge_result['resolution_relevance']}/5"
            )

            print(
                f"Grounding: "
                f"{judge_result['grounding']}/5"
            )

            print(
                f"Helpfulness: "
                f"{judge_result['helpfulness']}/5"
            )

            print(
                f"Unsupported claims safety: "
                f"{judge_result['unsupported_claims']}/5"
            )

            print(
                f"Overall: "
                f"{judge_result['overall']}/5"
            )

            print(
                f"Reason: "
                f"{judge_result['reason']}"
            )


if __name__ == "__main__":
    interactive_mode()