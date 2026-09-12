import pandas as pd

from src.agent import AmazonSupportAgent


INPUT_PATH = "data/sample_input.csv"
OUTPUT_PATH = "agent_output.csv"


def main():
    print("Loading input data...")

    df = pd.read_csv(INPUT_PATH)

    if "customer_message" not in df.columns:
        raise ValueError(
            "Input CSV must contain a 'customer_message' column."
        )

    print(f"Found {len(df)} customer messages.\n")

    agent = AmazonSupportAgent()

    results = []

    for i, message in enumerate(df["customer_message"], start=1):

        print(f"Processing message {i}/{len(df)}...")

        result = agent.run(message)

        results.append({
            "customer_message": result["customer_message"],
            "intent": result["intent"],
            "intent_confidence": result["intent_confidence"],
            "response": result["response"],
            "decision": result["decision"],
            "escalation_reason": result["escalation_reason"]
        })

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nAgent run complete.")
    print(f"Results saved to: {OUTPUT_PATH}")

    print("\nResults:")
    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()