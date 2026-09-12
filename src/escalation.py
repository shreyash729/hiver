import json

from langchain_core.prompts import ChatPromptTemplate


escalation_prompt = ChatPromptTemplate.from_template("""
You are deciding whether an Amazon customer-support message can be safely
handled automatically using historical support evidence.

Customer message:
{customer_message}

Predicted intent:
{intent}

Historical support evidence:
{historical_examples}

Decision rules:

AUTO_HANDLE:
Choose this when the historical responses provide a clear and applicable
resolution or next step that can be communicated without accessing the
customer's account, order, payment, or other private information.

ESCALATE:
Choose this when resolving the issue requires account/order/payment-specific
investigation, human intervention, or information that is not available in
the customer message and historical evidence.

Do not make assumptions about the customer's account.

Return ONLY valid JSON in exactly this format:

{
  "decision": "AUTO_HANDLE" or "ESCALATE",
  "reason": "brief explanation of why this decision was made"
}
""")


def decide_escalation(llm, query, predicted_intent, retrieved_cases):

    historical_examples = "\n\n".join(
        [
            f"Customer: {row['cleaned_customer_message']}\n"
            f"Amazon response: {row['cleaned_amazon_response']}"
            for _, row in retrieved_cases.iterrows()
        ]
    )

    chain = escalation_prompt | llm

    result = chain.invoke({
        "customer_message": query,
        "intent": predicted_intent,
        "historical_examples": historical_examples
    })

    try:
        return json.loads(result.content)

    except json.JSONDecodeError:
        return {
            "decision": "ESCALATE",
            "reason": "The escalation decision could not be parsed reliably."
        }