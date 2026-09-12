import json

from langchain_core.prompts import ChatPromptTemplate


judge_prompt = ChatPromptTemplate.from_template("""
You are an evaluator for an AI customer-support agent.

Evaluate the generated response using ONLY the customer message and
historical support evidence provided below.

Customer message:
{customer_message}

Historical support evidence:
{historical_examples}

Generated response:
{generated_response}

Score each criterion from 1 to 5.

1. resolution_relevance:
Does the response address the customer's actual problem?

2. grounding:
Are the claims and guidance supported by the historical support evidence?

3. helpfulness:
Does the response provide a useful and appropriate next step?

4. unsupported_claims:
Does the response avoid unsupported claims, invented policies,
guarantees, or pretending to access the customer's account/order?

5. overall:
Overall quality of the support response.

Return ONLY valid JSON in exactly this format:

{
  "resolution_relevance": <1-5>,
  "grounding": <1-5>,
  "helpfulness": <1-5>,
  "unsupported_claims": <1-5>,
  "overall": <1-5>,
  "reason": "<brief explanation>"
}
""")


def evaluate_response(
    llm,
    query,
    retrieved_cases,
    generated_response
):
    historical_examples = "\n\n".join(
        [
            f"Customer: {row['cleaned_customer_message']}\n"
            f"Amazon response: {row['cleaned_amazon_response']}"
            for _, row in retrieved_cases.iterrows()
        ]
    )

    chain = judge_prompt | llm

    result = chain.invoke({
        "customer_message": query,
        "historical_examples": historical_examples,
        "generated_response": generated_response
    })

    try:
        return json.loads(result.content)

    except json.JSONDecodeError:
        return {
            "resolution_relevance": None,
            "grounding": None,
            "helpfulness": None,
            "unsupported_claims": None,
            "overall": None,
            "reason": "Judge output could not be parsed."
        }