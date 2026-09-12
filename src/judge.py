import os
import json

from langchain_groq import ChatGroq
import config

MODEL_NAME = config.LLM_MODEL


JUDGE_PROMPT = """
You are an evaluator for an Amazon customer support AI agent.

Evaluate the generated response using ONLY:
1. The customer's message
2. The retrieved historical Amazon support responses
3. The generated response

Score each criterion from 1 to 5.

Criteria:

1. resolution_relevance
How well does the response address the customer's actual issue?

2. grounding
How well is the response supported by the provided historical responses?

3. helpfulness
How useful, clear, empathetic, and actionable is the response?

4. unsupported_claims
How safe is the response with respect to unsupported claims?
5 = no unsupported claims
1 = many serious unsupported claims

5. overall
Overall quality of the response.

Return ONLY valid JSON:

{
  "resolution_relevance": 1,
  "grounding": 1,
  "helpfulness": 1,
  "unsupported_claims": 1,
  "overall": 1,
  "reason": "brief explanation"
}
"""


def create_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set."
        )

    return ChatGroq(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0,
        max_tokens=300,
        reasoning_format="parsed",
        max_retries=2
    )


def judge_response(
    customer_message,
    historical_cases,
    generated_response
):
    """
    Evaluate a generated support response using an LLM judge.
    """

    evidence = []

    for _, row in historical_cases.iterrows():
        evidence.append(
            f"Customer: {row['cleaned_customer_message']}\n"
            f"Amazon response: {row['cleaned_amazon_response']}"
        )

    evidence_text = "\n\n---\n\n".join(evidence)

    prompt = f"""
Customer message:
{customer_message}

Historical support evidence:
{evidence_text}

Generated response:
{generated_response}

Evaluate the generated response using the specified rubric.
Return only the JSON object.
"""

    llm = create_llm()

    response = llm.invoke([
        ("system", JUDGE_PROMPT),
        ("human", prompt)
    ])

    raw_output = response.content.strip()

    try:
        result = json.loads(raw_output)

        required_fields = [
            "resolution_relevance",
            "grounding",
            "helpfulness",
            "unsupported_claims",
            "overall",
            "reason"
        ]

        if not all(field in result for field in required_fields):
            raise ValueError("Missing required evaluation fields.")

        return result

    except (json.JSONDecodeError, ValueError):
        return {
            "resolution_relevance": None,
            "grounding": None,
            "helpfulness": None,
            "unsupported_claims": None,
            "overall": None,
            "reason": "Judge returned an invalid JSON response."
        }