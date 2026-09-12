import os
import json

from langchain_groq import ChatGroq
import config

MODEL_NAME = config.LLM_MODEL


ESCALATION_PROMPT = """
You are deciding whether an Amazon customer-support message can be
automatically handled or should be escalated to a human support agent.

Return ONLY valid JSON in this format:

{
  "decision": "AUTO_HANDLE" or "ESCALATE",
  "reason": "brief explanation"
}

Rules:

AUTO_HANDLE when the issue has a clear, general resolution or next step
that can be provided without accessing private customer information.

ESCALATE when resolving the issue requires:
- checking a specific order or account
- accessing payment or refund details
- investigating customer-specific information
- human intervention
- information that is not available in the provided context

Do not assume that you have access to the customer's account, order,
payment information, or internal Amazon systems.
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
        max_tokens=200,
        reasoning_format="parsed",
        max_retries=2
    )


def decide_escalation(customer_message, generated_response):
    """
    Decide whether the customer request should be auto-handled
    or escalated to a human.
    """

    prompt = f"""
Customer message:
{customer_message}

Proposed support response:
{generated_response}

Decide whether this case should be AUTO_HANDLE or ESCALATE.
Return only the requested JSON object.
"""

    llm = create_llm()

    response = llm.invoke([
        ("system", ESCALATION_PROMPT),
        ("human", prompt)
    ])

    raw_output = response.content.strip()

    try:
        result = json.loads(raw_output)

        if result.get("decision") not in {"AUTO_HANDLE", "ESCALATE"}:
            raise ValueError("Invalid escalation decision.")

        return result

    except (json.JSONDecodeError, ValueError):
        # Fail safely: if the model produces an invalid decision,
        # escalate rather than automatically handling the case.
        return {
            "decision": "ESCALATE",
            "reason": "The escalation model returned an invalid decision, so the case is being routed to human support."
        }