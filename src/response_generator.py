import os

from langchain_groq import ChatGroq
import config

MODEL_NAME = config.LLM_MODEL


SYSTEM_PROMPT = """
You are an Amazon customer support agent.

Your task is to write a helpful reply to the customer's message using
ONLY the provided historical Amazon support responses as evidence.

Rules:
1. Do not invent policies, procedures, refunds, timelines, links, or guarantees.
2. Do not claim that you checked the customer's account, order, or payment details.
3. You may combine or paraphrase information from multiple historical responses.
4. If the historical evidence does not provide a clear resolution, politely ask
   the customer to contact Amazon support for further assistance.
5. Keep the response concise, professional, and empathetic.
6. Do not mention that you are using historical examples.
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


def generate_response(customer_message, historical_cases):
    """
    Generate a support response grounded in retrieved historical cases.

    Parameters
    ----------
    customer_message : str
        Incoming customer message.

    historical_cases : pandas.DataFrame
        Retrieved historical cases containing:
        - cleaned_customer_message
        - cleaned_amazon_response
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

Historical Amazon support responses:
{evidence_text}

Write the best possible support reply to the customer.
Use only information supported by the historical responses.
"""

    llm = create_llm()

    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", prompt)
    ])

    return response.content