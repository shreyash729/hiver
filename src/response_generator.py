import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import config

MODEL_NAME = config.LLM_MODEL


def create_llm():
    """
    Create the Groq LLM.

    The API key must be supplied through the GROQ_API_KEY
    environment variable.
    """

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY environment variable is not set."
        )

    return ChatGroq(
        model=MODEL_NAME,
        temperature=0
    )


response_prompt = ChatPromptTemplate.from_template("""
You are an Amazon customer support agent.

Your task is to write a helpful reply to the customer's message.

IMPORTANT RULES:

1. Use ONLY the historical Amazon support responses provided below
   as evidence.

2. Do not invent policies, procedures, refunds, timelines, links,
   or guarantees.

3. Do not claim that you checked the customer's account or order.

4. You may combine and paraphrase information from multiple
   historical responses.

5. If the historical responses do not provide a clear resolution,
   politely ask the customer to contact Amazon support for further
   assistance.

6. Keep the reply concise, professional, and empathetic.

7. Do not mention these historical examples in your reply.

Customer message:
{customer_message}

Historical support examples:
{historical_examples}

Write only the final customer-facing reply.
""")


def generate_response(llm, query, retrieved_cases):

    historical_examples = "\n\n".join(
        [
            f"Customer: {row['cleaned_customer_message']}\n"
            f"Amazon response: {row['cleaned_amazon_response']}"
            for _, row in retrieved_cases.iterrows()
        ]
    )

    chain = response_prompt | llm

    response = chain.invoke({
        "customer_message": query,
        "historical_examples": historical_examples
    })

    return response.content