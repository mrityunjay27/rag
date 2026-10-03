import os

from dotenv import load_dotenv
from groq import Groq


# Load environment variables from .env file
load_dotenv()


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def evaluate_context_relevance(
    query: str,
    context: str,
) -> bool:
    """
    LLM as judge to determine whether the provided context is relevant to answering the user's question.
    """

    prompt = f"""
You are evaluating a RAG retrieval system.

Determine whether the provided context is relevant to answering
the user's question.

Return only:
YES
or
NO

Question:
{query}

Context:
{context}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    result = response.choices[0].message.content.strip().upper()

    return result == "YES"
