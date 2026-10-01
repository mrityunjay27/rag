import os

from dotenv import load_dotenv
from groq import Groq


# Load environment variables from .env file
load_dotenv()


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_answer(
    query: str,
    context: str,
) -> str:

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using only the provided context.

Rules:
1. Do not use knowledge outside the provided context.
2. If the answer cannot be found in the context,
   say "I don't know based on the provided context."
3. For every factual statement, include the source number
   that supports it.
4. Use citations in this format: [Source 1]
5. Only cite sources that actually support the statement.
6. Do not invent source numbers.

Context:
{context}

Question:
{query}

Answer:
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

    return response.choices[0].message.content