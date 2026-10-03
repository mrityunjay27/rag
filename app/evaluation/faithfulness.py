import os

from dotenv import load_dotenv
from groq import Groq


# Load environment variables from .env file
load_dotenv()


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def evaluate_faithfulness(
    context: str,
    generated_answer: str,
) -> bool:

    prompt = f"""
You are evaluating a RAG system.

Determine whether the generated answer is fully supported
by the provided context.

The answer must not contain factual claims that are not
supported by the context.

Return only:
YES
or
NO

Context:
{context}

Generated Answer:
{generated_answer}
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