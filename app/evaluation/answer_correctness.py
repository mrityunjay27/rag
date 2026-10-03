import os

from dotenv import load_dotenv
from groq import Groq


# Load environment variables from .env file
load_dotenv()


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def evaluate_answer_correctness(
    query: str,
    expected_answer: str,
    generated_answer: str,
) -> bool:

    prompt = f"""
You are evaluating a RAG system.

Determine whether the generated answer correctly answers
the user's question when compared with the expected answer.

The wording does not need to be identical.
Focus on whether the meaning and factual content are correct.

Return only:
YES
or
NO

Question:
{query}

Expected Answer:
{expected_answer}

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