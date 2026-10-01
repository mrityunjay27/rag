import os
from re import search

from dotenv import load_dotenv
from groq import Groq
from app.retrieval.search import search
from app.generation.context_builder import build_context


query = "What happens when an access token expires?"

results = search(
    query=query,
    top_k=2,
    similarity_threshold=0.3,
)

context = build_context(results)


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

If the answer cannot be found in the context,
say "I don't know based on the provided context."

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


if __name__ == "__main__":
    # Example usage
    query = "What happens when an access token expires?"
    query_1 = "How to play a game of chess?"  # Response from AI was: "I don't know based on the provided context." woohoo!
    
    # Step 1: Search for relevant context from the database (R)
    results = search(
        query=query_1,
        top_k=2,
        similarity_threshold=0.3,
    )

    # Step 2: Build context from the search results (A)
    context = build_context(results)

    # Step 3: Generate an answer using the query and context (G)
    answer = generate_answer(query_1, context)
    print("Answer:")
    print(answer)