from app.db.database import get_connection

from app.embeddings.embedder import generate_embedding


def search(
    query: str,
    top_k: int = 2,
    similarity_threshold: float = 0.5,
    source: str | None = None,
):

    # Generate embedding for the user's query
    query_embedding = generate_embedding(query)

    # Convert Python list into pgvector format
    query_vector = "[" + ",".join(map(str, query_embedding)) + "]"

    connection = get_connection()

    cursor = connection.cursor()

    # <=> operator is used for cosine distance in pgvector.
    # 1 - cosine distance gives us similarity score.

    if source:

        cursor.execute(
            """
            SELECT
                content,
                source,
                chunk_index,
                1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            WHERE source = %s
              AND 1 - (embedding <=> %s::vector) >= %s
            ORDER BY similarity DESC
            LIMIT %s
            """,
            (
                query_vector,
                source,
                query_vector,
                similarity_threshold,
                top_k,
            ),
        )

    else:

        cursor.execute(
            """
            SELECT
                content,
                source,
                chunk_index,
                1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            WHERE 1 - (embedding <=> %s::vector) >= %s
            ORDER BY similarity DESC
            LIMIT %s
            """,
            (
                query_vector,
                query_vector,
                similarity_threshold,
                top_k,
            ),
        )

    results = cursor.fetchall()

    cursor.close()

    connection.close()

    return results


if __name__ == "__main__":

    queries = [
        "What happens when an access token expires?",
        "How should refresh tokens be stored?",
        "What happens if authentication fails?",
        "How do I cook pasta?",
    ]

    """
    For example:

        "How do I cook pasta?"

        It will still return three authentication chunks.

        That's a real RAG problem.

        Next we'll solve it with similarity thresholds:

        Similarity threshold helps us reject results
        that are not semantically relevant.
    """

    answers = []

    for query in queries:

        results = search(
            query,
            similarity_threshold=0.3,
            source="data/documents/authentication.md",
        )

        answers.append((query, results))

    for query, results in answers:

        print(f"\nQuery: {query}")

        for content, source, chunk_index, similarity in results:

            print(f"Similarity: {similarity}")
            print(f"Source: {source}")
            print(f"Chunk: {chunk_index}")

            print(
                f"Content: {content[:50]}..."
            )  # Print first 50 characters of the content

            print("-" * 80)