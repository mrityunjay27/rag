from app.retrieval.search import search
from app.evaluation.context_relevance import evaluate_context_relevance


def calculate_context_relevance(
    query: str,
    k: int,
) -> float:

    results = search(
        query=query,
        top_k=k,
        similarity_threshold=0.0,
    )

    if not results:
        return 0.0

    relevant_chunks = 0

    for (
        content,
        source,
        chunk_index,
        similarity,
    ) in results:

        is_relevant = evaluate_context_relevance(
            query=query,
            context=content,
        )

        if is_relevant:
            relevant_chunks += 1

        print(
            f"\nChunk {chunk_index} "
            f"Context Relevance: "
            f"{'YES' if is_relevant else 'NO'}"
        )

    return relevant_chunks / len(results)
