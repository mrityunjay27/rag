from app.retrieval.search import search


def calculate_precision_at_k(
    query: str,
    expected_evidence: str,
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

    for content, source, chunk_index, similarity in results:

        if expected_evidence.lower() in content.lower():
            relevant_chunks += 1

    return relevant_chunks / len(results)
