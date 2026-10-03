from app.retrieval.search import search


def calculate_reciprocal_rank(
    query: str,
    expected_evidence: str,
    k: int,
) -> float:

    results = search(
        query=query,
        top_k=k,
        similarity_threshold=0.0,
    )

    for rank, (
        content,
        source,
        chunk_index,
        similarity,
    ) in enumerate(results, start=1):

        if expected_evidence.lower() in content.lower():
            return 1 / rank

    return 0.0
