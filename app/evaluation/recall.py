from app.retrieval.search import search


def calculate_recall_at_k(
    query: str,
    expected_evidence: str,
    k: int,
) -> int:

    results = search(
        query=query,
        top_k=k,
        similarity_threshold=0.0,
    )

    for content, source, chunk_index, similarity in results:

        if expected_evidence.lower() in content.lower():
            return 1

    return 0
