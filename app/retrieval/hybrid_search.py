from app.retrieval.search import search
from app.retrieval.keyword_search import keyword_search


def reciprocal_rank_fusion(
    vector_results: list[tuple],
    keyword_results: list[tuple],
    k: int = 60,
) -> list[tuple]:

    scores = {}
    documents = {}

    # Process vector search results
    for rank, result in enumerate(vector_results, start=1):

        content, source, chunk_index, similarity = result

        key = (source, chunk_index)

        documents[key] = result

        scores[key] = scores.get(key, 0) + (
            1 / (k + rank)
        )

    # Process keyword search results
    for rank, result in enumerate(keyword_results, start=1):

        content, source, chunk_index, score = result

        key = (source, chunk_index)

        documents[key] = result

        scores[key] = scores.get(key, 0) + (
            1 / (k + rank)
        )

    # Sort documents by combined RRF score
    ranked_documents = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    results = []

    for key, rrf_score in ranked_documents:

        result = documents[key]

        results.append(
            (
                result,
                rrf_score,
            )
        )

    return results

def hybrid_search(
    query: str,
    top_k: int = 5,
) -> list[tuple]:

    vector_results = search(
        query=query,
        top_k=top_k,
        similarity_threshold=0.0,
    )

    keyword_results = keyword_search(
        query=query,
        top_k=top_k,
    )

    return reciprocal_rank_fusion(
        vector_results=vector_results,
        keyword_results=keyword_results,
    )