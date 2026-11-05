from app.retrieval.hybrid_search import hybrid_search


results = hybrid_search(
    query="What happens when an access token expires?",
    top_k=3,
)


for rank, (
    result,
    rrf_score,
) in enumerate(results, start=1):

    content, source, chunk_index, score = result

    print(f"\nRank: {rank}")
    print(f"RRF Score: {rrf_score}")
    print(f"Source: {source}")
    print(f"Chunk: {chunk_index}")
    print(f"Content:\n{content}")