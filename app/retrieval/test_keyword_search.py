from app.retrieval.keyword_search import keyword_search


results = keyword_search(
    query="access token expires",
    top_k=3,
)


for rank, (
    content,
    source,
    chunk_index,
    score,
) in enumerate(results, start=1):

    print(f"\nRank: {rank}")
    print(f"Score: {score}")
    print(f"Source: {source}")
    print(f"Chunk: {chunk_index}")
    print(f"Content:\n{content}")