from app.retrieval.search import search
from app.generation.context_builder import build_context


query = "What happens when an access token expires?"

results = search(
    query=query,
    top_k=2,
    similarity_threshold=0.3,
)

context = build_context(results)

print("QUERY:")
print(query)

print("\nCONTEXT:")
print(context)