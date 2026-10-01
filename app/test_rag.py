from app.rag import ask


queries = [
    "What happens when an access token expires?",
    # "How should refresh tokens be stored?",
    # "What happens if authentication fails?",
    # "How do I cook pasta?",
]


for query in queries:

    # Run the complete RAG pipeline
    ask(
        query=query,
        top_k=2,
        similarity_threshold=0.3,
    )