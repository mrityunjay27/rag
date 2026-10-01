import re

from app.retrieval.search import search
from app.generation.context_builder import build_context
from app.generation.generator import generate_answer


def resolve_citations(
    answer: str,
    results: list[tuple],
) -> str:

    # The LLM can emit Unicode spaces (e.g. U+202F) inside citations,
    # so normalize horizontal whitespace to plain ASCII spaces.
    answer = re.sub(r"[^\S\n]+", " ", answer)

    sources = []

    for index, (
        content,
        source,
        chunk_index,
        similarity,
    ) in enumerate(results, start=1):

        citation = f"[Source {index}]"

        if citation in answer:

            source_info = (
                f"{citation} {source} — Chunk {chunk_index}"
            )

            sources.append(source_info)

    if sources:

        answer += "\n\nSources:\n"
        answer += "\n".join(sources)

    return answer


def ask(
    query: str,
    top_k: int = 2,
    similarity_threshold: float = 0.3,
):
    print("\n")
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(query)

    # Retrieve relevant chunks from the vector database
    results = search(
        query=query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    print("\n")
    print("=" * 80)
    print("RETRIEVED CHUNKS")
    print("=" * 80)

    if results:

        for index, (
            content,
            source,
            chunk_index,
            similarity,
        ) in enumerate(results, start=1):

            print(f"\nChunk {index}")
            print(f"Similarity: {similarity}")
            print(f"Source: {source}")
            print(f"Chunk Index: {chunk_index}")
            print(f"Content:\n{content}")

    else:

        print("No relevant chunks found.")

        print("\n")
        print("=" * 80)
        print("FINAL ANSWER")
        print("=" * 80)

        print("I don't know based on the provided context.")

        return "I don't know based on the provided context."

    # Build context from the retrieved chunks
    context = build_context(results)

    print("\n")
    print("=" * 80)
    print("CONTEXT SENT TO LLM")
    print("=" * 80)

    print(context)

    # Generate an answer using the retrieved context
    answer = generate_answer(
        query=query,
        context=context,
    )

    # Resolve LLM citation references to actual source metadata
    answer = resolve_citations(
        answer=answer,
        results=results,
    )

    print("\n")
    print("=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print(answer)

    return answer