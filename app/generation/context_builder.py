def build_context(results: list[tuple]) -> str:

    context_parts = []

    for index, (
        content,
        source,
        chunk_index,
        similarity,
    ) in enumerate(results, start=1):

        context_parts.append(
            f"""
[Source {index}]
File: {source}
Chunk: {chunk_index}

Content:
{content}
""".strip()
        )

    return "\n\n".join(context_parts)