def build_context(results: list[tuple]) -> str:
    context_parts = []

    for content, similarity in results:
        context_parts.append(content)

    return "\n\n".join(context_parts)