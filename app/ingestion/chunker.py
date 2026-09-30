# Too Naive. CHUNK SIZE is not considered. Just splits by paragraphs.
def chunk_text_v1(text: str) -> list[str]:
    """Take a string and return a list of paragraphs as chunks.
    Paragraph based chunker.
    """
    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if paragraph:
            chunks.append(paragraph)

    return chunks

# A more sophisticated chunker that considers chunk size and overlap.
# Production RAG chunking are token aware, not character based.
def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + "\n\n"

        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            overlap_text = current_chunk[-overlap:]

            current_chunk = overlap_text + "\n\n" + paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

