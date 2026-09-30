from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import generate_embedding


file_path = "data/documents/authentication.md"

text = load_document(file_path)
chunks = chunk_text(text)

for index, chunk in enumerate(chunks):
    embedding = generate_embedding(chunk)

    print(f"\nChunk {index + 1}")
    print("Text:", chunk)
    print("Vector dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])