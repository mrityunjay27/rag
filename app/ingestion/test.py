from .chunker import chunk_text
from .loader import load_document


FILE_PATH = "data/documents/authentication.md"


text = load_document(FILE_PATH)

print("Document loaded successfully!")
print(f"Characters: {len(text)}")

chunks = chunk_text(text)

print(f"\nNumber of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print(chunk)