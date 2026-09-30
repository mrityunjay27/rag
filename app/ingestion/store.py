import uuid

from app.db.database import get_connection
from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import generate_embedding


def store_document(file_path: str):
    text = load_document(file_path)
    chunks = chunk_text(text)

    document_id = str(uuid.uuid4())

    connection = get_connection()
    cursor = connection.cursor()

    try:
        for index, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)

            cursor.execute(
                """
                INSERT INTO document_chunks (
                    id,
                    document_id,
                    source,
                    chunk_index,
                    content,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid.uuid4()),
                    document_id,
                    file_path,
                    index,
                    chunk,
                    embedding,
                ),
            )

        connection.commit()

        print(f"Stored {len(chunks)} chunks successfully!")
        print(f"Document ID: {document_id}")

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    store_document("data/documents/authentication.md")