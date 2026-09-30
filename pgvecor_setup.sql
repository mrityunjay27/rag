-- docker exec -it rag-postgres psql -U rag_user -d rag_db
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(384)
);

ALTER TABLE document_chunks
ADD COLUMN document_id UUID,
ADD COLUMN source TEXT,
ADD COLUMN chunk_index INTEGER;


\d document_chunks
SELECT id, content
FROM document_chunks;
SELECT id, embedding
FROM document_chunks
LIMIT 1;

SELECT
    content,
    embedding <=> %s AS distance
FROM document_chunks
ORDER BY embedding <=> %s
LIMIT 3;

