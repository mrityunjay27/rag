from app.db.database import get_connection


def keyword_search(
    query: str,
    top_k: int = 5,
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            content,
            source,
            chunk_index,
            ts_rank_cd(
                to_tsvector('english', content),
                websearch_to_tsquery('english', %s)
            ) AS score
        FROM document_chunks
        WHERE to_tsvector('english', content)
              @@ websearch_to_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s
        """,
        (
            query,
            query,
            top_k,
        ),
    )

    results = cursor.fetchall()

    cursor.close()
    connection.close()

    return results