import psycopg2


def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5435,
        database="rag_db",
        user="rag_user",
        password="rag_password",
    )