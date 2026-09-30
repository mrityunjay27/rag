from app.db.database import get_connection


connection = get_connection()

print("Connected to PostgreSQL!")

connection.close()