import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vectordb")


def init_database():
    print("Connecting to PostgreSQL container...")
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Read schema.sql
            with open("schema.sql", "r") as f:
                schema_sql = f.read()
            
            print("Applying schema migrations (pgvector, tables, GIN & HNSW indexes)...")
            cur.execute(schema_sql)
            conn.commit()

            # Verify pgvector is enabled
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            version = cur.fetchone()
            print(f"PostgreSQL initialized successfully. pgvector version: {version[0]}")


if __name__ == "__main__":
    init_database()
