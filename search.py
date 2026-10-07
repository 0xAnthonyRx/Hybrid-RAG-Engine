import os
import re
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional
from dotenv import load_dotenv
import litellm
from pgvector import Vector
from pgvector.psycopg import register_vector
import psycopg
from psycopg.rows import dict_row

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vectordb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini/gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))


@contextmanager
def get_db_connection() -> Generator[psycopg.Connection, None, None]:
    """Yields a psycopg connection registered with pgvector and dict_row."""
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        yield conn


def sanitize_tsquery(query: str) -> str:
    """
    Cleans raw user queries for PostgreSQL full-text search:
    1. Replaces hyphens with spaces to prevent accidental 'NOT' (!) negation.
    2. Strips conversational commands (find, show, search).
    3. Joins terms with OR (|) so partial matches surface for ts_rank.
    """
    clean = re.sub(r"[^\w\s]", " ", query)
    tokens = [t.strip() for t in clean.split() if len(t.strip()) > 1]

    stop_verbs = {"find", "show", "get", "search", "where", "what", "how", "who"}
    meaningful = [t for t in tokens if t.lower() not in stop_verbs]

    if not meaningful:
        meaningful = tokens

    if not meaningful:
        return ""

    return " | ".join(meaningful)


def get_query_embedding(query: str, dimensions: int = EMBEDDING_DIM) -> Vector:
    """Generates the query vector wrapped in pgvector.Vector with auto-retries."""
    response = litellm.embedding(
        model=EMBEDDING_MODEL,
        input=[query],
        dimensions=dimensions,
        num_retries=3,  # Automatically retry this bih! TLS handshake dropping every damn time sighs...
    )
    return Vector(response.data[0]["embedding"])


def vector_search(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Pure dense vector search using cosine distance (<=>)."""
    query_vector = get_query_embedding(query)

    sql = """
    SELECT 
        id, 
        content, 
        metadata, 
        ROUND((embedding <=> %(vec)s::vector)::numeric, 4) AS distance
    FROM documents
    ORDER BY embedding <=> %(vec)s::vector ASC
    LIMIT %(limit)s;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"vec": query_vector, "limit": limit})
            rows = cur.fetchall()
            return [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "metadata": r["metadata"] or {},
                    "score": float(r["distance"]),
                }
                for r in rows
            ]


def keyword_search(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Pure sparse keyword search using tsvector, GIN index, and ts_rank."""
    ts_query = sanitize_tsquery(query)

    if not ts_query:
        return []

    sql = """
    SELECT 
        id, 
        content, 
        metadata, 
        ROUND(ts_rank(tsv, to_tsquery('english', %(ts_query)s))::numeric, 4) AS rank_score
    FROM documents
    WHERE tsv @@ to_tsquery('english', %(ts_query)s)
    ORDER BY rank_score DESC
    LIMIT %(limit)s;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"ts_query": ts_query, "limit": limit})
            rows = cur.fetchall()
            return [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "metadata": r["metadata"] or {},
                    "score": float(r["rank_score"]),
                }
                for r in rows
            ]


def hybrid_search(
    query: str, 
    limit: int = 3, 
    k: int = 60, 
    candidates_limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Hybrid Search using Reciprocal Rank Fusion (RRF).
    Combines dense vector search and sparse keyword search in a single SQL query.
    """
    query_vector = get_query_embedding(query)
    ts_query = sanitize_tsquery(query)

    # Fallback to plain text search if sanitization strips all tokens
    keyword_condition = (
        "to_tsquery('english', %(ts_query)s)" if ts_query else "plainto_tsquery('english', %(query)s)"
    )

    sql = f"""
    WITH vector_candidates AS (
        SELECT 
            id,
            content,
            metadata,
            ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector ASC) AS rank
        FROM documents
        ORDER BY embedding <=> %(vec)s::vector ASC
        LIMIT %(candidates)s
    ),
    keyword_candidates AS (
        SELECT 
            id,
            content,
            metadata,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank(tsv, {keyword_condition}) DESC
            ) AS rank
        FROM documents
        WHERE tsv @@ {keyword_condition}
        ORDER BY ts_rank(tsv, {keyword_condition}) DESC
        LIMIT %(candidates)s
    )
    SELECT 
        COALESCE(v.id, k.id) AS id,
        COALESCE(v.content, k.content) AS content,
        COALESCE(v.metadata, k.metadata) AS metadata,
        v.rank AS vec_rank,
        k.rank AS key_rank,
        ROUND((
            COALESCE(1.0 / (%(k)s + v.rank), 0.0) +
            COALESCE(1.0 / (%(k)s + k.rank), 0.0)
        )::numeric, 5) AS rrf_score
    FROM vector_candidates v
    FULL OUTER JOIN keyword_candidates k ON v.id = k.id
    ORDER BY rrf_score DESC
    LIMIT %(limit)s;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                {
                    "vec": query_vector,
                    "ts_query": ts_query,
                    "query": query,
                    "k": k,
                    "candidates": candidates_limit,
                    "limit": limit,
                },
            )
            rows = cur.fetchall()
            return [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "metadata": r["metadata"] or {},
                    "vec_rank": r["vec_rank"],
                    "key_rank": r["key_rank"],
                    "rrf_score": float(r["rrf_score"]),
                }
                for r in rows
            ]


def print_results(title: str, results: List[Dict[str, Any]], score_key: str = "score"):
    print(f"\n--- {title} ---")
    if not results:
        print("  [No matching documents found]")
        return

    for idx, doc in enumerate(results, start=1):
        meta = doc.get("metadata") or {}
        doc_id = meta.get("doc_id", doc.get("id", "N/A"))
        score = doc.get(score_key, "N/A")

        extra_ranks = ""
        if "vec_rank" in doc or "key_rank" in doc:
            v_rk = doc.get("vec_rank") or "-"
            k_rk = doc.get("key_rank") or "-"
            extra_ranks = f" | Vec Rank: {v_rk}, Key Rank: {k_rk}"

        content_preview = (doc.get("content") or "").replace("\n", " ").strip()[:90]
        print(f"[{idx}] ({score_key}: {score}{extra_ranks}) Doc ID: {doc_id}")
        print(f"    Passage: {content_preview}...")
