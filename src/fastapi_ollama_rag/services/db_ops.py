import json

import asyncpg
import structlog

from fastapi_ollama_rag.models.chat import SearchResult
from fastapi_ollama_rag.models.chunk import DocumentChunk

logger = structlog.get_logger(__name__)


async def is_file_processed(conn: asyncpg.Connection, file_hash: str) -> bool:
    """
    Lightning-fast check to see if a file fingerprint already exists in the database.
    """
    query = """
            SELECT 1
            FROM documents
            WHERE metadata ->>'file_hash' = $1
                LIMIT 1; \
            """
    result = await conn.fetchval(query, file_hash)
    return bool(result)


async def bulk_insert_chunks(
    conn: asyncpg.Connection, chunks: list[DocumentChunk], embeddings: list[list[float]]
) -> None:
    """
    Efficiently bulk-inserts the chunks and their vectors into DB.
    """
    if len(chunks) != len(embeddings):
        logger.error("Mismatch between chunks and embeddings count")
        raise ValueError("Mismatch between chunks and embeddings count.")

    query = """
            INSERT INTO documents (content, metadata, embedding)
            VALUES ($1, $2::jsonb, $3); \
            """

    # Prepare the data matrix for EXECUTEMANY (Comparatively faster)
    # json.dumps the metadata dict so Postgres can cast it to JSONB
    records: list[tuple[str, str, list[float]]] = [
        (chunk.text, json.dumps(chunk.metadata), emb)
        for chunk, emb in zip(chunks, embeddings)
    ]

    try:
        logger.info("Starting bulk DB insertion", record_count=len(records))
        await conn.executemany(query, records)
        logger.info("Bulk insertion successful")
    except Exception as e:
        logger.error("Failed to insert records into database", error=str(e))
        raise


async def search_similar_documents(
    conn: asyncpg.Connection, query_embedding: list[float], limit: int = 5
) -> list[SearchResult]:
    """
    Performs a vector similarity search using the HNSW index.
    Returns the top 'limit' most semantically similar chunks.
    """
    # The <=> operator calculates Cosine Distance.
    # Cosine Similarity = 1 - Cosine Distance.
    # Order by distance ascending (closest first).
    query = """
            SELECT content, \
                   metadata, \
                   1 - (embedding <=> $1::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> $1::vector
                LIMIT $2; \
            """

    try:
        logger.info("Executing vector search", limit=limit)

        # The pgvector asyncpg codec handles the binary serialization automatically.
        records = await conn.fetch(query, query_embedding, limit)

        results = [
            SearchResult(
                text=record["content"],
                metadata=json.loads(record["metadata"])
                if isinstance(record["metadata"], str)
                else record["metadata"],
                similarity_score=record["similarity"],
            )
            for record in records
        ]

        logger.info("Vector search complete", results_found=len(results))
        return results

    except Exception as e:
        logger.error("Failed to execute vector search", error=str(e))
        raise
