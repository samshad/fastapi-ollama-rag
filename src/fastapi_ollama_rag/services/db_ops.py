import json
from pathlib import Path

import aiosql
import asyncpg
import structlog

from fastapi_ollama_rag.models.chat import SearchResult
from fastapi_ollama_rag.models.chunk import DocumentChunk

logger = structlog.get_logger(__name__)

sql_dir = Path(__file__).parent.parent / "core" / "sql"
queries = aiosql.from_path(sql_dir / "documents.sql", "asyncpg")

BULK_INSERT_SQL = (sql_dir / "bulk_insert.sql").read_text(encoding="utf-8")


async def is_file_processed(
    conn: asyncpg.Connection, file_hash: str, user_id: str
) -> bool:
    """
    Lightning-fast check to see if a file fingerprint already exists for this user.
    """
    result = await queries.check_file_exists(conn, file_hash=file_hash, user_id=user_id)
    logger.info(
        "Check if this file fingerprint already exists for this user!",
        file_hash=file_hash,
        user_id=user_id,
    )
    return bool(result)


async def bulk_insert_chunks(
    conn: asyncpg.Connection,
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
    user_id: str,
    filename: str,
    file_hash: str,
) -> None:
    """Efficiently bulk-inserts the file and its chunks into the multi-tenant DB."""
    if len(chunks) != len(embeddings):
        logger.error("Mismatch between chunks and embeddings count")
        raise ValueError("Mismatch between chunks and embeddings count.")

    try:
        logger.info("Starting multi-tenant DB insertion", record_count=len(chunks))

        file_record = await queries.create_file_record(
            conn, user_id=user_id, filename=filename, file_hash=file_hash
        )
        file_id = file_record["id"]

        # raw_sql = await queries.get_bulk_insert_sql(c)

        records = [
            (chunk.text, json.dumps(chunk.metadata), emb, user_id, file_id)
            for chunk, emb in zip(chunks, embeddings)
        ]

        await conn.executemany(BULK_INSERT_SQL, records)
        logger.info("Bulk insertion successful...")

    except Exception as e:
        logger.error("Failed to insert records into database", error=str(e))
        raise


async def search_similar_documents(
    conn: asyncpg.Connection, query_embedding: list[float], user_id: str, limit: int = 5
) -> list[SearchResult]:
    """Performs an isolated vector similarity search for the logged-in user."""
    try:
        logger.info("Executing secured vector search", limit=limit, user_id=user_id)

        records = []
        async for record in queries.search_vectors(
            conn, embedding=query_embedding, user_id=user_id, limit_val=limit
        ):
            records.append(record)

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


async def get_files_for_user(conn: asyncpg.Connection, user_id: str) -> list[dict]:
    """Fetches a list of all files uploaded by the user."""
    logger.info("Fetching files for user", user_id=user_id)

    records = []
    async for record in queries.get_user_files(conn, user_id=user_id):
        records.append(
            {
                "id": str(record["id"]),
                "filename": record["filename"],
                "file_hash": record["file_hash"],
                "created_at": record["created_at"].isoformat(),
            }
        )

    return records


async def delete_file_and_chunks(
    conn: asyncpg.Connection, file_id: str, user_id: str
) -> bool:
    """Safely deletes a file and all its vector chunks using a transaction."""
    logger.info("Attempting to delete file", file_id=file_id, user_id=user_id)

    try:
        async with conn.transaction():
            await queries.delete_file_chunks(conn, file_id=file_id, user_id=user_id)
            await queries.delete_file_record(conn, file_id=file_id, user_id=user_id)

        logger.info("Successfully deleted file and chunks", file_id=file_id)
        return True
    except Exception as e:
        logger.error("Failed to delete file", file_id=file_id, error=str(e))
        raise
