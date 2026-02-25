from pathlib import Path

import structlog

from fastapi_ollama_rag.core import database
from fastapi_ollama_rag.core.config import settings

logger = structlog.get_logger(__name__)


async def run_migrations() -> None:
    """
    Executes idempotent database migrations.
    Creates the documents table, ensures all columns exist,
    and builds the HNSW vector index.
    Sets up the multi-tenant architecture:
    Users, OTPs, and linking users to Documents table.
    """
    if database.pool is None:
        logger.error("Database pool is not initialized. Cannot run migrations.")
        raise RuntimeError("Database pool is not initialized. Cannot run migrations.")

    schema_path = Path(__file__).parent / "sql" / "schema.sql"

    sql_content = schema_path.read_text(encoding="utf-8")

    sql_content = sql_content.replace("{dimension}", str(settings.embedding_dimension))

    try:
        logger.info("Running database migrations from SQL file...")
        async with database.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(sql_content)

        logger.info("Database migrations completed successfully...")
    except Exception as e:
        logger.error("Failed to execute database migrations!!", error=str(e))
        raise
