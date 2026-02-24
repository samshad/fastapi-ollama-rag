import structlog

from fastapi_ollama_rag.core import database
from fastapi_ollama_rag.core.config import settings

logger = structlog.get_logger(__name__)


async def run_migrations() -> None:
    """
    Executes idempotent database migrations.
    Creates the documents table, ensures all columns exist,
    and builds the HNSW vector index.
    """
    if database.pool is None:
        logger.error("Database pool is not initialized. Cannot run migrations.")
        raise RuntimeError("Database pool is not initialized. Cannot run migrations.")

    dimension = settings.embedding_dimension

    # Table schema with UUID, standard text,
    # JSONB metadata, and the VECTOR column
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        content TEXT NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        embedding VECTOR({dimension}),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """

    add_created_at_sql = """
    ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(); \
    """

    # HNSW Index optimized for cosine similarity search (<=>)
    # While an exact nearest neighbor search requires
    # scanning every single row O(N),
    # HNSW achieves O(log N) search complexity.
    create_index_sql = """
    CREATE INDEX IF NOT EXISTS idx_documents_embedding
        ON documents
        USING hnsw (embedding vector_cosine_ops);
    """

    # GIN Index for fast filtering on JSONB metadata
    create_metadata_index_sql = """
    CREATE INDEX IF NOT EXISTS idx_documents_metadata
        ON documents
        USING gin (metadata);
    """

    try:
        logger.info("Running database migrations...")
        async with database.pool.acquire() as conn:
            # Wrap all schema changes in a single transaction
            async with conn.transaction():
                await conn.execute(create_table_sql)
                await conn.execute(add_created_at_sql)
                await conn.execute(create_index_sql)
                await conn.execute(create_metadata_index_sql)
        logger.info("Database migrations completed successfully.")
    except Exception as e:
        logger.error("Failed to execute database migrations", error=str(e))
        raise
