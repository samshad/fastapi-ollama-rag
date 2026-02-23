from collections.abc import AsyncGenerator

import asyncpg
import structlog
from pgvector.asyncpg import register_vector

from fastapi_ollama_rag.core.config import settings

logger = structlog.get_logger(__name__)

# Global connection pool
pool: asyncpg.Pool | None = None


async def init_connection(conn: asyncpg.Connection) -> None:
    """
    Hook to configure each new connection created by the pool.
    Registers pgvector so asyncpg can natively handle vector arrays.
    """
    await register_vector(conn)


async def connect_to_db() -> None:
    """Initializes the asyncpg connection pool."""
    global pool
    try:
        logger.info("Initializing database connection pool...")
        pool = await asyncpg.create_pool(
            dsn=str(settings.database_url),
            init=init_connection,
            min_size=2,
            max_size=10,
        )
        logger.info("Database pool initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise


async def close_db_connection() -> None:
    """Closes the connection pool gracefully."""
    global pool
    if pool is not None:
        logger.info("Closing database connection pool...")
        await pool.close()
        logger.info("Database pool closed.")


async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    """
    FastAPI dependency to yield a database connection from the pool.
    Ensures connections are acquired and released safely.
    """
    if pool is None:
        raise RuntimeError("Database pool is not initialized")

    # The async context manager automatically returns the connection to the pool
    async with pool.acquire() as conn:
        yield conn
