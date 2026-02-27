import uuid

import asyncpg
import pytest_asyncio

from fastapi_ollama_rag.core import database
from fastapi_ollama_rag.core.database import connect_to_db, close_db_connection
from fastapi_ollama_rag.core.migrations import run_migrations


# ───────────────────────────────────────────────────────────────────
# Session: boot pool + run migrations once
# ───────────────────────────────────────────────────────────────────
@pytest_asyncio.fixture(scope="session", autouse=True)
async def _db_pool():
    """
    Starts the real asyncpg pool and runs migrations once per session.
    Saves a reference so unit tests that set database.pool = None
    can't destroy it.
    """
    await connect_to_db()
    await run_migrations()
    pool_ref = database.pool
    yield pool_ref
    # Restore in case a unit test nullified it
    database.pool = pool_ref
    await close_db_connection()


# ───────────────────────────────────────────────────────────────────
# Per-test: convenience fixture for a pre-inserted test user
# ───────────────────────────────────────────────────────────────────
@pytest_asyncio.fixture(scope="function")
async def test_user(db_conn: asyncpg.Connection) -> dict:
    """
    Inserts a verified test user and returns ``{"id": ..., "email": ...}``.
    No manual cleanup needed — the enclosing transaction rolls back.
    """
    email = f"test_{uuid.uuid4()}@example.com"
    hashed_pwd = "dummy_hashed_password"

    record = await db_conn.fetchrow(
        """
        INSERT INTO users (email, hashed_password, is_verified)
        VALUES ($1, $2, $3) RETURNING id, email;
        """,
        email,
        hashed_pwd,
        True,
    )

    return {
        "id": str(record["id"]),
        "email": record["email"],
    }
