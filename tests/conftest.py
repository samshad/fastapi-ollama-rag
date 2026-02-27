import uuid
import pytest_asyncio
import asyncpg
from typing import AsyncGenerator

from fastapi_ollama_rag.core.database import connect_to_db, close_db_connection, get_db
from fastapi_ollama_rag.core import database


@pytest_asyncio.fixture(scope="session", autouse=True)
async def db_pool_lifecycle():
    """
    Automatically starts the asyncpg connection pool before any tests run,
    and cleanly shuts it down after all tests finish.

    After all tests complete, truncates test-generated data from otps, users
    (which cascades to files and documents) so that test runs don't pollute
    the real database.
    """
    await connect_to_db()
    # Save pool reference before tests run, because some unit tests
    # (e.g., test_database.py) set database.pool = None as part of their mocking.
    pool_ref = database.pool
    yield
    # --- Cleanup all test-created data ---
    # Restore pool reference in case unit tests nullified it.
    if pool_ref is not None:
        database.pool = pool_ref
        async with pool_ref.acquire() as conn:
            await conn.execute("DELETE FROM otps")
            await conn.execute("DELETE FROM documents")
            await conn.execute("DELETE FROM files")
            await conn.execute("DELETE FROM users")
    await close_db_connection()


@pytest_asyncio.fixture(scope="function")
async def db_conn() -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Provides a real database connection for integration tests.
    Uses your existing get_db dependency.
    """
    async for conn in get_db():
        yield conn


@pytest_asyncio.fixture(scope="function")
async def test_user(db_conn: asyncpg.Connection) -> dict:
    """
    Injects a real test user into the database and returns their details so
    we can test foreign key relationships (like file.user_id).
    """
    email = f"test_{uuid.uuid4()}@example.com"
    hashed_pwd = "dummy_hashed_password"

    # 1. Insert the test user into the DB
    query = """
            INSERT INTO users (email, hashed_password, is_verified)
            VALUES ($1, $2, $3) RETURNING id, email; \
            """
    record = await db_conn.fetchrow(query, email, hashed_pwd, True)

    # 2. Yield the user for the test to use
    user_dict = {
        "id": str(record["id"]),
        "email": record["email"]
    }
    yield user_dict

    # 3. Cleanup: Delete the user (and cascade delete their files) after test finishes
    await db_conn.execute("DELETE FROM users WHERE id = $1", record["id"])
