import uuid
import pytest_asyncio
import asyncpg
from typing import AsyncGenerator

from fastapi_ollama_rag.core.database import connect_to_db, close_db_connection, get_db


@pytest_asyncio.fixture(scope="session", autouse=True)
async def db_pool_lifecycle():
    """
    Automatically starts the asyncpg connection pool before any tests run,
    and cleanly shuts it down after all tests finish.
    """
    await connect_to_db()
    yield
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
