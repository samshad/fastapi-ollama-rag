import uuid
import pytest_asyncio
import asyncpg
from typing import AsyncGenerator

from fastapi_ollama_rag.core.database import connect_to_db, close_db_connection, get_db
from fastapi_ollama_rag.core import database
from fastapi_ollama_rag.core.migrations import run_migrations
from helpers import track_test_email, get_tracked_emails, clear_tracked_emails


@pytest_asyncio.fixture(scope="session", autouse=True)
async def db_pool_lifecycle():
    """
    Starts the asyncpg pool + runs migrations before any tests,
    then deletes ONLY test-created rows and shuts the pool down.

    Safety: cleanup targets rows whose email appears in the tracked registry.
    Real user data is never touched.
    """
    await connect_to_db()
    await run_migrations()
    # Save pool reference before tests run, because some unit tests
    # (e.g., test_database.py) set database.pool = None as part of their mocking.
    pool_ref = database.pool
    yield
    # --- Cleanup ONLY test-created data ---
    emails = list(get_tracked_emails())
    if pool_ref is not None and emails:
        database.pool = pool_ref
        async with pool_ref.acquire() as conn:
            # OTPs reference email (text), not a FK to users
            await conn.execute(
                "DELETE FROM otps WHERE email = ANY($1::text[])", emails
            )
            # files & documents cascade from users (ON DELETE CASCADE)
            await conn.execute(
                "DELETE FROM users WHERE email = ANY($1::text[])", emails
            )
        clear_tracked_emails()
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
    track_test_email(email)

    # 2. Yield the user for the test to use
    user_dict = {
        "id": str(record["id"]),
        "email": record["email"]
    }
    yield user_dict

    # 3. Per-function cleanup (belt-and-suspenders alongside session teardown)
    await db_conn.execute("DELETE FROM users WHERE id = $1", record["id"])
