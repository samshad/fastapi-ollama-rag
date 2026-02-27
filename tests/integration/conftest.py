"""
Transaction-rollback isolation for all integration tests.

Every test in this directory is automatically wrapped in a database
transaction that rolls back at the end — zero residue, zero cleanup
code, impossible to accidentally delete real data.
"""

import asyncpg
import pytest_asyncio
from typing import AsyncGenerator

from fastapi_ollama_rag.core import database
from helpers import _SingleConnectionProxy


@pytest_asyncio.fixture(autouse=True)
async def db_conn(_db_pool: asyncpg.Pool) -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Autouse fixture that:

    1. Acquires a real connection from the session pool.
    2. Opens a transaction (BEGIN).
    3. Monkey-patches ``database.pool`` with a single-connection proxy so
       that **all** code calling ``database.pool.acquire()`` internally
       (user_repo, auth service, etc.) receives the same connection and
       participates in the same transaction.
    4. Yields the connection for tests that need direct DB access
       (e.g. ``test_db_ops.py`` functions that accept a ``conn`` argument).
    5. After the test, **rolls back** the transaction.  Every INSERT,
       UPDATE, DELETE vanishes.  Real data is never touched.

    Nested ``conn.transaction()`` calls in service code (e.g.
    ``delete_file_and_chunks``) become PostgreSQL SAVEPOINTs, which
    roll back correctly with the outer transaction.
    """
    conn = await _db_pool.acquire()
    tx = conn.transaction()
    await tx.start()

    original_pool = database.pool
    database.pool = _SingleConnectionProxy(conn)

    yield conn

    await tx.rollback()
    database.pool = original_pool
    await _db_pool.release(conn)



