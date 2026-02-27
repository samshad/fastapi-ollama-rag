"""
Transaction-rollback test isolation.

Provides a proxy pool that makes every ``database.pool.acquire()`` call
return the *same* connection with an open transaction.  At test teardown
the transaction is rolled back — zero residue, zero cleanup code,
impossible to accidentally delete real data.

This works because PostgreSQL treats ``conn.transaction()`` calls inside
an already-open transaction as **savepoints**, so service code that uses
``async with conn.transaction():`` (e.g. ``delete_file_and_chunks``)
still works correctly.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import asyncpg


class _SingleConnectionProxy:
    """
    A fake asyncpg pool whose ``acquire()`` always yields the same connection.

    This lets service-layer code like ``user_repo.py`` that calls
    ``database.pool.acquire()`` internally participate in the test's
    outer transaction — because they receive the same connection object.
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        """Yield the pinned connection without releasing it."""
        yield self._conn

    # Forward any other pool attributes that production code may read.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

