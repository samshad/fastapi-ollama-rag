import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi_ollama_rag.core.migrations import run_migrations
from fastapi_ollama_rag.core import database
from fastapi_ollama_rag.core.config import settings


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_db_pool():
    """Ensure the global pool state is safely restored between tests."""
    original_pool = database.pool
    yield
    database.pool = original_pool


def _make_mock_pool(mock_conn):
    """Helper to wire up a mock pool → acquire → conn → transaction chain."""
    mock_pool = MagicMock()

    mock_transaction_ctx = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=mock_transaction_ctx)

    mock_acquire_ctx = AsyncMock()
    mock_acquire_ctx.__aenter__.return_value = mock_conn
    mock_pool.acquire = MagicMock(return_value=mock_acquire_ctx)

    return mock_pool


# ===================================================================
# pool is None guard (L19-21)
# ===================================================================


@pytest.mark.asyncio
async def test_run_migrations_pool_is_none():
    """Test that migrations instantly fail if the pool isn't initialized."""
    database.pool = None

    with pytest.raises(RuntimeError, match="Database pool is not initialized"):
        await run_migrations()


@pytest.mark.asyncio
async def test_run_migrations_pool_is_none_exact_message():
    """The RuntimeError message should be the exact fixed string."""
    database.pool = None

    with pytest.raises(RuntimeError) as exc_info:
        await run_migrations()

    assert str(exc_info.value) == "Database pool is not initialized. Cannot run migrations."


# ===================================================================
# schema file reading (L23-25)
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_reads_with_utf8_encoding(mock_read_text):
    """
    L25: read_text(encoding="utf-8") — verify the encoding kwarg is passed.
    """
    mock_read_text.return_value = "SELECT 1;"

    mock_conn = AsyncMock()
    database.pool = _make_mock_pool(mock_conn)

    await run_migrations()

    mock_read_text.assert_called_once_with(encoding="utf-8")


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.core.migrations.Path.read_text",
    side_effect=FileNotFoundError("schema.sql not found"),
)
async def test_run_migrations_schema_file_missing(mock_read_text):
    """
    Edge Case: schema.sql doesn't exist → FileNotFoundError propagates.
    This happens BEFORE the try/except block (L25 is outside try), so
    the exception is NOT caught — it propagates directly.
    """
    mock_conn = AsyncMock()
    database.pool = _make_mock_pool(mock_conn)

    with pytest.raises(FileNotFoundError, match="schema.sql not found"):
        await run_migrations()

    # conn.execute should never be called since the file read failed
    mock_conn.execute.assert_not_awaited()


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.core.migrations.Path.read_text",
    side_effect=PermissionError("Permission denied"),
)
async def test_run_migrations_schema_file_permission_error(mock_read_text):
    """
    Edge Case: schema.sql exists but is not readable → PermissionError propagates.
    Also outside the try/except, so not caught.
    """
    mock_conn = AsyncMock()
    database.pool = _make_mock_pool(mock_conn)

    with pytest.raises(PermissionError, match="Permission denied"):
        await run_migrations()

    mock_conn.execute.assert_not_awaited()


# ===================================================================
# {dimension} replacement (L27)
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_dimension_replacement(mock_read_text):
    """
    L27: {dimension} is replaced with str(settings.embedding_dimension).
    Verify the executed SQL has the numeric dimension, not the placeholder.
    """
    mock_read_text.return_value = "CREATE TABLE docs (vec VECTOR({dimension}));"

    mock_conn = AsyncMock()
    database.pool = _make_mock_pool(mock_conn)

    await run_migrations()

    executed_sql = mock_conn.execute.call_args[0][0]
    expected_dim = str(settings.embedding_dimension)

    assert expected_dim in executed_sql
    assert "{dimension}" not in executed_sql
    assert f"VECTOR({expected_dim})" in executed_sql


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_no_dimension_placeholder(mock_read_text):
    """
    Edge Case: SQL has no {dimension} placeholder at all.
    .replace is a no-op — the SQL should be passed through unchanged.
    """
    raw_sql = "CREATE TABLE users (id UUID PRIMARY KEY);"
    mock_read_text.return_value = raw_sql

    mock_conn = AsyncMock()
    database.pool = _make_mock_pool(mock_conn)

    await run_migrations()

    executed_sql = mock_conn.execute.call_args[0][0]
    assert executed_sql == raw_sql


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_multiple_dimension_placeholders(mock_read_text):
    """
    Edge Case: SQL has multiple {dimension} placeholders.
    str.replace replaces ALL occurrences — verify all are substituted.
    """
    mock_read_text.return_value = (
        "CREATE TABLE a (v1 VECTOR({dimension}));\n"
        "CREATE TABLE b (v2 VECTOR({dimension}));\n"
        "CREATE TABLE c (v3 VECTOR({dimension}));"
    )

    mock_conn = AsyncMock()
    database.pool = _make_mock_pool(mock_conn)

    await run_migrations()

    executed_sql = mock_conn.execute.call_args[0][0]
    expected_dim = str(settings.embedding_dimension)

    # No raw placeholders should remain
    assert "{dimension}" not in executed_sql
    # All three should be replaced
    assert executed_sql.count(expected_dim) == 3


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
@patch("fastapi_ollama_rag.core.migrations.settings")
async def test_run_migrations_custom_embedding_dimension(mock_settings, mock_read_text):
    """
    Edge Case: settings.embedding_dimension is a custom value (not the default 1024).
    Verify the custom value is injected into the SQL.
    """
    mock_settings.embedding_dimension = 384
    mock_read_text.return_value = "VECTOR({dimension})"

    mock_conn = AsyncMock()
    database.pool = _make_mock_pool(mock_conn)

    await run_migrations()

    executed_sql = mock_conn.execute.call_args[0][0]
    assert "384" in executed_sql
    assert "{dimension}" not in executed_sql


# ===================================================================
# Success path — execute within transaction (L29-35)
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_success(mock_read_text):
    """Test that the SQL file is read, dimensions are injected, and it executes."""
    mock_read_text.return_value = "CREATE TABLE docs (vec vector({dimension}));"

    mock_conn = AsyncMock()
    database.pool = _make_mock_pool(mock_conn)

    await run_migrations()

    mock_conn.execute.assert_awaited_once()
    executed_sql = mock_conn.execute.call_args[0][0]
    expected_dimension = str(settings.embedding_dimension)

    assert expected_dimension in executed_sql
    assert "{dimension}" not in executed_sql


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_returns_none(mock_read_text):
    """run_migrations() should return None on success."""
    mock_read_text.return_value = "SELECT 1;"

    mock_conn = AsyncMock()
    database.pool = _make_mock_pool(mock_conn)

    result = await run_migrations()

    assert result is None


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_execute_called_exactly_once(mock_read_text):
    """conn.execute should be called exactly once with the full SQL content."""
    mock_read_text.return_value = "CREATE TABLE t1 (id INT); CREATE TABLE t2 (id INT);"

    mock_conn = AsyncMock()
    database.pool = _make_mock_pool(mock_conn)

    await run_migrations()

    assert mock_conn.execute.await_count == 1


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_uses_transaction(mock_read_text):
    """
    L32: The execute must happen inside conn.transaction().
    Verify transaction() is called on the connection.
    """
    mock_read_text.return_value = "SELECT 1;"

    mock_conn = AsyncMock()
    database.pool = _make_mock_pool(mock_conn)

    await run_migrations()

    mock_conn.transaction.assert_called_once()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_acquires_connection_from_pool(mock_read_text):
    """
    L31: pool.acquire() should be called to get a connection.
    """
    mock_read_text.return_value = "SELECT 1;"

    mock_conn = AsyncMock()
    mock_pool = _make_mock_pool(mock_conn)
    database.pool = mock_pool

    await run_migrations()

    mock_pool.acquire.assert_called_once()


# ===================================================================
# Failure path — exception handling (L36-38)
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_execution_failure(mock_read_text):
    """Test that SQL execution errors are properly bubbled up."""
    mock_read_text.return_value = "INVALID SQL STATEMENT;"

    mock_conn = AsyncMock()
    mock_conn.execute.side_effect = Exception("Simulated DB Crash")
    database.pool = _make_mock_pool(mock_conn)

    with pytest.raises(Exception, match="Simulated DB Crash"):
        await run_migrations()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_preserves_original_exception_type(mock_read_text):
    """
    L38: `raise` re-raises the original exception, preserving its type.
    A ValueError from execute should remain a ValueError, not wrapped.
    """
    mock_read_text.return_value = "SELECT 1;"

    mock_conn = AsyncMock()
    mock_conn.execute.side_effect = ValueError("bad value")
    database.pool = _make_mock_pool(mock_conn)

    with pytest.raises(ValueError, match="bad value"):
        await run_migrations()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_timeout_error_propagates(mock_read_text):
    """Edge Case: TimeoutError from DB execute should propagate."""
    mock_read_text.return_value = "SELECT 1;"

    mock_conn = AsyncMock()
    mock_conn.execute.side_effect = TimeoutError("query timed out")
    database.pool = _make_mock_pool(mock_conn)

    with pytest.raises(TimeoutError, match="query timed out"):
        await run_migrations()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_pool_acquire_failure(mock_read_text):
    """
    Edge Case: pool.acquire() itself raises (e.g., pool exhausted).
    This happens inside the try/except, so it's caught and re-raised.
    """
    mock_read_text.return_value = "SELECT 1;"

    mock_pool = MagicMock()
    mock_acquire_ctx = AsyncMock()
    mock_acquire_ctx.__aenter__.side_effect = OSError("pool exhausted")
    mock_pool.acquire = MagicMock(return_value=mock_acquire_ctx)

    database.pool = mock_pool

    with pytest.raises(OSError, match="pool exhausted"):
        await run_migrations()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_transaction_failure(mock_read_text):
    """
    Edge Case: conn.transaction() raises (e.g., isolation level error).
    The error is inside the try/except, so it's caught and re-raised.
    """
    mock_read_text.return_value = "SELECT 1;"

    mock_conn = AsyncMock()
    mock_transaction_ctx = AsyncMock()
    mock_transaction_ctx.__aenter__.side_effect = RuntimeError("transaction failed")
    mock_conn.transaction = MagicMock(return_value=mock_transaction_ctx)

    mock_pool = MagicMock()
    mock_acquire_ctx = AsyncMock()
    mock_acquire_ctx.__aenter__.return_value = mock_conn
    mock_pool.acquire = MagicMock(return_value=mock_acquire_ctx)

    database.pool = mock_pool

    with pytest.raises(RuntimeError, match="transaction failed"):
        await run_migrations()

    # execute should never be called since transaction setup failed
    mock_conn.execute.assert_not_awaited()


# ===================================================================
# Edge: empty SQL content
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.core.migrations.Path.read_text")
async def test_run_migrations_empty_sql_file(mock_read_text):
    """
    Edge Case: schema.sql is empty. read_text returns "".
    .replace on "" is a no-op. conn.execute("") is called — whether it
    succeeds or fails depends on the DB driver, but the function proceeds.
    """
    mock_read_text.return_value = ""

    mock_conn = AsyncMock()
    database.pool = _make_mock_pool(mock_conn)

    await run_migrations()

    mock_conn.execute.assert_awaited_once_with("")

