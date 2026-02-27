from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import fastapi_ollama_rag.core.database as db_module
from fastapi_ollama_rag.core.config import settings


@pytest.fixture(autouse=True)
def reset_db_pool():
    """Ensure the global pool is None before and after every test."""
    db_module.pool = None
    yield
    db_module.pool = None


@pytest.mark.asyncio
async def test_init_connection():
    """Test that pgvector is registered on new connections."""
    with patch(
        "fastapi_ollama_rag.core.database.register_vector", new_callable=AsyncMock
    ) as mock_register:
        mock_conn = AsyncMock()

        await db_module.init_connection(mock_conn)

        mock_register.assert_awaited_once_with(mock_conn)


@pytest.mark.asyncio
async def test_init_connection_propagates_register_vector_error():
    """Test that an error in register_vector propagates out of init_connection."""
    with patch(
        "fastapi_ollama_rag.core.database.register_vector",
        new_callable=AsyncMock,
        side_effect=TypeError("unsupported vector extension"),
    ):
        mock_conn = AsyncMock()
        with pytest.raises(TypeError, match="unsupported vector extension"):
            await db_module.init_connection(mock_conn)


@pytest.mark.asyncio
async def test_connect_to_db_success():
    """Test successfully creating a connection pool."""
    mock_bootstrap = AsyncMock()
    with patch(
        "fastapi_ollama_rag.core.database.asyncpg.connect", new_callable=AsyncMock,
        return_value=mock_bootstrap,
    ), patch(
        "fastapi_ollama_rag.core.database.asyncpg.create_pool", new_callable=AsyncMock
    ) as mock_create_pool:
        mock_pool_instance = AsyncMock()
        mock_create_pool.return_value = mock_pool_instance

        await db_module.connect_to_db()

        mock_create_pool.assert_awaited_once()
        assert db_module.pool is mock_pool_instance
        mock_bootstrap.execute.assert_awaited_once_with(
            "CREATE EXTENSION IF NOT EXISTS vector"
        )
        mock_bootstrap.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_to_db_passes_correct_parameters():
    """
    Test that create_pool is called with the expected DSN, init hook, and pool sizes.
    """
    with patch(
        "fastapi_ollama_rag.core.database.asyncpg.connect", new_callable=AsyncMock,
        return_value=AsyncMock(),
    ), patch(
        "fastapi_ollama_rag.core.database.asyncpg.create_pool", new_callable=AsyncMock
    ) as mock_create_pool:
        mock_create_pool.return_value = AsyncMock()

        await db_module.connect_to_db()

        mock_create_pool.assert_awaited_once_with(
            dsn=str(settings.database_url),
            init=db_module.init_connection,
            min_size=2,
            max_size=10,
        )


@pytest.mark.asyncio
async def test_connect_to_db_failure():
    """Test that database connection failures are bubbled up."""
    with patch(
        "fastapi_ollama_rag.core.database.asyncpg.connect", new_callable=AsyncMock,
        return_value=AsyncMock(),
    ), patch(
        "fastapi_ollama_rag.core.database.asyncpg.create_pool", new_callable=AsyncMock
    ) as mock_create_pool:
        mock_create_pool.side_effect = Exception("Simulated DB Connection Error")

        with pytest.raises(Exception, match="Simulated DB Connection Error"):
            await db_module.connect_to_db()


@pytest.mark.asyncio
async def test_connect_to_db_failure_leaves_pool_none():
    """Test that pool remains None when connect_to_db fails."""
    with patch(
        "fastapi_ollama_rag.core.database.asyncpg.connect", new_callable=AsyncMock,
        return_value=AsyncMock(),
    ), patch(
        "fastapi_ollama_rag.core.database.asyncpg.create_pool", new_callable=AsyncMock
    ) as mock_create_pool:
        mock_create_pool.side_effect = OSError("network unreachable")

        with pytest.raises(OSError):
            await db_module.connect_to_db()

        assert db_module.pool is None


@pytest.mark.asyncio
async def test_connect_to_db_called_twice_overwrites_pool():
    """
    Test that calling connect_to_db twice replaces the pool reference (potential leak).
    """
    with patch(
        "fastapi_ollama_rag.core.database.asyncpg.connect", new_callable=AsyncMock,
        return_value=AsyncMock(),
    ), patch(
        "fastapi_ollama_rag.core.database.asyncpg.create_pool", new_callable=AsyncMock
    ) as mock_create_pool:
        first_pool = AsyncMock()
        second_pool = AsyncMock()
        mock_create_pool.side_effect = [first_pool, second_pool]

        await db_module.connect_to_db()
        assert db_module.pool is first_pool

        await db_module.connect_to_db()
        assert db_module.pool is second_pool
        first_pool.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_db_connection():
    """Test that closing the connection actually awaits pool.close()."""
    mock_pool = AsyncMock()
    db_module.pool = mock_pool

    await db_module.close_db_connection()

    mock_pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_db_connection_when_pool_is_none():
    """Test that close_db_connection is a safe no-op when pool is None."""
    db_module.pool = None

    await db_module.close_db_connection()


@pytest.mark.asyncio
async def test_close_db_connection_propagates_close_error():
    """Test that an error during pool.close() propagates to the caller."""
    mock_pool = AsyncMock()
    mock_pool.close.side_effect = RuntimeError("close failed")
    db_module.pool = mock_pool

    with pytest.raises(RuntimeError, match="close failed"):
        await db_module.close_db_connection()


@pytest.mark.asyncio
async def test_get_db_uninitialized():
    """Test that get_db raises an error if the pool hasn't been created."""
    db_module.pool = None

    with pytest.raises(RuntimeError, match="Database pool is not initialized"):
        async for _ in db_module.get_db():
            pass


@pytest.mark.asyncio
async def test_get_db_yields_connection():
    """Test that get_db successfully yields a connection from the pool."""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()

    mock_acquire_context = AsyncMock()
    mock_acquire_context.__aenter__.return_value = mock_conn

    mock_pool.acquire.return_value = mock_acquire_context
    db_module.pool = mock_pool

    connections_yielded = []
    async for conn in db_module.get_db():
        connections_yielded.append(conn)

    assert len(connections_yielded) == 1
    assert connections_yielded[0] is mock_conn


@pytest.mark.asyncio
async def test_get_db_releases_connection_after_use():
    """
    Test that the async context manager __aexit__ is called.
    """
    mock_pool = MagicMock()
    mock_conn = AsyncMock()

    mock_acquire_context = AsyncMock()
    mock_acquire_context.__aenter__.return_value = mock_conn

    mock_pool.acquire.return_value = mock_acquire_context
    db_module.pool = mock_pool

    async for _ in db_module.get_db():
        pass

    mock_acquire_context.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_db_releases_connection_on_consumer_error():
    """
    Test that the connection is released even when the consumer raises an exception.
    """
    mock_pool = MagicMock()
    mock_conn = AsyncMock()

    mock_acquire_context = AsyncMock()
    mock_acquire_context.__aenter__.return_value = mock_conn

    mock_pool.acquire.return_value = mock_acquire_context
    db_module.pool = mock_pool

    gen = db_module.get_db()
    with pytest.raises(ValueError, match="consumer boom"):
        _ = await gen.__anext__()
        raise ValueError("consumer boom")

    await gen.aclose()

    mock_acquire_context.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_db_acquire_raises():
    """Test that get_db propagates errors from pool.acquire()."""
    mock_pool = MagicMock()

    mock_acquire_context = AsyncMock()
    mock_acquire_context.__aenter__.side_effect = OSError("cannot acquire")

    mock_pool.acquire.return_value = mock_acquire_context
    db_module.pool = mock_pool

    with pytest.raises(OSError, match="cannot acquire"):
        async for _ in db_module.get_db():
            pass
