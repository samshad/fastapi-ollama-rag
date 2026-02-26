import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi_ollama_rag.main import app, lifespan


# -------------------------------------------------------------------
# Lifespan tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.main.run_migrations", new_callable=AsyncMock)
@patch("fastapi_ollama_rag.main.connect_to_db", new_callable=AsyncMock)
@patch("fastapi_ollama_rag.main.close_db_connection", new_callable=AsyncMock)
async def test_lifespan_calls_startup_and_shutdown(
    mock_close, mock_connect, mock_migrate
):
    """The lifespan context manager should call connect, migrate on startup, and close on shutdown."""
    async with lifespan(app):
        mock_connect.assert_awaited_once()
        mock_migrate.assert_awaited_once()

    mock_close.assert_awaited_once()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.main.run_migrations", new_callable=AsyncMock)
@patch("fastapi_ollama_rag.main.connect_to_db", new_callable=AsyncMock)
@patch("fastapi_ollama_rag.main.close_db_connection", new_callable=AsyncMock)
async def test_lifespan_startup_order(mock_close, mock_connect, mock_migrate):
    """connect_to_db must be called before run_migrations."""
    call_order = []
    mock_connect.side_effect = lambda: call_order.append("connect")
    mock_migrate.side_effect = lambda: call_order.append("migrate")

    async with lifespan(app):
        pass

    assert call_order == ["connect", "migrate"]


# -------------------------------------------------------------------
# App wiring tests
# -------------------------------------------------------------------


def test_app_title():
    """App title should match settings.project_name."""
    from fastapi_ollama_rag.core.config import settings

    assert app.title == settings.project_name


def test_app_has_auth_routes():
    """App should have /auth routes registered."""
    routes = [r.path for r in app.routes]
    assert any("/auth" in r for r in routes)


def test_app_has_chat_routes():
    """App should have /chat routes registered."""
    routes = [r.path for r in app.routes]
    assert any("/chat" in r for r in routes)


def test_app_has_documents_routes():
    """App should have /documents routes registered."""
    routes = [r.path for r in app.routes]
    assert any("/documents" in r for r in routes)


def test_app_has_health_route():
    """App should have /health route registered."""
    routes = [r.path for r in app.routes]
    assert "/health" in routes


# -------------------------------------------------------------------
# Health endpoint test
# -------------------------------------------------------------------


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.main.get_db")
async def test_health_check(mock_get_db):
    """Test the /health endpoint returns correct structure."""
    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = "PostgreSQL 16.1"

    async def override_get_db():
        yield mock_conn

    app.dependency_overrides[mock_get_db] = override_get_db

    # We need to override using the actual get_db import used in the module
    from fastapi_ollama_rag.core.database import get_db as real_get_db

    app.dependency_overrides[real_get_db] = override_get_db

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "project_name" in data
        assert data["database_version"] == "PostgreSQL 16.1"
    finally:
        app.dependency_overrides.clear()

