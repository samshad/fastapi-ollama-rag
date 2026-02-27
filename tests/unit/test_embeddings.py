import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi_ollama_rag.services.embeddings import generate_embedding
from fastapi_ollama_rag.core.config import settings


# ===================================================================
# Happy path
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.embeddings.httpx.AsyncClient")
async def test_generate_embedding_success(MockClient):
    """Test happy path: Ollama returns a valid embedding vector."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = None
    MockClient.return_value = mock_client_instance

    result = await generate_embedding("Hello world")

    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.embeddings.httpx.AsyncClient")
async def test_generate_embedding_url_and_payload(MockClient):
    """Verify the correct URL and payload are sent to Ollama."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.5]}
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = None
    MockClient.return_value = mock_client_instance

    await generate_embedding("Test text")

    expected_url = f"{settings.ollama_base_url}/api/embeddings"
    expected_payload = {
        "model": settings.ollama_embedding_model,
        "prompt": "Test text",
    }
    mock_client_instance.post.assert_awaited_once_with(
        expected_url, json=expected_payload
    )


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.embeddings.httpx.AsyncClient")
async def test_generate_embedding_timeout_is_30(MockClient):
    """L17: httpx.AsyncClient is created with timeout=30.0."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1]}
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = None
    MockClient.return_value = mock_client_instance

    await generate_embedding("test")

    MockClient.assert_called_once_with(timeout=30.0)


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.embeddings.httpx.AsyncClient")
async def test_generate_embedding_calls_raise_for_status(MockClient):
    """L20: response.raise_for_status() must be called."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1]}
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = None
    MockClient.return_value = mock_client_instance

    await generate_embedding("test")

    mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.embeddings.httpx.AsyncClient")
async def test_generate_embedding_returns_list_of_floats(MockClient):
    """Return type should be a list of floats."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4]}
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = None
    MockClient.return_value = mock_client_instance

    result = await generate_embedding("test")

    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


# ===================================================================
# Error handling (L25-27)
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.embeddings.httpx.AsyncClient")
async def test_generate_embedding_http_error_raises_runtime_error(MockClient):
    """L25-27: httpx.HTTPError → caught → RuntimeError."""
    mock_client_instance = AsyncMock()
    mock_client_instance.post.side_effect = httpx.HTTPError("Connection refused")
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = None
    MockClient.return_value = mock_client_instance

    with pytest.raises(RuntimeError, match="Ollama embedding failed"):
        await generate_embedding("test")


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.embeddings.httpx.AsyncClient")
async def test_generate_embedding_timeout_raises_runtime_error(MockClient):
    """httpx.ReadTimeout is a subclass of HTTPError → RuntimeError."""
    mock_client_instance = AsyncMock()
    mock_client_instance.post.side_effect = httpx.ReadTimeout("Read timed out")
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = None
    MockClient.return_value = mock_client_instance

    with pytest.raises(RuntimeError, match="Ollama embedding failed"):
        await generate_embedding("test")


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.embeddings.httpx.AsyncClient")
async def test_generate_embedding_connect_error_raises_runtime_error(MockClient):
    """httpx.ConnectError → RuntimeError."""
    mock_client_instance = AsyncMock()
    mock_client_instance.post.side_effect = httpx.ConnectError("No route to host")
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = None
    MockClient.return_value = mock_client_instance

    with pytest.raises(RuntimeError, match="Ollama embedding failed"):
        await generate_embedding("test")


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.embeddings.httpx.AsyncClient")
async def test_generate_embedding_status_error_raises_runtime_error(MockClient):
    """L20: raise_for_status raises httpx.HTTPStatusError → RuntimeError."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Server Error",
        request=MagicMock(),
        response=MagicMock(),
    )

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = None
    MockClient.return_value = mock_client_instance

    with pytest.raises(RuntimeError, match="Ollama embedding failed"):
        await generate_embedding("test")

