import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI

# Import the router and dependencies
from fastapi_ollama_rag.api.routes.chat import router
from fastapi_ollama_rag.api.dependencies import get_current_user
from fastapi_ollama_rag.core.database import get_db
from fastapi_ollama_rag.models.chat import SearchResult

# -------------------------------------------------------------------
# Setup a "Mini" FastAPI App with Dependency Overrides
# -------------------------------------------------------------------
mini_app = FastAPI()
mini_app.include_router(router)

# A separate app WITHOUT the auth override — for testing 401 behavior
unauth_app = FastAPI()
unauth_app.include_router(router)


# 1. Mock the authenticated user
async def override_get_current_user():
    return {"id": "mock_user_123", "email": "test@example.com"}


# 2. Mock the database connection
async def override_get_db():
    yield AsyncMock()


# 3. Apply the overrides to our mini app
mini_app.dependency_overrides[get_current_user] = override_get_current_user
mini_app.dependency_overrides[get_db] = override_get_db

# Only override DB for unauth app — leave auth dependency intact
unauth_app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
async def client():
    """Provides an async HTTP client connected to our mini FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=mini_app), base_url="http://testserver"
    ) as ac:
        yield ac


@pytest.fixture
async def unauth_client():
    """Provides a client WITHOUT auth overrides — requests arrive unauthenticated."""
    async with AsyncClient(
        transport=ASGITransport(app=unauth_app), base_url="http://testserver"
    ) as ac:
        yield ac


# -------------------------------------------------------------------
# Fake Generators and Data for Mocks
# -------------------------------------------------------------------
async def fake_streaming_generator(*args, **kwargs):
    """Simulates an LLM streaming chunks of text over time."""
    yield "Hello "
    yield "World"
    yield "!"


async def fake_empty_generator(*args, **kwargs):
    """Simulates an LLM that yields nothing."""
    return
    yield  # makes it an async generator


def get_fake_search_results():
    """Returns a list of fake SearchResult Pydantic objects."""
    return [
        SearchResult(
            text="This is a relevant chunk.",
            metadata={"page": 1},
            similarity_score=0.95,
        ),
        SearchResult(
            text="This is another chunk.",
            metadata={"page": 2},
            similarity_score=0.82,
        ),
    ]


# ===================================================================
# POST /chat/search — happy path
# ===================================================================


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_raw_vector_search(mock_embed, mock_search, client: AsyncClient):
    """Test that the /search endpoint successfully embeds and retrieves chunks."""
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_search.return_value = get_fake_search_results()

    payload = {"query": "What is the capital of France?", "limit": 2}
    response = await client.post("/chat/search", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert len(data) == 2
    assert data[0]["text"] == "This is a relevant chunk."
    assert data[0]["similarity_score"] == 0.95

    # Verify the user ID from our dependency override was passed to the DB search
    mock_search.assert_awaited_once()
    kwargs = mock_search.call_args[1]
    assert kwargs["user_id"] == "mock_user_123"


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_search_calls_generate_embedding_with_query(
    mock_embed, mock_search, client: AsyncClient
):
    """L36: generate_embedding must be called with request.query."""
    mock_embed.return_value = [0.1]
    mock_search.return_value = []

    await client.post("/chat/search", json={"query": "My specific question"})

    mock_embed.assert_awaited_once_with("My specific question")


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_search_passes_all_kwargs_to_db(
    mock_embed, mock_search, client: AsyncClient
):
    """L39-41: Verify all keyword args passed to search_similar_documents."""
    mock_embed.return_value = [0.5, 0.6]
    mock_search.return_value = []

    await client.post("/chat/search", json={"query": "test", "limit": 7})

    kwargs = mock_search.call_args[1]
    assert kwargs["query_embedding"] == [0.5, 0.6]
    assert kwargs["user_id"] == "mock_user_123"
    assert kwargs["limit"] == 7
    assert "conn" in kwargs


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_search_default_limit(mock_embed, mock_search, client: AsyncClient):
    """ChatRequest.limit defaults to 5 when not provided."""
    mock_embed.return_value = [0.1]
    mock_search.return_value = []

    await client.post("/chat/search", json={"query": "test"})

    kwargs = mock_search.call_args[1]
    assert kwargs["limit"] == 5


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_search_empty_results(mock_embed, mock_search, client: AsyncClient):
    """Edge Case: No matching documents → 200 with empty list."""
    mock_embed.return_value = [0.1]
    mock_search.return_value = []

    response = await client.post("/chat/search", json={"query": "unknown topic"})

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_search_response_is_json(mock_embed, mock_search, client: AsyncClient):
    """Response content-type should be application/json."""
    mock_embed.return_value = [0.1]
    mock_search.return_value = get_fake_search_results()

    response = await client.post("/chat/search", json={"query": "test"})

    assert "application/json" in response.headers["content-type"]


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_search_result_structure(mock_embed, mock_search, client: AsyncClient):
    """Each result must have text, metadata, and similarity_score keys."""
    mock_embed.return_value = [0.1]
    mock_search.return_value = get_fake_search_results()

    response = await client.post("/chat/search", json={"query": "test"})

    for item in response.json():
        assert set(item.keys()) == {"text", "metadata", "similarity_score"}
        assert isinstance(item["text"], str)
        assert isinstance(item["metadata"], dict)
        assert isinstance(item["similarity_score"], float)


# ===================================================================
# POST /chat/search — validation errors
# ===================================================================


@pytest.mark.asyncio
async def test_search_missing_query(client: AsyncClient):
    """Missing 'query' field → 422."""
    response = await client.post("/chat/search", json={"limit": 5})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_empty_body(client: AsyncClient):
    """Completely empty body → 422."""
    response = await client.post("/chat/search", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_null_query(client: AsyncClient):
    """Edge Case: query is null → 422."""
    response = await client.post("/chat/search", json={"query": None})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_no_json_body(client: AsyncClient):
    """No body at all → 422."""
    response = await client.post(
        "/chat/search",
        content=b"",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


# ===================================================================
# POST /chat/search — service failures
# ===================================================================


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
    side_effect=RuntimeError("Ollama down"),
)
async def test_search_embedding_failure(mock_embed, client: AsyncClient):
    """Edge Case: generate_embedding raises → propagates."""
    with pytest.raises(RuntimeError, match="Ollama down"):
        await client.post("/chat/search", json={"query": "test"})


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
    side_effect=RuntimeError("DB connection lost"),
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_search_db_failure(mock_embed, mock_search, client: AsyncClient):
    """Edge Case: search_similar_documents raises → propagates."""
    mock_embed.return_value = [0.1]

    with pytest.raises(RuntimeError, match="DB connection lost"):
        await client.post("/chat/search", json={"query": "test"})


# ===================================================================
# POST /chat/search — authentication
# ===================================================================


@pytest.mark.asyncio
async def test_search_no_auth_token(unauth_client: AsyncClient):
    """
    Edge Case: No Authorization header → 401.
    The unauth_client does NOT override get_current_user, so the real
    OAuth2 dependency rejects the request.
    """
    response = await unauth_client.post(
        "/chat/search", json={"query": "test"}
    )
    assert response.status_code == 401


# ===================================================================
# POST /chat/completions — happy path
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.routes.chat.generate_rag_response")
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_chat_completions_streaming(
    mock_embed, mock_search, mock_generate, client: AsyncClient
):
    """Test that the /completions endpoint returns a StreamingResponse."""
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_search.return_value = get_fake_search_results()
    mock_generate.return_value = fake_streaming_generator()

    payload = {"query": "Tell me a story.", "limit": 5}
    response = await client.post("/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.text == "Hello World!"
    assert response.headers["content-type"] == "text/plain; charset=utf-8"


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.routes.chat.generate_rag_response")
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_completions_calls_generate_embedding_with_query(
    mock_embed, mock_search, mock_generate, client: AsyncClient
):
    """L65: generate_embedding must be called with request.query."""
    mock_embed.return_value = [0.1]
    mock_search.return_value = []
    mock_generate.return_value = fake_empty_generator()

    await client.post(
        "/chat/completions", json={"query": "Specific user question"}
    )

    mock_embed.assert_awaited_once_with("Specific user question")


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.routes.chat.generate_rag_response")
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_completions_passes_all_kwargs_to_search(
    mock_embed, mock_search, mock_generate, client: AsyncClient
):
    """L67-69: Verify all keyword args passed to search_similar_documents."""
    mock_embed.return_value = [0.5]
    mock_search.return_value = []
    mock_generate.return_value = fake_empty_generator()

    await client.post(
        "/chat/completions", json={"query": "test", "limit": 3}
    )

    kwargs = mock_search.call_args[1]
    assert kwargs["query_embedding"] == [0.5]
    assert kwargs["user_id"] == "mock_user_123"
    assert kwargs["limit"] == 3
    assert "conn" in kwargs


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.routes.chat.generate_rag_response")
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_completions_passes_query_and_context_to_generator(
    mock_embed, mock_search, mock_generate, client: AsyncClient
):
    """L71: generate_rag_response called with query=request.query and context=results."""
    results = get_fake_search_results()
    mock_embed.return_value = [0.1]
    mock_search.return_value = results
    mock_generate.return_value = fake_empty_generator()

    await client.post(
        "/chat/completions", json={"query": "What is RAG?"}
    )

    mock_generate.assert_called_once_with(query="What is RAG?", context=results)


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.routes.chat.generate_rag_response")
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_completions_generate_not_awaited(
    mock_embed, mock_search, mock_generate, client: AsyncClient
):
    """
    L71: generate_rag_response is called WITHOUT await — it's a sync function
    that returns an async generator. Verify it's called (not awaited).
    """
    mock_embed.return_value = [0.1]
    mock_search.return_value = []
    mock_generate.return_value = fake_empty_generator()

    await client.post("/chat/completions", json={"query": "test"})

    # It should be called once but NOT awaited (it's sync returning async gen)
    mock_generate.assert_called_once()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.routes.chat.generate_rag_response")
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_completions_default_limit(
    mock_embed, mock_search, mock_generate, client: AsyncClient
):
    """ChatRequest.limit defaults to 5 when not provided."""
    mock_embed.return_value = [0.1]
    mock_search.return_value = []
    mock_generate.return_value = fake_empty_generator()

    await client.post("/chat/completions", json={"query": "test"})

    kwargs = mock_search.call_args[1]
    assert kwargs["limit"] == 5


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.routes.chat.generate_rag_response")
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_completions_empty_context(
    mock_embed, mock_search, mock_generate, client: AsyncClient
):
    """
    Edge Case: No matching documents (empty context).
    The LLM should still be called with an empty context list.
    """
    mock_embed.return_value = [0.1]
    mock_search.return_value = []
    mock_generate.return_value = fake_streaming_generator()

    response = await client.post(
        "/chat/completions", json={"query": "unknown topic"}
    )

    assert response.status_code == 200
    mock_generate.assert_called_once_with(query="unknown topic", context=[])


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.routes.chat.generate_rag_response")
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_completions_empty_generator(
    mock_embed, mock_search, mock_generate, client: AsyncClient
):
    """Edge Case: Generator yields nothing → 200 with empty body."""
    mock_embed.return_value = [0.1]
    mock_search.return_value = []
    mock_generate.return_value = fake_empty_generator()

    response = await client.post(
        "/chat/completions", json={"query": "test"}
    )

    assert response.status_code == 200
    assert response.text == ""
    assert response.headers["content-type"] == "text/plain; charset=utf-8"


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.routes.chat.generate_rag_response")
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_completions_media_type_is_text_plain(
    mock_embed, mock_search, mock_generate, client: AsyncClient
):
    """L79: StreamingResponse media_type='text/plain'."""
    mock_embed.return_value = [0.1]
    mock_search.return_value = []
    mock_generate.return_value = fake_empty_generator()

    response = await client.post(
        "/chat/completions", json={"query": "test"}
    )

    assert response.headers["content-type"] == "text/plain; charset=utf-8"


# ===================================================================
# POST /chat/completions — validation errors
# ===================================================================


@pytest.mark.asyncio
async def test_completions_missing_query(client: AsyncClient):
    """Missing 'query' field → 422."""
    response = await client.post("/chat/completions", json={"limit": 5})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_completions_empty_body(client: AsyncClient):
    """Completely empty body → 422."""
    response = await client.post("/chat/completions", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_completions_null_query(client: AsyncClient):
    """Edge Case: query is null → 422."""
    response = await client.post(
        "/chat/completions", json={"query": None}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_completions_no_json_body(client: AsyncClient):
    """No body at all → 422."""
    response = await client.post(
        "/chat/completions",
        content=b"",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


# ===================================================================
# POST /chat/completions — service failures
# ===================================================================


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
    side_effect=RuntimeError("Ollama down"),
)
async def test_completions_embedding_failure(mock_embed, client: AsyncClient):
    """Edge Case: generate_embedding raises → propagates."""
    with pytest.raises(RuntimeError, match="Ollama down"):
        await client.post("/chat/completions", json={"query": "test"})


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.chat.search_similar_documents",
    new_callable=AsyncMock,
    side_effect=RuntimeError("DB connection lost"),
)
@patch(
    "fastapi_ollama_rag.api.routes.chat.generate_embedding",
    new_callable=AsyncMock,
)
async def test_completions_db_failure(
    mock_embed, mock_search, client: AsyncClient
):
    """Edge Case: search_similar_documents raises → propagates."""
    mock_embed.return_value = [0.1]

    with pytest.raises(RuntimeError, match="DB connection lost"):
        await client.post("/chat/completions", json={"query": "test"})


# ===================================================================
# POST /chat/completions — authentication
# ===================================================================


@pytest.mark.asyncio
async def test_completions_no_auth_token(unauth_client: AsyncClient):
    """
    Edge Case: No Authorization header → 401.
    """
    response = await unauth_client.post(
        "/chat/completions", json={"query": "test"}
    )
    assert response.status_code == 401


# ===================================================================
# Cross-cutting: wrong HTTP methods
# ===================================================================


@pytest.mark.asyncio
async def test_search_get_not_allowed(client: AsyncClient):
    """GET on a POST-only endpoint → 405."""
    response = await client.get("/chat/search")
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_completions_get_not_allowed(client: AsyncClient):
    """GET on a POST-only endpoint → 405."""
    response = await client.get("/chat/completions")
    assert response.status_code == 405


# ===================================================================
# Cross-cutting: nonexistent route
# ===================================================================


@pytest.mark.asyncio
async def test_nonexistent_chat_route(client: AsyncClient):
    """A route that doesn't exist → 404."""
    response = await client.post(
        "/chat/does-not-exist", json={"query": "test"}
    )
    assert response.status_code == 404

