import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi_ollama_rag.services.generation import generate_rag_response
from fastapi_ollama_rag.models.chat import SearchResult
from fastapi_ollama_rag.core.config import settings


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def get_fake_context():
    return [
        SearchResult(text="Paris is the capital.", metadata={"page": 1}, similarity_score=0.95),
        SearchResult(text="France is in Europe.", metadata={"page": 2}, similarity_score=0.80),
    ]


def make_ollama_line(content: str, done: bool = False) -> str:
    """Create a single Ollama streaming JSON line."""
    return json.dumps({"message": {"content": content}, "done": done})


class FakeAsyncLineIterator:
    """Simulates response.aiter_lines() yielding lines one at a time."""

    def __init__(self, lines: list[str]):
        self._lines = lines
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


class FakeStreamResponse:
    """Simulates an httpx streaming response async context manager."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code} Error",
                request=MagicMock(),
                response=MagicMock(),
            )

    def aiter_lines(self):
        return FakeAsyncLineIterator(self._lines)


class FakeErrorStreamContext:
    """Simulates client.stream() that raises HTTPError inside async with."""

    def __init__(self, error: Exception):
        self._error = error

    async def __aenter__(self):
        raise self._error

    async def __aexit__(self, *args):
        pass


def _make_mock_client(lines=None, error=None):
    """Build a mock httpx.AsyncClient that returns a FakeStreamResponse or raises."""
    mock_client_instance = MagicMock()

    if error is not None:
        mock_client_instance.stream = MagicMock(return_value=FakeErrorStreamContext(error))
    else:
        mock_client_instance.stream = MagicMock(return_value=FakeStreamResponse(lines or []))

    # Support `async with httpx.AsyncClient(...) as client:`
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=None)

    return mock_client_instance


# ===================================================================
# Happy path — streaming tokens
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.generation.httpx.AsyncClient")
async def test_generate_rag_response_streams_tokens(MockClient):
    """Test that the generator yields content tokens from Ollama lines."""
    lines = [
        make_ollama_line("Hello "),
        make_ollama_line("World"),
        make_ollama_line("!", done=True),
    ]

    MockClient.return_value = _make_mock_client(lines)

    tokens = []
    async for token in generate_rag_response(query="Hi", context=[]):
        tokens.append(token)

    assert tokens == ["Hello ", "World", "!"]


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.generation.httpx.AsyncClient")
async def test_generate_rag_response_stops_on_done(MockClient):
    """L70-72: Generator breaks when done=True, ignoring subsequent lines."""
    lines = [
        make_ollama_line("First", done=False),
        make_ollama_line("", done=True),
        make_ollama_line("Should not appear", done=False),
    ]

    MockClient.return_value = _make_mock_client(lines)

    tokens = []
    async for token in generate_rag_response(query="Hi", context=[]):
        tokens.append(token)

    assert tokens == ["First"]
    assert "Should not appear" not in tokens


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.generation.httpx.AsyncClient")
async def test_generate_rag_response_skips_empty_content(MockClient):
    """L66: Only non-empty content strings are yielded."""
    lines = [
        make_ollama_line(""),
        make_ollama_line("Token"),
        make_ollama_line(""),
        make_ollama_line("!", done=True),
    ]

    MockClient.return_value = _make_mock_client(lines)

    tokens = []
    async for token in generate_rag_response(query="Q", context=[]):
        tokens.append(token)

    assert tokens == ["Token", "!"]


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.generation.httpx.AsyncClient")
async def test_generate_rag_response_skips_empty_lines(MockClient):
    """L60: Empty lines from aiter_lines are skipped (the `if line:` guard)."""
    lines = [
        "",
        make_ollama_line("Data"),
        "",
        make_ollama_line("", done=True),
    ]

    MockClient.return_value = _make_mock_client(lines)

    tokens = []
    async for token in generate_rag_response(query="Q", context=[]):
        tokens.append(token)

    assert tokens == ["Data"]


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.generation.httpx.AsyncClient")
async def test_generate_rag_response_payload_structure(MockClient):
    """Verify the payload sent to Ollama has correct model, messages, and stream flag."""
    lines = [make_ollama_line("ok", done=True)]

    mock_client = _make_mock_client(lines)
    MockClient.return_value = mock_client

    context = get_fake_context()

    # Consume the generator
    async for _ in generate_rag_response(query="What is the capital?", context=context):
        pass

    # Verify stream() was called with correct args
    call_args = mock_client.stream.call_args
    assert call_args[0][0] == "POST"  # method

    url = call_args[0][1]
    assert url == f"{settings.ollama_base_url}/api/chat"

    payload = call_args[1]["json"]
    assert payload["model"] == settings.ollama_generation_model
    assert payload["stream"] is True
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"] == "What is the capital?"


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.generation.httpx.AsyncClient")
async def test_generate_rag_response_context_in_system_prompt(MockClient):
    """L23-37: Context chunks are compiled into the system prompt."""
    lines = [make_ollama_line("ok", done=True)]

    mock_client = _make_mock_client(lines)
    MockClient.return_value = mock_client

    context = get_fake_context()

    async for _ in generate_rag_response(query="Q", context=context):
        pass

    payload = mock_client.stream.call_args[1]["json"]
    system_prompt = payload["messages"][0]["content"]

    assert "Paris is the capital." in system_prompt
    assert "France is in Europe." in system_prompt
    assert "0.95" in system_prompt
    assert "0.80" in system_prompt


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.generation.httpx.AsyncClient")
async def test_generate_rag_response_empty_context(MockClient):
    """Edge Case: Empty context list → system prompt still valid, no crash."""
    lines = [make_ollama_line("I don't know.", done=True)]

    MockClient.return_value = _make_mock_client(lines)

    tokens = []
    async for token in generate_rag_response(query="Unknown?", context=[]):
        tokens.append(token)

    assert tokens == ["I don't know."]


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.generation.httpx.AsyncClient")
async def test_generate_rag_response_timeout_is_none(MockClient):
    """L53: httpx.AsyncClient is created with timeout=None."""
    lines = [make_ollama_line("ok", done=True)]

    MockClient.return_value = _make_mock_client(lines)

    async for _ in generate_rag_response(query="Q", context=[]):
        pass

    MockClient.assert_called_once_with(timeout=None)


# ===================================================================
# Error handling (L74-76)
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.generation.httpx.AsyncClient")
async def test_generate_rag_response_http_error_yields_error_message(MockClient):
    """L74-76: httpx.HTTPError → yields an error message string instead of crashing."""
    MockClient.return_value = _make_mock_client(error=httpx.HTTPError("Connection refused"))

    tokens = []
    async for token in generate_rag_response(query="Q", context=[]):
        tokens.append(token)

    assert len(tokens) == 1
    assert "[Error:" in tokens[0]
    assert "Connection to LLM generation service failed" in tokens[0]


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.generation.httpx.AsyncClient")
async def test_generate_rag_response_connect_error(MockClient):
    """httpx.ConnectError is a subclass of HTTPError → error message yielded."""
    MockClient.return_value = _make_mock_client(error=httpx.ConnectError("No route to host"))

    tokens = []
    async for token in generate_rag_response(query="Q", context=[]):
        tokens.append(token)

    assert len(tokens) == 1
    assert "[Error:" in tokens[0]
