import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import FastAPI

# Import the router and dependencies
from fastapi_ollama_rag.api.routes.documents import router
from fastapi_ollama_rag.api.dependencies import get_current_user
from fastapi_ollama_rag.core.database import get_db

# -------------------------------------------------------------------
# Setup a "Mini" FastAPI App with Dependency Overrides
# -------------------------------------------------------------------
mini_app = FastAPI()
mini_app.include_router(router)

# A separate app WITHOUT the auth override — for testing 401 behavior
unauth_app = FastAPI()
unauth_app.include_router(router)


async def override_get_current_user():
    return {"id": "mock_user_123", "email": "test@example.com"}


async def override_get_db():
    yield AsyncMock()


mini_app.dependency_overrides[get_current_user] = override_get_current_user
mini_app.dependency_overrides[get_db] = override_get_db

# Only override DB for unauth app — leave auth dependency intact
unauth_app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=mini_app), base_url="http://testserver"
    ) as ac:
        yield ac


@pytest.fixture
async def unauth_client():
    """Client WITHOUT auth overrides — requests arrive unauthenticated."""
    async with AsyncClient(
        transport=ASGITransport(app=unauth_app), base_url="http://testserver"
    ) as ac:
        yield ac


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _setup_full_ingest_mocks(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    *, num_chunks=2,
):
    """Wire up all ingest pipeline mocks for a successful run."""
    mock_is_processed.return_value = False

    mock_parsed_doc = MagicMock()
    mock_parsed_doc.metadata = {}
    mock_parse.return_value = mock_parsed_doc

    chunks = []
    for i in range(num_chunks):
        c = MagicMock()
        c.text = f"Chunk {i}"
        chunks.append(c)
    mock_chunk.return_value = chunks

    mock_embed.return_value = [0.1, 0.2, 0.3]

    return mock_parsed_doc, chunks


INGEST_PATCHES = [
    "fastapi_ollama_rag.api.routes.documents.bulk_insert_chunks",
    "fastapi_ollama_rag.api.routes.documents.generate_embedding",
    "fastapi_ollama_rag.api.routes.documents.chunk_document",
    "fastapi_ollama_rag.api.routes.documents.parse_pdf",
    "fastapi_ollama_rag.api.routes.documents.is_file_processed",
]


# ===================================================================
# POST /documents/ingest — content type validation (L50-53)
# ===================================================================


@pytest.mark.asyncio
async def test_ingest_invalid_file_type(client: AsyncClient):
    """Uploading a non-PDF file → 400 Bad Request."""
    files = {"file": ("test.txt", b"Hello World", "text/plain")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 400
    assert "Only application/pdf is supported" in response.json()["detail"]


@pytest.mark.asyncio
async def test_ingest_image_file_type(client: AsyncClient):
    """Edge Case: Uploading an image → 400."""
    files = {"file": ("photo.png", b"\x89PNG\r\n", "image/png")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 400
    assert "Only application/pdf is supported" in response.json()["detail"]


@pytest.mark.asyncio
async def test_ingest_no_content_type(client: AsyncClient):
    """
    Edge Case: File uploaded with no/unknown content_type.
    content_type != "application/pdf" → 400.
    """
    files = {"file": ("test.pdf", b"Fake PDF Bytes", "application/octet-stream")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 400


# ===================================================================
# POST /documents/ingest — duplicate detection (L61-72)
# ===================================================================


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.documents.is_file_processed",
    new_callable=AsyncMock,
)
async def test_ingest_already_processed_file(mock_is_processed, client: AsyncClient):
    """Uploading an identical file → skipped."""
    mock_is_processed.return_value = True

    files = {"file": ("test.pdf", b"Fake PDF Bytes", "application/pdf")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "skipped"
    assert data["filename"] == "test.pdf"
    assert data["chunks_processed"] == 0
    assert "already exists" in data["detail"]


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.documents.is_file_processed",
    new_callable=AsyncMock,
)
async def test_ingest_skipped_passes_hash_and_user_id(
    mock_is_processed, client: AsyncClient
):
    """
    L59+L61: is_file_processed is called with (conn, file_hash, user_id).
    Verify user_id is the one from the overridden dependency.
    """
    mock_is_processed.return_value = True

    files = {"file": ("test.pdf", b"Fake PDF Bytes", "application/pdf")}
    await client.post("/documents/ingest", files=files)

    args = mock_is_processed.call_args[0]
    # args[0] = conn (mock), args[1] = file_hash (sha256 hex), args[2] = user_id
    assert args[2] == "mock_user_123"
    # file_hash should be a valid 64-char hex string
    assert len(args[1]) == 64
    assert all(c in "0123456789abcdef" for c in args[1])


# ===================================================================
# POST /documents/ingest — success path (L74-102)
# ===================================================================


@pytest.mark.asyncio
@patch(INGEST_PATCHES[0], new_callable=AsyncMock)
@patch(INGEST_PATCHES[1], new_callable=AsyncMock)
@patch(INGEST_PATCHES[2])  # chunk_document is sync
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_success(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    client: AsyncClient,
):
    """Test the complete happy-path ingestion pipeline."""
    _setup_full_ingest_mocks(
        mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    )

    files = {"file": ("test.pdf", b"Fake PDF Bytes", "application/pdf")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "test.pdf"
    assert data["chunks_processed"] == 2

    assert mock_embed.call_count == 2
    mock_insert.assert_awaited_once()


@pytest.mark.asyncio
@patch(INGEST_PATCHES[0], new_callable=AsyncMock)
@patch(INGEST_PATCHES[1], new_callable=AsyncMock)
@patch(INGEST_PATCHES[2])
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_response_structure(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    client: AsyncClient,
):
    """IngestResponse must have status, filename, chunks_processed; detail is optional."""
    _setup_full_ingest_mocks(
        mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    )

    files = {"file": ("report.pdf", b"PDF data", "application/pdf")}
    response = await client.post("/documents/ingest", files=files)

    data = response.json()
    assert "status" in data
    assert "filename" in data
    assert "chunks_processed" in data
    # detail is optional (None on success), so may or may not be present
    assert data["filename"] == "report.pdf"


@pytest.mark.asyncio
@patch(INGEST_PATCHES[0], new_callable=AsyncMock)
@patch(INGEST_PATCHES[1], new_callable=AsyncMock)
@patch(INGEST_PATCHES[2])
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_metadata_injection(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    client: AsyncClient,
):
    """
    L75-76: parsed_doc.metadata should be updated with file_hash and filename.
    """
    parsed_doc, _ = _setup_full_ingest_mocks(
        mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    )

    files = {"file": ("meta.pdf", b"PDF data", "application/pdf")}
    await client.post("/documents/ingest", files=files)

    assert "file_hash" in parsed_doc.metadata
    assert parsed_doc.metadata["filename"] == "meta.pdf"
    assert len(parsed_doc.metadata["file_hash"]) == 64


@pytest.mark.asyncio
@patch(INGEST_PATCHES[0], new_callable=AsyncMock)
@patch(INGEST_PATCHES[1], new_callable=AsyncMock)
@patch(INGEST_PATCHES[2])
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_chunk_document_called_with_settings(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    client: AsyncClient,
):
    """
    L78-82: chunk_document called with parsed_doc, chunk_size, chunk_overlap from settings.
    """
    from fastapi_ollama_rag.core.config import settings

    parsed_doc, _ = _setup_full_ingest_mocks(
        mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    )

    files = {"file": ("test.pdf", b"PDF data", "application/pdf")}
    await client.post("/documents/ingest", files=files)

    mock_chunk.assert_called_once()
    args, kwargs = mock_chunk.call_args
    assert args[0] is parsed_doc
    assert kwargs["chunk_size"] == settings.chunk_size
    assert kwargs["chunk_overlap"] == settings.chunk_overlap


@pytest.mark.asyncio
@patch(INGEST_PATCHES[0], new_callable=AsyncMock)
@patch(INGEST_PATCHES[1], new_callable=AsyncMock)
@patch(INGEST_PATCHES[2])
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_embed_called_with_chunk_text(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    client: AsyncClient,
):
    """L91: generate_embedding is called with chunk.text for each chunk."""
    _, chunks = _setup_full_ingest_mocks(
        mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    )

    files = {"file": ("test.pdf", b"PDF data", "application/pdf")}
    await client.post("/documents/ingest", files=files)

    calls = mock_embed.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0] == "Chunk 0"
    assert calls[1][0][0] == "Chunk 1"


@pytest.mark.asyncio
@patch(INGEST_PATCHES[0], new_callable=AsyncMock)
@patch(INGEST_PATCHES[1], new_callable=AsyncMock)
@patch(INGEST_PATCHES[2])
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_bulk_insert_args(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    client: AsyncClient,
):
    """
    L96-98: bulk_insert_chunks called with (conn, chunks, embeddings, user_id, filename, file_hash).
    """
    _, chunks = _setup_full_ingest_mocks(
        mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    )

    files = {"file": ("insert.pdf", b"PDF data", "application/pdf")}
    await client.post("/documents/ingest", files=files)

    args = mock_insert.call_args[0]
    # args: conn, chunks, embeddings, user_id, filename, file_hash
    assert args[1] is chunks  # chunks list
    assert len(args[2]) == 2  # embeddings list length matches chunks
    assert args[3] == "mock_user_123"  # user_id
    assert args[4] == "insert.pdf"  # filename
    assert len(args[5]) == 64  # file_hash is sha256 hex


@pytest.mark.asyncio
@patch(INGEST_PATCHES[0], new_callable=AsyncMock)
@patch(INGEST_PATCHES[1], new_callable=AsyncMock)
@patch(INGEST_PATCHES[2])
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_many_chunks_hits_progress_log(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    client: AsyncClient,
):
    """
    L93-94: When (i+1) % 10 == 0, a progress log is emitted.
    Use 12 chunks so the branch triggers at i=9 (chunk 10).
    """
    _setup_full_ingest_mocks(
        mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
        num_chunks=12,
    )

    files = {"file": ("big.pdf", b"PDF data", "application/pdf")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 200
    assert response.json()["chunks_processed"] == 12
    assert mock_embed.call_count == 12


@pytest.mark.asyncio
@patch(INGEST_PATCHES[0], new_callable=AsyncMock)
@patch(INGEST_PATCHES[1], new_callable=AsyncMock)
@patch(INGEST_PATCHES[2])
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_single_chunk(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    client: AsyncClient,
):
    """Boundary: File that produces exactly 1 chunk."""
    _setup_full_ingest_mocks(
        mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
        num_chunks=1,
    )

    files = {"file": ("single.pdf", b"PDF data", "application/pdf")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 200
    assert response.json()["chunks_processed"] == 1
    assert mock_embed.call_count == 1


@pytest.mark.asyncio
@patch(INGEST_PATCHES[0], new_callable=AsyncMock)
@patch(INGEST_PATCHES[1], new_callable=AsyncMock)
@patch(INGEST_PATCHES[2])
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_file_hash_is_deterministic(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    client: AsyncClient,
):
    """
    L59: Same file_bytes should produce the same SHA-256 hash.
    Verify the hash passed to is_file_processed matches expected value.
    """
    import hashlib

    _setup_full_ingest_mocks(
        mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    )

    file_bytes = b"Deterministic test content"
    expected_hash = hashlib.sha256(file_bytes).hexdigest()

    files = {"file": ("test.pdf", file_bytes, "application/pdf")}
    await client.post("/documents/ingest", files=files)

    actual_hash = mock_is_processed.call_args[0][1]
    assert actual_hash == expected_hash


# ===================================================================
# POST /documents/ingest — empty chunks (L84-86)
# ===================================================================


@pytest.mark.asyncio
@patch(INGEST_PATCHES[2])  # chunk_document
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)  # parse_pdf
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)  # is_file_processed
async def test_ingest_empty_chunks_raises_422(
    mock_is_processed, mock_parse, mock_chunk, client: AsyncClient
):
    """
    L84-86: chunk_document returns empty list → ValueError →
    caught by L104-106 → 422 with detail.
    """
    mock_is_processed.return_value = False
    mock_parsed_doc = MagicMock()
    mock_parsed_doc.metadata = {}
    mock_parse.return_value = mock_parsed_doc
    mock_chunk.return_value = []  # no chunks

    files = {"file": ("empty.pdf", b"PDF data", "application/pdf")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 422
    assert "no extractable text" in response.json()["detail"]


# ===================================================================
# POST /documents/ingest — error handling (L104-111)
# ===================================================================


@pytest.mark.asyncio
@patch(INGEST_PATCHES[2])
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_value_error_caught_as_422(
    mock_is_processed, mock_parse, mock_chunk, client: AsyncClient
):
    """
    L104-106: Any ValueError inside the try block → 422.
    """
    mock_is_processed.return_value = False
    mock_parsed_doc = MagicMock()
    mock_parsed_doc.metadata = {}
    mock_parse.return_value = mock_parsed_doc
    mock_chunk.side_effect = ValueError("Custom parsing error")

    files = {"file": ("test.pdf", b"PDF data", "application/pdf")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 422
    assert "Custom parsing error" in response.json()["detail"]


@pytest.mark.asyncio
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_parse_pdf_raises_500(
    mock_is_processed, mock_parse, client: AsyncClient
):
    """
    L107-111: parse_pdf raises a generic Exception → caught → 500.
    """
    mock_is_processed.return_value = False
    mock_parse.side_effect = RuntimeError("PDF corrupted")

    files = {"file": ("bad.pdf", b"bad data", "application/pdf")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 500
    assert "Internal server error" in response.json()["detail"]


@pytest.mark.asyncio
@patch(INGEST_PATCHES[1], new_callable=AsyncMock)  # generate_embedding
@patch(INGEST_PATCHES[2])  # chunk_document
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_embedding_failure_raises_500(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, client: AsyncClient
):
    """
    L107-111: generate_embedding raises → caught → 500.
    """
    mock_is_processed.return_value = False
    mock_parsed_doc = MagicMock()
    mock_parsed_doc.metadata = {}
    mock_parse.return_value = mock_parsed_doc
    chunk = MagicMock()
    chunk.text = "Some text"
    mock_chunk.return_value = [chunk]
    mock_embed.side_effect = RuntimeError("Ollama down")

    files = {"file": ("test.pdf", b"PDF data", "application/pdf")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 500
    assert "Internal server error" in response.json()["detail"]


@pytest.mark.asyncio
@patch(INGEST_PATCHES[0], new_callable=AsyncMock)  # bulk_insert_chunks
@patch(INGEST_PATCHES[1], new_callable=AsyncMock)
@patch(INGEST_PATCHES[2])
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_bulk_insert_failure_raises_500(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    client: AsyncClient,
):
    """L107-111: bulk_insert_chunks raises → caught → 500."""
    _setup_full_ingest_mocks(
        mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    )
    mock_insert.side_effect = RuntimeError("DB write failed")

    files = {"file": ("test.pdf", b"PDF data", "application/pdf")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 500
    assert "Internal server error" in response.json()["detail"]


@pytest.mark.asyncio
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_is_file_processed_failure_raises_500(
    mock_is_processed, client: AsyncClient
):
    """L107-111: is_file_processed raises → caught → 500."""
    mock_is_processed.side_effect = RuntimeError("DB read failed")

    files = {"file": ("test.pdf", b"PDF data", "application/pdf")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 500
    assert "Internal server error" in response.json()["detail"]


@pytest.mark.asyncio
async def test_ingest_no_file_uploaded(client: AsyncClient):
    """Edge Case: No file field in the request → 422."""
    response = await client.post("/documents/ingest")
    assert response.status_code == 422


@pytest.mark.asyncio
@patch(INGEST_PATCHES[0], new_callable=AsyncMock)
@patch(INGEST_PATCHES[1], new_callable=AsyncMock)
@patch(INGEST_PATCHES[2])
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_empty_pdf_bytes(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    client: AsyncClient,
):
    """
    Edge Case: A 0-byte PDF file.
    sha256 of empty bytes is a valid hash. parse_pdf will receive b"".
    """
    import hashlib

    _setup_full_ingest_mocks(
        mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    )

    expected_hash = hashlib.sha256(b"").hexdigest()

    files = {"file": ("empty.pdf", b"", "application/pdf")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 200
    actual_hash = mock_is_processed.call_args[0][1]
    assert actual_hash == expected_hash


@pytest.mark.asyncio
@patch(INGEST_PATCHES[0], new_callable=AsyncMock)
@patch(INGEST_PATCHES[1], new_callable=AsyncMock)
@patch(INGEST_PATCHES[2])
@patch(INGEST_PATCHES[3], new_callable=AsyncMock)
@patch(INGEST_PATCHES[4], new_callable=AsyncMock)
async def test_ingest_500_detail_does_not_leak_internals(
    mock_is_processed, mock_parse, mock_chunk, mock_embed, mock_insert,
    client: AsyncClient,
):
    """
    L109-110: The 500 response should use the fixed generic message,
    NOT leak the internal exception message.
    """
    mock_is_processed.return_value = False
    mock_parse.side_effect = RuntimeError("super secret stack trace info")

    files = {"file": ("test.pdf", b"PDF data", "application/pdf")}
    response = await client.post("/documents/ingest", files=files)

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Internal server error during document ingestion."
    assert "super secret" not in detail


# ===================================================================
# POST /documents/ingest — authentication
# ===================================================================


@pytest.mark.asyncio
async def test_ingest_no_auth_token(unauth_client: AsyncClient):
    """No Authorization header → 401."""
    files = {"file": ("test.pdf", b"Fake PDF Bytes", "application/pdf")}
    response = await unauth_client.post("/documents/ingest", files=files)

    assert response.status_code == 401


# ===================================================================
# GET /documents/ — happy path
# ===================================================================


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.documents.get_files_for_user",
    new_callable=AsyncMock,
)
async def test_list_user_files(mock_get_files, client: AsyncClient):
    """Test fetching the user's uploaded files."""
    mock_get_files.return_value = [
        {"id": "file_1", "filename": "report.pdf"},
        {"id": "file_2", "filename": "invoice.pdf"},
    ]

    response = await client.get("/documents/")

    assert response.status_code == 200
    data = response.json()
    assert len(data["files"]) == 2
    assert data["files"][0]["filename"] == "report.pdf"


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.documents.get_files_for_user",
    new_callable=AsyncMock,
)
async def test_list_user_files_passes_user_id(mock_get_files, client: AsyncClient):
    """L121-122: get_files_for_user called with correct conn + user_id."""
    mock_get_files.return_value = []

    await client.get("/documents/")

    args = mock_get_files.call_args[0]
    assert args[1] == "mock_user_123"


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.documents.get_files_for_user",
    new_callable=AsyncMock,
)
async def test_list_user_files_empty(mock_get_files, client: AsyncClient):
    """Edge Case: User has no files → {"files": []}."""
    mock_get_files.return_value = []

    response = await client.get("/documents/")

    assert response.status_code == 200
    assert response.json() == {"files": []}


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.documents.get_files_for_user",
    new_callable=AsyncMock,
)
async def test_list_user_files_response_structure(mock_get_files, client: AsyncClient):
    """Response must be JSON with a top-level 'files' key."""
    mock_get_files.return_value = [{"id": "1", "filename": "a.pdf"}]

    response = await client.get("/documents/")

    assert "application/json" in response.headers["content-type"]
    assert set(response.json().keys()) == {"files"}


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.documents.get_files_for_user",
    new_callable=AsyncMock,
    side_effect=RuntimeError("DB error"),
)
async def test_list_user_files_db_failure(mock_get_files, client: AsyncClient):
    """
    Edge Case: get_files_for_user raises → no try/except in route,
    so it propagates through FastAPI.
    """
    with pytest.raises(RuntimeError, match="DB error"):
        await client.get("/documents/")


# ===================================================================
# GET /documents/ — authentication
# ===================================================================


@pytest.mark.asyncio
async def test_list_user_files_no_auth(unauth_client: AsyncClient):
    """No Authorization header → 401."""
    response = await unauth_client.get("/documents/")
    assert response.status_code == 401


# ===================================================================
# DELETE /documents/{file_id} — happy path
# ===================================================================


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.documents.delete_file_and_chunks",
    new_callable=AsyncMock,
)
async def test_delete_user_file(mock_delete, client: AsyncClient):
    """Test deleting a file by ID."""
    mock_delete.return_value = True

    response = await client.delete("/documents/file_123")

    assert response.status_code == 200
    assert "successfully" in response.json()["message"]

    mock_delete.assert_awaited_once()
    kwargs = mock_delete.call_args[1]
    assert kwargs["user_id"] == "mock_user_123"
    assert kwargs["file_id"] == "file_123"


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.documents.delete_file_and_chunks",
    new_callable=AsyncMock,
)
async def test_delete_exact_response_message(mock_delete, client: AsyncClient):
    """L142: Response message should be the exact fixed string."""
    mock_delete.return_value = True

    response = await client.delete("/documents/some_id")

    assert (
        response.json()["message"]
        == "File and associated vector chunks deleted successfully."
    )


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.documents.delete_file_and_chunks",
    new_callable=AsyncMock,
)
async def test_delete_response_is_json(mock_delete, client: AsyncClient):
    """Response should be JSON."""
    mock_delete.return_value = True

    response = await client.delete("/documents/file_1")

    assert "application/json" in response.headers["content-type"]


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.documents.delete_file_and_chunks",
    new_callable=AsyncMock,
)
async def test_delete_uuid_file_id(mock_delete, client: AsyncClient):
    """Edge Case: file_id is a UUID (typical real-world case)."""
    mock_delete.return_value = True
    uuid_id = "550e8400-e29b-41d4-a716-446655440000"

    response = await client.delete(f"/documents/{uuid_id}")

    assert response.status_code == 200
    kwargs = mock_delete.call_args[1]
    assert kwargs["file_id"] == uuid_id


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.documents.delete_file_and_chunks",
    new_callable=AsyncMock,
    side_effect=RuntimeError("DB error"),
)
async def test_delete_db_failure(mock_delete, client: AsyncClient):
    """
    Edge Case: delete_file_and_chunks raises → no try/except in route,
    so it propagates through FastAPI.
    """
    with pytest.raises(RuntimeError, match="DB error"):
        await client.delete("/documents/file_1")


# ===================================================================
# DELETE /documents/{file_id} — authentication
# ===================================================================


@pytest.mark.asyncio
async def test_delete_no_auth_token(unauth_client: AsyncClient):
    """No Authorization header → 401."""
    response = await unauth_client.delete("/documents/file_1")
    assert response.status_code == 401


# ===================================================================
# Cross-cutting: wrong HTTP methods
# ===================================================================


@pytest.mark.asyncio
async def test_ingest_get_not_allowed(client: AsyncClient):
    """GET on POST-only /ingest → 405."""
    response = await client.get("/documents/ingest")
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_list_files_post_not_allowed(client: AsyncClient):
    """POST on GET-only /documents/ → 405."""
    response = await client.post("/documents/", json={})
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_delete_post_not_allowed(client: AsyncClient):
    """POST on DELETE-only /documents/{file_id} → 405."""
    response = await client.post("/documents/file_1", json={})
    assert response.status_code == 405


# ===================================================================
# Cross-cutting: nonexistent route
# ===================================================================


@pytest.mark.asyncio
async def test_nonexistent_document_route(client: AsyncClient):
    """A route that doesn't exist → 404."""
    response = await client.get("/documents/nonexistent/nested")
    assert response.status_code in (404, 405)
