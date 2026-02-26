import uuid
import pytest
import asyncpg

from fastapi_ollama_rag.services.db_ops import (
    bulk_insert_chunks,
    is_file_processed,
    search_similar_documents,
    get_files_for_user,
    delete_file_and_chunks,
)
from fastapi_ollama_rag.models.chunk import DocumentChunk
from fastapi_ollama_rag.models.chat import SearchResult

# -------------------------------------------------------------------
# Helper Data
# -------------------------------------------------------------------
# Must match the dimension of your pgvector column (1024 for mxbai-embed-large)
DUMMY_EMBEDDING = [0.05] * 1024


def get_dummy_chunks(n: int = 2) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            text=f"This is test chunk {i}",
            chunk_index=i,
            metadata={"page": i + 1},
        )
        for i in range(n)
    ]


async def insert_test_file(
    db_conn: asyncpg.Connection,
    user_id: str,
    filename: str = "test.pdf",
    file_hash: str | None = None,
    num_chunks: int = 2,
) -> str:
    """Helper to insert a file with chunks. Returns the file_hash."""
    file_hash = file_hash or str(uuid.uuid4())
    chunks = get_dummy_chunks(num_chunks)
    embeddings = [DUMMY_EMBEDDING] * num_chunks
    await bulk_insert_chunks(
        conn=db_conn,
        chunks=chunks,
        embeddings=embeddings,
        user_id=user_id,
        filename=filename,
        file_hash=file_hash,
    )
    return file_hash


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bulk_insert_and_is_processed(db_conn: asyncpg.Connection, test_user: dict):
    """Test inserting chunks and verifying the file hash is recorded."""
    user_id = test_user["id"]
    file_hash = str(uuid.uuid4())  # Generate a random hash for isolation
    filename = "test_document.pdf"

    chunks = get_dummy_chunks()
    embeddings = [DUMMY_EMBEDDING, DUMMY_EMBEDDING]

    # 1. Ensure it is NOT processed initially
    is_processed_before = await is_file_processed(db_conn, file_hash, user_id)
    assert is_processed_before is False

    # 2. Insert the document
    await bulk_insert_chunks(
        conn=db_conn,
        chunks=chunks,
        embeddings=embeddings,
        user_id=user_id,
        filename=filename,
        file_hash=file_hash
    )

    # 3. Ensure it IS processed now
    is_processed_after = await is_file_processed(db_conn, file_hash, user_id)
    assert is_processed_after is True


@pytest.mark.asyncio
async def test_bulk_insert_mismatch_raises_error(db_conn: asyncpg.Connection, test_user: dict):
    """Test the fail-fast validation for mismatched chunk/embedding lengths."""
    chunks = get_dummy_chunks()
    # 2 chunks, but only 1 embedding
    embeddings = [DUMMY_EMBEDDING]

    with pytest.raises(ValueError, match="Mismatch between chunks and embeddings count"):
        await bulk_insert_chunks(
            conn=db_conn,
            chunks=chunks,
            embeddings=embeddings,
            user_id=test_user["id"],
            filename="bad_doc.pdf",
            file_hash="bad_hash"
        )


@pytest.mark.asyncio
async def test_search_similar_documents(db_conn: asyncpg.Connection, test_user: dict):
    """Test vector search returns the correct chunks for the specific user."""
    user_id = test_user["id"]
    file_hash = str(uuid.uuid4())

    # 1. Insert data first
    await bulk_insert_chunks(
        conn=db_conn,
        chunks=get_dummy_chunks(),
        embeddings=[DUMMY_EMBEDDING, DUMMY_EMBEDDING],
        user_id=user_id,
        filename="search_doc.pdf",
        file_hash=file_hash
    )

    # 2. Search for it
    results = await search_similar_documents(
        conn=db_conn,
        query_embedding=DUMMY_EMBEDDING,
        user_id=user_id,
        limit=2
    )

    # 3. Verify results
    assert len(results) > 0
    assert isinstance(results[0].text, str)
    assert "This is test chunk" in results[0].text
    assert hasattr(results[0], "similarity_score")


@pytest.mark.asyncio
async def test_file_management_lifecycle(db_conn: asyncpg.Connection, test_user: dict):
    """Tests fetching the user's files, and then safely deleting them via cascading transaction."""
    user_id = test_user["id"]
    file_hash = str(uuid.uuid4())
    filename = "lifecycle_doc.pdf"

    # 1. Insert
    await bulk_insert_chunks(
        conn=db_conn,
        chunks=get_dummy_chunks(),
        embeddings=[DUMMY_EMBEDDING, DUMMY_EMBEDDING],
        user_id=user_id,
        filename=filename,
        file_hash=file_hash
    )

    # 2. Fetch the file list
    files = await get_files_for_user(db_conn, user_id)

    # Find the specific file we just inserted
    inserted_file = next((f for f in files if f["file_hash"] == file_hash), None)
    assert inserted_file is not None
    assert inserted_file["filename"] == filename

    file_id = inserted_file["id"]

    # 3. Delete the file
    delete_success = await delete_file_and_chunks(db_conn, file_id, user_id)
    assert delete_success is True

    # 4. Verify it's gone from the file list
    files_after_delete = await get_files_for_user(db_conn, user_id)
    assert not any(f["id"] == file_id for f in files_after_delete)

    # 5. Verify the chunks are gone (search should return nothing or irrelevant stuff)
    # We check is_file_processed as a proxy for file existence
    assert await is_file_processed(db_conn, file_hash, user_id) is False


# ===================================================================
# is_file_processed – edge cases
# ===================================================================


@pytest.mark.asyncio
async def test_is_file_processed_returns_bool(db_conn: asyncpg.Connection, test_user: dict):
    """Return type should be exactly bool, not int or truthy value."""
    result = await is_file_processed(db_conn, str(uuid.uuid4()), test_user["id"])
    assert isinstance(result, bool)
    assert result is False


@pytest.mark.asyncio
async def test_is_file_processed_multi_tenant_isolation(
    db_conn: asyncpg.Connection, test_user: dict
):
    """
    A file inserted by user A should NOT be visible to user B.
    Same file_hash, different user_id -> must return False.
    """
    file_hash = await insert_test_file(db_conn, test_user["id"])

    email2 = f"other_{uuid.uuid4()}@example.com"
    rec = await db_conn.fetchrow(
        "INSERT INTO users (email, hashed_password, is_verified) "
        "VALUES ($1, $2, $3) RETURNING id",
        email2, "pwd", True,
    )
    other_user_id = str(rec["id"])

    try:
        assert await is_file_processed(db_conn, file_hash, other_user_id) is False
    finally:
        await db_conn.execute("DELETE FROM users WHERE id = $1", rec["id"])


@pytest.mark.asyncio
async def test_is_file_processed_same_user_different_hash(
    db_conn: asyncpg.Connection, test_user: dict
):
    """Same user, different hash -> should return False."""
    await insert_test_file(db_conn, test_user["id"], file_hash="hash_AAA")
    result = await is_file_processed(db_conn, "hash_ZZZ", test_user["id"])
    assert result is False


# ===================================================================
# bulk_insert_chunks – edge cases
# ===================================================================


@pytest.mark.asyncio
async def test_bulk_insert_empty_chunks(db_conn: asyncpg.Connection, test_user: dict):
    """
    Edge Case: 0 chunks, 0 embeddings -> len check passes (0 == 0).
    A file record should still be created, but no document rows.
    """
    file_hash = str(uuid.uuid4())
    await bulk_insert_chunks(
        conn=db_conn,
        chunks=[],
        embeddings=[],
        user_id=test_user["id"],
        filename="empty.pdf",
        file_hash=file_hash,
    )

    # File record exists
    assert await is_file_processed(db_conn, file_hash, test_user["id"]) is True

    # But no chunks in documents table for this file
    files = await get_files_for_user(db_conn, test_user["id"])
    target = next(f for f in files if f["file_hash"] == file_hash)
    count = await db_conn.fetchval(
        "SELECT count(*) FROM documents WHERE file_id = $1::uuid", target["id"]
    )
    assert count == 0


@pytest.mark.asyncio
async def test_bulk_insert_single_chunk(db_conn: asyncpg.Connection, test_user: dict):
    """Edge Case: Single chunk + single embedding."""
    file_hash = str(uuid.uuid4())
    await bulk_insert_chunks(
        conn=db_conn,
        chunks=get_dummy_chunks(1),
        embeddings=[DUMMY_EMBEDDING],
        user_id=test_user["id"],
        filename="single.pdf",
        file_hash=file_hash,
    )
    assert await is_file_processed(db_conn, file_hash, test_user["id"]) is True


@pytest.mark.asyncio
async def test_bulk_insert_duplicate_file_hash_raises(
    db_conn: asyncpg.Connection, test_user: dict
):
    """
    Edge Case: Inserting the same (user_id, file_hash) twice should violate
    the UNIQUE(user_id, file_hash) constraint and raise an exception.
    """
    file_hash = str(uuid.uuid4())
    await insert_test_file(db_conn, test_user["id"], file_hash=file_hash)

    with pytest.raises(asyncpg.UniqueViolationError):
        await insert_test_file(db_conn, test_user["id"], file_hash=file_hash)


@pytest.mark.asyncio
async def test_bulk_insert_same_hash_different_users(
    db_conn: asyncpg.Connection, test_user: dict
):
    """
    Two different users CAN upload a file with the same hash.
    UNIQUE(user_id, file_hash) is per-user, not global.
    """
    file_hash = str(uuid.uuid4())
    await insert_test_file(db_conn, test_user["id"], file_hash=file_hash)

    email2 = f"other_{uuid.uuid4()}@example.com"
    rec = await db_conn.fetchrow(
        "INSERT INTO users (email, hashed_password, is_verified) "
        "VALUES ($1, $2, $3) RETURNING id",
        email2, "pwd", True,
    )
    other_user_id = str(rec["id"])

    try:
        await insert_test_file(db_conn, other_user_id, file_hash=file_hash)
        assert await is_file_processed(db_conn, file_hash, other_user_id) is True
    finally:
        await db_conn.execute("DELETE FROM users WHERE id = $1", rec["id"])


@pytest.mark.asyncio
async def test_bulk_insert_metadata_is_json_serialized(
    db_conn: asyncpg.Connection, test_user: dict
):
    """Verify that chunk metadata is stored as JSONB and can be read back correctly."""
    file_hash = str(uuid.uuid4())
    chunks = [
        DocumentChunk(
            text="chunk with metadata",
            chunk_index=0,
            metadata={"page": 1, "source": "test.pdf", "nested": {"key": "value"}},
        )
    ]
    await bulk_insert_chunks(
        conn=db_conn,
        chunks=chunks,
        embeddings=[DUMMY_EMBEDDING],
        user_id=test_user["id"],
        filename="meta.pdf",
        file_hash=file_hash,
    )

    results = await search_similar_documents(
        conn=db_conn,
        query_embedding=DUMMY_EMBEDDING,
        user_id=test_user["id"],
        limit=10,
    )
    matched = [r for r in results if r.text == "chunk with metadata"]
    assert len(matched) >= 1
    assert matched[0].metadata["source"] == "test.pdf"
    assert matched[0].metadata["nested"] == {"key": "value"}


@pytest.mark.asyncio
async def test_bulk_insert_more_embeddings_than_chunks_raises(
    db_conn: asyncpg.Connection, test_user: dict
):
    """Edge Case: More embeddings than chunks should also raise ValueError."""
    chunks = get_dummy_chunks(1)
    embeddings = [DUMMY_EMBEDDING, DUMMY_EMBEDDING]

    with pytest.raises(ValueError, match="Mismatch between chunks and embeddings count"):
        await bulk_insert_chunks(
            conn=db_conn,
            chunks=chunks,
            embeddings=embeddings,
            user_id=test_user["id"],
            filename="x.pdf",
            file_hash="x",
        )


# ===================================================================
# search_similar_documents – edge cases
# ===================================================================


@pytest.mark.asyncio
async def test_search_returns_search_result_instances(
    db_conn: asyncpg.Connection, test_user: dict
):
    """Every result should be a SearchResult model instance."""
    await insert_test_file(db_conn, test_user["id"])

    results = await search_similar_documents(
        conn=db_conn,
        query_embedding=DUMMY_EMBEDDING,
        user_id=test_user["id"],
        limit=5,
    )

    for r in results:
        assert isinstance(r, SearchResult)
        assert isinstance(r.text, str)
        assert isinstance(r.metadata, dict)
        assert isinstance(r.similarity_score, float)


@pytest.mark.asyncio
async def test_search_returns_empty_for_user_with_no_data(db_conn: asyncpg.Connection):
    """A user with no uploaded documents should get an empty result list."""
    email = f"empty_{uuid.uuid4()}@example.com"
    rec = await db_conn.fetchrow(
        "INSERT INTO users (email, hashed_password, is_verified) "
        "VALUES ($1, $2, $3) RETURNING id",
        email, "pwd", True,
    )
    fresh_user_id = str(rec["id"])

    try:
        results = await search_similar_documents(
            conn=db_conn,
            query_embedding=DUMMY_EMBEDDING,
            user_id=fresh_user_id,
            limit=5,
        )
        assert results == []
    finally:
        await db_conn.execute("DELETE FROM users WHERE id = $1", rec["id"])


@pytest.mark.asyncio
async def test_search_multi_tenant_isolation(
    db_conn: asyncpg.Connection, test_user: dict
):
    """User A's documents should NOT appear in user B's search results."""
    await insert_test_file(db_conn, test_user["id"], filename="secret.pdf")

    email2 = f"other_{uuid.uuid4()}@example.com"
    rec = await db_conn.fetchrow(
        "INSERT INTO users (email, hashed_password, is_verified) "
        "VALUES ($1, $2, $3) RETURNING id",
        email2, "pwd", True,
    )
    other_user_id = str(rec["id"])

    try:
        results = await search_similar_documents(
            conn=db_conn,
            query_embedding=DUMMY_EMBEDDING,
            user_id=other_user_id,
            limit=10,
        )
        assert results == []
    finally:
        await db_conn.execute("DELETE FROM users WHERE id = $1", rec["id"])


@pytest.mark.asyncio
async def test_search_respects_limit_parameter(
    db_conn: asyncpg.Connection, test_user: dict
):
    """The limit parameter should cap the number of returned results."""
    file_hash = str(uuid.uuid4())
    await bulk_insert_chunks(
        conn=db_conn,
        chunks=get_dummy_chunks(5),
        embeddings=[DUMMY_EMBEDDING] * 5,
        user_id=test_user["id"],
        filename="limit_test.pdf",
        file_hash=file_hash,
    )

    results_all = await search_similar_documents(
        conn=db_conn,
        query_embedding=DUMMY_EMBEDDING,
        user_id=test_user["id"],
        limit=100,
    )

    results_limited = await search_similar_documents(
        conn=db_conn,
        query_embedding=DUMMY_EMBEDDING,
        user_id=test_user["id"],
        limit=2,
    )

    assert len(results_limited) == 2
    assert len(results_all) >= 5


@pytest.mark.asyncio
async def test_search_similarity_score_range(
    db_conn: asyncpg.Connection, test_user: dict
):
    """
    Similarity score (1 - cosine distance) should be between -1 and 1.
    For identical embeddings it should be 1.0 or very close.
    """
    await insert_test_file(db_conn, test_user["id"])

    results = await search_similar_documents(
        conn=db_conn,
        query_embedding=DUMMY_EMBEDDING,
        user_id=test_user["id"],
        limit=5,
    )

    assert len(results) > 0
    for r in results:
        assert -1.0 <= r.similarity_score <= 1.0

    # Searching with the exact same embedding should give score ~= 1.0
    assert results[0].similarity_score == pytest.approx(1.0, abs=0.01)


@pytest.mark.asyncio
async def test_search_results_ordered_by_similarity(
    db_conn: asyncpg.Connection, test_user: dict
):
    """Results should be ordered by similarity, highest first."""
    await insert_test_file(db_conn, test_user["id"], num_chunks=3)

    results = await search_similar_documents(
        conn=db_conn,
        query_embedding=DUMMY_EMBEDDING,
        user_id=test_user["id"],
        limit=10,
    )

    if len(results) >= 2:
        for i in range(len(results) - 1):
            assert results[i].similarity_score >= results[i + 1].similarity_score


# ===================================================================
# get_files_for_user – edge cases
# ===================================================================


@pytest.mark.asyncio
async def test_get_files_returns_correct_structure(
    db_conn: asyncpg.Connection, test_user: dict
):
    """Each returned dict must have id, filename, file_hash, created_at."""
    await insert_test_file(db_conn, test_user["id"], filename="struct.pdf")

    files = await get_files_for_user(db_conn, test_user["id"])

    assert len(files) >= 1
    for f in files:
        assert set(f.keys()) == {"id", "filename", "file_hash", "created_at"}
        assert isinstance(f["id"], str)
        assert isinstance(f["filename"], str)
        assert isinstance(f["file_hash"], str)
        assert isinstance(f["created_at"], str)


@pytest.mark.asyncio
async def test_get_files_empty_for_new_user(db_conn: asyncpg.Connection):
    """A fresh user with no uploads should get an empty list."""
    email = f"fresh_{uuid.uuid4()}@example.com"
    rec = await db_conn.fetchrow(
        "INSERT INTO users (email, hashed_password, is_verified) "
        "VALUES ($1, $2, $3) RETURNING id",
        email, "pwd", True,
    )
    fresh_id = str(rec["id"])

    try:
        files = await get_files_for_user(db_conn, fresh_id)
        assert files == []
    finally:
        await db_conn.execute("DELETE FROM users WHERE id = $1", rec["id"])


@pytest.mark.asyncio
async def test_get_files_multi_tenant_isolation(
    db_conn: asyncpg.Connection, test_user: dict
):
    """User A's files should NOT appear in user B's file list."""
    await insert_test_file(db_conn, test_user["id"], filename="private.pdf")

    email2 = f"other_{uuid.uuid4()}@example.com"
    rec = await db_conn.fetchrow(
        "INSERT INTO users (email, hashed_password, is_verified) "
        "VALUES ($1, $2, $3) RETURNING id",
        email2, "pwd", True,
    )
    other_id = str(rec["id"])

    try:
        files = await get_files_for_user(db_conn, other_id)
        assert not any(f["filename"] == "private.pdf" for f in files)
    finally:
        await db_conn.execute("DELETE FROM users WHERE id = $1", rec["id"])


@pytest.mark.asyncio
async def test_get_files_multiple_files(
    db_conn: asyncpg.Connection, test_user: dict
):
    """User with multiple files should see all of them."""
    for name in ["a.pdf", "b.pdf", "c.pdf"]:
        await insert_test_file(db_conn, test_user["id"], filename=name)

    files = await get_files_for_user(db_conn, test_user["id"])
    filenames = [f["filename"] for f in files]

    assert "a.pdf" in filenames
    assert "b.pdf" in filenames
    assert "c.pdf" in filenames


@pytest.mark.asyncio
async def test_get_files_ordered_by_created_at_desc(
    db_conn: asyncpg.Connection, test_user: dict
):
    """Files should be returned newest-first (ORDER BY created_at DESC)."""
    import asyncio

    for name in ["first.pdf", "second.pdf", "third.pdf"]:
        await insert_test_file(db_conn, test_user["id"], filename=name)
        await asyncio.sleep(0.05)

    files = await get_files_for_user(db_conn, test_user["id"])
    our = [f for f in files if f["filename"] in ("first.pdf", "second.pdf", "third.pdf")]
    assert len(our) == 3
    assert our[0]["filename"] == "third.pdf"
    assert our[2]["filename"] == "first.pdf"


@pytest.mark.asyncio
async def test_get_files_created_at_is_iso_format(
    db_conn: asyncpg.Connection, test_user: dict
):
    """created_at should be a valid ISO 8601 string."""
    from datetime import datetime

    await insert_test_file(db_conn, test_user["id"], filename="iso.pdf")

    files = await get_files_for_user(db_conn, test_user["id"])
    target = next(f for f in files if f["filename"] == "iso.pdf")

    parsed = datetime.fromisoformat(target["created_at"])
    assert parsed is not None


@pytest.mark.asyncio
async def test_get_files_id_is_valid_uuid(
    db_conn: asyncpg.Connection, test_user: dict
):
    """The 'id' field should be a valid UUID string."""
    await insert_test_file(db_conn, test_user["id"], filename="uuid_check.pdf")

    files = await get_files_for_user(db_conn, test_user["id"])
    target = next(f for f in files if f["filename"] == "uuid_check.pdf")

    parsed = uuid.UUID(target["id"])
    assert str(parsed) == target["id"]


# ===================================================================
# delete_file_and_chunks – edge cases
# ===================================================================


@pytest.mark.asyncio
async def test_delete_returns_true_for_nonexistent_file(
    db_conn: asyncpg.Connection, test_user: dict
):
    """
    Edge Case: Deleting a non-existent file_id.
    DELETE ... WHERE ... succeeds with 0 rows affected.
    The function should still return True (no exception).
    """
    fake_file_id = str(uuid.uuid4())
    result = await delete_file_and_chunks(db_conn, fake_file_id, test_user["id"])
    assert result is True


@pytest.mark.asyncio
async def test_delete_multi_tenant_isolation(
    db_conn: asyncpg.Connection, test_user: dict
):
    """
    User B should NOT be able to delete user A's file.
    The WHERE clause includes user_id, so the DELETE affects 0 rows,
    and user A's file remains.
    """
    file_hash = await insert_test_file(
        db_conn, test_user["id"], filename="protected.pdf"
    )

    files = await get_files_for_user(db_conn, test_user["id"])
    target = next(f for f in files if f["file_hash"] == file_hash)
    file_id = target["id"]

    email2 = f"attacker_{uuid.uuid4()}@example.com"
    rec = await db_conn.fetchrow(
        "INSERT INTO users (email, hashed_password, is_verified) "
        "VALUES ($1, $2, $3) RETURNING id",
        email2, "pwd", True,
    )
    other_user_id = str(rec["id"])

    try:
        await delete_file_and_chunks(db_conn, file_id, other_user_id)
        assert await is_file_processed(db_conn, file_hash, test_user["id"]) is True
    finally:
        await db_conn.execute("DELETE FROM users WHERE id = $1", rec["id"])


@pytest.mark.asyncio
async def test_delete_removes_both_file_and_chunks(
    db_conn: asyncpg.Connection, test_user: dict
):
    """After deletion, both the file record AND document rows should be gone."""
    file_hash = str(uuid.uuid4())
    await bulk_insert_chunks(
        conn=db_conn,
        chunks=get_dummy_chunks(3),
        embeddings=[DUMMY_EMBEDDING] * 3,
        user_id=test_user["id"],
        filename="full_delete.pdf",
        file_hash=file_hash,
    )

    files = await get_files_for_user(db_conn, test_user["id"])
    target = next(f for f in files if f["file_hash"] == file_hash)
    file_id = target["id"]

    chunk_count_before = await db_conn.fetchval(
        "SELECT count(*) FROM documents WHERE file_id = $1::uuid", file_id
    )
    assert chunk_count_before == 3

    await delete_file_and_chunks(db_conn, file_id, test_user["id"])

    assert await is_file_processed(db_conn, file_hash, test_user["id"]) is False

    chunk_count_after = await db_conn.fetchval(
        "SELECT count(*) FROM documents WHERE file_id = $1::uuid", file_id
    )
    assert chunk_count_after == 0


@pytest.mark.asyncio
async def test_delete_returns_bool(db_conn: asyncpg.Connection, test_user: dict):
    """delete_file_and_chunks should return exactly True (bool)."""
    file_hash = await insert_test_file(db_conn, test_user["id"])

    files = await get_files_for_user(db_conn, test_user["id"])
    target = next(f for f in files if f["file_hash"] == file_hash)

    result = await delete_file_and_chunks(db_conn, target["id"], test_user["id"])
    assert result is True
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_delete_one_file_does_not_affect_another(
    db_conn: asyncpg.Connection, test_user: dict
):
    """Deleting file A should not affect file B for the same user."""
    hash_a = await insert_test_file(db_conn, test_user["id"], filename="del_a.pdf")
    hash_b = await insert_test_file(db_conn, test_user["id"], filename="del_b.pdf")

    files = await get_files_for_user(db_conn, test_user["id"])
    file_a = next(f for f in files if f["file_hash"] == hash_a)

    await delete_file_and_chunks(db_conn, file_a["id"], test_user["id"])

    assert await is_file_processed(db_conn, hash_b, test_user["id"]) is True
    assert await is_file_processed(db_conn, hash_a, test_user["id"]) is False


# ===================================================================
# Exception path: search_similar_documents (db_ops.py L97-99)
# ===================================================================


@pytest.mark.asyncio
async def test_search_similar_documents_raises_on_db_error(
    db_conn, test_user,
):
    """
    Force a DB error during search_similar_documents to cover L97-99.
    Patch the aiosql query to raise, simulating a broken DB connection.
    """
    from unittest.mock import patch

    user_id = test_user["id"]

    with patch(
        "fastapi_ollama_rag.services.db_ops.queries.search_vectors",
        side_effect=RuntimeError("Simulated DB error"),
    ):
        with pytest.raises(RuntimeError, match="Simulated DB error"):
            await search_similar_documents(
                conn=db_conn,
                query_embedding=DUMMY_EMBEDDING,
                user_id=user_id,
                limit=5,
            )


# ===================================================================
# Exception path: delete_file_and_chunks (db_ops.py L133-135)
# ===================================================================


@pytest.mark.asyncio
async def test_delete_file_and_chunks_raises_on_db_error(
    db_conn, test_user,
):
    """
    Force a DB error during delete_file_and_chunks to cover L133-135.
    Patch the aiosql query to raise, simulating a DB failure.
    """
    from unittest.mock import patch

    user_id = test_user["id"]

    with patch(
        "fastapi_ollama_rag.services.db_ops.queries.delete_file_chunks",
        side_effect=RuntimeError("Simulated delete error"),
    ):
        with pytest.raises(RuntimeError, match="Simulated delete error"):
            await delete_file_and_chunks(
                db_conn, file_id=str(uuid.uuid4()), user_id=user_id
            )
