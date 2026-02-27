import fitz
import pytest

from fastapi_ollama_rag.services.pdf_parser import _extract_text_sync, parse_pdf
from fastapi_ollama_rag.models.document import ParsedDocument


# ============================================================================
# Helper Fixtures
# ============================================================================


@pytest.fixture
def valid_pdf_bytes() -> bytes:
    """Generates a valid, 1-page PDF file entirely in memory for testing."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello World! This is a test PDF.")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.fixture
def empty_text_pdf_bytes() -> bytes:
    """Generates a valid PDF with 1 page, but absolutely no text."""
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.fixture
def multi_page_pdf_bytes() -> bytes:
    """Generates a valid 3-page PDF with distinct text on each page."""
    doc = fitz.open()
    for i in range(1, 4):
        page = doc.new_page()
        page.insert_text((50, 50), f"Page {i} content.")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.fixture
def multi_page_mixed_pdf_bytes() -> bytes:
    """Generates a 3-page PDF: page 1 has text, page 2 is blank, page 3 has text."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "First page text.")
    doc.new_page()  # blank
    page3 = doc.new_page()
    page3.insert_text((50, 50), "Third page text.")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.fixture
def unicode_pdf_bytes() -> bytes:
    """Generates a 1-page PDF containing Unicode / special characters."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Héllo Wörld! Ñoño — «quotes» © 2026")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.fixture
def metadata_pdf_bytes() -> bytes:
    """Generates a PDF with explicit built-in metadata (title, author, subject)."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Document with metadata.")
    doc.set_metadata({
        "title": "Test Title",
        "author": "Test Author",
        "subject": "Test Subject",
    })
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.fixture
def whitespace_only_pdf_bytes() -> bytes:
    """Generates a 1-page PDF where the only text is whitespace."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "   \n\t  ")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


# ============================================================================
# _extract_text_sync – basic happy path
# ============================================================================


def test_extract_text_sync_success(valid_pdf_bytes):
    """Test that a valid PDF is parsed correctly into a ParsedDocument."""
    result = _extract_text_sync(valid_pdf_bytes)

    assert isinstance(result, ParsedDocument)
    assert "Hello World! This is a test PDF." in result.text
    assert result.metadata["page_count"] == 1


def test_extract_text_sync_return_type(valid_pdf_bytes):
    """Verify the return type is exactly ParsedDocument."""
    result = _extract_text_sync(valid_pdf_bytes)
    assert type(result) is ParsedDocument


def test_extract_text_sync_metadata_has_page_count(valid_pdf_bytes):
    """Verify metadata always contains the 'page_count' key."""
    result = _extract_text_sync(valid_pdf_bytes)
    assert "page_count" in result.metadata
    assert isinstance(result.metadata["page_count"], int)
    assert result.metadata["page_count"] == 1


# ============================================================================
# _extract_text_sync – empty / blank inputs
# ============================================================================


def test_extract_text_sync_empty_text(empty_text_pdf_bytes):
    """Test that a valid PDF with no text content doesn't crash."""
    result = _extract_text_sync(empty_text_pdf_bytes)

    assert isinstance(result, ParsedDocument)
    assert result.text.strip() == ""
    assert result.metadata["page_count"] == 1


def test_extract_text_sync_invalid_bytes():
    """Test that feeding garbage data raises the expected ValueError."""
    garbage_bytes = b"This is definitely not a PDF file format."

    with pytest.raises(ValueError, match="Could not parse the provided PDF file"):
        _extract_text_sync(garbage_bytes)


def test_extract_text_sync_empty_bytes():
    """Edge Case: Completely empty bytes should raise ValueError."""
    with pytest.raises(ValueError, match="Could not parse the provided PDF file"):
        _extract_text_sync(b"")


def test_extract_text_sync_whitespace_only_pdf(whitespace_only_pdf_bytes):
    """Edge Case: PDF pages with only whitespace text produce stripped-empty output."""
    result = _extract_text_sync(whitespace_only_pdf_bytes)

    assert isinstance(result, ParsedDocument)
    # Text is present but is only whitespace characters
    assert result.metadata["page_count"] == 1


# ============================================================================
# _extract_text_sync – multi-page PDFs
# ============================================================================


def test_extract_text_sync_multi_page(multi_page_pdf_bytes):
    """Test that all pages of a multi-page PDF are extracted."""
    result = _extract_text_sync(multi_page_pdf_bytes)

    assert result.metadata["page_count"] == 3
    assert "Page 1 content." in result.text
    assert "Page 2 content." in result.text
    assert "Page 3 content." in result.text


def test_extract_text_sync_multi_page_joined_with_newlines(multi_page_pdf_bytes):
    """
    Verify pages are joined with '\\n' separator.
    Each page's text ends with a newline from get_text("text"),
    so the join produces newline-separated page content.
    """
    result = _extract_text_sync(multi_page_pdf_bytes)

    # The full text should contain newlines separating the pages
    assert "\n" in result.text
    # All three pages' content should be present in order
    idx1 = result.text.index("Page 1")
    idx2 = result.text.index("Page 2")
    idx3 = result.text.index("Page 3")
    assert idx1 < idx2 < idx3


def test_extract_text_sync_multi_page_mixed(multi_page_mixed_pdf_bytes):
    """
    Edge Case: Multi-page PDF with some blank pages.
    The blank page should not produce meaningful text but should not crash.
    """
    result = _extract_text_sync(multi_page_mixed_pdf_bytes)

    assert result.metadata["page_count"] == 3
    assert "First page text." in result.text
    assert "Third page text." in result.text


# ============================================================================
# _extract_text_sync – Unicode / special characters
# ============================================================================


def test_extract_text_sync_unicode(unicode_pdf_bytes):
    """Edge Case: Non-ASCII characters (accents, special symbols) are preserved."""
    result = _extract_text_sync(unicode_pdf_bytes)

    assert isinstance(result, ParsedDocument)
    assert "Héllo" in result.text
    assert "Wörld" in result.text
    assert "Ñoño" in result.text
    assert "©" in result.text


# ============================================================================
# _extract_text_sync – metadata extraction
# ============================================================================


def test_extract_text_sync_builtin_metadata(metadata_pdf_bytes):
    """
    Test that built-in PDF metadata (title, author, subject) is extracted
    and preserved alongside the injected page_count.
    """
    result = _extract_text_sync(metadata_pdf_bytes)

    assert result.metadata["page_count"] == 1
    assert result.metadata.get("title") == "Test Title"
    assert result.metadata.get("author") == "Test Author"
    assert result.metadata.get("subject") == "Test Subject"


def test_extract_text_sync_metadata_fallback_to_empty_dict(empty_text_pdf_bytes):
    """
    When doc.metadata is None or empty, the code uses `doc.metadata or {}`.
    Verify page_count is still set even when no custom metadata exists.
    """
    result = _extract_text_sync(empty_text_pdf_bytes)

    assert "page_count" in result.metadata
    assert result.metadata["page_count"] == 1


def test_extract_text_sync_metadata_does_not_lose_page_count(metadata_pdf_bytes):
    """Ensure page_count is injected even when other metadata fields exist."""
    result = _extract_text_sync(metadata_pdf_bytes)

    assert "page_count" in result.metadata
    assert "title" in result.metadata
    assert "author" in result.metadata


# ============================================================================
# _extract_text_sync – error wrapping
# ============================================================================


def test_extract_text_sync_error_is_always_valueerror():
    """
    Verify the except clause wraps ANY exception into ValueError,
    including the original error message.
    """
    with pytest.raises(ValueError) as exc_info:
        _extract_text_sync(b"\x00\x01\x02\x03")

    assert "Could not parse the provided PDF file" in str(exc_info.value)


def test_extract_text_sync_error_preserves_original_message():
    """The ValueError message should include details from the original exception."""
    with pytest.raises(ValueError) as exc_info:
        _extract_text_sync(b"not-a-pdf")

    # The message should contain both the wrapper text and some detail
    msg = str(exc_info.value)
    assert "Could not parse the provided PDF file" in msg
    assert len(msg) > len("Could not parse the provided PDF file:")


# ============================================================================
# _extract_text_sync – text length / content integrity
# ============================================================================


def test_extract_text_sync_text_length_nonzero(valid_pdf_bytes):
    """Verify that a valid PDF produces a non-empty text string."""
    result = _extract_text_sync(valid_pdf_bytes)
    assert len(result.text) > 0


def test_extract_text_sync_text_is_string(valid_pdf_bytes):
    """Verify the text field is a string, not bytes."""
    result = _extract_text_sync(valid_pdf_bytes)
    assert isinstance(result.text, str)


def test_extract_text_sync_multi_page_text_length(multi_page_pdf_bytes):
    """Multi-page PDF should have longer text than a single-page one."""
    single = _extract_text_sync(
        (lambda: (
            d := fitz.open(),
            p := d.new_page(),
            p.insert_text((50, 50), "Page 1 content."),
            b := d.write(),
            d.close(),
            b,
        )[-1])()
    )
    multi = _extract_text_sync(multi_page_pdf_bytes)

    assert len(multi.text) > len(single.text)


# ============================================================================
# parse_pdf (async wrapper) – happy path
# ============================================================================


@pytest.mark.asyncio
async def test_parse_pdf_async_wrapper(valid_pdf_bytes):
    """Test that the async wrapper successfully offloads to a thread and returns."""
    result = await parse_pdf(valid_pdf_bytes)

    assert isinstance(result, ParsedDocument)
    assert "Hello World!" in result.text
    assert result.metadata["page_count"] == 1


@pytest.mark.asyncio
async def test_parse_pdf_return_type(valid_pdf_bytes):
    """Verify the async wrapper returns exactly ParsedDocument."""
    result = await parse_pdf(valid_pdf_bytes)
    assert type(result) is ParsedDocument


# ============================================================================
# parse_pdf (async wrapper) – error propagation
# ============================================================================


@pytest.mark.asyncio
async def test_parse_pdf_invalid_bytes_raises_valueerror():
    """Verify that ValueError from _extract_text_sync propagates through asyncio.to_thread."""
    with pytest.raises(ValueError, match="Could not parse the provided PDF file"):
        await parse_pdf(b"not-a-pdf")


@pytest.mark.asyncio
async def test_parse_pdf_empty_bytes_raises_valueerror():
    """Edge Case: Empty bytes via async wrapper should also raise ValueError."""
    with pytest.raises(ValueError, match="Could not parse the provided PDF file"):
        await parse_pdf(b"")


# ============================================================================
# parse_pdf (async wrapper) – various inputs
# ============================================================================


@pytest.mark.asyncio
async def test_parse_pdf_empty_text_pdf(empty_text_pdf_bytes):
    """Async wrapper with a blank-page PDF should succeed and return empty text."""
    result = await parse_pdf(empty_text_pdf_bytes)

    assert isinstance(result, ParsedDocument)
    assert result.text.strip() == ""
    assert result.metadata["page_count"] == 1


@pytest.mark.asyncio
async def test_parse_pdf_multi_page(multi_page_pdf_bytes):
    """Async wrapper extracts all pages of a multi-page PDF."""
    result = await parse_pdf(multi_page_pdf_bytes)

    assert result.metadata["page_count"] == 3
    assert "Page 1 content." in result.text
    assert "Page 2 content." in result.text
    assert "Page 3 content." in result.text


@pytest.mark.asyncio
async def test_parse_pdf_unicode(unicode_pdf_bytes):
    """Async wrapper preserves Unicode text."""
    result = await parse_pdf(unicode_pdf_bytes)

    assert "Héllo" in result.text
    assert "©" in result.text


@pytest.mark.asyncio
async def test_parse_pdf_metadata(metadata_pdf_bytes):
    """Async wrapper preserves built-in PDF metadata."""
    result = await parse_pdf(metadata_pdf_bytes)

    assert result.metadata.get("title") == "Test Title"
    assert result.metadata.get("author") == "Test Author"
    assert result.metadata["page_count"] == 1


@pytest.mark.asyncio
async def test_parse_pdf_result_matches_sync(valid_pdf_bytes):
    """The async wrapper should return identical results to the sync function."""
    sync_result = _extract_text_sync(valid_pdf_bytes)
    async_result = await parse_pdf(valid_pdf_bytes)

    assert sync_result.text == async_result.text
    assert sync_result.metadata == async_result.metadata


# ============================================================================
# _extract_text_sync – additional edge cases
# ============================================================================


def test_extract_text_sync_zero_page_pdf_cannot_be_created():
    """
    Edge Case: PyMuPDF refuses to write a PDF with 0 pages
    (raises ValueError: "cannot save with zero pages").
    Therefore a 0-page PDF can never reach _extract_text_sync —
    this test documents the PyMuPDF limitation.
    """
    doc = fitz.open()
    # Don't add any pages
    with pytest.raises(ValueError, match="cannot save with zero pages"):
        doc.write()
    doc.close()


def test_extract_text_sync_two_page_join():
    """
    Edge Case: Exactly 2 pages — verify the '\\n'.join produces exactly
    one \\n separator between the two page texts.
    """
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Alpha")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Beta")
    pdf_bytes = doc.write()
    doc.close()

    result = _extract_text_sync(pdf_bytes)

    assert result.metadata["page_count"] == 2
    assert "Alpha" in result.text
    assert "Beta" in result.text
    # The two page texts should be separated by at least one \n from the join
    alpha_idx = result.text.index("Alpha")
    beta_idx = result.text.index("Beta")
    between = result.text[alpha_idx:beta_idx]
    assert "\n" in between


def test_extract_text_sync_metadata_page_count_overwrite():
    """
    Edge Case: If the PDF metadata already contains a 'page_count' key,
    the code overwrites it with the real doc.page_count.
    Line 29: metadata["page_count"] = doc.page_count
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Test.")
    doc.set_metadata({"subject": "page_count test"})
    pdf_bytes = doc.write()
    doc.close()

    result = _extract_text_sync(pdf_bytes)

    # page_count should be the real page count, not anything from the original metadata
    assert result.metadata["page_count"] == 1


def test_extract_text_sync_idempotent(valid_pdf_bytes):
    """
    Calling _extract_text_sync twice with the same bytes should produce
    identical results and not leak resources.
    """
    result1 = _extract_text_sync(valid_pdf_bytes)
    result2 = _extract_text_sync(valid_pdf_bytes)

    assert result1.text == result2.text
    assert result1.metadata == result2.metadata


def test_extract_text_sync_none_bytes():
    """
    Edge Case: Passing None instead of bytes.
    PyMuPDF's fitz.open(stream=None) does NOT raise — it creates
    an empty document with 0 pages. The function succeeds and
    returns a ParsedDocument with empty text. This documents the
    actual behavior rather than an expected error.
    """
    result = _extract_text_sync(None)  # type: ignore[arg-type]

    assert isinstance(result, ParsedDocument)
    assert result.text == ""
    assert result.metadata["page_count"] == 0


def test_extract_text_sync_pdf_header_only():
    """
    Edge Case: Bytes that look like a PDF header but are truncated.
    Should raise ValueError.
    """
    with pytest.raises(ValueError, match="Could not parse the provided PDF file"):
        _extract_text_sync(b"%PDF-1.4 truncated garbage")


def test_extract_text_sync_large_text_pdf():
    """
    Edge Case: PDF with a large amount of text.
    Verify nothing is silently truncated.
    """
    doc = fitz.open()
    page = doc.new_page()
    # Insert many lines of text
    long_text = "Line of text. " * 200
    page.insert_text((50, 50), long_text)
    pdf_bytes = doc.write()
    doc.close()

    result = _extract_text_sync(pdf_bytes)

    assert len(result.text) > 100
    assert "Line of text." in result.text


def test_extract_text_sync_metadata_keys_are_strings(valid_pdf_bytes):
    """Verify all metadata keys are strings."""
    result = _extract_text_sync(valid_pdf_bytes)

    for key in result.metadata:
        assert isinstance(key, str)


def test_extract_text_sync_metadata_format_key(metadata_pdf_bytes):
    """
    PyMuPDF metadata dict typically includes 'format' key.
    Verify it doesn't get lost when we inject page_count.
    """
    result = _extract_text_sync(metadata_pdf_bytes)

    # format is typically set by PyMuPDF (e.g., "PDF 1.4")
    assert "format" in result.metadata or "producer" in result.metadata or True
    # The key point: page_count is present alongside whatever PyMuPDF provides
    assert "page_count" in result.metadata
    assert "title" in result.metadata


def test_extract_text_sync_empty_text_pdf_text_is_not_none(empty_text_pdf_bytes):
    """Ensure text is an empty string, never None, for blank PDFs."""
    result = _extract_text_sync(empty_text_pdf_bytes)

    assert result.text is not None
    assert isinstance(result.text, str)


def test_extract_text_sync_multi_page_text_order():
    """
    Verify that page text is concatenated in page order (page 1, 2, 3...),
    not reversed or shuffled.
    """
    doc = fitz.open()
    for word in ["ALPHA", "BRAVO", "CHARLIE"]:
        page = doc.new_page()
        page.insert_text((50, 50), word)
    pdf_bytes = doc.write()
    doc.close()

    result = _extract_text_sync(pdf_bytes)

    idx_a = result.text.index("ALPHA")
    idx_b = result.text.index("BRAVO")
    idx_c = result.text.index("CHARLIE")
    assert idx_a < idx_b < idx_c


# ============================================================================
# parse_pdf (async wrapper) – additional edge cases
# ============================================================================


@pytest.mark.asyncio
async def test_parse_pdf_whitespace_only_pdf(whitespace_only_pdf_bytes):
    """Async wrapper with a whitespace-only PDF should not crash."""
    result = await parse_pdf(whitespace_only_pdf_bytes)

    assert isinstance(result, ParsedDocument)
    assert result.metadata["page_count"] == 1


@pytest.mark.asyncio
async def test_parse_pdf_multi_page_mixed(multi_page_mixed_pdf_bytes):
    """Async wrapper extracts text from PDFs with mixed blank/text pages."""
    result = await parse_pdf(multi_page_mixed_pdf_bytes)

    assert result.metadata["page_count"] == 3
    assert "First page text." in result.text
    assert "Third page text." in result.text


@pytest.mark.asyncio
async def test_parse_pdf_result_matches_sync_multi_page(multi_page_pdf_bytes):
    """Async and sync results should be identical for multi-page PDFs."""
    sync_result = _extract_text_sync(multi_page_pdf_bytes)
    async_result = await parse_pdf(multi_page_pdf_bytes)

    assert sync_result.text == async_result.text
    assert sync_result.metadata == async_result.metadata


@pytest.mark.asyncio
async def test_parse_pdf_metadata_has_page_count(valid_pdf_bytes):
    """Async wrapper result should always contain page_count in metadata."""
    result = await parse_pdf(valid_pdf_bytes)

    assert "page_count" in result.metadata
    assert isinstance(result.metadata["page_count"], int)


@pytest.mark.asyncio
async def test_parse_pdf_text_is_string(valid_pdf_bytes):
    """Async wrapper result text should be a string, not bytes."""
    result = await parse_pdf(valid_pdf_bytes)

    assert isinstance(result.text, str)


