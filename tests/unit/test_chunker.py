import pytest

from fastapi_ollama_rag.models.chunk import DocumentChunk
from fastapi_ollama_rag.models.document import ParsedDocument
from fastapi_ollama_rag.services.chunker import _append_chunk, chunk_document

# ============================================================================
# chunk_document – empty / trivial inputs
# ============================================================================


def test_chunk_empty_document():
    """Edge Case: Document has no text."""
    doc = ParsedDocument(text="", metadata={"source": "empty.pdf"})
    chunks = chunk_document(doc)

    assert len(chunks) == 0
    assert isinstance(chunks, list)


def test_chunk_whitespace_only_document():
    """Edge Case: Document text is only whitespace – should produce 0 chunks."""
    doc = ParsedDocument(text="   \n\t\n   ", metadata={"source": "blank.pdf"})
    chunks = chunk_document(doc, chunk_size=5, chunk_overlap=1)

    assert len(chunks) == 0


def test_chunk_single_character():
    """Edge Case: Document contains exactly one non-space character."""
    doc = ParsedDocument(text="X", metadata={})
    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)

    assert len(chunks) == 1
    assert chunks[0].text == "X"
    assert chunks[0].chunk_index == 0


# ============================================================================
# chunk_document – size / boundary basics
# ============================================================================


def test_chunk_single_small_chunk():
    """Edge Case: Text is smaller than the chunk size."""
    doc = ParsedDocument(text="This is a short text.", metadata={"page": 1})
    chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 1
    assert chunks[0].text == "This is a short text."
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata == {"page": 1}


def test_chunk_text_exactly_equal_to_chunk_size():
    """Edge Case: len(text) == chunk_size, so end >= text_len on the first pass."""
    text = "A" * 20
    doc = ParsedDocument(text=text, metadata={})
    chunks = chunk_document(doc, chunk_size=20, chunk_overlap=5)

    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_text_one_byte_over_chunk_size():
    """Edge Case: len(text) == chunk_size + 1 – forces exactly two chunks."""
    text = "A" * 21
    doc = ParsedDocument(text=text, metadata={})
    chunks = chunk_document(doc, chunk_size=20, chunk_overlap=5)

    assert len(chunks) == 2
    # With no natural boundaries, first chunk hard-splits at 20
    assert chunks[0].text == "A" * 20
    # Remaining after overlap: start = 20 - 5 = 15, text[15:21]
    assert chunks[1].text == "A" * 6


def test_chunk_no_natural_boundaries():
    """
    Edge Case: A massive string with NO spaces or punctuation.
    The algorithm should fallback to a hard split at exactly `chunk_size`.
    """
    doc = ParsedDocument(text="A" * 50, metadata={})

    # Size 20, overlap 5.
    # C1: index 0 to 20 -> start becomes 20 - 5 = 15
    # C2: index 15 to 35 -> start becomes 35 - 5 = 30
    # C3: index 30 to 50
    chunks = chunk_document(doc, chunk_size=20, chunk_overlap=5)

    assert len(chunks) == 3
    assert chunks[0].text == "A" * 20
    assert chunks[1].text == "A" * 20
    assert chunks[2].text == "A" * 20


def test_chunk_with_natural_boundaries():
    """
    Core Logic: Ensure it snaps back to a space/punctuation so it
    doesn't cut words in half.
    """
    # Total length is 36 characters.
    # "This is a sentence. This is another."
    #  012345678901234567890123456789012345
    text = "This is a sentence. This is another."
    doc = ParsedDocument(text=text, metadata={})

    # Chunk_size 25.
    # Index 25 is the 'i' in the second "is".
    # It should look backwards and split at the space before it (Index 24).
    chunks = chunk_document(doc, chunk_size=25, chunk_overlap=5)

    assert len(chunks) == 2
    assert chunks[0].text == "This is a sentence. This"
    assert chunks[1].text == "This is another."


# ============================================================================
# chunk_document – boundary character types (\n . ? !)
# ============================================================================


def test_chunk_snaps_at_newline_boundary():
    """Edge Case: Algorithm should snap at '\\n' as a natural boundary."""
    text = "Line one content\nLine two content here."
    doc = ParsedDocument(text=text, metadata={})

    # chunk_size=18, overlap=0
    # Iter 1: end=18, text[18]='i', scan back → '\n' at 16. Chunk text[0:16]="Line one content".
    #         start=16, fast-forward '\n' → start=17.
    # Iter 2: end=35, text[35]='e', scan back → ' ' at 33. Chunk text[17:33]="Line two content".
    #         start=33, fast-forward ' ' → start=34.
    # Iter 3: end=52 >= 39 → take rest text[34:39]="here.".
    chunks = chunk_document(doc, chunk_size=18, chunk_overlap=0)

    assert len(chunks) == 3
    assert chunks[0].text == "Line one content"
    assert chunks[1].text == "Line two content"
    assert chunks[2].text == "here."


def test_chunk_snaps_at_question_mark():
    """Edge Case: Algorithm should snap at '?' as a natural boundary."""
    text = "Is this working? Yes it is."
    doc = ParsedDocument(text=text, metadata={})

    # chunk_size 17 → index 17 is space after '?'; snap back to '?' at 15 or space at 16
    chunks = chunk_document(doc, chunk_size=17, chunk_overlap=0)

    assert len(chunks) == 2
    # Should split at a boundary character
    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert c.text.strip() != ""


def test_chunk_snaps_at_exclamation_mark():
    """Edge Case: Algorithm should snap at '!' as a natural boundary."""
    text = "Hello world! Goodbye world."
    doc = ParsedDocument(text=text, metadata={})

    # "Hello world! Goodbye world." (27 chars)
    # chunk_size=13, overlap=0
    # Iter 1: end=13, text[13]='G'. Scan back → ' ' at 12, boundary. Chunk text[0:12]="Hello world!".
    #         start=12, fast-forward ' ' → start=13.
    # Iter 2: end=26, text[26]='.'. '.' is boundary → break_point=26.
    #         Chunk text[13:26]="Goodbye world". start=26, fast-forward: text[26]='.' not space → start=26.
    # Iter 3: end=39 >= 27 → take rest text[26:27]=".".
    chunks = chunk_document(doc, chunk_size=13, chunk_overlap=0)

    assert len(chunks) == 3
    assert chunks[0].text == "Hello world!"
    assert chunks[1].text == "Goodbye world"
    assert chunks[2].text == "."


def test_chunk_snaps_at_period():
    """Edge Case: Algorithm should snap at '.' as a natural boundary."""
    text = "First sentence. Second sentence."
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=16, chunk_overlap=0)

    assert len(chunks) == 2
    assert chunks[0].text == "First sentence."
    assert chunks[1].text == "Second sentence."


def test_chunk_text_made_entirely_of_boundary_characters():
    """Edge Case: Text is all boundary characters like spaces and periods."""
    text = ". . . . . . . . . ."
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=5, chunk_overlap=1)

    # Every character is a boundary; after stripping, chunks may be '.' only
    # The important thing is no infinite loop and no crash.
    assert isinstance(chunks, list)
    for c in chunks:
        assert c.text.strip() != ""


# ============================================================================
# chunk_document – whitespace fast-forward
# ============================================================================


def test_chunk_fast_forward_spaces():
    """
    Edge Case: Ensure the `while start < text_len and text[start].isspace():`
    logic successfully skips large blocks of whitespace after overlap calculation.
    """
    # Lots of spaces in the middle
    text = "Word." + (" " * 20) + "Next."
    doc = ParsedDocument(text=text, metadata={})

    # Make chunk size small enough that it gets trapped in the spaces
    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)

    # We should not get chunks that are purely empty strings due to the fast-forward
    for chunk in chunks:
        assert chunk.text.strip() != ""
    assert "Next." in [c.text for c in chunks]


def test_chunk_leading_whitespace_stripped():
    """Edge Case: Text starts with whitespace – first chunk should be stripped."""
    text = "   Hello world. Goodbye."
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=0)

    assert len(chunks) == 1
    assert chunks[0].text == "Hello world. Goodbye."


def test_chunk_trailing_whitespace_stripped():
    """Edge Case: Text ends with whitespace – last chunk should be stripped."""
    text = "Hello world. Goodbye.   \n\n"
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=0)

    assert len(chunks) == 1
    assert chunks[0].text == "Hello world. Goodbye."


# ============================================================================
# chunk_document – overlap edge cases
# ============================================================================


def test_chunk_overlap_zero():
    """Edge Case: No overlap – chunks should be strictly non-overlapping."""
    text = "AAAAA BBBBB CCCCC"
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=6, chunk_overlap=0)

    # With overlap=0, no text should repeat across chunks.
    full_reconstructed = " ".join(c.text for c in chunks)
    # Each original word should appear exactly once in the output
    for word in ["AAAAA", "BBBBB", "CCCCC"]:
        assert sum(1 for c in chunks if word in c.text) == 1


def test_chunk_overlap_larger_than_chunk_size():
    """
    Edge Case: overlap > chunk_size is a misconfiguration.
    start = break_point - overlap can go deeply negative, causing
    text[start] to index from the end of the string. If the resulting
    negative index falls outside [-text_len, -1], an IndexError is raised.
    This test documents the current (buggy) behavior.
    """
    text = "Hello world. Goodbye world."
    doc = ParsedDocument(text=text, metadata={})

    with pytest.raises(IndexError):
        chunk_document(doc, chunk_size=10, chunk_overlap=50)


def test_chunk_size_one():
    """Edge Case: chunk_size=1 – each character becomes its own chunk (excluding spaces)."""
    text = "AB CD"
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=1, chunk_overlap=0)

    # Characters: 'A','B',' ','C','D'. Spaces are boundaries; after strip, non-empty chars remain.
    chunk_texts = [c.text for c in chunks]
    # Each non-space character should appear
    for ch in ["A", "B", "C", "D"]:
        assert ch in chunk_texts


def test_chunk_indices_are_sequential():
    """Verify that chunk_index values are 0, 1, 2, … across all chunks."""
    text = "A" * 100
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=20, chunk_overlap=5)

    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_metadata_propagated_to_all_chunks():
    """Verify every chunk receives the same metadata from the parent document."""
    metadata = {"source": "report.pdf", "author": "Alice", "pages": 42}
    text = "Word. " * 50
    doc = ParsedDocument(text=text, metadata=metadata)

    chunks = chunk_document(doc, chunk_size=30, chunk_overlap=5)

    assert len(chunks) > 1  # must produce multiple chunks to be meaningful
    for chunk in chunks:
        assert chunk.metadata == metadata


def test_all_chunks_are_document_chunk_instances():
    """Verify the return type of every element is DocumentChunk."""
    text = "Hello world. Foo bar. Baz qux."
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=15, chunk_overlap=3)

    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)


def test_chunk_full_text_is_recoverable():
    """
    All original non-whitespace content should appear in at least one chunk.
    This guards against silent data loss from the sliding window.
    """
    text = "The quick brown fox jumps over the lazy dog."
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=15, chunk_overlap=5)

    concatenated = " ".join(c.text for c in chunks)
    for word in text.split():
        # Strip punctuation for matching
        clean_word = word.strip(".")
        assert clean_word in concatenated, f"'{clean_word}' missing from chunks"


def test_chunk_unicode_text():
    """Edge Case: Non-ASCII / Unicode text should be chunked correctly."""
    text = "日本語のテスト文章です。これはテストです。"
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)

    assert len(chunks) >= 1
    concatenated = "".join(c.text for c in chunks)
    # Every original character (except potential whitespace) should appear
    for ch in text.replace(" ", ""):
        assert ch in concatenated


def test_chunk_with_mixed_whitespace():
    """Edge Case: Tabs, newlines, and spaces mixed into the text."""
    text = "Hello\t\tworld\n\nGoodbye\t\nworld"
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=15, chunk_overlap=3)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.text.strip() != ""


def test_chunk_uses_default_parameters():
    """Verify chunk_document works with default chunk_size=1000 and chunk_overlap=200."""
    text = "A " * 600  # 1200 characters
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc)

    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 1000


def test_chunk_whitespace_only_longer_than_chunk_size():
    """
    Edge Case: All-whitespace text longer than chunk_size.
    The while-loop iterates multiple times, but every _append_chunk
    receives only whitespace that strips to empty → 0 chunks.
    """
    text = " " * 50
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=3)

    assert len(chunks) == 0


def test_chunk_only_newlines():
    """Edge Case: Text made entirely of newlines – should produce 0 chunks."""
    text = "\n" * 30
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)

    assert len(chunks) == 0


def test_chunk_exact_multiple_of_effective_step():
    """
    Edge Case: Text length is an exact multiple of (chunk_size - chunk_overlap).
    With no natural boundaries, the window slides perfectly without leftover.
    """
    # chunk_size=10, overlap=5 → step = 5
    # 30 / 5 = 6 iterations, but last chunk might overlap differently.
    text = "X" * 30
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=5)

    # All text should be covered, no crash, all chunks non-empty
    assert len(chunks) >= 1
    for c in chunks:
        assert c.text.strip() != ""
    # Verify full coverage
    total_chars = sum(len(c.text) for c in chunks)
    assert total_chars >= 30  # overlap means total chars >= original length


def test_chunk_overlap_content_shared_between_consecutive_chunks():
    """
    Verify that overlap actually causes shared content between consecutive chunks.
    With no natural boundaries, chunks should overlap by exactly `chunk_overlap` chars.
    """
    text = "ABCDEFGHIJKLMNOPQRST"  # 20 chars
    doc = ParsedDocument(text=text, metadata={})

    # chunk_size=10, overlap=4
    # C1: [0:10] = "ABCDEFGHIJ", start = 10 - 4 = 6
    # C2: [6:16] = "GHIJKLMNOP", start = 16 - 4 = 12
    # C3: [12:20] = "MNOPQRST" (end >= text_len, takes rest)
    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=4)

    assert len(chunks) == 3
    assert chunks[0].text == "ABCDEFGHIJ"
    assert chunks[1].text == "GHIJKLMNOP"
    assert chunks[2].text == "MNOPQRST"

    # Verify overlap: last 4 chars of chunk[0] == first 4 chars of chunk[1]
    assert chunks[0].text[-4:] == chunks[1].text[:4]
    # And last 4 of chunk[1] == first 4 of chunk[2]
    assert chunks[1].text[-4:] == chunks[2].text[:4]


def test_chunk_index_increments_even_when_append_rejects():
    """
    chunk_index increments on every loop iteration regardless of whether
    _append_chunk actually adds a chunk (e.g., when the slice is all whitespace).
    This means chunk indices might have gaps. Verify the behavior.
    """
    # "AB" + 20 spaces + "CD"
    # With chunk_size=10, overlap=0:
    # First iteration: "AB" + 8 spaces → stripped to "AB" → index 0
    # After fast-forward past whitespace, start jumps to "CD"
    # Next iteration: "CD" → index depends on how many iterations ran
    text = "AB" + (" " * 20) + "CD"
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=0)

    # Both "AB" and "CD" should be present
    chunk_texts = [c.text for c in chunks]
    assert "AB" in chunk_texts
    assert "CD" in chunk_texts


def test_chunk_boundary_at_position_start():
    """
    Edge Case: The character at index `start` is itself a boundary.
    The backward scan `while break_point > start` should stop before going
    past start and the code should handle this correctly.
    """
    # Put a period right where start would be after first chunk
    # "AAAAAAAAAA.BBBBBBBBB" (10 A's, 1 period, 9 B's = 20 chars)
    text = "A" * 10 + "." + "B" * 9
    doc = ParsedDocument(text=text, metadata={})

    # chunk_size=11, overlap=0
    # First pass: end = 11, text[11] = 'B', scan back → '.' at index 10 (boundary)
    # break_point = 10, chunk = text[0:10] = "AAAAAAAAAA"
    chunks = chunk_document(doc, chunk_size=11, chunk_overlap=0)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.text.strip() != ""


def test_chunk_negative_start_from_large_overlap():
    """
    Edge Case: When break_point is small and overlap is huge,
    start = break_point - overlap becomes a large negative number.
    text[start] may then index from the end of the string, and if the
    magnitude exceeds text length, an IndexError is raised.
    This test documents the current (buggy) behavior.
    """
    text = " AB"
    doc = ParsedDocument(text=text, metadata={})

    with pytest.raises(IndexError):
        chunk_document(doc, chunk_size=2, chunk_overlap=100)


def test_chunk_consecutive_boundary_characters():
    """
    Edge Case: Multiple consecutive boundary characters ("..." or "?!").
    The backward scan should stop at the first boundary from the end.
    """
    text = "Hello...World"
    doc = ParsedDocument(text=text, metadata={})

    # chunk_size=9 → end=9, text[9]='o', scan back: text[8]='W', text[7]='.', boundary!
    chunks = chunk_document(doc, chunk_size=9, chunk_overlap=0)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.text.strip() != ""


def test_chunk_emoji_multibyte_text():
    """Edge Case: Emoji / multi-byte characters should be handled at char level."""
    text = "Hello 🌍🌎🌏 World! More text here for chunking."
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=15, chunk_overlap=3)

    assert len(chunks) >= 1
    combined = " ".join(c.text for c in chunks)
    assert "🌍" in combined
    assert "🌎" in combined
    assert "🌏" in combined


def test_chunk_all_boundaries_except_last_char():
    """
    Edge Case: Every character is a boundary except the very last one.
    The backward scan from end should stop immediately at the last boundary.
    """
    text = " " * 19 + "X"
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=0)

    # After fast-forward past leading spaces, "X" at end should form a chunk
    assert len(chunks) >= 1
    chunk_texts = [c.text for c in chunks]
    assert "X" in chunk_texts


def test_chunk_metadata_is_copied_by_pydantic():
    """
    Pydantic's BaseModel copies dict fields during model construction.
    This means mutating one chunk's metadata does NOT affect the others.
    Verify that all chunks have equal metadata values but are independent copies.
    """
    metadata = {"source": "test.pdf"}
    text = "A" * 50
    doc = ParsedDocument(text=text, metadata=metadata)

    chunks = chunk_document(doc, chunk_size=20, chunk_overlap=5)

    assert len(chunks) >= 2
    # All chunks should have equal metadata values
    for chunk in chunks:
        assert chunk.metadata == metadata

    # Pydantic copies the dict, so mutation of one chunk's metadata should NOT
    # affect the others (they are independent copies).
    chunks[0].metadata["extra"] = "modified"
    assert "extra" not in chunks[1].metadata


def test_chunk_very_large_overlap_short_text():
    """
    Edge Case: overlap is much larger than the text itself.
    start = break_point - overlap goes deeply negative.
    Should terminate without error.
    """
    text = "Hi."
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=5, chunk_overlap=1000)

    assert len(chunks) == 1
    assert chunks[0].text == "Hi."


def test_chunk_text_with_only_one_boundary_at_end():
    """
    Edge Case: text = "ABCDEFGHIJ." (11 chars)
    chunk_size=10, end=10 → text[10]='.' (boundary) → break_point stays at 10
    → chunk is text[0:10] = "ABCDEFGHIJ", then remainder "." → stripped to "."
    """
    text = "ABCDEFGHIJ."
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=0)

    assert len(chunks) >= 1
    combined = "".join(c.text for c in chunks)
    assert "ABCDEFGHIJ" in combined


def test_chunk_overlap_creates_duplicate_first_chunk_index():
    """
    Verify that chunk_index starts at 0 and the first chunk always has index 0.
    """
    text = "Hello world. Goodbye world. More text here."
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=15, chunk_overlap=5)

    assert len(chunks) >= 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_chunk_single_boundary_character():
    """Edge Case: Text is just a single boundary character like '.'."""
    doc = ParsedDocument(text=".", metadata={})

    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)

    assert len(chunks) == 1
    assert chunks[0].text == "."


def test_chunk_text_with_no_spaces_but_punctuation():
    """
    Edge Case: No spaces, but punctuation boundaries exist.
    'aaa.bbb.ccc' should split at periods.
    """
    text = "aaa.bbb.ccc"
    doc = ParsedDocument(text=text, metadata={})

    # chunk_size=5 → end=5, text[5]='b', scan back → '.' at 3 (boundary)
    chunks = chunk_document(doc, chunk_size=5, chunk_overlap=0)

    assert len(chunks) >= 2
    combined = "".join(c.text for c in chunks)
    # All original non-boundary chars should be present
    for part in ["aaa", "bbb", "ccc"]:
        assert part in combined


def test_chunk_return_type_is_always_list():
    """Verify that chunk_document always returns a list, even for edge cases."""
    for text_input in ["", " ", "X", "Hello world."]:
        doc = ParsedDocument(text=text_input, metadata={})
        result = chunk_document(doc)
        assert isinstance(result, list)


def test_chunk_boundary_exactly_at_end_position():
    """
    Edge Case: text[end] is itself a boundary character.
    The backward scan should stop immediately (break_point stays at end).
    """
    # "ABCDE FGHIJ" (11 chars). chunk_size=5 → end=5, text[5]=' ' → boundary!
    # break_point stays at 5, chunk = text[0:5] = "ABCDE"
    text = "ABCDE FGHIJ"
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=5, chunk_overlap=0)

    assert len(chunks) == 2
    assert chunks[0].text == "ABCDE"
    assert chunks[1].text == "FGHIJ"


def test_chunk_boundary_period_exactly_at_end_position():
    """
    Edge Case: A period sits exactly at the `end` index.
    """
    # "ABCDE.FGHIJ" (11 chars). chunk_size=5, end=5, text[5]='.' → boundary.
    text = "ABCDE.FGHIJ"
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=5, chunk_overlap=0)

    assert len(chunks) >= 2
    assert chunks[0].text == "ABCDE"


def test_chunk_fast_forward_consumes_all_remaining_text():
    """
    Edge Case: After a mid-split, the remainder is entirely whitespace.
    The fast-forward `while start < text_len and text[start].isspace()`
    pushes `start` to `text_len`, exiting the outer loop with no more chunks.
    """
    # "Hello world     " (16 chars). chunk_size=6, overlap=0.
    # Iter 1: end=6, text[6]='o'. Scan back: text[5]=' ' → boundary.
    #         Chunk text[0:5]="Hello". start=5, fast-forward: text[5]=' ' → start=6.
    # Iter 2: end=12, text[12]=' '. boundary, break_point=12.
    #         Chunk text[6:12]="world". start=12, fast-forward: all spaces → start=16.
    # start=16 >= text_len=16 → exit.
    text = "Hello world     "
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=6, chunk_overlap=0)

    assert len(chunks) == 2
    assert chunks[0].text == "Hello"
    assert chunks[1].text == "world"


def test_chunk_remaining_tail_is_whitespace():
    """
    Edge Case: The final chunk picked up by `end >= text_len` branch
    is all whitespace. _append_chunk should reject it, producing
    fewer chunks than loop iterations.
    """
    text = "Hello.     "  # 11 chars
    doc = ParsedDocument(text=text, metadata={})

    # chunk_size=7, overlap=0. Iter1: end=7, text[7]=' ' → boundary.
    # Chunk text[0:7]="Hello." (stripped to "Hello."). start=7, fast-forward→start=11.
    # start=11 >= text_len=11 → exit.
    chunks = chunk_document(doc, chunk_size=7, chunk_overlap=0)

    # Only "Hello." should appear, the trailing spaces are consumed by fast-forward
    assert len(chunks) == 1
    assert chunks[0].text == "Hello."


def test_chunk_backward_scan_does_not_check_text_at_start():
    """
    The backward scan condition is `while break_point > start`, meaning
    it never checks `text[start]` itself. If text[start] is a boundary
    but nothing between start+1 and end is, break_point reaches start
    and the code force-splits at end.
    """
    # text[0]=' ', text[1..9]='AAAAAAAAA' (no boundary)
    # chunk_size=5, start=0, end=5. Scan back from 5: text[5]='A',4='A',3='A',2='A',1='A'.
    # break_point=0 == start → force split at end=5.
    text = " AAAAAAAAA"  # 10 chars: space then 9 A's
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=5, chunk_overlap=0)

    # The first chunk is text[0:5]=" AAAA" → stripped to "AAAA" (or includes space? strip removes it)
    # Then start=5, fast-forward: text[5]='A' not space → start=5.
    # Iter 2: end=10 >= 10 → rest text[5:10]="AAAAA".
    assert len(chunks) == 2
    assert chunks[0].text == "AAAA"
    assert chunks[1].text == "AAAAA"


def test_chunk_overlap_equal_to_chunk_size_with_boundaries():
    """
    Edge Case: overlap == chunk_size with text that HAS natural boundaries.

    When break_point snaps to a boundary *before* `end`, we get:
        start = break_point - chunk_overlap
    Since break_point < end and chunk_overlap == chunk_size, start goes negative.
    Python's negative indexing makes `text[start]` silently resolve to a
    character near the end of the string, causing either an infinite loop
    or unpredictable behavior — the same class of bug as overlap > chunk_size.

    Trace for "Word. Another. Last." (20 chars), chunk_size=6, overlap=6:
      Iter 1: start=0, end=6. text[6]='A'. Scan back → text[5]=' ' (boundary).
              break_point=5. Chunk text[0:5]="Word.".
              start = 5 - 6 = -1. text[-1]='.' → not space → start stays -1.
              -1 < 20 → loop continues. end = -1+6 = 5 → same window → infinite loop.

    This test documents the known buggy behavior. The overlap >= chunk_size
    configuration is unsupported and results in an infinite loop when the
    text contains boundary characters.
    """
    # We only verify the *shorter* variant that hits an IndexError before
    # looping forever (when the negative index magnitude exceeds text length).
    text = "A B"  # 3 chars, chunk_size=3, overlap=3
    doc = ParsedDocument(text=text, metadata={})

    # start=0, end=3 >= 3 → takes rest on first iteration → single chunk, no issue.
    # That path works. But with longer text and a split:
    text2 = "AB CD"  # 5 chars, chunk_size=3, overlap=3
    doc2 = ParsedDocument(text=text2, metadata={})

    # Iter 1: end=3, text[3]='C'. Scan back: text[2]=' ' → boundary. break_point=2.
    #         Chunk text[0:2]="AB". start = 2-3 = -1. text[-1]='D' not space → start=-1.
    #         -1 < 5 → loop. end = -1+3 = 2. 2 < 5 → backward scan.
    #         text[2]=' ' → boundary. break_point=2. Chunk text[-1:2]="" → stripped → rejected.
    #         start = 2-3 = -1 → same state → infinite loop.
    # We can't run this without hanging, so just document it.
    # For the IndexError variant (large overlap on short text with boundaries):
    text3 = "A B C D E F G H I J K L"  # 23 chars
    doc3 = ParsedDocument(text=text3, metadata={})

    # chunk_size=3, overlap=100. break_point=2 (space), start=2-100=-98.
    # text[-98] → IndexError because |-98| > 23.
    with pytest.raises(IndexError):
        chunk_document(doc3, chunk_size=3, chunk_overlap=100)


def test_chunk_huge_chunk_size():
    """Edge Case: chunk_size much larger than text → single chunk."""
    text = "Short."
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=999999, chunk_overlap=100)

    assert len(chunks) == 1
    assert chunks[0].text == "Short."


def test_chunk_empty_metadata():
    """Edge Case: Empty metadata dict propagated to all chunks."""
    text = "A" * 30
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)

    for c in chunks:
        assert c.metadata == {}


def test_chunk_text_with_tab_as_boundary():
    """
    Edge Case: Tab character is NOT in the boundaries set ({'\\n', '.', '?', '!', ' '}).
    The algorithm should NOT snap at a tab. It's treated as non-boundary.
    However, tab IS whitespace, so fast-forward will skip past it.
    """
    text = "AAAA\tBBBB"  # 9 chars: 4 A's, tab, 4 B's
    doc = ParsedDocument(text=text, metadata={})

    # chunk_size=5, end=5. text[5]='B' not boundary. Scan back:
    # text[4]='\t' → NOT in boundaries set. text[3]='A' → not boundary.
    # Eventually break_point == start → force split at end=5.
    chunks = chunk_document(doc, chunk_size=5, chunk_overlap=0)

    # Force split at 5: text[0:5]="AAAA\t" → stripped to "AAAA".
    # start=5. text[5]='B' not space → no fast-forward.
    # Iter 2: end=10 >= 9 → rest text[5:9]="BBBB".
    assert len(chunks) == 2
    assert chunks[0].text == "AAAA"
    assert chunks[1].text == "BBBB"


def test_chunk_text_with_carriage_return():
    """
    Edge Case: '\\r' is NOT in the boundaries set. But it IS whitespace.
    The backward scan won't snap at it, but the fast-forward will skip it.
    """
    text = "Hello\r\nWorld test."
    doc = ParsedDocument(text=text, metadata={})

    # '\r' is at index 5, '\n' is at index 6.
    # chunk_size=7, end=7. text[7]='o'. Scan back: text[6]='\n' → boundary!
    chunks = chunk_document(doc, chunk_size=7, chunk_overlap=0)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.text.strip() != ""
        assert "\r" not in c.text  # stripped by _append_chunk


def test_chunk_overlap_one():
    """Edge Case: Minimal non-zero overlap of 1 character."""
    text = "ABCDEFGHIJ"  # 10 chars
    doc = ParsedDocument(text=text, metadata={})

    # chunk_size=5, overlap=1. No boundaries → force split.
    # Iter 1: text[0:5]="ABCDE". start=5-1=4.
    # Iter 2: text[4:9]="EFGHI". start=9-1=8.
    # Iter 3: end=13 >= 10 → text[8:10]="IJ".
    chunks = chunk_document(doc, chunk_size=5, chunk_overlap=1)

    assert len(chunks) == 3
    assert chunks[0].text == "ABCDE"
    assert chunks[1].text == "EFGHI"
    assert chunks[2].text == "IJ"
    # Verify overlap of 1 char between consecutive chunks
    assert chunks[0].text[-1] == chunks[1].text[0]
    assert chunks[1].text[-1] == chunks[2].text[0]


def test_chunk_text_two_chars_no_boundary():
    """Edge Case: Minimal multi-char text with no boundaries."""
    text = "AB"
    doc = ParsedDocument(text=text, metadata={})

    chunks = chunk_document(doc, chunk_size=1, chunk_overlap=0)

    # chunk_size=1. Iter 1: end=1, text[1]='B' not boundary. Scan: break_point=0 == start → force end=1.
    # Chunk text[0:1]="A". start=1.
    # Iter 2: end=2 >= 2 → text[1:2]="B".
    assert len(chunks) == 2
    assert chunks[0].text == "A"
    assert chunks[1].text == "B"


def test_chunk_multiline_paragraph():
    """Integration: Realistic multi-paragraph text with mixed boundaries."""
    text = (
        "First paragraph here.\n"
        "Second paragraph with more content!\n"
        "Third? Yes, third.\n"
        "Final paragraph."
    )
    doc = ParsedDocument(text=text, metadata={"type": "article"})

    chunks = chunk_document(doc, chunk_size=30, chunk_overlap=5)

    assert len(chunks) >= 2
    # All text should be recoverable
    combined = " ".join(c.text for c in chunks)
    for word in ["First", "Second", "Third", "Final"]:
        assert word in combined
    # All chunks should have metadata
    for c in chunks:
        assert c.metadata == {"type": "article"}


# ============================================================================
# _append_chunk helper
# ============================================================================


def test_append_chunk_ignores_whitespace():
    """
    Helper Logic: Ensure _append_chunk rejects strings that are empty after stripping.
    """
    chunks: list[DocumentChunk] = []
    metadata = {"test": True}

    # Valid string
    _append_chunk(chunks, " Valid text ", 0, metadata)
    assert len(chunks) == 1
    assert chunks[0].text == "Valid text"

    # Invalid strings (spaces, tabs, newlines)
    _append_chunk(chunks, "   ", 1, metadata)
    _append_chunk(chunks, "\n\t\n", 2, metadata)

    # Length should still be 1!
    assert len(chunks) == 1


def test_append_chunk_ignores_empty_string():
    """Helper Logic: Ensure _append_chunk rejects a completely empty string."""
    chunks: list[DocumentChunk] = []
    _append_chunk(chunks, "", 0, {})

    assert len(chunks) == 0


def test_append_chunk_preserves_index_and_metadata():
    """Helper Logic: Verify _append_chunk correctly sets chunk_index and metadata."""
    chunks: list[DocumentChunk] = []
    meta = {"page": 7}

    _append_chunk(chunks, "some text", 42, meta)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 42
    assert chunks[0].metadata == meta
    assert chunks[0].text == "some text"


def test_append_chunk_multiple_valid():
    """Helper Logic: Successive valid appends should each increase the list length."""
    chunks: list[DocumentChunk] = []

    _append_chunk(chunks, "first", 0, {})
    _append_chunk(chunks, "second", 1, {})
    _append_chunk(chunks, "third", 2, {})

    assert len(chunks) == 3
    assert [c.text for c in chunks] == ["first", "second", "third"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_append_chunk_returns_none():
    """Helper Logic: _append_chunk should return None (modifies list in place)."""
    chunks: list[DocumentChunk] = []
    result = _append_chunk(chunks, "text", 0, {})

    assert result is None
    assert len(chunks) == 1


def test_append_chunk_unicode_text():
    """Helper Logic: _append_chunk should handle Unicode text correctly."""
    chunks: list[DocumentChunk] = []
    _append_chunk(chunks, "  日本語テスト  ", 0, {"lang": "ja"})

    assert len(chunks) == 1
    assert chunks[0].text == "日本語テスト"
    assert chunks[0].metadata == {"lang": "ja"}


def test_append_chunk_single_newline():
    """Helper Logic: A single newline should be stripped to empty and rejected."""
    chunks: list[DocumentChunk] = []
    _append_chunk(chunks, "\n", 0, {})

    assert len(chunks) == 0


def test_append_chunk_text_with_internal_whitespace():
    """Helper Logic: Internal whitespace should be preserved, only leading/trailing stripped."""
    chunks: list[DocumentChunk] = []
    _append_chunk(chunks, "  hello   world  ", 0, {})

    assert len(chunks) == 1
    assert chunks[0].text == "hello   world"
