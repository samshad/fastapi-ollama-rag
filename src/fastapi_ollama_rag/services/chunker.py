import structlog

from fastapi_ollama_rag.models.chunk import DocumentChunk
from fastapi_ollama_rag.models.document import ParsedDocument

logger = structlog.get_logger(__name__)


def chunk_document(
    document: ParsedDocument, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[DocumentChunk]:
    """
    Splits a ParsedDocument into smaller DocumentChunks using a SLIDING WINDOW O(N).
    Snaps to the nearest natural boundary (punctuation or space) to preserve semantics.
    """
    logger.info("Starting text chunking", text_length=len(document.text))

    text = document.text
    text_len = len(text)
    chunks: list[DocumentChunk] = []

    if text_len == 0:
        logger.warning("Attempted to chunk an empty document.")
        return chunks

    start = 0
    chunk_index = 0
    # Define natural boundaries to snap to
    boundaries = set(["\n", ".", "?", "!", " "])

    while start < text_len:
        end = start + chunk_size

        # If we reach the end of the text, take the rest and break
        if end >= text_len:
            _append_chunk(chunks, text[start:text_len], chunk_index, document.metadata)
            break

        # Look backward from the 'end' limit to find a natural boundary
        break_point = end
        while break_point > start and text[break_point] not in boundaries:
            break_point -= 1

        # If no natural boundary was found, force split
        if break_point == start:
            break_point = end

        # Extract the text and append the chunk
        _append_chunk(chunks, text[start:break_point], chunk_index, document.metadata)

        # Slide the window forward, factoring in the overlap
        start = break_point - chunk_overlap
        chunk_index += 1

        # Fast-forward 'start' past any trailing
        # whitespaces/newlines to avoid empty overlapping starts
        while start < text_len and text[start].isspace():
            start += 1

    logger.info("Chunking complete", total_chunks=len(chunks))
    return chunks


def _append_chunk(
    chunks: list[DocumentChunk], text_segment: str, index: int, metadata: dict
) -> None:
    """Helper to cleanly instantiate and append a DocumentChunk if it contains text."""
    clean_text = text_segment.strip()
    if clean_text:
        chunks.append(
            DocumentChunk(text=clean_text, chunk_index=index, metadata=metadata)
        )
