import asyncio

import fitz  # PyMuPDF
import structlog

from fastapi_ollama_rag.models.document import ParsedDocument

logger = structlog.get_logger(__name__)


def _extract_text_sync(file_bytes: bytes) -> ParsedDocument:
    """
    Synchronous CPU-bound function to parse the PDF.
    Opens the PDF from memory to avoid disk I/O.
    """
    try:
        # Open document from a memory stream
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        text_chunks = []
        for page in doc:
            # Extract plain text; PyMuPDF is generally good at preserving basic layout
            text_chunks.append(page.get_text("text"))

        full_text = "\n".join(text_chunks)

        # Safely extract built-in metadata
        metadata = doc.metadata or {}
        metadata["page_count"] = doc.page_count

        doc.close()

        logger.info(
            f"PDF text extraction complete with "
            f"text length: {len(full_text)}"
            f" and metadata: {metadata}"
        )

        return ParsedDocument(text=full_text, metadata=metadata)
    except Exception as e:
        logger.error("Failed to parse PDF bytes", error=str(e))
        raise ValueError(f"Could not parse the provided PDF file: {e}")


async def parse_pdf(file_bytes: bytes) -> ParsedDocument:
    """
    Asynchronous wrapper for PDF extraction.
    Offloads the CPU-bound PyMuPDF extraction to a separate thread,
    preventing the FastAPI event loop from blocking on large files.
    """
    logger.info("Starting PDF text extraction...")

    # Offload the blocking sync function to the default ThreadPoolExecutor
    parsed_doc = await asyncio.to_thread(_extract_text_sync, file_bytes)

    logger.info(
        "PDF extraction complete",
        page_count=parsed_doc.metadata.get("page_count"),
        text_length=len(parsed_doc.text),
    )

    return parsed_doc
