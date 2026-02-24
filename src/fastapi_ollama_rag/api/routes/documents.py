import hashlib

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile

from fastapi_ollama_rag.models.document import ParsedDocument
from fastapi_ollama_rag.services.pdf_parser import parse_pdf

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/parse", response_model=ParsedDocument)
async def upload_and_parse_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF, calculates its SHA-256 fingerprint for deduplication,
    extracts the text, and returns the structured data.
    """
    if file.content_type != "application/pdf":
        logger.warning("Invalid file type uploaded", content_type=file.content_type)
        raise HTTPException(
            status_code=400, detail="Only application/pdf is supported."
        )

    try:
        file_bytes = await file.read()

        file_hash = hashlib.sha256(file_bytes).hexdigest()
        logger.info("File upload received", filename=file.filename, file_hash=file_hash)

        parsed_doc = await parse_pdf(file_bytes)

        parsed_doc.metadata["file_hash"] = file_hash
        parsed_doc.metadata["filename"] = file.filename

        return parsed_doc

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error("Unexpected error during PDF upload", error=str(e))
        raise HTTPException(
            status_code=500, detail="Internal server error while parsing PDF."
        )
