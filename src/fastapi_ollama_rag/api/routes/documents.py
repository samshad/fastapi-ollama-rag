import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile

from fastapi_ollama_rag.models.document import ParsedDocument
from fastapi_ollama_rag.services.pdf_parser import parse_pdf

logger = structlog.get_logger(__name__)

# Group these endpoints under the "Documents" tag in Swagger UI
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/parse", response_model=ParsedDocument)
async def upload_and_parse_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, extracts the text entirely in-memory,
    and returns the structured text and metadata.
    """
    # Validate the file type
    if file.content_type != "application/pdf":
        logger.warning("Invalid file type uploaded", content_type=file.content_type)
        raise HTTPException(
            status_code=400, detail="Only application/pdf is supported."
        )

    try:
        # Await the file stream into RAM
        file_bytes = await file.read()

        # Pass the raw bytes to the decoupled PyMuPDF service
        parsed_doc = await parse_pdf(file_bytes)

        return parsed_doc

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error("Unexpected error during PDF upload", error=str(e))
        raise HTTPException(
            status_code=500, detail="Internal server error while parsing PDF."
        )
