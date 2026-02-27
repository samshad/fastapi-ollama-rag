import hashlib
from typing import Annotated

import asyncpg
import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from fastapi_ollama_rag.api.dependencies import get_current_user
from fastapi_ollama_rag.core.config import settings
from fastapi_ollama_rag.core.database import get_db
from fastapi_ollama_rag.services.chunker import chunk_document
from fastapi_ollama_rag.services.db_ops import (
    bulk_insert_chunks,
    delete_file_and_chunks,
    get_files_for_user,
    is_file_processed,
)
from fastapi_ollama_rag.services.embeddings import generate_embedding
from fastapi_ollama_rag.services.pdf_parser import parse_pdf

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

DBConnection = Annotated[asyncpg.Connection, Depends(get_db)]


class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_processed: int
    detail: str | None = None


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    db: DBConnection = None,
    current_user: asyncpg.Record = Depends(get_current_user),
) -> IngestResponse:
    """
    End-to-end RAG ingestion pipeline:
    1. Hash PDF for deduplication.
    2. Extract text (PyMuPDF).
    3. Semantic chunking.
    4. Generate embeddings (Ollama).
    5. Bulk upsert into Neon (pgvector).
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, detail="Only application/pdf is supported."
        )

    user_id = current_user["id"]

    try:
        file_bytes = await file.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if await is_file_processed(db, file_hash, user_id):
            logger.info(
                "File already processed for user, skipping.",
                file_hash=file_hash,
                user_id=user_id,
            )
            return IngestResponse(
                status="skipped",
                filename=file.filename,
                chunks_processed=0,
                detail="File already exists in your account.",
            )

        parsed_doc = await parse_pdf(file_bytes)
        parsed_doc.metadata["file_hash"] = file_hash
        parsed_doc.metadata["filename"] = file.filename

        chunks = chunk_document(
            parsed_doc,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            logger.error("Ingestion pipeline failed", chunks=chunks)
            raise ValueError("Document yielded no extractable text.")

        embeddings = []
        logger.info("Generating embeddings via Ollama...", chunk_count=len(chunks))
        for i, chunk in enumerate(chunks):
            vector = await generate_embedding(chunk.text)
            embeddings.append(vector)
            if (i + 1) % 10 == 0:
                logger.info("Embedding progress", completed=i + 1, total=len(chunks))

        await bulk_insert_chunks(
            db, chunks, embeddings, user_id, file.filename, file_hash
        )

        return IngestResponse(
            status="success", filename=file.filename, chunks_processed=len(chunks)
        )

    except ValueError as ve:
        logger.error("Ingestion pipeline failed", error=str(ve))
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error("Ingestion pipeline failed", error=str(e))
        raise HTTPException(
            status_code=500, detail="Internal server error during document ingestion."
        )


@router.get("/")
async def list_user_files(
    db: DBConnection = None, current_user: asyncpg.Record = Depends(get_current_user)
):
    """
    Returns a list of all PDFs uploaded by the authenticated user.
    """
    user_id = current_user["id"]
    files = await get_files_for_user(db, user_id)
    logger.info("Successfully retrieved files for user.", total_files=len(files))
    return {"files": files}


@router.delete("/{file_id}")
async def delete_user_file(
    file_id: str,
    db: DBConnection = None,
    current_user: asyncpg.Record = Depends(get_current_user),
):
    """
    Deletes a specific file and all of its associated vector chunks.
    """
    user_id = current_user["id"]

    await delete_file_and_chunks(db, file_id=file_id, user_id=user_id)
    logger.info(
        "File and associated vector chunks deleted successfully.", file_id=file_id
    )
    return {"message": "File and associated vector chunks deleted successfully."}
