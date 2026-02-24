from typing import Annotated

import asyncpg
import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from fastapi_ollama_rag.core.database import get_db
from fastapi_ollama_rag.models.chat import SearchResult
from fastapi_ollama_rag.services.db_ops import search_similar_documents
from fastapi_ollama_rag.services.generation import generate_rag_response
from fastapi_ollama_rag.services.embeddings import generate_embedding

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

DBConnection = Annotated[asyncpg.Connection, Depends(get_db)]


class ChatRequest(BaseModel):
    query: str
    limit: int = 5


@router.post("/search", response_model=list[SearchResult])
async def raw_vector_search(
    request: ChatRequest, db: DBConnection = None
) -> list[SearchResult]:
    """
    Directly queries the vector database using semantic similarity.
    Returns the raw chunks and cosine similarity scores without LLM generation.
    """
    logger.info("Received search request", query=request.query)

    query_embedding = await generate_embedding(request.query)

    # The HNSW index search in Postgres
    results = await search_similar_documents(db, query_embedding, request.limit)

    logger.info("Received search results", results=results)

    return results


@router.post("/completions")
async def chat_completions(
        request: ChatRequest,
        db: DBConnection = None
):
    """
    End-to-end RAG Chat Endpoint.
    1. Embeds the user's query.
    2. Retrieves the top matching chunks from the database.
    3. Streams the LLM's grounded response back to the client.
    """
    logger.info("Received chat completion request...", query=request.query)

    query_embedding = await generate_embedding(request.query)
    context = await search_similar_documents(db, query_embedding, request.limit)

    response_generator = generate_rag_response(query=request.query, context=context)

    logger.info("Received rag response", query=request.query, context=context)

    return StreamingResponse(
        response_generator,
        media_type="text/plain"
    )
