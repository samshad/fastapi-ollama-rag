from typing import Annotated

import asyncpg
import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from fastapi_ollama_rag.api.dependencies import get_current_user
from fastapi_ollama_rag.core.database import get_db
from fastapi_ollama_rag.models.chat import ChatRequest, SearchResult
from fastapi_ollama_rag.services.db_ops import search_similar_documents
from fastapi_ollama_rag.services.embeddings import generate_embedding
from fastapi_ollama_rag.services.generation import generate_rag_response

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

DBConnection = Annotated[asyncpg.Connection, Depends(get_db)]


@router.post("/search", response_model=list[SearchResult])
async def raw_vector_search(
    request: ChatRequest,
    db: DBConnection = None,
    current_user: asyncpg.Record = Depends(get_current_user),
) -> list[SearchResult]:
    """
    Directly queries the vector database using semantic similarity.
    Returns the raw chunks and cosine similarity scores without LLM generation.
    Only searches the authenticated user's documents.
    """
    user_id = current_user["id"]
    logger.info("Received search request", query=request.query, user_id=user_id)

    query_embedding = await generate_embedding(request.query)

    # 3. The HNSW index search in Postgres
    results = await search_similar_documents(
        conn=db, query_embedding=query_embedding, user_id=user_id, limit=request.limit
    )

    logger.info("Received search results", results_count=len(results))

    return results


@router.post("/completions")
async def chat_completions(
    request: ChatRequest,
    db: DBConnection = None,
    current_user: asyncpg.Record = Depends(get_current_user),
):
    """
    End-to-end RAG Chat Endpoint.
    1. Embeds the user's query.
    2. Retrieves the top matching chunks from the database (isolated to user).
    3. Streams the LLM's grounded response back to the client.
    """
    user_id = current_user["id"]
    logger.info(
        "Received chat completion request...", query=request.query, user_id=user_id
    )

    query_embedding = await generate_embedding(request.query)

    context = await search_similar_documents(
        conn=db, query_embedding=query_embedding, user_id=user_id, limit=request.limit
    )

    response_generator = generate_rag_response(query=request.query, context=context)

    logger.info(
        "Streaming RAG response initialized",
        query=request.query,
        context_chunks=len(context),
    )

    return StreamingResponse(response_generator, media_type="text/plain")
