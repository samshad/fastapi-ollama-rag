from contextlib import asynccontextmanager
from typing import Annotated

import asyncpg
import structlog
from fastapi import Depends, FastAPI

from fastapi_ollama_rag.api.routes import auth, chat, documents
from fastapi_ollama_rag.core.config import settings
from fastapi_ollama_rag.core.database import close_db_connection, connect_to_db, get_db
from fastapi_ollama_rag.core.logger import setup_logging
from fastapi_ollama_rag.core.migrations import run_migrations

setup_logging()

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application lifecycle")

    await connect_to_db()

    await run_migrations()

    yield

    logger.info("Shutting down application lifecycle")
    await close_db_connection()


app = FastAPI(title=settings.project_name, lifespan=lifespan)

app.include_router(documents.router, prefix=settings.api_v1_prefix)
app.include_router(chat.router, prefix=settings.api_v1_prefix)
app.include_router(auth.router, prefix=settings.api_v1_prefix)

DBConnection = Annotated[asyncpg.Connection, Depends(get_db)]


@app.get("/health")
async def health_check(db: DBConnection):
    version = await db.fetchval("SELECT version();")
    logger.info("Health check executed", db_version=version)
    return {
        "status": "healthy",
        "project_name": str(settings.project_name),
        "database_version": version,
    }
