from fastapi import FastAPI

from fastapi_ollama_rag.core.config import settings

app = FastAPI(title=settings.project_name)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "db_host": str(settings.database_url)}
