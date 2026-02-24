import httpx
import structlog

from fastapi_ollama_rag.core.config import settings

logger = structlog.get_logger(__name__)


async def generate_embedding(text: str) -> list[float]:
    """
    Calls the Ollama instance to generate a vector embedding for a single text chunk.
    """
    url = f"{settings.ollama_base_url}/api/embeddings"
    payload = {"model": settings.ollama_embedding_model, "prompt": text}

    # Custom timeout because LLM cold-starts can take a few seconds
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()
            return data["embedding"]

        except httpx.HTTPError as e:
            logger.error("Failed to connect to Ollama", error=str(e), url=url)
            raise RuntimeError(f"Ollama embedding failed: {e}")
