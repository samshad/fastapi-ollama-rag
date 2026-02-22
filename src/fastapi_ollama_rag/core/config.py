from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Core application settings. Fails fast if required variables are missing.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    project_name: str = "FastAPI Ollama RAG"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    database_url: PostgresDsn

    # Ollama Local Service
    # Using host.docker.internal allows Docker containers to hit the host machine's Ollama instance.
    ollama_base_url: str = Field(default="http://host.docker.internal:11434")
    ollama_generation_model: str = Field(default="deepseek-r1:8b")
    ollama_embedding_model: str = Field(default="mxbai-embed-large")


# Instantiate settings to be imported across the application
settings = Settings()
