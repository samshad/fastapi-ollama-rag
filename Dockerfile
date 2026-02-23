# syntax=docker/dockerfile:1
# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Copy the compiled uv binary directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# UV_COMPILE_BYTECODE: Compiles .py files to .pyc for faster container startups.
# UV_LINK_MODE: Prevents hardlink errors in Docker.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy dependency manifests first to maximize Docker layer caching
COPY pyproject.toml uv.lock README.md ./

# Mount a cache for uv to dramatically speed up subsequent rebuilds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application source code
COPY src ./src

# Sync again to install the local project code into the environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Create a non-root user for security compliance (OWASP best practice)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy only the compiled virtual environment and app code from the builder stage
COPY --from=builder --chown=appuser:appuser /app /app

# Pre-pend the virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

# Drop root privileges
USER appuser

EXPOSE 8000

# Execute the FastAPI application
CMD ["uvicorn", "fastapi_ollama_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]