# FastAPI RAG System

[![CI](https://github.com/samshad/fastapi-ollama-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/samshad/fastapi-ollama-rag/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A production-grade, fully asynchronous, multi-tenant Retrieval-Augmented Generation (RAG) API. Built with FastAPI, Neon Serverless Postgres (`pgvector`), and Ollama.

---

## System Architecture

<p align="center">
  <img src="docs/rag-architecture.webp" alt="System Architecture Diagram" width="100%"/>
</p>

### Authentication & Multi-Tenancy
- OAuth2 Password Bearer with JWT access tokens and bcrypt password hashing.
- OTP-based email verification via async SMTP (`aiosmtplib`) for registration and password reset.
- Every document and vector is scoped to a `user_id` — queries are isolated at the database level.

### Document Ingestion
- **Deduplication** — SHA-256 file fingerprinting prevents redundant embedding computations.
- **Non-blocking parsing** — CPU-bound PyMuPDF work is offloaded via `asyncio.to_thread`.
- **Chunking** — Zero-dependency, O(N) sliding-window algorithm that snaps to natural punctuation boundaries.
- **Bulk insert** — `asyncpg.executemany` over binary protocol for optimized vector storage.

### Retrieval & Generation
- **HNSW vector search** — Cosine similarity via `pgvector` (`<=>` operator), filtered by authenticated user.
- **Grounded generation** — Retrieved chunks are compiled into a system prompt that constrains the LLM to provided context only.
- **Streaming** — Token-by-token LLM output piped directly into `StreamingResponse`.

### Testing & CI/CD
- **468 tests** across unit, integration, and API route layers — **100% line coverage**.
- **Real database tests** — GitHub Actions spins up ephemeral `pgvector/pgvector:pg16` containers. No mocking SQL.
- **Lint gate** — Ruff linter + formatter enforced before tests run.

### Observability
- `structlog` JSON logging with rotating file handler and optional Better Stack cloud aggregation.

---

## Tech Stack

| Layer | Technology |
|:---|:---|
| Framework | FastAPI, Uvicorn, Python 3.12+ |
| Database | Neon PostgreSQL 16, `pgvector`, `asyncpg` (raw SQL) |
| AI / LLM | Ollama (`mxbai-embed-large`, `deepseek-r1:8b`), `httpx` |
| Auth | PyJWT, bcrypt, `aiosmtplib` |
| Parsing | PyMuPDF, `python-multipart` |
| Validation | Pydantic v2, `pydantic-settings` |
| Testing | pytest, `pytest-asyncio`, `pytest-cov`, `pytest-mock` |
| CI/CD | GitHub Actions (lint → test with ephemeral pgvector DB) |
| Packaging | `uv`, Docker (multi-stage, non-root) |
| Observability | `structlog`, Better Stack |

---

## API Endpoints

All `/documents` and `/chat` endpoints require a valid `Bearer` token. Interactive docs available at `/docs`.

| Category | Method | Endpoint | Description |
|:---|:---|:---|:---|
| Auth | `POST` | `/api/v1/auth/request-otp` | Email a 6-digit registration OTP |
| Auth | `POST` | `/api/v1/auth/register` | Verify OTP and create account |
| Auth | `POST` | `/api/v1/auth/login` | Authenticate and issue JWT |
| Auth | `POST` | `/api/v1/auth/request-reset-otp` | Email a password reset OTP |
| Auth | `POST` | `/api/v1/auth/reset-password` | Verify OTP and update password |
| Docs | `POST` | `/api/v1/documents/ingest` | Upload, chunk, embed, and store a PDF |
| Docs | `GET` | `/api/v1/documents/` | List user's uploaded files |
| Docs | `DELETE` | `/api/v1/documents/{file_id}` | Delete file and all associated chunks |
| Chat | `POST` | `/api/v1/chat/search` | Semantic vector search against user's documents |
| Chat | `POST` | `/api/v1/chat/completions` | Full RAG — retrieve context and stream LLM response |
| Health | `GET` | `/health` | Database connectivity check |

---

## Project Structure

```
src/fastapi_ollama_rag/
├── main.py                  # Lifespan hooks, health check
├── api/
│   ├── dependencies.py      # JWT token validation
│   └── routes/              # Auth, Chat, Documents
├── core/
│   ├── config.py            # Pydantic settings (env-driven)
│   ├── database.py          # asyncpg connection pool
│   ├── migrations.py        # Idempotent SQL schema runner
│   ├── security.py          # bcrypt + JWT helpers
│   ├── logger.py            # structlog configuration
│   └── sql/                 # Raw SQL (schema, queries)
├── models/                  # Pydantic request/response models
├── repository/              # Database access layer
└── services/                # Business logic (auth, chunker, embeddings, etc.)
```

---

## Design Decisions

| Decision | Why |
|:---|:---|
| `asyncpg` raw SQL over SQLAlchemy | Fastest Python Postgres driver. Direct control over `pgvector` operators and bulk binary insertions. No ORM overhead. |
| Custom chunker over LangChain | Zero dependencies. Deterministic. Testable. No framework bloat for a simple sliding-window algorithm. |
| Real DB integration tests over mocks | Mocking raw SQL hides syntax errors. Ephemeral `pgvector` containers catch real failures. |
| Custom JWT/OTP over Auth0 | Full ownership of user data and auth flows. No vendor lock-in. Precise control over multi-tenant schema. |
| Single Postgres over separate vector DB | `pgvector` unifies relational data and vectors in one ACID-compliant store. No split-brain infrastructure. |

---

## Getting Started

### Prerequisites

- [Docker & Docker Compose](https://docs.docker.com/engine/install/)
- [Ollama](https://ollama.com/download)
- A [Neon Serverless Postgres](https://neon.tech/) database (or any Postgres 16+ with `pgvector`)
- SMTP credentials for OTP emails (e.g., Gmail App Password)

### 1. Configure Environment

```bash
cp .env.example .env
```

```ini
# Database
DATABASE_URL="postgres://user:password@ep-your-db.region.aws.neon.tech/neondb?sslmode=require"

# AI / LLM
OLLAMA_BASE_URL="http://host.docker.internal:11434"  # Docker → host Ollama

# Security
SECRET_KEY="generate-a-strong-random-key"

# SMTP (any provider — Gmail, SES, SendGrid, etc.)
SMTP_SERVER="smtp.gmail.com"
SMTP_PORT=587
SMTP_USERNAME="you@gmail.com"
SMTP_PASSWORD="your-app-password"
```

> **Note:** `host.docker.internal` routes from the Docker container to the host machine's Ollama instance.

### 2. Pull AI Models

```bash
ollama pull mxbai-embed-large
ollama pull deepseek-r1:8b
```

### 3. Run

```bash
docker compose up --build -d
```

The API is available at `http://localhost:8000`. Migrations run automatically on startup.

### 4. Run Tests

```bash
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## License

[MIT](LICENSE)
