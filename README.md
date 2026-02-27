# High-Performance FastAPI RAG System

A production-grade, fully asynchronous, and multi-tenant Retrieval-Augmented Generation (RAG) API built with FastAPI, Neon Serverless Postgres (`pgvector`), and Ollama models.

This project demonstrates enterprise-level backend architecture, focusing on secure user isolation, non-blocking event loops, optimal database interactions, strict data contracts, and memory-efficient document processing.


---

## 🏗 System Architecture

The system is divided into two highly decoupled, asynchronous pipelines:

### 1. Authentication & Multi-Tenancy (`/api/v1/auth/*`)
* **Secure JWT Access:** Implements OAuth2 Password Bearer flows with short-lived JSON Web Tokens (JWT) and robust bcrypt password hashing.
* **OTP Email Verification:** Utilizes asynchronous SMTP dispatching (`aiosmtplib`) for secure, 6-digit OTP-based user registration and password reset flows, preventing spam and phantom accounts.
* **Strict Data Isolation:** Every document, chunk, and vector is cryptographically bound to a specific `user_id`. Queries and retrievals are strictly scoped at the database level to guarantee users can only access their own uploaded context.

### 2. Document Ingestion Pipeline (`/api/v1/documents/*`)
* **Cryptographic Deduplication:** Calculates SHA-256 fingerprints of incoming byte streams to prevent redundant GPU embedding computations and database bloat.
* **Non-Blocking Parsing:** Offloads CPU-bound PyMuPDF C-extension executions to a background thread pool (`asyncio.to_thread`), preventing the FastAPI event loop from stalling during massive PDF uploads.
* **Semantic Chunking:** Utilizes a zero-dependency, O(N) sliding-window algorithm to chunk text. Snaps to natural punctuation boundaries to ensure words and semantic thoughts are not severed.
* **Vector Embedding & Bulk Upsert:** Communicates asynchronously with Ollama (`mxbai-embed-large`) to generate 1024-dimensional vectors. Utilizes `asyncpg.executemany` over a binary protocol for highly optimized bulk insertions into Postgres.

### 3. Retrieval & Generation Pipeline (`/api/v1/chat/*`)
* **HNSW Vector Search:** Executes lightning-fast semantic similarity queries using `pgvector`'s Hierarchical Navigable Small World (HNSW) index and cosine distance (`<=>`), strictly filtered by the authenticated user's ID.
* **Strict Grounding:** Compiles retrieved chunks into a fortified system prompt, strictly instructing the LLM (`deepseek-r1:8b` / `llama3.2`) to avoid hallucinating outside the provided context.
* **Real-Time Streaming:** Pipes the LLM's token-by-token generation directly into FastAPI's `StreamingResponse`, dropping perceived latency to sub-second levels.

### 4. Testing & Observability
* **Comprehensive Testing Suite:** Features isolated Unit, Integration, and API route tests using `pytest` and `httpx.AsyncClient`. Validates everything from asynchronous context managers to Pydantic payload rejections.
* **CI/CD Database Ephemerality:** GitHub Actions pipeline spins up temporary Dockerized Postgres containers with `pgvector` to run real SQL integration tests on every pull request.
* **Structured JSON Logging:** Completely bypasses standard Python text logging in favor of `structlog`. Every event is emitted as a strictly formatted JSON object containing contextual metadata.
* **Cloud Log Aggregation:** Streams logs asynchronously to **Better Stack**, providing a centralized, highly searchable observability plane for real-time monitoring and debugging.

---

## 🧠 Key Architectural Decisions

| Decision                     | Alternative Considered | Rationale |
|:-----------------------------| :--- | :--- |
| **Custom Auth & OTP** | Auth0 / Firebase | Rolling a custom JWT/OTP flow ensures complete ownership of user data, zero vendor lock-in, and precise control over the multi-tenant database schema. |
| **Real DB Integration Tests**| Mocking Database Calls | Mocking raw SQL completely blinds the test suite to syntax errors. Spinning up a real `pgvector` pool in the test session guarantees the SQL behaves exactly as it will in production. |
| **Pure Python Chunker** | LangChain / LlamaIndex | Avoiding framework bloat. Heavy orchestration libraries abstract away critical control flows. A custom chunker is faster, deterministic, and easier to test. |
| **`asyncpg` (Raw SQL)** | SQLAlchemy / SQLModel | `asyncpg` is demonstrably the fastest Postgres driver for Python. Bypassing ORM overhead allows for precise tuning of `pgvector` operators and bulk binary insertions. |
| **Neon Serverless Postgres** | ChromaDB / Milvus | Postgres with `pgvector` unifies relational metadata (Users, OTPs, Files) and vector data in a single ACID-compliant database, eliminating the "split-brain" infrastructure problem. |

---

## 🛠 Tech Stack

* **Framework:** FastAPI, Python 3.12+
* **Database:** Neon (PostgreSQL 16), `pgvector`, `asyncpg`
* **AI / LLM:** Ollama (Models: Generation & Embeddings), `httpx`
* **Security & Auth:** PyJWT, `bcrypt`, `aiosmtplib`
* **Document Processing:** PyMuPDF (`fitz`), `python-multipart`
* **Validation & Config:** Pydantic v2, `pydantic-settings`
* **Testing & CI/CD:** Pytest, `pytest-asyncio`, GitHub Actions
* **Infrastructure:** Docker, Docker Compose, `uv`
* **Observability:** `structlog`, Better Stack (Cloud Log Management)

---

## 📡 API Reference

Interactive Swagger documentation is automatically generated and available at `/docs` when the server is running. All `/documents` and `/chat` endpoints require a valid `Bearer` token.

| Category | Method | Endpoint | Description |
| :--- | :--- | :--- | :--- |
| **Auth** | `POST` | `/api/v1/auth/request-otp` | Initiates registration by emailing a 6-digit OTP. |
| **Auth** | `POST` | `/api/v1/auth/register` | Consumes OTP and registers user with hashed password. |
| **Auth** | `POST` | `/api/v1/auth/login` | Authenticates credentials and issues a JWT access token. |
| **Auth** | `POST` | `/api/v1/auth/request-reset-otp` | Initiates the password reset flow. |
| **Auth** | `POST` | `/api/v1/auth/reset-password` | Consumes OTP and updates user's password. |
| **Docs** | `POST` | `/api/v1/documents/ingest` | Uploads, chunks, embeds, and saves a PDF to the user's account. |
| **Docs** | `GET` | `/api/v1/documents/` | Lists all PDFs uploaded by the authenticated user. |
| **Docs** | `DELETE` | `/api/v1/documents/{file_id}` | Cascading deletion of a file and all its associated vector chunks. |
| **Chat** | `POST` | `/api/v1/chat/search` | Raw semantic vector search against the user's isolated documents. |
| **Chat** | `POST` | `/api/v1/chat/completions` | End-to-end RAG. Retrieves user context and streams LLM response. |

---

## 🚀 Development Setup

### Prerequisites
1. [Docker & Docker Compose](https://docs.docker.com/engine/install/)
2. [Ollama](https://ollama.com/download)
3. A [Neon Serverless Postgres](https://neon.tech/) database URL
4. Basic SMTP Credentials (e.g., Gmail App Passwords) for sending OTPs.

### 1. Configure the Environment
Clone the repository and create a `.env` file in the root directory:
```bash
cp .env.example .env
```

Populate the `.env` file with your database connection string, AI provider routing, and email credentials.

* **Database:** Provide your Neon Serverless Postgres URL.
* **AI Routing:** Point to a local Dockerized Ollama instance, or swap it out for a cloud-hosted LLM endpoint.
* **SMTP Service:** The example below uses Gmail App Passwords, but the system is entirely provider-agnostic. You can plug in any standard SMTP service (e.g., AWS SES, SendGrid, Mailgun) by simply updating the variables.

```ini
# Core Database
DATABASE_URL="postgres://user:password@ep-your-db-id.region.aws.neon.tech/neondb?sslmode=require"

# AI / LLM Routing
OLLAMA_BASE_URL="[https://your-cloud-llm-instance.com](https://your-cloud-llm-instance.com)" # Or use [http://host.docker.internal:11434](http://host.docker.internal:11434) for local Docker

# Security
SECRET_KEY="generate-a-super-secret-random-key-here"

# SMTP / Email Settings (Replace with any SMTP provider)
SMTP_SERVER="smtp.gmail.com"
SMTP_PORT=587
SMTP_USERNAME="your-email@gmail.com"
SMTP_PASSWORD="your-app-password"
```
**Note:** host.docker.internal is required for the Dockerized FastAPI app to communicate with Ollama running on the host machine's localhost; otherwise, use the hosted address.

### 2. Pull Local AI Models
Ensure Ollama is running, then pull the required models for embeddings and generation:
```bash
ollama pull mxbai-embed-large
ollama pull deepseek-r1:8b  # or llama3.2
```

### 3. Build and Run via Docker
```bash
docker compose up --build -d
```
The FastAPI application will be available at `http://localhost:8000`. Database migrations (creating tables and HNSW indices) are executed automatically on startup via FastAPI lifecycle hooks.
