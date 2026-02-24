# High-Performance FastAPI RAG System

A production-grade, fully asynchronous Retrieval-Augmented Generation (RAG) API built with FastAPI, Neon Serverless Postgres (`pgvector`), and Ollama models. 

This project demonstrates enterprise-level backend architecture, focusing on non-blocking event loops, optimal database interactions, strict data contracts, and memory-efficient document processing.



---

## 🏗 System Architecture

The system is divided into two highly decoupled, asynchronous pipelines:

### 1. Document Ingestion Pipeline (`/api/v1/documents/ingest`)
* **Cryptographic Deduplication:** Calculates SHA-256 fingerprints of incoming byte streams to prevent redundant GPU embedding computations and database bloat.
* **Non-Blocking Parsing:** Offloads CPU-bound PyMuPDF C-extension executions to a background thread pool (`asyncio.to_thread`), preventing the FastAPI event loop from stalling during massive PDF uploads.
* **Semantic Chunking:** Utilizes a zero-dependency, $O(N)$ sliding-window algorithm to chunk text. Snaps to natural punctuation boundaries to ensure words and semantic thoughts are not severed.
* **Vector Embedding & Bulk Upsert:** Communicates asynchronously with Ollama (`mxbai-embed-large`) to generate 1024-dimensional vectors. Utilizes `asyncpg.executemany` over a binary protocol for highly optimized bulk insertions into Postgres.

### 2. Retrieval & Generation Pipeline (`/api/v1/chat/completions`)
* **HNSW Vector Search:** Executes lightning-fast semantic similarity queries using `pgvector`'s Hierarchical Navigable Small World (HNSW) index and cosine distance (`<=>`).
* **Strict Grounding:** Compiles retrieved chunks into a fortified system prompt, strictly instructing the LLM (`deepseek-r1:8b` / `llama3.2`) to avoid hallucinating outside the provided context.
* **Real-Time Streaming:** Pipes the LLM's token-by-token generation directly into FastAPI's `StreamingResponse`, dropping perceived latency to sub-second levels.

### 3. Observability & Telemetry
* **Structured JSON Logging:** Completely bypasses standard Python text logging in favor of `structlog`. Every event is emitted as a strictly formatted JSON object containing contextual metadata (e.g., execution time, chunk counts, model names).
* **Cloud Log Aggregation:** Streams logs asynchronously to **Better Stack**, providing a centralized, highly searchable observability plane for real-time monitoring, debugging, and alerting without blocking the FastAPI event loop.

---

## 🧠 Key Architectural Decisions

| Decision                     | Alternative Considered | Rationale |
|:-----------------------------| :--- | :--- |
| **Pure Python Chunker**      | LangChain / LlamaIndex | Avoiding framework bloat. Heavy orchestration libraries abstract away critical control flows. A custom $O(N)$ chunker is faster, deterministic, and easier to test. |
| **`asyncpg` (Raw SQL)**      | SQLAlchemy / SQLModel | `asyncpg` is demonstrably the fastest Postgres driver for Python. Bypassing ORM overhead allows for precise tuning of `pgvector` operators and bulk binary insertions. |
| **Neon Serverless Postgres** | ChromaDB / Milvus | Postgres with `pgvector` unifies relational metadata and vector data in a single ACID-compliant database, eliminating the "split-brain" infrastructure problem common in AI stacks. |
| **`uv` Package Manager**     | pip / Poetry | `uv` is written in Rust, offering near-instant dependency resolution and installation, significantly speeding up CI/CD pipelines and Docker build times. |
| **Structured Logging (```structlog``` + Better Stack)**      | Standard Python logging | Standard logging produces flat, unstructured text that is impossible to query effectively at scale. ```structlog``` enforces JSON-bound context, allowing Better Stack to parse and filter logs via SQL-like queries instantly during an incident. |

---

## 🛠 Tech Stack

* **Framework:** FastAPI, Python 3.12+
* **Database:** Neon (PostgreSQL 16), `pgvector`, `asyncpg`
* **AI / LLM:** Ollama (Models: Generation & Embeddings), `httpx`
* **Document Processing:** PyMuPDF (`fitz`), `python-multipart`
* **Validation & Config:** Pydantic v2, `pydantic-settings`
* **Infrastructure:** Docker, Docker Compose, `uv`
* **Observability:** `structlog`, Better Stack (Cloud Log Management)

---

## 📡 API Reference

Interactive Swagger documentation is automatically generated and available at `/docs` when the server is running.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/documents/ingest` | Uploads a PDF, hashes it, chunks text, generates embeddings, and bulk-inserts to Postgres. |
| `POST` | `/api/v1/chat/search` | Performs a raw semantic vector search and returns chunks + cosine similarity scores. |
| `POST` | `/api/v1/chat/completions` | End-to-end RAG. Embeds query, retrieves context, and streams the grounded LLM response. |
| `GET` | `/health` | Application health and database connection status check. |

---

## 🚀 Development Setup

### Prerequisites
1. [Docker & Docker Compose](https://docs.docker.com/engine/install/)
2. [Ollama](https://ollama.com/download)
3. A [Neon Serverless Postgres](https://neon.tech/) database URL

### 1. Configure the Environment
Clone the repository and create a `.env` file in the root directory:
```bash
cp .env.example .env
```

Populate the `.env` with your Neon database URL and local Ollama routing:
```bash
DATABASE_URL="postgres://user:password@ep-your-db-id.region.aws.neon.tech/neondb?sslmode=require"
OLLAMA_BASE_URL="[http://host.docker.internal:11434](http://host.docker.internal:11434)"
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
The FastAPI application will be available at ```http://localhost:8000```. Database migrations (creating tables and HNSW indices) are executed automatically on startup via FastAPI lifecycle hooks.
