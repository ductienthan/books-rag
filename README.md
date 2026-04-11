# BookRAG

Open-source book ingestion and Q&A system powered by local LLMs. Ingest PDF and EPUB books, then ask natural-language questions against them — fully local, no external API calls required.

## Table of Contents

- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Installation](#installation)
- [Commands](#commands)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Ingestion Pipeline](#ingestion-pipeline)
- [RAG Pipeline](#rag-pipeline)
- [Database Schema](#database-schema)
- [Testing](#testing)
- [Development](#development)
- [License](#license)

---

## Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env

# 2. Start services (PostgreSQL + Ollama)
docker compose up -d

# 3. First-time setup: runs migrations, pulls models
bookrag setup

# 4. Ingest a book and wait for completion
bookrag add my-book.pdf --wait

# 5. Ask a question
bookrag ask "What is the main argument of this book?"

# Or enter interactive Q&A mode
bookrag ask
```

---

## Requirements

- Docker + Docker Compose
- Python 3.11+
- 8 GB RAM minimum (16 GB recommended for large books)
- ~5 GB disk for models (Gemma 4 via Ollama + nomic-embed-text)

---

## Installation

### With Docker (recommended)

```bash
git clone <repo>
cd book-rag-training
cp .env.example .env
docker compose up -d
```

The `app` container starts the background worker automatically. Use `bookrag` CLI from your host machine.

### Local Development

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e .
```

Ensure PostgreSQL and Ollama are running and accessible (see [Configuration](#configuration)).

---

## Commands

All commands are exposed via the `bookrag` entry point.

### `bookrag setup`

First-time initialization wizard.

- Runs Alembic database migrations
- Verifies Ollama connectivity and pulls `gemma4:e4b` if needed
- Downloads the embedding model (~270 MB)
- Creates a default conversation session

### `bookrag add <file>`

Ingest a PDF or EPUB book into the system.

```bash
bookrag add my-book.pdf
bookrag add my-book.epub --title "Custom Title" --author "Author Name"
bookrag add my-book.pdf --wait    # block until ingestion completes
bookrag add my-book.pdf --local   # run ingestion in-process (no Docker worker)
bookrag add my-book.pdf --force   # re-ingest even if already exists
```

| Option | Description |
|---|---|
| `--title, -t TEXT` | Override detected book title |
| `--author, -a TEXT` | Override detected author |
| `--wait, -w` | Block until ingestion job finishes |
| `--local, -l` | Run ingestion in current process |
| `--force, -f` | Re-ingest even if a duplicate hash is found |

Files are deduplicated by SHA256 hash. A copy is stored in `./data/books/` for worker access.

### `bookrag list`

Display all ingested books with metadata.

```
ID        Title                Author          Type  Pages  Status   Added
--------  -------------------  --------------  ----  -----  -------  ----------
a1b2c3d4  The Great Gatsby     F. Scott Fitz.  PDF   180    ready    2024-01-15
e5f6g7h8  Sapiens              Yuval N. Harari EPUB  443    ready    2024-01-16
```

### `bookrag status [job_id]`

Check ingestion job progress.

```bash
bookrag status               # show all active jobs
bookrag status a1b2c3d4      # check specific job by ID prefix
bookrag status a1b2 --watch  # live progress bar until completion
```

| Option | Description |
|---|---|
| `--watch, -w` | Poll until job completes |

Output includes: job ID, book title, current phase (extracting/chunking/embedding/indexing), progress %, and ETA.

### `bookrag ask [question]`

Ask a question against your ingested books.

```bash
bookrag ask "What does the author say about consciousness?"
bookrag ask                                          # interactive mode
bookrag ask --session <uuid>                        # use a specific session
bookrag ask --books a1b2c3d4,e5f6g7h8               # scope to specific books
```

| Option | Description |
|---|---|
| `--session, -s UUID` | Use a specific session ID |
| `--books, -b IDS` | Comma-separated book ID prefixes to search |

The system automatically scopes searches based on your question (e.g., mentioning a book title). In interactive mode, type `exit` or press Ctrl+C to quit.

Responses include inline source citations: `[Book: Title, Chapter N, p. 42]`

### `bookrag session new`

Start a new conversation session.

```bash
bookrag session new
bookrag session new --books a1b2c3d4,e5f6g7h8
```

Outputs a session UUID stored locally in `~/.bookrag_session`.

### `bookrag session scope <ids>`

Scope the current session to specific books.

```bash
bookrag session scope a1b2c3d4,e5f6g7h8
```

All future queries in this session will only search the specified books.

### `bookrag remove <book_id>`

Delete a book and all its associated data (chunks, embeddings, summaries).

```bash
bookrag remove a1b2c3d4
bookrag remove a1b2 --yes   # skip confirmation
```

---

## Configuration

All settings are loaded from `.env` via `pydantic-settings`. Copy `.env.example` to get started.

### Database

```env
POSTGRES_URL=postgresql+psycopg://bookrag:bookrag@localhost:5432/bookrag
ASYNC_POSTGRES_URL=postgresql+asyncpg://bookrag:bookrag@localhost:5432/bookrag
```

Two URLs are required: one for synchronous CLI operations, one for the async background worker. When using Docker Compose, the host port is mapped to **5433** to avoid conflict with any local PostgreSQL instance.

### LLM (Ollama)

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_LLM_MODEL=gemma4:e4b
```

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_LLM_MODEL` | `gemma4:e4b` | Model for query rewriting, answer generation, and chapter summarization |

### Embedding Model

```env
EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5
EMBED_BATCH_SIZE=32
EMBED_DIM=768
```

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | HuggingFace model ID (sentence-transformers) |
| `EMBED_BATCH_SIZE` | `32` | Batch size for embedding (tune for GPU/CPU memory) |
| `EMBED_DIM` | `768` | Vector dimension — must match model |

The model uses task-specific prefixes: `search_document:` when storing, `search_query:` when retrieving.

### Ingestion

```env
OCR_CHAR_THRESHOLD=100
MAX_BOOKS=50
```

| Variable | Default | Description |
|---|---|---|
| `OCR_CHAR_THRESHOLD` | `100` | Chars per page below which EasyOCR fallback is triggered |
| `MAX_BOOKS` | `50` | Maximum books allowed in the system |

### Chunking

```env
CHILD_CHUNK_MIN_TOKENS=80
CHILD_CHUNK_MAX_TOKENS=120
CHILD_CHUNK_OVERLAP_TOKENS=20
PARENT_CHUNK_MIN_TOKENS=300
PARENT_CHUNK_MAX_TOKENS=400
PARENT_CHUNK_OVERLAP_TOKENS=40
SUMMARY_MIN_TOKENS=500
SUMMARY_MAX_TOKENS=800
```

All values are in tokens (using `tiktoken` `cl100k_base` encoding).

| Level | Range | Overlap | Purpose |
|---|---|---|---|
| Child | 80–120 tokens | 20 | Dense, specific chunks for vector search |
| Parent | 300–400 tokens | 40 | Rich context blocks fed to the LLM |
| Summary | 500–800 tokens | — | Chapter summaries for broad context |

### Retrieval

```env
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
COSINE_THRESHOLD=0.35
HNSW_EF_SEARCH=40
```

| Variable | Default | Description |
|---|---|---|
| `RETRIEVAL_TOP_K` | `20` | Candidate chunks fetched from pgvector |
| `RERANK_TOP_K` | `5` | Final chunks passed to LLM after reranking |
| `COSINE_THRESHOLD` | `0.35` | Cosine similarity floor; results below are flagged low-confidence |
| `HNSW_EF_SEARCH` | `40` | HNSW search effort (higher = more accurate but slower) |

### Agent

```env
LLM_MAX_TOKENS=512
SESSION_MEMORY_WINDOW=6
LOG_LEVEL=INFO
```

| Variable | Default | Description |
|---|---|---|
| `LLM_MAX_TOKENS` | `512` | Max tokens in generated answers |
| `SESSION_MEMORY_WINDOW` | `6` | Number of past conversation turns injected as context |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Architecture

BookRAG is organized into five layers:

```
L4: CLI (typer + rich)
         │
L3: Agent (router → rewriter → searcher → reranker → generator → grounding)
         │
L2: Storage (PostgreSQL 16 + pgvector HNSW)
         │
L1: Ingestion (extractor → chunker → embedder → summarizer)
         │
L5: Quality (grounding checks · source citations · session memory)
```

### Services (Docker Compose)

| Service | Image | Purpose | Port |
|---|---|---|---|
| `db` | `pgvector/pgvector:pg16` | Vector-capable PostgreSQL | 5433 (host) |
| `ollama` | `ollama/ollama:latest` | Local LLM inference | 11434 |
| `app` | Built from `Dockerfile` | Background ingestion worker | — |

GPU support for Ollama is available via an optional `deploy.resources` block in `docker-compose.yml` (commented out by default).

### Project Structure

```
bookrag/
├── cli/
│   └── main.py              # Typer CLI entry point
├── ingestion/
│   ├── extractor.py         # PDF/EPUB text extraction
│   ├── chunker.py           # Hierarchical chunking
│   ├── embedder.py          # Sentence-transformers encoding
│   └── worker.py            # Background job processor
├── retrieval/
│   ├── router.py            # Query scope resolution
│   ├── rewriter.py          # LLM query expansion
│   ├── searcher.py          # pgvector HNSW search
│   └── reranker.py          # Cross-encoder reranking
├── agent/
│   ├── loop.py              # ReAct orchestration
│   ├── generator.py         # Answer generation (Gemma 4)
│   └── memory.py            # Session memory management
├── db/
│   ├── models.py            # SQLAlchemy ORM models
│   └── session.py           # Connection pooling
├── quality/
│   └── grounding.py         # Hallucination detection
└── config.py                # Centralized pydantic settings
```

---

## Ingestion Pipeline

Ingestion runs in five sequential phases managed by the background worker.

### Phase 1: Extraction

**PDF** — Uses `pdfplumber` to stream pages. If a page yields fewer than `OCR_CHAR_THRESHOLD` characters, `EasyOCR` is run as a fallback at 200 DPI. Chapters are detected from the PDF outline (TOC); if none exists, regex heading detection is used (`chapter`, `part`, `section`, `prologue`, etc.).

**EPUB** — Uses `ebooklib` to iterate spine items. Each item is parsed with `BeautifulSoup4` — block-level HTML elements are converted to newlines, and the first `h1/h2/h3` tag becomes the chapter title.

Both formats go through the same text cleaning step: NUL bytes removed, 3+ consecutive newlines collapsed to 2, whitespace trimmed.

### Phase 2: Chunking

Hierarchical chunking creates two levels of granularity from each chapter:

1. **Parent chunks** (300–400 tokens) — split by paragraph boundaries with 40-token overlap. Store char positions for source attribution.
2. **Child chunks** (80–120 tokens) — split by sentence boundaries within each parent, with 20-token overlap. Child chunks store a SHA256 hash for deduplication.

Tokens are counted with `tiktoken` (`cl100k_base`). Sentences longer than 360 tokens fall back to word-level splitting.

### Phase 3: Embedding

Child chunks are encoded in batches of 32 using `nomic-ai/nomic-embed-text-v1.5` with the `search_document:` prefix. Embeddings are normalized (cosine similarity). The process is resumable — only chunks where `embedded_at IS NULL` are processed.

### Phase 4: Summarization

Each chapter is summarized by Gemma 4 (truncated to ~12,000 characters). The summary is embedded and stored in `chapter_summaries` for broader context during retrieval. Falls back to the first 2,000 characters of raw text if the LLM call fails.

### Phase 5: Indexing

Runs `VACUUM ANALYZE` on `child_chunks` and `chapter_summaries` to update planner statistics. Marks the job as `done`.

---

## RAG Pipeline

Each query goes through eight steps:

### Step 1: Scope Resolution

Determines which books to search, in priority order:

1. **Explicit** — fuzzy-matches query text against book titles/authors (≥80% threshold via RapidFuzz)
2. **Cross-book** — detects keywords like `compare`, `across all books`, `between` → searches all books
3. **Inferred** — uses the current session's `book_scope` setting
4. **Default** — all books with `status = completed`

### Step 2: Query Rewriting

Gemma 4 expands and optionally decomposes the query into 1–3 sub-queries. Pronouns and implicit references are resolved using the last 6 conversation turns. Falls back to the original question on error.

### Step 3: Vector Search

Each sub-query is embedded with the `search_query:` prefix and searched against `child_chunks` using pgvector's `<=>` (cosine distance) operator with the HNSW index. Results below `COSINE_THRESHOLD` are discarded. Candidates are de-duplicated by parent chunk (keeping the highest-scoring child per parent). Up to `RETRIEVAL_TOP_K` (20) unique parent contexts are returned.

### Step 4: Confidence Check

If no candidates pass the threshold, or the top score is below `COSINE_THRESHOLD`, the response is flagged as low-confidence and the LLM is instructed to say so rather than guess.

### Step 5: Reranking

A `cross-encoder/ms-marco-MiniLM-L-6-v2` model rescores `(query, parent_chunk_text)` pairs. If there are more candidates than `RERANK_TOP_K`, the cross-encoder is run and the top 5 are kept. This step is skipped if the candidate count is already ≤ 5.

### Step 6: Answer Generation

Gemma 4 is called with:

- A system prompt instructing it to answer only from provided excerpts and cite sources inline (`[Book: Title, Chapter N, p. X]`)
- Formatted context blocks (up to 5 parent chunks with source metadata)
- Session history (last 6 turns)
- Settings: temperature 0.1, max 512 tokens

### Step 7: Grounding Check

A second Gemma 4 call verifies the answer is supported by the retrieved excerpts. Returns `GROUNDED` or `UNGROUNDED`. Fails open (assumes grounded) on error.

### Step 8: Persistence

The question and answer are saved to the `messages` table with referenced book IDs and chunk IDs. The session's `last_active_at` timestamp is updated. Sources are deduplicated by `(book_id, chapter_index)` before display.

---

## Database Schema

Migrations are managed with Alembic. Run `bookrag setup` or `alembic upgrade head` to apply.

| Table | Purpose |
|---|---|
| `books` | Ingested books with metadata and status |
| `ingestion_jobs` | Per-book async job tracking with progress |
| `chapters` | Extracted chapters with raw text |
| `chapter_summaries` | LLM-generated summaries with 768-dim embeddings |
| `parent_chunks` | Context blocks (300–400 tokens) with char positions |
| `child_chunks` | Search chunks (80–120 tokens) with 768-dim HNSW-indexed embeddings |
| `sessions` | Conversation sessions with book scope |
| `messages` | Session history with referenced book/chunk IDs |

Key indexes:
- `child_chunks.embedding` — HNSW vector index (pgvector)
- `child_chunks` partial index on `embedded_at IS NULL` — enables resumable embedding
- Trigram index on book titles via `pg_trgm` — used by query router for fuzzy matching

---

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=bookrag

# Specific file
pytest tests/test_chunker.py -v
```

| File | Coverage |
|---|---|
| `tests/test_router.py` | Query scope resolution (explicit, cross-book, inferred, default) |
| `tests/test_chunker.py` | Token ranges, child/parent consistency, hash generation, empty input |
| `tests/test_extractor.py` | PDF TOC detection, EPUB heading extraction, text cleaning |
| `tests/test_embedder.py` | Batch encoding, prefix handling, vector dimensions |

---

## Development

### Setup

```bash
uv sync --all-extras
```

### Linting & Formatting

```bash
ruff check bookrag/
ruff format bookrag/
```

### Type Checking

```bash
mypy bookrag/
```

### Adding a Migration

```bash
alembic revision --autogenerate -m "description"
alembic upgrade head
```

### Running the Worker Locally

```bash
python -m bookrag.worker
```

---

## License

MIT
