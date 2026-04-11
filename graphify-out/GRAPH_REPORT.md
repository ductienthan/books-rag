# Graph Report - .  (2026-04-07)

## Corpus Check
- Corpus is ~11,132 words - fits in a single context window. You may not need a graph.

## Summary
- 262 nodes · 510 edges · 20 communities detected
- Extraction: 54% EXTRACTED · 46% INFERRED · 0% AMBIGUOUS · INFERRED: 237 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `Book` - 30 edges
2. `Session` - 25 edges
3. `IngestionJob` - 21 edges
4. `RetrievedChunk` - 17 edges
5. `RawChapter` - 15 edges
6. `Chapter` - 14 edges
7. `ChapterSummary` - 14 edges
8. `ParentChunk` - 14 edges
9. `ChildChunk` - 14 edges
10. `_get_db()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `Tests for the file extractor (no real files needed — mocked).` --uses--> `RawPage`  [INFERRED]
  tests/test_extractor.py → bookrag/ingestion/extractor.py
- `Tests for the query router.` --uses--> `ScopeType`  [INFERRED]
  tests/test_router.py → bookrag/retrieval/router.py
- `Tests for the query router.` --uses--> `Book`  [INFERRED]
  tests/test_router.py → bookrag/db/models.py
- `Tests for the query router.` --uses--> `Session`  [INFERRED]
  tests/test_router.py → bookrag/db/models.py
- `pdfplumber` --conceptually_related_to--> `extractor.py`  [INFERRED]
  README.md → SKILLS.md

## Hyperedges (group relationships)
- **Ingestion Pipeline: Extract to Chunk to Embed to Summarise to Index** — skills_extractor, skills_chunker, skills_embedder, skills_worker [EXTRACTED 1.00]
- **Query Pipeline: Router to Rewriter to Searcher to Reranker to Generator to Grounding** — skills_router, skills_rewriter, skills_searcher, skills_reranker, skills_agent_loop [EXTRACTED 1.00]
- **Data Model Hierarchy: Book to Chapter to ParentChunk to ChildChunk** — skills_book_model, skills_chapter_model, skills_parent_chunk_model, skills_child_chunk_model [EXTRACTED 1.00]

## Communities

### Community 0 - "System Overview & Architecture"
Cohesion: 0.06
Nodes (53): BeautifulSoup4, BookRAG System, EasyOCR, ebooklib, Gemma 4 (LLM), Ingestion Pipeline, nomic-embed-text-v1.5, pdfplumber (+45 more)

### Community 1 - "Text Chunking & Tokenization"
Cohesion: 0.08
Nodes (38): ChildChunkData, chunk_chapter(), ChunkResult, _count_tokens(), _make_child_chunks(), _make_parent_chunks(), ParentChunkData, Layer 1 — Hierarchical chunking.  Produces three levels from raw chapter text: (+30 more)

### Community 2 - "ORM Models & Quality Grounding"
Cohesion: 0.2
Nodes (25): DeclarativeBase, check_grounding(), Layer 5 — Grounding check.  Sends a tiny Gemma 4 call to verify the answer is su, Returns True if the answer is grounded in the retrieved chunks.     Falls back t, Base, Book, Chapter, ChapterSummary (+17 more)

### Community 3 - "CLI & Command Interface"
Cohesion: 0.17
Nodes (29): add(), ask(), _detect_author(), _detect_title(), _get_db(), _get_session_id(), _interactive_loop(), list_books() (+21 more)

### Community 4 - "Answer Generation"
Cohesion: 0.21
Nodes (12): build_context_block(), generate_answer(), GeneratedAnswer, Layer 3 — Answer generation via Gemma 4 (Ollama).  Builds a structured prompt fr, Format retrieved chunks into a numbered context block., Generate an answer using Gemma 4 via Ollama.     history: list of {role, content, AgentResponse, ask() (+4 more)

### Community 5 - "Session & Conversation Memory"
Cohesion: 0.25
Nodes (10): create_session(), get_history(), Layer 3 / 5 — Session memory.  Loads the rolling window of recent messages from, Return the last N messages for a session as a list of {role, content} dicts., Persist a message and update session.last_active_at., Create a new session, optionally scoped to specific books., Update the book scope for an existing session., save_message() (+2 more)

### Community 6 - "Router Test Suite"
Cohesion: 0.56
Nodes (8): _make_book(), _make_db(), _make_session(), Tests for the query router., test_cross_book_signal(), test_default_all_books(), test_explicit_scope_by_title(), test_inferred_from_session()

### Community 7 - "Ingestion Worker Pipeline"
Cohesion: 0.56
Nodes (8): _phase_chunk(), _phase_embed(), _phase_extract(), _phase_index(), _phase_summarise(), process_job(), run_worker(), _update_job()

### Community 8 - "Embedding & Vector Encoding"
Cohesion: 0.31
Nodes (8): embed_batch(), embed_documents(), embed_query(), _get_model(), Layer 1 — Embedding engine.  Uses nomic-embed-text-v1.5 via sentence-transformer, Embed a list of document passages (child chunks / summaries).     Returns a list, Embed a single user query with the correct task prefix., Alias for embed_documents — preferred name in the worker.

### Community 9 - "Embedder Test Suite"
Cohesion: 0.25
Nodes (5): Tests for the embedding engine., nomic-embed-text-v1.5 with normalize=True should return unit vectors., A query and a matching document should have high cosine similarity., test_document_query_similarity(), test_embeddings_are_normalized()

### Community 10 - "Query Scope Routing"
Cohesion: 0.32
Nodes (7): Enum, QueryScope, Layer 3 — Query router.  Classifies each query into one of three scopes:   expli, Determine which books to search for this query.     Priority: explicit > cross-b, resolve_scope(), ScopeType, str

### Community 11 - "Extractor Test Suite"
Cohesion: 0.29
Nodes (1): Tests for the file extractor (no real files needed — mocked).

### Community 12 - "Database Session Management"
Cohesion: 0.33
Nodes (1): Database session factories — sync (for worker) and async (for CLI / API).

### Community 13 - "Configuration & Settings"
Cohesion: 0.5
Nodes (4): BaseSettings, get_settings(), Central settings — loaded from environment / .env file., Settings

### Community 14 - "Cross-Encoder Reranking"
Cohesion: 0.5
Nodes (4): _get_cross_encoder(), Layer 3 — Cross-encoder reranker.  Takes the top-K bi-encoder candidates and re-, Re-rank candidate parent chunks using a cross-encoder.     Returns up to top_k c, rerank()

### Community 15 - "Query Rewriting"
Cohesion: 0.5
Nodes (3): Layer 3 — Query rewriter.  Uses Gemma 4 (via Ollama) to: 1. Expand pronouns / im, Rewrite a query using recent conversation context.     Falls back to the origina, rewrite_query()

### Community 16 - "Database Schema Migration"
Cohesion: 0.5
Nodes (1): Initial schema  Revision ID: 001 Revises: Create Date: 2026-04-06

### Community 17 - "Alembic Migration Environment"
Cohesion: 0.67
Nodes (0): 

### Community 18 - "Package Initialization"
Cohesion: 1.0
Nodes (1): BookRAG — open-source book Q&A system.

### Community 19 - "Sample Prompts"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **34 isolated node(s):** `Tests for the embedding engine.`, `nomic-embed-text-v1.5 with normalize=True should return unit vectors.`, `A query and a matching document should have high cosine similarity.`, `Central settings — loaded from environment / .env file.`, `BookRAG — open-source book Q&A system.` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Package Initialization`** (2 nodes): `__init__.py`, `BookRAG — open-source book Q&A system.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sample Prompts`** (1 nodes): `prompt-sample.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Book` connect `ORM Models & Quality Grounding` to `Query Scope Routing`, `CLI & Command Interface`, `Router Test Suite`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `Session` connect `CLI & Command Interface` to `ORM Models & Quality Grounding`, `Answer Generation`, `Session & Conversation Memory`, `Router Test Suite`, `Query Scope Routing`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `RetrievedChunk` connect `ORM Models & Quality Grounding` to `Answer Generation`, `Cross-Encoder Reranking`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `Book` (e.g. with `Tests for the query router.` and `Layer 1 — Background ingestion worker.  Polls ingestion_jobs for queued jobs and`) actually correct?**
  _`Book` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `Session` (e.g. with `Tests for the query router.` and `Layer 3 / 5 — Session memory.  Loads the rolling window of recent messages from`) actually correct?**
  _`Session` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `IngestionJob` (e.g. with `Layer 1 — Background ingestion worker.  Polls ingestion_jobs for queued jobs and` and `Extract and persist chapters. Returns list of Chapter ORM objects.`) actually correct?**
  _`IngestionJob` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `RetrievedChunk` (e.g. with `search()` and `Layer 5 — Grounding check.  Sends a tiny Gemma 4 call to verify the answer is su`) actually correct?**
  _`RetrievedChunk` has 16 INFERRED edges - model-reasoned connections that need verification._