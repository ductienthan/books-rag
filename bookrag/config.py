"""Central settings — loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  model_config = SettingsConfigDict(
      env_file=".env", env_file_encoding="utf-8", extra="ignore")

  # ── Database ──────────────────────────────────────────────────────────────
  postgres_url: str = "postgresql+psycopg://bookrag:bookrag@localhost:5432/bookrag"
  async_postgres_url: str = "postgresql+asyncpg://bookrag:bookrag@localhost:5432/bookrag"

  # ── Ollama ────────────────────────────────────────────────────────────────
  ollama_host: str = "http://localhost:11434"
  ollama_llm_model: str = "gemma4:e4b"

  # ── Embedding model ───────────────────────────────────────────────────────
  embed_model: str = "nomic-ai/nomic-embed-text-v1.5"
  embed_batch_size: int = 32
  embed_dim: int = 768

  # ── pgvector HNSW ─────────────────────────────────────────────────────────
  hnsw_ef_search: int = 40

  # ── Ingestion ─────────────────────────────────────────────────────────────
  data_books_dir: str = "./data/books"
  ocr_char_threshold: int = 100
  max_books: int = 50
  # Chapters (content type) with fewer chars than this are merged forward into
  # the next chapter during the post-extraction consolidation pass.
  min_chapter_chars: int = 400
  # When True, only the shallowest TOC level is used as chapter boundaries.
  # Keep False (default): the front/back matter consolidation already handles
  # tiny stubs, and level filtering would collapse individual chapters into
  # their parent Parts, producing enormous 50-100 page "chapters".
  toc_top_level_only: bool = False

  # ── Chunking ──────────────────────────────────────────────────────────────
  child_chunk_min_tokens: int = 80
  child_chunk_max_tokens: int = 120
  child_chunk_overlap_tokens: int = 20
  parent_chunk_min_tokens: int = 300
  parent_chunk_max_tokens: int = 400
  parent_chunk_overlap_tokens: int = 40
  summary_min_tokens: int = 500
  summary_max_tokens: int = 800

  # ── Retrieval ─────────────────────────────────────────────────────────────
  # Reduced from 20 → 10: the cross-encoder reranker runs on every candidate,
  # so halving the pool roughly halves reranking time with negligible quality loss.
  retrieval_top_k: int = 10
  rerank_top_k: int = 5
  cosine_threshold: float = 0.35

  # ── Phase 2: Context Expansion ────────────────────────────────────────────
  # Maximum tokens to include in LLM context (up from ~1,750 in Phase 1)
  max_context_tokens: int = 4000
  # Maximum chunks to retrieve for detailed questions (before reranking)
  max_retrieval_chunks: int = 12
  # Minimum chunks for simple queries
  min_retrieval_chunks: int = 5
  # Enable adaptive context sizing based on question complexity
  adaptive_context: bool = True

  # ── Phase 3: Caching ──────────────────────────────────────────────────────
  # Enable answer caching (reduces repeated query time from 90-150s to <5s)
  enable_answer_cache: bool = True
  # Enable embedding caching (saves 2-5s per query)
  enable_embedding_cache: bool = True
  # Cache directory (relative to project root)
  cache_dir: str = ".cache"
  # Answer cache TTL in seconds (default: 1 hour)
  answer_cache_ttl: int = 3600
  # Maximum cache size in MB (0 = unlimited)
  max_cache_size_mb: int = 500

  # ── Phase 3: BM25 Hybrid Search ───────────────────────────────────────────
  # Enable BM25 keyword-based search alongside vector search
  enable_bm25: bool = True
  # BM25 index directory
  bm25_index_dir: str = ".indexes/bm25"
  # Weight for BM25 scores in hybrid search (0.0-1.0)
  bm25_weight: float = 0.3
  # Weight for vector search scores in hybrid search (0.0-1.0)
  vector_weight: float = 0.7
  # Number of candidates to retrieve from BM25 (before fusion)
  bm25_top_k: int = 40

  # ── Agent / LLM ───────────────────────────────────────────────────────────
  llm_max_tokens: int = 512
  session_memory_window: int = 6
  # Grounding check adds a full Ollama round-trip (~10-20s) after generation.
  # Disabled by default for speed; enable when answer faithfulness is critical.
  enable_grounding_check: bool = False

  # ── Phase 3.3: Streaming ──────────────────────────────────────────────────
  # Enable streaming LLM output (better UX, see progress in real-time)
  enable_streaming: bool = True

  # ── Logging ───────────────────────────────────────────────────────────────
  log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
  return Settings()
