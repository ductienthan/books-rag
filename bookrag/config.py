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

  # ── Agent / LLM ───────────────────────────────────────────────────────────
  llm_max_tokens: int = 512
  session_memory_window: int = 6
  # Grounding check adds a full Ollama round-trip (~10-20s) after generation.
  # Disabled by default for speed; enable when answer faithfulness is critical.
  enable_grounding_check: bool = False

  # ── Logging ───────────────────────────────────────────────────────────────
  log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
  return Settings()
