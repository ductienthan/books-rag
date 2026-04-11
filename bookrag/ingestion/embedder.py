"""
Layer 1 — Embedding engine.

Uses nomic-embed-text-v1.5 via sentence-transformers (local, no Ollama needed).
Supports batched embedding with progress tracking and resumability.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from bookrag.config import get_settings

log = logging.getLogger(__name__)
_settings = get_settings()

# ── Model singleton (loaded once per process) ─────────────────────────────────
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", _settings.embed_model)
        _model = SentenceTransformer(
            _settings.embed_model,
            trust_remote_code=True,   # required for nomic-embed-text-v1.5
        )
        log.info("Embedding model loaded (dim=%d)", _settings.embed_dim)
    return _model


# ── nomic-embed-text-v1.5 uses task prefixes ──────────────────────────────────
# "search_document:" for passages being stored
# "search_query:"    for the user's question at query time

def embed_documents(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """
    Embed a list of document passages (child chunks / summaries).
    Returns a list of float vectors (len = embed_dim).
    """
    model = _get_model()
    bs = batch_size or _settings.embed_batch_size
    prefixed = [f"search_document: {t}" for t in texts]

    log.debug("Embedding %d documents in batches of %d", len(prefixed), bs)
    vecs = model.encode(
        prefixed,
        batch_size=bs,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    """
    Embed a single user query with the correct task prefix.
    """
    model = _get_model()
    prefixed = f"search_query: {text}"
    vec = model.encode(
        [prefixed],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec[0].tolist()


def embed_batch(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """Alias for embed_documents — preferred name in the worker."""
    return embed_documents(texts, batch_size)
