"""
Layer 3 — Cross-encoder reranker.

Takes the top-K bi-encoder candidates and re-scores them with a
cross-encoder (ms-marco-MiniLM-L-6-v2) for higher precision.
"""
from __future__ import annotations

import logging
from typing import Sequence

from bookrag.config import get_settings
from bookrag.retrieval.searcher import RetrievedChunk

log = logging.getLogger(__name__)
_settings = get_settings()

# ── Cross-encoder singleton ───────────────────────────────────────────────────
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        log.info("Loading cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        log.info("Cross-encoder loaded")
    return _cross_encoder


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Re-rank candidate parent chunks using a cross-encoder.
    Returns up to top_k chunks sorted by rerank score (descending).
    """
    k = top_k or _settings.rerank_top_k
    if not candidates:
        return []
    if len(candidates) <= k:
        # Not enough to bother reranking — return as-is sorted by bi-encoder score
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:k]

    cross_encoder = _get_cross_encoder()
    pairs = [(query, c.parent_text) for c in candidates]

    try:
        scores = cross_encoder.predict(pairs)
    except Exception as exc:
        log.warning("Reranking failed (falling back to bi-encoder order): %s", exc)
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:k]

    # Attach rerank scores and sort
    for chunk, score in zip(candidates, scores):
        chunk.score = float(score)   # override with cross-encoder score

    reranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    return reranked[:k]
