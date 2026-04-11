"""
Layer 3 — ReAct agent loop.

Orchestrates:
  1. Scope resolution (query router)
  2. Query rewriting (LLM-based expansion + decomposition)
  3. Scoped vector search + parent expansion
  4. Cross-encoder reranking
  5. Answer generation (Gemma 4)
  6. Session memory persistence
  7. Grounding check (Layer 5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session as DBSession

from bookrag.agent.generator import GeneratedAnswer, generate_answer
from bookrag.agent.memory import get_history, save_message
from bookrag.config import get_settings
from bookrag.db.models import Session as DBSessionModel
from bookrag.quality.grounding import check_grounding
from bookrag.retrieval.reranker import rerank
from bookrag.retrieval.router import resolve_scope
from bookrag.retrieval.rewriter import rewrite_query
from bookrag.retrieval.searcher import RetrievedChunk, search

log = logging.getLogger(__name__)
_settings = get_settings()


@dataclass
class AgentResponse:
    answer: str
    sources: list[dict]      # [{book, chapter, pages}, ...]
    session_id: str
    is_grounded: bool
    scope_type: str


def ask(
    question: str,
    session_id: str,
    db: DBSession,
) -> AgentResponse:
    """
    Main entry point for the agent loop. Takes a question + session ID,
    runs the full ReAct pipeline, persists messages, and returns the response.
    """
    # ── 0. Load session ───────────────────────────────────────────────────────
    session = db.query(DBSessionModel).filter(DBSessionModel.id == session_id).one()
    history = get_history(session_id, db)

    # ── 1. Resolve scope ──────────────────────────────────────────────────────
    scope = resolve_scope(question, session, db)
    log.info("Scope: %s → %d books", scope.scope_type, len(scope.book_ids))

    if not scope.book_ids:
        answer_text = "No books have been ingested yet. Please add a book with `bookrag add <file>`."
        _persist(question, answer_text, session_id, db, [], [])
        return AgentResponse(
            answer=answer_text,
            sources=[],
            session_id=session_id,
            is_grounded=False,
            scope_type=scope.scope_type,
        )

    # ── 2. Rewrite queries ────────────────────────────────────────────────────
    queries = rewrite_query(question, history)
    log.info("Rewritten queries: %s", queries)

    # ── 3. Vector search ──────────────────────────────────────────────────────
    candidates: list[RetrievedChunk] = search(queries, scope.book_ids, db, top_k=_settings.retrieval_top_k)
    log.info("Retrieved %d candidate chunks", len(candidates))

    # ── 4. Detect low confidence ──────────────────────────────────────────────
    low_confidence = not candidates or (
        candidates and max(c.score for c in candidates) < _settings.cosine_threshold
    )

    # ── 5. Rerank ─────────────────────────────────────────────────────────────
    top_chunks = rerank(question, candidates, top_k=_settings.rerank_top_k) if candidates else []
    log.info("Reranked to %d chunks", len(top_chunks))

    # ── 6. Generate answer ────────────────────────────────────────────────────
    result: GeneratedAnswer = generate_answer(
        question=question,
        chunks=top_chunks,
        history=history,
        low_confidence=low_confidence,
    )

    # ── 7. Grounding check ────────────────────────────────────────────────────
    is_grounded = True
    if top_chunks and not low_confidence:
        is_grounded = check_grounding(question, result.answer, top_chunks)

    # ── 8. Persist messages ───────────────────────────────────────────────────
    _persist(question, result.answer, session_id, db, result.book_ids, result.chunk_ids)

    # ── 9. Build sources list ─────────────────────────────────────────────────
    sources = _build_sources(top_chunks)

    return AgentResponse(
        answer=result.answer,
        sources=sources,
        session_id=session_id,
        is_grounded=is_grounded,
        scope_type=scope.scope_type,
    )


def _persist(
    question: str,
    answer: str,
    session_id: str,
    db: DBSession,
    book_ids: list[str],
    chunk_ids: list[str],
) -> None:
    save_message(session_id, "user", question, db)
    save_message(session_id, "assistant", answer, db, book_ids=book_ids, chunk_ids=chunk_ids)
    db.commit()


def _build_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    seen = set()
    sources = []
    for c in chunks:
        key = (c.book_id, c.chapter_index)
        if key not in seen:
            seen.add(key)
            sources.append({
                "book": c.book_title,
                "chapter": c.chapter_title or f"Chapter {c.chapter_index + 1}",
                "pages": f"pp. {c.page_start}–{c.page_end}" if c.page_start else "N/A",
                "score": round(c.score, 3),
            })
    return sources
