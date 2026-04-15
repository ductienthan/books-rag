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
import re
from dataclasses import dataclass

from sqlalchemy.orm import Session as DBSession

from bookrag.agent.generator import GeneratedAnswer, generate_answer
from bookrag.agent.memory import get_history, save_message
from bookrag.config import get_settings
from bookrag.db.models import Book, Chapter, ChapterSummary, Session as DBSessionModel
from bookrag.quality.grounding import check_grounding
from bookrag.retrieval.reranker import rerank
from bookrag.retrieval.router import resolve_scope
from bookrag.retrieval.rewriter import rewrite_query
from bookrag.retrieval.searcher import RetrievedChunk, search, search_chapter_summaries

log = logging.getLogger(__name__)
_settings = get_settings()

# Typo-tolerant: sum[ma]+r covers "summary", "sumary", "summry", etc.
_SUMMARY_RE = re.compile(
    r'\b(sum[ma]+r\w*|overview|outline|what.+cover|what.+about|describe)\b',
    re.IGNORECASE,
)
_CHAPTER_RE = re.compile(r'\b(chapter|part|section)\b', re.IGNORECASE)
_CHAPTER_NUM_RE = re.compile(
    r'\b(?:chapter|part|section)\s+'
    r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|'
    r'first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b',
    re.IGNORECASE,
)
_ORDINAL_MAP = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
}

# Chapters with fewer raw characters than this are front/title matter, not real chapters
_MIN_CHAPTER_RAW_CHARS = 500

# Titles that identify front/back matter — excluded from numbered chapter lookups
_FRONT_BACK_MATTER_RE = re.compile(
    r'^\s*(series\s+page|half\s+title|title\s+page|copyright|dedication|epigraph|'
    r'table\s+of\s+contents|contents|foreword|acknowledgments?|about\s+the\s+author|'
    r'index|bibliography|back\s+cover|colophon|permissions?)\s*$',
    re.IGNORECASE,
)


def _is_chapter_summary_request(question: str) -> bool:
    return bool(_SUMMARY_RE.search(question) and _CHAPTER_RE.search(question))


def _extract_chapter_number(question: str) -> int | None:
    """Return the 1-based chapter number from the question, or None."""
    m = _CHAPTER_NUM_RE.search(question)
    if not m:
        return None
    tok = m.group(1).lower()
    return int(tok) if tok.isdigit() else _ORDINAL_MAP.get(tok)


def _chapter_summaries_as_chunks(
    question: str,
    book_ids: list[str],
    db: DBSession,
) -> list[RetrievedChunk]:
    """Fetch pre-generated chapter summaries and return them as RetrievedChunk objects.

    When a specific chapter number is mentioned, does a direct DB lookup by
    chapter_index (reliable). Otherwise falls back to semantic search.
    """
    chapter_num = _extract_chapter_number(question)

    if chapter_num is not None:
        # Find the Nth real chapter, skipping front/back matter.
        # A chapter is "real" if its summary has enough content (not a cover/title page).
        all_summaries = (
            db.query(ChapterSummary)
            .join(Chapter, ChapterSummary.chapter_id == Chapter.id)
            .filter(ChapterSummary.book_id.in_(book_ids))
            .order_by(Chapter.chapter_index)
            .all()
        )
        real_chapters = [
            cs for cs in all_summaries
            if len(cs.chapter.raw_text.strip()) >= _MIN_CHAPTER_RAW_CHARS
            and not (cs.chapter.title and _FRONT_BACK_MATTER_RE.match(cs.chapter.title))
        ]
        # chapter_num is 1-based
        summaries = [real_chapters[chapter_num - 1]] if chapter_num <= len(real_chapters) else []
    else:
        summaries = search_chapter_summaries(question, book_ids, db, top_k=5)

    if not summaries:
        return []

    books_map = {b.id: b for b in db.query(Book).filter(Book.id.in_(book_ids)).all()}
    chunks = []
    for cs in summaries:
        chapter: Chapter = cs.chapter
        book = books_map.get(cs.book_id)
        if not chapter or not book:
            continue
        chunks.append(RetrievedChunk(
            child_id=cs.id,
            parent_id=cs.chapter_id,
            book_id=cs.book_id,
            book_title=book.title,
            chapter_index=chapter.chapter_index,
            chapter_title=chapter.title,
            page_start=chapter.page_start,
            page_end=chapter.page_end,
            child_text=cs.summary_text,
            parent_text=cs.summary_text,
            score=1.0,
        ))
    return chunks


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

    # ── 2. Route: chapter summary vs. regular search ──────────────────────────
    if _is_chapter_summary_request(question):
        log.info("Detected chapter summary request — using pre-generated summaries")
        top_chunks = _chapter_summaries_as_chunks(question, scope.book_ids, db)
        if top_chunks:
            log.info("Returning %d pre-generated chapter summaries directly", len(top_chunks))
            # Return the pre-generated summary text directly — no LLM needed.
            answer_parts = []
            for c in top_chunks:
                label = c.chapter_title or f"Chapter {c.chapter_index + 1}"
                answer_parts.append(f"## {label}\n\n{c.parent_text}")
            answer_text = "\n\n---\n\n".join(answer_parts)
            book_ids_used = list({c.book_id for c in top_chunks})
            chunk_ids_used = [c.child_id for c in top_chunks]
            _persist(question, answer_text, session_id, db, book_ids_used, chunk_ids_used)
            return AgentResponse(
                answer=answer_text,
                sources=_build_sources(top_chunks),
                session_id=session_id,
                is_grounded=True,
                scope_type=scope.scope_type,
            )
        log.info("No chapter summaries found — falling back to vector search")

    # ── 3. Rewrite queries ────────────────────────────────────────────────────
    queries = rewrite_query(question, history)
    log.info("Rewritten queries: %s", queries)

    # ── 4. Vector search ──────────────────────────────────────────────────────
    candidates: list[RetrievedChunk] = search(queries, scope.book_ids, db, top_k=_settings.retrieval_top_k)
    log.info("Retrieved %d candidate chunks", len(candidates))

    # ── 5. Detect low confidence ──────────────────────────────────────────────
    low_confidence = not candidates or (
        candidates and max(c.score for c in candidates) < _settings.cosine_threshold
    )

    # ── 6. Rerank ─────────────────────────────────────────────────────────────
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
