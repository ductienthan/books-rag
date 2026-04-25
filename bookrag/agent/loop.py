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
  8. Phase 3.1: Query result caching
  9. Phase 3.3: Streaming responses
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from sqlalchemy.orm import Session as DBSession

from bookrag.agent.generator import GeneratedAnswer, generate_answer, generate_answer_stream
from bookrag.agent.memory import get_history, save_message
from bookrag.config import get_settings
from bookrag.db.models import Book, Chapter, ChapterSummary, Session as DBSessionModel
from bookrag.quality.grounding import check_grounding
from bookrag.retrieval.cache import CacheManager
from bookrag.retrieval.reranker import rerank
from bookrag.retrieval.router import resolve_scope
from bookrag.retrieval.rewriter import rewrite_query
from bookrag.retrieval.searcher import RetrievedChunk, search, search_chapter_summaries, extract_page_numbers

log = logging.getLogger(__name__)
_settings = get_settings()

# Phase 3: Initialize cache manager (singleton pattern)
_cache_manager: CacheManager | None = None

def _get_cache_manager() -> CacheManager:
    """Get or create cache manager singleton."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(
            cache_dir=Path(_settings.cache_dir),
            answer_ttl=_settings.answer_cache_ttl,
            enable_embedding_cache=_settings.enable_embedding_cache,
            enable_answer_cache=_settings.enable_answer_cache,
        )
    return _cache_manager

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

# Patterns for simple queries that don't need LLM rewriting
# Page-based queries (already extracted by extract_page_numbers)
_PAGE_RE = re.compile(r'\b(?:page|pages|pg|p\.?)\s*\d+', re.IGNORECASE)
# Simple question words at the start
_SIMPLE_QUESTION_RE = re.compile(r'^\s*(?:who|what|when|where|which|how\s+many)\b', re.IGNORECASE)
# Very short queries (5 words or less) - likely simple lookups
_SHORT_QUERY_THRESHOLD = 5

# Phase 2: Patterns for detailed questions that need expanded context
_DETAILED_PATTERNS = [
    re.compile(r'\b(explain|describe|detail|elaborate|comprehensive|thorough)\b', re.IGNORECASE),
    re.compile(r'\b(all|every|entire|complete|full|everything)\b', re.IGNORECASE),
    re.compile(r'\b(compare|contrast|analyze|analyse|evaluate|assess|examine)\b', re.IGNORECASE),
    re.compile(r'\b(how|why)\b.*\b(work|function|develop|relate|differ)\b', re.IGNORECASE),
    re.compile(r'\b(relationship|difference|similarity|connection)\b', re.IGNORECASE),
    re.compile(r'\b(process|step|stage|phase)\b.*\b(work|involve|include)\b', re.IGNORECASE),
]
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


def _is_simple_query(question: str) -> bool:
    """
    Detect if a query is simple enough to skip LLM rewriting.
    Simple queries include:
    - Page-based lookups ("page 81 and 82")
    - Short factual questions ("Who is the author?")
    - Very short queries (5 words or less)
    """
    # Page-based queries
    if _PAGE_RE.search(question) or extract_page_numbers(question):
        return True

    # Simple question patterns
    if _SIMPLE_QUESTION_RE.match(question):
        word_count = len(question.split())
        if word_count <= 10:  # Simple questions should be concise
            return True

    # Very short queries
    if len(question.split()) <= _SHORT_QUERY_THRESHOLD:
        return True

    return False


def _needs_expanded_context(question: str) -> bool:
    """
    Phase 2: Detect if a question needs expanded context (more chunks).

    Detailed/analytical questions benefit from more context chunks:
    - Explanatory questions ("explain...", "describe in detail...")
    - Comprehensive questions ("all", "every", "complete...")
    - Analytical questions ("compare", "contrast", "analyze...")
    - How/why questions about processes

    Returns:
        True if the question appears to need comprehensive context
    """
    # Check against detailed question patterns
    for pattern in _DETAILED_PATTERNS:
        if pattern.search(question):
            return True

    # Long questions (>15 words) are likely complex
    if len(question.split()) > 15:
        return True

    return False


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
    chunks: list[RetrievedChunk]  # Actual retrieved chunks with text
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

    Phase 3: Now includes caching for faster repeated queries.
    """
    start_time = time.time()

    # ── 0. Load session ───────────────────────────────────────────────────────
    session = db.query(DBSessionModel).filter(DBSessionModel.id == session_id).one()
    history = get_history(session_id, db)

    # ── 1. Resolve scope ──────────────────────────────────────────────────────
    scope = resolve_scope(question, session, db)
    log.info("Scope: %s → %d books", scope.scope_type, len(scope.book_ids))

    # ── Phase 3: Check answer cache ───────────────────────────────────────────
    cache_mgr = _get_cache_manager()
    cached_answer = cache_mgr.get_answer(question, scope.book_ids)

    if cached_answer:
        elapsed_ms = (time.time() - start_time) * 1000
        log.info(
            f"✓ Answer cache HIT! Returning cached result (saved ~{cached_answer.generation_time_ms:.0f}ms, "
            f"actual: {elapsed_ms:.0f}ms)"
        )

        # Deserialize chunks back to RetrievedChunk objects
        chunks = [
            RetrievedChunk(
                child_id=c["child_id"],
                parent_id=c["parent_id"],
                book_id=c["book_id"],
                book_title=c["book_title"],
                chapter_index=c["chapter_index"],
                chapter_title=c["chapter_title"],
                page_start=c["page_start"],
                page_end=c["page_end"],
                child_text=c["child_text"],
                parent_text=c["parent_text"],
                score=c["score"],
            )
            for c in cached_answer.chunks
        ]

        # Don't persist cached answers (already persisted when first generated)
        return AgentResponse(
            answer=cached_answer.answer,
            sources=_build_sources(chunks),
            chunks=chunks,
            session_id=session_id,
            is_grounded=True,  # Assume cached answers were grounded
            scope_type=scope.scope_type,
        )

    if not scope.book_ids:
        answer_text = "No books have been ingested yet. Please add a book with `bookrag add <file>`."
        _persist(question, answer_text, session_id, db, [], [])
        return AgentResponse(
            answer=answer_text,
            sources=[],
            chunks=[],
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
                chunks=top_chunks,
                session_id=session_id,
                is_grounded=True,
                scope_type=scope.scope_type,
            )
        log.info("No chapter summaries found — falling back to vector search")

    # ── 3. Rewrite queries (skip for simple queries to save time) ─────────────
    if _is_simple_query(question):
        log.info("Simple query detected — skipping LLM rewriting")
        queries = [question]
    else:
        log.info("Complex query — using LLM rewriting")
        queries = rewrite_query(question, history)
    log.info("Queries for search: %s", queries)

    # ── 4. Phase 2: Adaptive chunk retrieval ──────────────────────────────────
    # Determine how many chunks to retrieve based on question complexity
    if _settings.adaptive_context and _needs_expanded_context(question):
        retrieval_k = _settings.max_retrieval_chunks
        rerank_k = _settings.max_retrieval_chunks  # Keep all after reranking, let generator trim
        log.info("Detected detailed question — using expanded context (up to %d chunks)", retrieval_k)
    else:
        retrieval_k = _settings.retrieval_top_k
        rerank_k = _settings.rerank_top_k
        log.info("Standard retrieval — using %d chunks", rerank_k)

    # ── 5. Vector search ──────────────────────────────────────────────────────
    candidates: list[RetrievedChunk] = search(queries, scope.book_ids, db, top_k=retrieval_k)
    log.info("Retrieved %d candidate chunks", len(candidates))

    # ── 6. Detect low confidence ──────────────────────────────────────────────
    low_confidence = not candidates or (
        candidates and max(c.score for c in candidates) < _settings.cosine_threshold
    )

    # ── 7. Rerank ─────────────────────────────────────────────────────────────
    top_chunks = rerank(question, candidates, top_k=rerank_k) if candidates else []
    log.info("Reranked to %d chunks", len(top_chunks))

    # ── 8. Generate answer ────────────────────────────────────────────────────
    # Phase 2: Generator will trim chunks to fit max_context_tokens budget
    result: GeneratedAnswer = generate_answer(
        question=question,
        chunks=top_chunks,
        history=history,
        low_confidence=low_confidence,
    )

    # ── 9. Grounding check (optional) ─────────────────────────────────────────
    # Disabled by default (enable_grounding_check=False) because it adds a
    # full Ollama round-trip (~10-20s) after generation.  Enable via .env or
    # config when answer faithfulness verification is worth the latency cost.
    is_grounded = True
    if _settings.enable_grounding_check and top_chunks and not low_confidence:
        is_grounded = check_grounding(question, result.answer, top_chunks)

    # ── 10. Persist messages ──────────────────────────────────────────────────
    _persist(question, result.answer, session_id, db, result.book_ids, result.chunk_ids)

    # ── 11. Build sources list ────────────────────────────────────────────────
    sources = _build_sources(top_chunks)

    # ── Phase 3: Store in answer cache ────────────────────────────────────────
    total_time_ms = (time.time() - start_time) * 1000

    # Determine query type for analytics
    if extract_page_numbers(question):
        query_type = "page"
    elif _needs_expanded_context(question):
        query_type = "detailed"
    else:
        query_type = "simple"

    # Cache the result
    cache_mgr.set_answer(
        query=question,
        book_ids=scope.book_ids,
        answer=result.answer,
        chunks=top_chunks,
        query_type=query_type,
        generation_time_ms=total_time_ms
    )
    log.info(f"Cached answer for future queries (query_type={query_type}, time={total_time_ms:.0f}ms)")

    return AgentResponse(
        answer=result.answer,
        sources=sources,
        chunks=top_chunks,
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


@dataclass
class StreamUpdate:
    """Update message for streaming responses."""
    type: str  # "progress" | "chunk" | "complete"
    data: str | AgentResponse  # Progress message, answer chunk, or final response


def ask_stream(
    question: str,
    session_id: str,
    db: DBSession,
) -> Iterator[StreamUpdate]:
    """
    Streaming version of ask() that yields progress updates and answer chunks.

    Phase 3.3: Provides real-time feedback during answer generation.

    Yields:
        StreamUpdate objects with type and data:
        - "progress": Phase completion messages (str)
        - "chunk": Answer text chunks as generated (str)
        - "complete": Final AgentResponse with full answer and metadata

    Usage:
        for update in ask_stream(question, session_id, db):
            if update.type == "progress":
                print(f"Status: {update.data}")
            elif update.type == "chunk":
                print(update.data, end='', flush=True)
            elif update.type == "complete":
                response = update.data
                # Process final response
    """
    start_time = time.time()

    # ── 0. Load session ───────────────────────────────────────────────────────
    yield StreamUpdate("progress", "Loading session...")
    session = db.query(DBSessionModel).filter(DBSessionModel.id == session_id).one()
    history = get_history(session_id, db)

    # ── 1. Resolve scope ──────────────────────────────────────────────────────
    yield StreamUpdate("progress", "Resolving scope...")
    scope = resolve_scope(question, session, db)
    log.info("Scope: %s → %d books", scope.scope_type, len(scope.book_ids))

    # ── Phase 3: Check answer cache ───────────────────────────────────────────
    yield StreamUpdate("progress", "Checking cache...")
    cache_mgr = _get_cache_manager()
    cached_answer = cache_mgr.get_answer(question, scope.book_ids)

    if cached_answer:
        elapsed_ms = (time.time() - start_time) * 1000
        log.info(f"✓ Answer cache HIT! (saved ~{cached_answer.generation_time_ms:.0f}ms)")

        # Deserialize chunks
        chunks = [
            RetrievedChunk(
                child_id=c["child_id"],
                parent_id=c["parent_id"],
                book_id=c["book_id"],
                book_title=c["book_title"],
                chapter_index=c["chapter_index"],
                chapter_title=c["chapter_title"],
                page_start=c["page_start"],
                page_end=c["page_end"],
                child_text=c["child_text"],
                parent_text=c["parent_text"],
                score=c["score"],
            )
            for c in cached_answer.chunks
        ]

        # Return cached answer immediately
        response = AgentResponse(
            answer=cached_answer.answer,
            sources=_build_sources(chunks),
            chunks=chunks,
            session_id=session_id,
            is_grounded=True,
            scope_type=scope.scope_type,
        )
        yield StreamUpdate("complete", response)
        return

    if not scope.book_ids:
        answer_text = "No books have been ingested yet. Please add a book with `bookrag add <file>`."
        _persist(question, answer_text, session_id, db, [], [])
        response = AgentResponse(
            answer=answer_text,
            sources=[],
            chunks=[],
            session_id=session_id,
            is_grounded=False,
            scope_type=scope.scope_type,
        )
        yield StreamUpdate("complete", response)
        return

    # ── 2. Route: chapter summary vs. regular search ──────────────────────────
    if _is_chapter_summary_request(question):
        yield StreamUpdate("progress", "Fetching chapter summaries...")
        top_chunks = _chapter_summaries_as_chunks(question, scope.book_ids, db)
        if top_chunks:
            answer_parts = []
            for c in top_chunks:
                label = c.chapter_title or f"Chapter {c.chapter_index + 1}"
                answer_parts.append(f"## {label}\n\n{c.parent_text}")
            answer_text = "\n\n---\n\n".join(answer_parts)
            book_ids_used = list({c.book_id for c in top_chunks})
            chunk_ids_used = [c.child_id for c in top_chunks]
            _persist(question, answer_text, session_id, db, book_ids_used, chunk_ids_used)
            response = AgentResponse(
                answer=answer_text,
                sources=_build_sources(top_chunks),
                chunks=top_chunks,
                session_id=session_id,
                is_grounded=True,
                scope_type=scope.scope_type,
            )
            yield StreamUpdate("complete", response)
            return

    # ── 3. Rewrite queries ────────────────────────────────────────────────────
    if _is_simple_query(question):
        queries = [question]
    else:
        yield StreamUpdate("progress", "Expanding query...")
        queries = rewrite_query(question, history)
    log.info("Queries for search: %s", queries)

    # ── 4. Adaptive chunk retrieval ───────────────────────────────────────────
    if _settings.adaptive_context and _needs_expanded_context(question):
        retrieval_k = _settings.max_retrieval_chunks
        rerank_k = _settings.max_retrieval_chunks
    else:
        retrieval_k = _settings.retrieval_top_k
        rerank_k = _settings.rerank_top_k

    # ── 5. Vector search ──────────────────────────────────────────────────────
    yield StreamUpdate("progress", "Searching documents...")
    candidates = search(queries, scope.book_ids, db, top_k=retrieval_k)
    log.info("Retrieved %d candidate chunks", len(candidates))

    # ── 6. Detect low confidence ──────────────────────────────────────────────
    low_confidence = not candidates or (
        candidates and max(c.score for c in candidates) < _settings.cosine_threshold
    )

    # ── 7. Rerank ─────────────────────────────────────────────────────────────
    yield StreamUpdate("progress", "Reranking results...")
    top_chunks = rerank(question, candidates, top_k=rerank_k) if candidates else []
    log.info("Reranked to %d chunks", len(top_chunks))

    # ── 8. Generate answer with streaming ─────────────────────────────────────
    yield StreamUpdate("progress", "Generating answer...")

    answer_parts = []
    for chunk in generate_answer_stream(question, top_chunks, history, low_confidence):
        answer_parts.append(chunk)
        yield StreamUpdate("chunk", chunk)

    # Reconstruct complete answer
    answer_text = "".join(answer_parts)

    # ── 9. Grounding check (optional) ─────────────────────────────────────────
    is_grounded = True
    if _settings.enable_grounding_check and top_chunks and not low_confidence:
        yield StreamUpdate("progress", "Checking grounding...")
        is_grounded = check_grounding(question, answer_text, top_chunks)

    # ── 10. Persist messages ──────────────────────────────────────────────────
    book_ids_used = list({c.book_id for c in top_chunks})
    chunk_ids_used = [c.child_id for c in top_chunks]
    _persist(question, answer_text, session_id, db, book_ids_used, chunk_ids_used)

    # ── 11. Build sources list ────────────────────────────────────────────────
    sources = _build_sources(top_chunks)

    # ── Phase 3: Store in answer cache ────────────────────────────────────────
    total_time_ms = (time.time() - start_time) * 1000

    if extract_page_numbers(question):
        query_type = "page"
    elif _needs_expanded_context(question):
        query_type = "detailed"
    else:
        query_type = "simple"

    cache_mgr.set_answer(
        query=question,
        book_ids=scope.book_ids,
        answer=answer_text,
        chunks=top_chunks,
        query_type=query_type,
        generation_time_ms=total_time_ms
    )

    # ── Final response ────────────────────────────────────────────────────────
    response = AgentResponse(
        answer=answer_text,
        sources=sources,
        chunks=top_chunks,
        session_id=session_id,
        is_grounded=is_grounded,
        scope_type=scope.scope_type,
    )

    yield StreamUpdate("complete", response)
