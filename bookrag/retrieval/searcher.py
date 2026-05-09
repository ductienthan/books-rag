"""
Layer 3 — Scoped vector search with optional BM25 hybrid retrieval.

Searches child_chunks using pgvector HNSW with a book_id filter,
then fetches the corresponding parent chunks for richer context.

Phase 3: Optionally combines vector search with BM25 keyword search
using Reciprocal Rank Fusion (RRF) for improved retrieval quality.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from bookrag.config import get_settings
from bookrag.db.models import Chapter, ChapterSummary, ChildChunk, ParentChunk, Book
from bookrag.ingestion.embedder import embed_query

log = logging.getLogger(__name__)
_settings = get_settings()

# Pattern to extract page numbers from queries
# Matches: "page 81", "pages 81-82", "p. 81 and 82", "pg 81 to 82", "p.42", etc.
_PAGE_PATTERN = re.compile(
    r'\b(?:page|pages|pg|p\.?)\s*(\d+)',
    re.IGNORECASE
)


@dataclass
class RetrievedChunk:
    child_id: str
    parent_id: str
    book_id: str
    book_title: str
    chapter_index: int
    chapter_title: str | None
    page_start: int | None
    page_end: int | None
    child_text: str
    parent_text: str
    score: float       # cosine similarity (0–1)


def extract_page_numbers(query: str) -> list[int]:
    """
    Extract page numbers from queries like:
    - "page 81 and 82"
    - "pages 81-82"
    - "p. 81 to 82"
    - "give me page 81, 82, and 83"

    Returns a list of unique page numbers.
    """
    pages = set()

    # First, find the page keyword and extract all following numbers
    match = _PAGE_PATTERN.search(query)
    if match:
        # Get the position after the keyword
        start_pos = match.end() - len(match.group(1))
        # Extract the rest of the query after the keyword
        rest_of_query = query[start_pos:]

        # Find all numbers in the rest of the query (up to first non-page-related word)
        # Stop at common words that indicate end of page range
        stop_words = r'\b(in|from|of|about|for|the|chapter|section|book|text|content|tell|show|give|what|how|why)\b'
        stop_match = re.search(stop_words, rest_of_query, re.IGNORECASE)
        if stop_match:
            rest_of_query = rest_of_query[:stop_match.start()]

        # Extract all numbers from this segment
        number_pattern = re.compile(r'\b(\d+)\b')
        for num_match in number_pattern.finditer(rest_of_query):
            page_num = int(num_match.group(1))
            # Only accept reasonable page numbers (1-9999)
            if 1 <= page_num <= 9999:
                pages.add(page_num)

    return sorted(list(pages))


def reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Combine vector search and BM25 results using Reciprocal Rank Fusion (RRF).

    RRF formula: score = sum(weight / (k + rank))
    where k=60 is a constant that reduces the impact of high-ranked outliers.

    Args:
        vector_results: List of (parent_id, score) from vector search
        bm25_results: List of (parent_id, score) from BM25 search
        vector_weight: Weight for vector scores (default 0.7)
        bm25_weight: Weight for BM25 scores (default 0.3)
        k: RRF constant (default 60)

    Returns:
        List of (parent_id, fused_score) sorted by fused score descending
    """
    scores: dict[str, float] = {}

    # Add vector search scores
    for rank, (parent_id, _) in enumerate(vector_results, start=1):
        rrf_score = vector_weight / (k + rank)
        scores[parent_id] = scores.get(parent_id, 0.0) + rrf_score

    # Add BM25 scores
    for rank, (parent_id, _) in enumerate(bm25_results, start=1):
        rrf_score = bm25_weight / (k + rank)
        scores[parent_id] = scores.get(parent_id, 0.0) + rrf_score

    # Sort by fused score descending
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    log.debug(f"RRF fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 → {len(fused)} unique results")

    return fused


def search_by_pages(
    page_numbers: list[int],
    book_ids: list[str],
    db: Session,
) -> list[RetrievedChunk]:
    """
    Direct lookup for page-specific queries.
    Returns all parent chunks that overlap with the requested pages.
    """
    if not page_numbers or not book_ids:
        return []

    log.info("Page-based retrieval for pages: %s", page_numbers)

    # Build book lookup
    books = {b.id: b for b in db.query(Book).filter(Book.id.in_(book_ids)).all()}

    results: dict[str, RetrievedChunk] = {}  # parent_id -> chunk

    # For each requested page, find all parent chunks that overlap
    for page_num in page_numbers:
        # A parent chunk overlaps if: page_start <= page_num <= page_end
        parent_chunks = (
            db.query(ParentChunk)
            .join(Chapter, ParentChunk.chapter_id == Chapter.id)
            .filter(
                ParentChunk.book_id.in_(book_ids),
                ParentChunk.page_start.isnot(None),
                ParentChunk.page_end.isnot(None),
                ParentChunk.page_start <= page_num,
                ParentChunk.page_end >= page_num,
                # Exclude front/back matter
                Chapter.chapter_type.notin_(['front_matter', 'back_matter'])
            )
            .all()
        )

        for parent in parent_chunks:
            if parent.id in results:
                continue  # Already added

            chapter = db.query(Chapter).filter(Chapter.id == parent.chapter_id).one_or_none()
            book = books.get(parent.book_id)

            if not chapter or not book:
                continue

            # Get the first child chunk for this parent (we need child_id for the response)
            child = db.query(ChildChunk).filter(ChildChunk.parent_chunk_id == parent.id).first()
            if not child:
                continue

            # Score based on page overlap (1.0 for exact match, decreases with distance)
            # This ensures requested pages appear first
            score = 1.0 / (1 + abs(parent.page_start - page_num) * 0.01)

            results[parent.id] = RetrievedChunk(
                child_id=child.id,
                parent_id=parent.id,
                book_id=parent.book_id,
                book_title=book.title,
                chapter_index=chapter.chapter_index,
                chapter_title=chapter.title,
                page_start=parent.page_start,
                page_end=parent.page_end,
                child_text=child.text,
                parent_text=parent.text,
                score=score,
            )

    # Sort by score descending (closest pages first)
    return sorted(results.values(), key=lambda c: c.score, reverse=True)


def _vector_search(
    query: str,
    book_ids: list[str],
    db: Session,
    top_k: int,
    chapter_ids: list[str] | None = None,
) -> list[tuple[str, float]]:
    """
    Run vector search and return (parent_id, score) tuples.

    Args:
        query: Search query
        book_ids: List of book IDs to search
        db: Database session
        top_k: Number of results to retrieve
        chapter_ids: Optional list of chapter IDs to restrict the search to

    Returns:
        List of (parent_id, score) tuples sorted by score descending
    """
    query_vec = embed_query(query)

    chapter_filter = "AND cc.chapter_id = ANY(:chapter_ids)" if chapter_ids else ""
    sql = text(f"""
        SELECT
            cc.parent_chunk_id,
            1 - (cc.embedding <=> CAST(:vec AS vector)) AS score
        FROM child_chunks cc
        JOIN chapters ch ON ch.id = cc.chapter_id
        WHERE
            cc.book_id = ANY(:book_ids)
            AND cc.embedding IS NOT NULL
            AND ch.chapter_type NOT IN ('front_matter', 'back_matter')
            {chapter_filter}
        ORDER BY cc.embedding <=> CAST(:vec AS vector)
        LIMIT :k
    """)

    params: dict = {
        "vec": str(query_vec),
        "book_ids": book_ids,
        "k": top_k,
    }
    if chapter_ids:
        params["chapter_ids"] = chapter_ids

    rows = db.execute(sql, params).fetchall()

    # Filter by threshold and deduplicate by parent_id (keep highest score)
    seen: dict[str, float] = {}
    for row in rows:
        score = float(row.score)
        if score < _settings.cosine_threshold:
            continue

        parent_id = row.parent_chunk_id
        if parent_id not in seen or score > seen[parent_id]:
            seen[parent_id] = score

    # Return sorted by score descending
    return sorted(seen.items(), key=lambda x: x[1], reverse=True)


def _bm25_search(
    query: str,
    book_ids: list[str],
    db: Session,
    top_k: int,
) -> list[tuple[str, float]]:
    """
    Run BM25 search with Pseudo-Relevance Feedback (PRF) query expansion.

    Two-pass strategy:
      1. Initial BM25 pass with the original query tokens.
      2. Extract distinctive terms from the top-3 results (words that appear
         frequently in those chunks but are uncommon across the index).
      3. Append the top expansion terms to the query and run a second BM25 pass.
      4. Merge both result lists, preferring second-pass scores when a parent_id
         appears in both.

    PRF is book-agnostic: expansion terms come from the indexed book's own
    vocabulary, so it works for any domain without a manual synonym map.

    Args:
        query: Search query
        book_ids: List of book IDs to search
        db: Database session
        top_k: Number of results to retrieve

    Returns:
        List of (parent_id, score) tuples sorted by score descending
    """
    from collections import Counter
    from bookrag.retrieval.bm25 import BM25IndexManager, simple_tokenize

    try:
        index_dir = Path(_settings.bm25_index_dir)
        manager = BM25IndexManager(index_dir=index_dir)

        # ── Pass 1: initial search ────────────────────────────────────────────
        initial_results = manager.search(book_ids, query, db, top_k=top_k)
        log.debug(f"BM25 pass-1 returned {len(initial_results)} results")

        if not initial_results:
            return []

        # ── PRF: extract expansion terms from top-3 chunks ───────────────────
        prf_top_n = 3
        expansion_terms: list[str] = []

        top_parent_ids = [pid for pid, _ in initial_results[:prf_top_n]]
        feedback_tokens: list[str] = []

        for parent_id in top_parent_ids:
            parent = db.query(ParentChunk).filter(ParentChunk.id == parent_id).one_or_none()
            if parent:
                feedback_tokens.extend(simple_tokenize(parent.text))

        if feedback_tokens:
            # Count term frequencies in feedback documents
            tf = Counter(feedback_tokens)

            # Remove terms already in the query
            query_tokens_set = set(simple_tokenize(query))

            # Pick the top expansion terms not already in the query
            # (limit to 5 terms to avoid query drift)
            expansion_terms = [
                term for term, _ in tf.most_common(20)
                if term not in query_tokens_set and len(term) > 2
            ][:5]

        if expansion_terms:
            expanded_query = query + " " + " ".join(expansion_terms)
            log.debug(f"PRF expansion terms: {expansion_terms}")

            # ── Pass 2: expanded query ────────────────────────────────────────
            expanded_results = manager.search(book_ids, expanded_query, db, top_k=top_k)
            log.debug(f"BM25 pass-2 returned {len(expanded_results)} results")

            # Merge: second-pass scores take precedence (expanded query is richer)
            merged: dict[str, float] = dict(initial_results)
            for parent_id, score in expanded_results:
                if parent_id not in merged or score > merged[parent_id]:
                    merged[parent_id] = score

            return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return initial_results

    except Exception as e:
        log.warning(f"BM25 search failed: {e}")
        return []


def search(
    queries: list[str],
    book_ids: list[str],
    db: Session,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Hybrid retrieval: checks for page-specific queries first, then uses
    semantic search (optionally combined with BM25 keyword search).

    Phase 1: Page-specific queries → direct page lookup
    Phase 2: Semantic queries → vector search
    Phase 3: If BM25 enabled → hybrid vector + BM25 search with RRF fusion

    De-duplicates by parent_id (multiple children may share a parent).
    Returns up to top_k unique parent contexts.
    """
    k = top_k or _settings.retrieval_top_k
    if not book_ids:
        return []

    # PHASE 1: Check if this is a page-specific query
    if queries:
        page_numbers = extract_page_numbers(queries[0])
        if page_numbers:
            log.info("Detected page-specific query: pages %s", page_numbers)
            return search_by_pages(page_numbers, book_ids, db)

    # PHASE 2 & 3: Semantic search (with optional BM25 hybrid)
    use_bm25 = _settings.enable_bm25
    log.info(f"Using {'hybrid (vector + BM25)' if use_bm25 else 'vector'} search for queries: {queries}")

    # Build a book_id lookup for titles
    books = {b.id: b for b in db.query(Book).filter(Book.id.in_(book_ids)).all()}

    # Collect all ranked parent IDs across all queries
    all_parent_ids: dict[str, float] = {}  # parent_id -> best score

    for query in queries:
        # Run vector search
        vector_results = _vector_search(query, book_ids, db, top_k=k)

        if use_bm25:
            # Run BM25 search
            bm25_results = _bm25_search(query, book_ids, db, top_k=_settings.bm25_top_k)

            # Fuse results with RRF
            fused_results = reciprocal_rank_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                vector_weight=_settings.vector_weight,
                bm25_weight=_settings.bm25_weight,
            )

            # Update all_parent_ids with fused scores
            for parent_id, score in fused_results:
                if parent_id not in all_parent_ids or score > all_parent_ids[parent_id]:
                    all_parent_ids[parent_id] = score
        else:
            # Just use vector results
            for parent_id, score in vector_results:
                if parent_id not in all_parent_ids or score > all_parent_ids[parent_id]:
                    all_parent_ids[parent_id] = score

    # Now fetch full chunk data for top-k parent IDs
    sorted_parent_ids = sorted(all_parent_ids.items(), key=lambda x: x[1], reverse=True)[:k]

    results: list[RetrievedChunk] = []
    for parent_id, score in sorted_parent_ids:
        parent = db.query(ParentChunk).filter(ParentChunk.id == parent_id).one_or_none()
        if not parent:
            continue

        chapter = db.query(Chapter).filter(Chapter.id == parent.chapter_id).one_or_none()
        book = books.get(parent.book_id)

        if not chapter or not book:
            continue

        # Get first child chunk for this parent
        child = db.query(ChildChunk).filter(ChildChunk.parent_chunk_id == parent_id).first()
        if not child:
            continue

        results.append(RetrievedChunk(
            child_id=child.id,
            parent_id=parent_id,
            book_id=parent.book_id,
            book_title=book.title,
            chapter_index=chapter.chapter_index,
            chapter_title=chapter.title,
            page_start=parent.page_start,
            page_end=parent.page_end,
            child_text=child.text,
            parent_text=parent.text,
            score=score,
        ))

    log.info(f"Returning {len(results)} chunks from hybrid search")
    return results


def search_chapter_summaries(
    query: str,
    book_ids: list[str],
    db: Session,
    top_k: int = 3,
) -> list[ChapterSummary]:
    """
    Search chapter summaries for 'what is this chapter about' style queries.
    Front- and back-matter chapters are excluded.
    """
    query_vec = embed_query(query)
    sql = text("""
        SELECT cs.id, 1 - (cs.embedding <=> CAST(:vec AS vector)) AS score
        FROM chapter_summaries cs
        JOIN chapters ch ON ch.id = cs.chapter_id
        WHERE cs.book_id = ANY(:book_ids)
          AND cs.embedding IS NOT NULL
          AND ch.chapter_type NOT IN ('front_matter', 'back_matter')
        ORDER BY cs.embedding <=> CAST(:vec AS vector)
        LIMIT :k
    """)
    rows = db.execute(sql, {"vec": str(query_vec), "book_ids": book_ids, "k": top_k}).fetchall()
    ids = [r.id for r in rows if float(r.score) >= _settings.cosine_threshold]
    return db.query(ChapterSummary).filter(ChapterSummary.id.in_(ids)).all() if ids else []


def search_within_chapter(
    queries: list[str],
    chapter_ids: list[str],
    book_ids: list[str],
    db: Session,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Vector-search restricted to specific chapters (used for TOPIC_IN_CHAPTER intent).

    Mirrors the logic of search() but passes chapter_ids as an additional filter
    to _vector_search(), so only child chunks from the target chapter are scored.
    BM25 is intentionally skipped here — chapter scope provides sufficient focus.
    """
    if not chapter_ids or not book_ids:
        return []

    books = {b.id: b for b in db.query(Book).filter(Book.id.in_(book_ids)).all()}
    all_parent_ids: dict[str, float] = {}

    for query in queries:
        vector_results = _vector_search(query, book_ids, db, top_k=top_k, chapter_ids=chapter_ids)
        for parent_id, score in vector_results:
            if parent_id not in all_parent_ids or score > all_parent_ids[parent_id]:
                all_parent_ids[parent_id] = score

    sorted_parents = sorted(all_parent_ids.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results: list[RetrievedChunk] = []
    for parent_id, score in sorted_parents:
        parent = db.query(ParentChunk).filter(ParentChunk.id == parent_id).one_or_none()
        if not parent:
            continue
        chapter = db.query(Chapter).filter(Chapter.id == parent.chapter_id).one_or_none()
        book = books.get(parent.book_id)
        if not chapter or not book:
            continue
        child = db.query(ChildChunk).filter(ChildChunk.parent_chunk_id == parent_id).first()
        if not child:
            continue
        results.append(RetrievedChunk(
            child_id=child.id,
            parent_id=parent_id,
            book_id=parent.book_id,
            book_title=book.title,
            chapter_index=chapter.chapter_index,
            chapter_title=chapter.title,
            page_start=parent.page_start,
            page_end=parent.page_end,
            child_text=child.text,
            parent_text=parent.text,
            score=score,
        ))

    log.info("search_within_chapter: %d results from chapter_ids=%s", len(results), chapter_ids)
    return results
