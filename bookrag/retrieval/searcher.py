"""
Layer 3 — Scoped vector search.

Searches child_chunks using pgvector HNSW with a book_id filter,
then fetches the corresponding parent chunks for richer context.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from bookrag.config import get_settings
from bookrag.db.models import Chapter, ChapterSummary, ChildChunk, ParentChunk, Book
from bookrag.ingestion.embedder import embed_query

log = logging.getLogger(__name__)
_settings = get_settings()


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


def search(
    queries: list[str],
    book_ids: list[str],
    db: Session,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Embed each query, search child_chunks restricted to book_ids,
    and fetch the parent context for each hit.

    De-duplicates by parent_id (multiple children may share a parent).
    Returns up to top_k unique parent contexts.
    """
    k = top_k or _settings.retrieval_top_k
    if not book_ids:
        return []

    # Build a book_id lookup for titles
    books = {b.id: b for b in db.query(Book).filter(Book.id.in_(book_ids)).all()}

    seen_parents: dict[str, RetrievedChunk] = {}   # parent_id -> best chunk

    for query in queries:
        query_vec = embed_query(query)

        # pgvector cosine distance (1 - cosine_similarity) via <=>
        # We want similarity, so 1 - distance
        sql = text("""
            SELECT
                cc.id            AS child_id,
                cc.parent_chunk_id,
                cc.book_id,
                cc.chapter_id,
                cc.text          AS child_text,
                1 - (cc.embedding <=> CAST(:vec AS vector)) AS score
            FROM child_chunks cc
            WHERE
                cc.book_id = ANY(:book_ids)
                AND cc.embedding IS NOT NULL
            ORDER BY cc.embedding <=> CAST(:vec AS vector)
            LIMIT :k
        """)

        rows = db.execute(sql, {
            "vec": str(query_vec),
            "book_ids": book_ids,
            "k": k,
        }).fetchall()

        for row in rows:
            score = float(row.score)
            if score < _settings.cosine_threshold:
                continue   # below relevance threshold

            parent_id = row.parent_chunk_id
            if parent_id in seen_parents:
                # Keep the highest-scoring hit per parent
                if score <= seen_parents[parent_id].score:
                    continue

            parent = db.query(ParentChunk).filter(ParentChunk.id == parent_id).one_or_none()
            chapter = db.query(Chapter).filter(Chapter.id == row.chapter_id).one_or_none()
            book = books.get(row.book_id)

            if not parent or not chapter or not book:
                continue

            seen_parents[parent_id] = RetrievedChunk(
                child_id=row.child_id,
                parent_id=parent_id,
                book_id=row.book_id,
                book_title=book.title,
                chapter_index=chapter.chapter_index,
                chapter_title=chapter.title,
                page_start=parent.page_start,
                page_end=parent.page_end,
                child_text=row.child_text,
                parent_text=parent.text,
                score=score,
            )

    # Sort by score descending, return top-k unique parents
    results = sorted(seen_parents.values(), key=lambda c: c.score, reverse=True)
    return results[:k]


def search_chapter_summaries(
    query: str,
    book_ids: list[str],
    db: Session,
    top_k: int = 3,
) -> list[ChapterSummary]:
    """
    Search chapter summaries for 'what is this chapter about' style queries.
    """
    query_vec = embed_query(query)
    sql = text("""
        SELECT cs.id, 1 - (cs.embedding <=> CAST(:vec AS vector)) AS score
        FROM chapter_summaries cs
        WHERE cs.book_id = ANY(:book_ids)
          AND cs.embedding IS NOT NULL
        ORDER BY cs.embedding <=> CAST(:vec AS vector)
        LIMIT :k
    """)
    rows = db.execute(sql, {"vec": str(query_vec), "book_ids": book_ids, "k": top_k}).fetchall()
    ids = [r.id for r in rows if float(r.score) >= _settings.cosine_threshold]
    return db.query(ChapterSummary).filter(ChapterSummary.id.in_(ids)).all() if ids else []
