"""
Layer 3 — Query router.

Classifies each query into one of three scopes:
  explicit  — user named a specific book
  inferred  — session has an active book scope
  cross     — query explicitly asks to search across all books
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum

from rapidfuzz import fuzz, process as rfprocess
from sqlalchemy.orm import Session

from bookrag.db.models import Book, Session as DBSession

log = logging.getLogger(__name__)

_CROSS_BOOK_SIGNALS = re.compile(
    r"\b(compare|across|all books?|every book|between books?|multiple books?|difference between)\b",
    re.IGNORECASE,
)


class ScopeType(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    CROSS = "cross"


@dataclass
class QueryScope:
    scope_type: ScopeType
    book_ids: list[str]   # empty = all books


def resolve_scope(
    query: str,
    session: DBSession,
    db: Session,
) -> QueryScope:
    """
    Determine which books to search for this query.
    Priority: explicit > cross-book signal > inferred session scope > all books.
    """
    all_books = db.query(Book).filter(Book.status == "completed").all()
    all_ids = [b.id for b in all_books]

    # 1. Explicit book mention — fuzzy-match titles and author names
    for book in all_books:
        candidates = [book.title]
        if book.author:
            candidates.append(book.author)
        for candidate in candidates:
            if fuzz.partial_ratio(candidate.lower(), query.lower()) >= 80:
                log.debug("Explicit scope: matched book '%s'", book.title)
                return QueryScope(scope_type=ScopeType.EXPLICIT, book_ids=[book.id])

    # 2. Cross-book signal
    if _CROSS_BOOK_SIGNALS.search(query):
        log.debug("Cross-book scope detected")
        return QueryScope(scope_type=ScopeType.CROSS, book_ids=all_ids)

    # 3. Inferred from session
    if session.book_scope:
        try:
            scope_ids = json.loads(session.book_scope)
            if scope_ids:
                log.debug("Inferred scope from session: %s", scope_ids)
                return QueryScope(scope_type=ScopeType.INFERRED, book_ids=scope_ids)
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. Default: all completed books
    log.debug("Default scope: all %d books", len(all_ids))
    return QueryScope(scope_type=ScopeType.CROSS, book_ids=all_ids)
