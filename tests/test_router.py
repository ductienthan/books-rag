"""Tests for the query router."""
import json
import pytest
from unittest.mock import MagicMock

from bookrag.retrieval.router import resolve_scope, ScopeType
from bookrag.db.models import Book, Session as DBSession


def _make_book(title: str, author: str | None = None) -> Book:
    b = Book()
    b.id = f"book-{title[:4]}"
    b.title = title
    b.author = author
    b.status = "completed"
    b.file_hash = title
    b.file_path = "/tmp/x"
    b.file_type = "pdf"
    return b


def _make_session(book_scope: list[str] | None = None) -> DBSession:
    s = DBSession()
    s.id = "sess-1"
    s.book_scope = json.dumps(book_scope) if book_scope else None
    return s


def _make_db(books: list[Book]) -> MagicMock:
    db = MagicMock()
    query_mock = MagicMock()
    query_mock.filter.return_value.all.return_value = books
    db.query.return_value = query_mock
    return db


def test_explicit_scope_by_title():
    books = [_make_book("Thinking Fast and Slow"), _make_book("Atomic Habits")]
    db = _make_db(books)
    session = _make_session()
    scope = resolve_scope("What does Thinking Fast and Slow say about decision making?", session, db)
    assert scope.scope_type == ScopeType.EXPLICIT
    assert "book-Thin" in scope.book_ids


def test_cross_book_signal():
    books = [_make_book("Book A"), _make_book("Book B")]
    db = _make_db(books)
    session = _make_session()
    scope = resolve_scope("Compare the approaches across all books", session, db)
    assert scope.scope_type == ScopeType.CROSS


def test_inferred_from_session():
    books = [_make_book("Clean Code"), _make_book("The Pragmatic Programmer")]
    db = _make_db(books)
    session = _make_session(book_scope=["book-Clea"])
    scope = resolve_scope("What does it say about naming variables?", session, db)
    assert scope.scope_type == ScopeType.INFERRED
    assert "book-Clea" in scope.book_ids


def test_default_all_books():
    books = [_make_book("Book A"), _make_book("Book B")]
    db = _make_db(books)
    session = _make_session()
    scope = resolve_scope("What is the main theme?", session, db)
    assert scope.scope_type == ScopeType.CROSS
    assert len(scope.book_ids) == 2
