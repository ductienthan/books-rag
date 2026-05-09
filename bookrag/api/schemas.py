"""
bookrag/api/schemas.py

Pydantic request / response models for the BookRAG HTTP API.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Request models ────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., description="Natural-language question to ask.")
    book_ids: list[str] = Field(
        default_factory=list,
        description="Restrict search to these book IDs. Empty = all books.",
    )
    session_id: Optional[str] = Field(
        None,
        description="Conversation session ID for multi-turn context.",
    )


# ── Response models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str               # "ok" | "degraded" | "error"
    db: str                   # "connected" | "error: <msg>"
    bm25_indexes: int         # number of book indexes loaded
    message: Optional[str] = None


class BookItem(BaseModel):
    id: str
    title: str
    author: Optional[str]
    num_pages: Optional[int]
    status: str


class BooksResponse(BaseModel):
    books: list[BookItem]
    total: int


class SourceChunk(BaseModel):
    parent_id: str
    chapter_title: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    session_id: Optional[str]
    sources: list[SourceChunk]
    book_ids_used: list[str]
