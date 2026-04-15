"""SQLAlchemy ORM models for BookRAG."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text as sa_text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


# ── Books ─────────────────────────────────────────────────────────────────────

class Book(Base):
    __tablename__ = "books"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[Optional[str]] = mapped_column(Text)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)  # 'pdf' | 'epub'
    total_pages: Mapped[Optional[int]] = mapped_column(Integer)
    total_chapters: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now, nullable=False)

    # relationships
    chapters: Mapped[list[Chapter]] = relationship("Chapter", back_populates="book", cascade="all, delete-orphan")
    ingestion_job: Mapped[Optional[IngestionJob]] = relationship("IngestionJob", back_populates="book", uselist=False, cascade="all, delete-orphan")
    child_chunks: Mapped[list[ChildChunk]] = relationship("ChildChunk", back_populates="book")

    __table_args__ = (
        Index("ix_books_status", "status"),
        Index("ix_books_file_hash", "file_hash"),
    )


# ── Ingestion Jobs ────────────────────────────────────────────────────────────

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    book_id: Mapped[str] = mapped_column(String(36), ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    progress_pct: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    chunks_total: Mapped[Optional[int]] = mapped_column(Integer)
    chunks_embedded: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    error_msg: Mapped[Optional[str]] = mapped_column(Text)
    estimated_finish: Mapped[Optional[datetime]] = mapped_column(DateTime)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    book: Mapped[Book] = relationship("Book", back_populates="ingestion_job")

    __table_args__ = (
        Index("ix_ingestion_jobs_book_id", "book_id"),
        Index("ix_ingestion_jobs_status", "status"),
    )


# ── Chapters ──────────────────────────────────────────────────────────────────

class Chapter(Base):
    __tablename__ = "chapters"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    book_id: Mapped[str] = mapped_column(String(36), ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    chapter_index: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text)
    page_start: Mapped[Optional[int]] = mapped_column(Integer)
    page_end: Mapped[Optional[int]] = mapped_column(Integer)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Classification set at extraction time: "content" | "front_matter" |
    # "back_matter" | "unknown".  Used by the worker to gate LLM summary calls
    # and by retrieval to exclude non-content chapters from search results.
    chapter_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="unknown"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)

    book: Mapped[Book] = relationship("Book", back_populates="chapters")
    summary: Mapped[Optional[ChapterSummary]] = relationship(
        "ChapterSummary", back_populates="chapter", uselist=False, cascade="all, delete-orphan"
    )
    parent_chunks: Mapped[list[ParentChunk]] = relationship(
        "ParentChunk", back_populates="chapter", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_chapters_book_id", "book_id"),
        UniqueConstraint("book_id", "chapter_index", name="uq_chapter_book_index"),
    )


# ── Chapter Summaries ─────────────────────────────────────────────────────────

class ChapterSummary(Base):
    __tablename__ = "chapter_summaries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    chapter_id: Mapped[str] = mapped_column(String(36), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False, unique=True)
    book_id: Mapped[str] = mapped_column(String(36), ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    embedding = Column(Vector(768))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)

    chapter: Mapped[Chapter] = relationship("Chapter", back_populates="summary")

    __table_args__ = (
        Index("ix_chapter_summaries_book_id", "book_id"),
    )


# ── Parent Chunks ─────────────────────────────────────────────────────────────

class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    book_id: Mapped[str] = mapped_column(String(36), ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    chapter_id: Mapped[str] = mapped_column(String(36), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    page_start: Mapped[Optional[int]] = mapped_column(Integer)
    page_end: Mapped[Optional[int]] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(BigInteger, nullable=False)
    char_end: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)

    chapter: Mapped[Chapter] = relationship("Chapter", back_populates="parent_chunks")
    child_chunks: Mapped[list[ChildChunk]] = relationship(
        "ChildChunk", back_populates="parent_chunk", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_parent_chunks_book_chapter", "book_id", "chapter_id"),
    )


# ── Child Chunks ──────────────────────────────────────────────────────────────

class ChildChunk(Base):
    __tablename__ = "child_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    book_id: Mapped[str] = mapped_column(String(36), ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    chapter_id: Mapped[str] = mapped_column(String(36), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False)
    parent_chunk_id: Mapped[str] = mapped_column(String(36), ForeignKey("parent_chunks.id", ondelete="CASCADE"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding = Column(Vector(768), nullable=True)
    embedded_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)

    book: Mapped[Book] = relationship("Book", back_populates="child_chunks")
    parent_chunk: Mapped[ParentChunk] = relationship("ParentChunk", back_populates="child_chunks")

    __table_args__ = (
        Index("ix_child_chunks_book_chapter", "book_id", "chapter_id"),
        Index("ix_child_chunks_unembedded", "embedded_at", postgresql_where=sa_text("embedded_at IS NULL")),
        Index("ix_child_chunks_text_hash", "text_hash"),
    )


# ── Sessions ──────────────────────────────────────────────────────────────────

class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # JSON-serialized list of book_ids; stored as text for portability
    book_scope: Mapped[Optional[str]] = mapped_column(Text)  # JSON array of book_ids
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)
    last_active_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)

    messages: Mapped[list[Message]] = relationship("Message", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_sessions_last_active", "last_active_at"),
    )


# ── Messages ──────────────────────────────────────────────────────────────────

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(10), nullable=False)   # 'user' | 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    book_ids_referenced: Mapped[Optional[str]] = mapped_column(Text)   # JSON array
    chunk_ids_used: Mapped[Optional[str]] = mapped_column(Text)        # JSON array
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)

    session: Mapped[Session] = relationship("Session", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_session_created", "session_id", "created_at"),
    )
