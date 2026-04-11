"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-04-06
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")  # for fuzzy title search

    # ── books ─────────────────────────────────────────────────────────────────
    op.create_table(
        "books",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("author", sa.Text),
        sa.Column("file_path", sa.Text, nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("file_type", sa.String(10), nullable=False),
        sa.Column("total_pages", sa.Integer),
        sa.Column("total_chapters", sa.Integer),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_books_status", "books", ["status"])
    op.create_index("ix_books_file_hash", "books", ["file_hash"])

    # ── ingestion_jobs ────────────────────────────────────────────────────────
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("book_id", sa.String(36), sa.ForeignKey("books.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("progress_pct", sa.SmallInteger, nullable=False, server_default="0"),
        sa.Column("chunks_total", sa.Integer),
        sa.Column("chunks_embedded", sa.Integer, server_default="0"),
        sa.Column("error_msg", sa.Text),
        sa.Column("estimated_finish", sa.DateTime),
        sa.Column("started_at", sa.DateTime),
        sa.Column("finished_at", sa.DateTime),
    )
    op.create_index("ix_ingestion_jobs_book_id", "ingestion_jobs", ["book_id"])
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status"])

    # ── chapters ──────────────────────────────────────────────────────────────
    op.create_table(
        "chapters",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("book_id", sa.String(36), sa.ForeignKey("books.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chapter_index", sa.Integer, nullable=False),
        sa.Column("title", sa.Text),
        sa.Column("page_start", sa.Integer),
        sa.Column("page_end", sa.Integer),
        sa.Column("raw_text", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
        sa.UniqueConstraint("book_id", "chapter_index", name="uq_chapter_book_index"),
    )
    op.create_index("ix_chapters_book_id", "chapters", ["book_id"])

    # ── chapter_summaries ─────────────────────────────────────────────────────
    op.create_table(
        "chapter_summaries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("chapter_id", sa.String(36), sa.ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("book_id", sa.String(36), sa.ForeignKey("books.id", ondelete="CASCADE"), nullable=False),
        sa.Column("summary_text", sa.Text, nullable=False),
        sa.Column("token_count", sa.SmallInteger, nullable=False),
        sa.Column("embedding", Vector(768)),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_chapter_summaries_book_id", "chapter_summaries", ["book_id"])

    # ── parent_chunks ─────────────────────────────────────────────────────────
    op.create_table(
        "parent_chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("book_id", sa.String(36), sa.ForeignKey("books.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chapter_id", sa.String(36), sa.ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.SmallInteger, nullable=False),
        sa.Column("page_start", sa.Integer),
        sa.Column("page_end", sa.Integer),
        sa.Column("char_start", sa.BigInteger, nullable=False),
        sa.Column("char_end", sa.BigInteger, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_parent_chunks_book_chapter", "parent_chunks", ["book_id", "chapter_id"])

    # ── child_chunks ──────────────────────────────────────────────────────────
    op.create_table(
        "child_chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("book_id", sa.String(36), sa.ForeignKey("books.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chapter_id", sa.String(36), sa.ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("parent_chunk_id", sa.String(36), sa.ForeignKey("parent_chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.SmallInteger, nullable=False),
        sa.Column("text_hash", sa.String(64), nullable=False),
        sa.Column("embedding", Vector(768)),
        sa.Column("embedded_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_child_chunks_book_chapter", "child_chunks", ["book_id", "chapter_id"])
    op.create_index("ix_child_chunks_text_hash", "child_chunks", ["text_hash"])
    op.execute(
        "CREATE INDEX ix_child_chunks_unembedded ON child_chunks (embedded_at) WHERE embedded_at IS NULL"
    )

    # ── HNSW vector indexes ───────────────────────────────────────────────────
    op.execute(
        "CREATE INDEX ix_child_chunks_embedding ON child_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
    op.execute(
        "CREATE INDEX ix_chapter_summaries_embedding ON chapter_summaries USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )

    # ── sessions ──────────────────────────────────────────────────────────────
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("book_scope", sa.Text),   # JSON array of book_ids
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("last_active_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_sessions_last_active", "sessions", ["last_active_at"])

    # ── messages ──────────────────────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("session_id", sa.String(36), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(10), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("book_ids_referenced", sa.Text),   # JSON array
        sa.Column("chunk_ids_used", sa.Text),         # JSON array
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_messages_session_created", "messages", ["session_id", "created_at"])


def downgrade() -> None:
    op.drop_table("messages")
    op.drop_table("sessions")
    op.drop_table("child_chunks")
    op.drop_table("parent_chunks")
    op.drop_table("chapter_summaries")
    op.drop_table("chapters")
    op.drop_table("ingestion_jobs")
    op.drop_table("books")
    op.execute("DROP EXTENSION IF EXISTS vector")
