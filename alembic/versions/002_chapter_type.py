"""Add chapter_type column to chapters

Revision ID: 002
Revises: 001
Create Date: 2026-04-13

Adds a chapter_type VARCHAR(20) column (default "unknown") that is populated
at extraction time to classify each chapter as:
  "content"      — main body chapters (LLM summarisation runs normally)
  "front_matter" — cover, title page, copyright, dedication, preface, etc.
  "back_matter"  — index, bibliography, glossary, appendix, etc.
  "unknown"      — unclassified; treated as content by the worker

Front- and back-matter chapters are merged into single aggregate chapters
during the _consolidate_chapters pass in extractor.py, so existing books
re-ingested after this migration will benefit immediately.
"""
from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chapters",
        sa.Column(
            "chapter_type",
            sa.String(20),
            nullable=False,
            server_default="unknown",
        ),
    )
    # Index makes filtering by type cheap (e.g. retrieval excluding front/back matter)
    op.create_index("ix_chapters_chapter_type", "chapters", ["chapter_type"])


def downgrade() -> None:
    op.drop_index("ix_chapters_chapter_type", table_name="chapters")
    op.drop_column("chapters", "chapter_type")
