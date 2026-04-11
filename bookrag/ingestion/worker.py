"""
Layer 1 — Background ingestion worker.

Polls ingestion_jobs for queued jobs and processes them in order.
Each phase writes progress checkpoints so the job can resume after a crash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from sqlalchemy import text

from bookrag.config import get_settings
from bookrag.db.models import (
    Book, Chapter, ChapterSummary, ChildChunk, IngestionJob, ParentChunk
)
from bookrag.db.session import sync_session
from bookrag.ingestion.chunker import chunk_chapter
from bookrag.ingestion.embedder import embed_batch
from bookrag.ingestion.extractor import extract

log = logging.getLogger(__name__)
_settings = get_settings()

POLL_INTERVAL = 5   # seconds between polls when idle


# ── Job phases ────────────────────────────────────────────────────────────────

def _update_job(db: Session, job: IngestionJob, **kwargs) -> None:
    for k, v in kwargs.items():
        setattr(job, k, v)
    db.flush()


def _phase_extract(db: Session, job: IngestionJob, book: Book) -> list:
    """Extract and persist chapters. Returns list of Chapter ORM objects."""
    _update_job(db, job, status="extracting")
    log.info("[job %s] Extracting %s", job.id[:8], book.file_path)

    raw_chapters = extract(Path(book.file_path))
    book.total_pages = max((c.page_end or 0) for c in raw_chapters if c.page_end) or None
    book.total_chapters = len(raw_chapters)
    db.flush()

    chapter_objs = []
    for raw in raw_chapters:
        ch = Chapter(
            book_id=book.id,
            chapter_index=raw.chapter_index,
            title=raw.title,
            page_start=raw.page_start,
            page_end=raw.page_end,
            raw_text=raw.raw_text,
        )
        db.add(ch)
        chapter_objs.append((ch, raw))

    db.flush()
    log.info("[job %s] Extracted %d chapters", job.id[:8], len(chapter_objs))
    return chapter_objs


def _phase_chunk(db: Session, job: IngestionJob, book: Book, chapter_pairs: list) -> int:
    """Chunk all chapters and persist parent + child chunks. Returns total child count."""
    _update_job(db, job, status="chunking")
    total_children = 0

    for ch_obj, raw_ch in chapter_pairs:
        result = chunk_chapter(raw_ch)

        for parent_data in result.parents:
            parent = ParentChunk(
                book_id=book.id,
                chapter_id=ch_obj.id,
                chunk_index=parent_data.chunk_index,
                text=parent_data.text,
                token_count=parent_data.token_count,
                page_start=parent_data.page_start,
                page_end=parent_data.page_end,
                char_start=parent_data.char_start,
                char_end=parent_data.char_end,
            )
            db.add(parent)
            db.flush()   # get parent.id

            for child_data in parent_data.children:
                child = ChildChunk(
                    book_id=book.id,
                    chapter_id=ch_obj.id,
                    parent_chunk_id=parent.id,
                    chunk_index=child_data.chunk_index,
                    text=child_data.text,
                    token_count=child_data.token_count,
                    text_hash=child_data.text_hash,
                    embedding=None,
                    embedded_at=None,
                )
                db.add(child)
                total_children += 1

        db.flush()

    _update_job(db, job, chunks_total=total_children, chunks_embedded=0)
    log.info("[job %s] Created %d child chunks", job.id[:8], total_children)
    return total_children


def _phase_embed(db: Session, job: IngestionJob, book: Book) -> None:
    """
    Embed all un-embedded child chunks in batches.
    Resumable: only processes chunks where embedded_at IS NULL.
    Updates progress every 50 chunks.
    """
    _update_job(db, job, status="embedding")
    batch_size = _settings.embed_batch_size
    start_time = time.monotonic()
    embedded_count = job.chunks_embedded or 0

    while True:
        # Fetch next batch of un-embedded chunks
        chunks = (
            db.query(ChildChunk)
            .filter(ChildChunk.book_id == book.id, ChildChunk.embedded_at.is_(None))
            .order_by(ChildChunk.created_at)
            .limit(batch_size)
            .all()
        )
        if not chunks:
            break

        texts = [c.text for c in chunks]
        vectors = embed_batch(texts, batch_size=batch_size)

        now = datetime.utcnow()
        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec
            chunk.embedded_at = now

        db.flush()
        embedded_count += len(chunks)

        # Progress update every 50 chunks
        if embedded_count % 50 == 0 or len(chunks) < batch_size:
            elapsed = time.monotonic() - start_time
            rate = embedded_count / elapsed if elapsed > 0 else 1
            remaining = (job.chunks_total or 0) - embedded_count
            eta = datetime.utcnow() + timedelta(seconds=remaining / rate) if rate > 0 else None
            pct = int(100 * embedded_count / max(job.chunks_total or 1, 1))
            _update_job(
                db, job,
                chunks_embedded=embedded_count,
                progress_pct=min(pct, 95),    # reserve last 5% for indexing
                estimated_finish=eta,
            )
            log.info("[job %s] Embedded %d/%d chunks (%.0f/s)",
                     job.id[:8], embedded_count, job.chunks_total or 0, rate)

    log.info("[job %s] Embedding complete: %d chunks", job.id[:8], embedded_count)


def _phase_summarise(db: Session, job: IngestionJob, book: Book) -> None:
    """
    Generate chapter summaries using Gemma 4 via Ollama, then embed them.
    Skips chapters that already have a summary.
    """
    import ollama

    _update_job(db, job, status="summarising")
    chapters = db.query(Chapter).filter(Chapter.book_id == book.id).order_by(Chapter.chapter_index).all()
    client = ollama.Client(host=_settings.ollama_host)

    for ch in chapters:
        # Skip if already done
        if db.query(ChapterSummary).filter(ChapterSummary.chapter_id == ch.id).first():
            continue

        # Truncate very long chapters to ~3000 tokens for the summary prompt
        text_sample = ch.raw_text[:12000]   # ~3000 tokens ≈ 12000 chars
        prompt = (
            f"Summarise the following book chapter in 3–5 paragraphs (500–800 words). "
            f"Focus on key ideas, arguments, and narrative.\n\n"
            f"Chapter title: {ch.title or 'Untitled'}\n\n"
            f"{text_sample}"
        )

        try:
            response = client.generate(
                model=_settings.ollama_llm_model,
                prompt=prompt,
                options={"num_predict": _settings.summary_max_tokens},
            )
            summary_text = response.response.strip()
        except Exception as exc:
            log.warning("[job %s] Summary generation failed for chapter %d: %s", job.id[:8], ch.chapter_index, exc)
            summary_text = ch.raw_text[:2000]   # fallback to raw excerpt

        from bookrag.ingestion.embedder import embed_documents
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(summary_text))
        embedding = embed_documents([summary_text])[0]

        summary = ChapterSummary(
            chapter_id=ch.id,
            book_id=book.id,
            summary_text=summary_text,
            token_count=token_count,
            embedding=embedding,
        )
        db.add(summary)
        db.flush()

    log.info("[job %s] Chapter summaries done", job.id[:8])


def _phase_index(db: Session, job: IngestionJob) -> None:
    """
    The HNSW index is already defined — pgvector updates it incrementally
    as rows are inserted. This phase just runs VACUUM ANALYZE to keep
    statistics fresh and marks the job complete.
    """
    _update_job(db, job, status="indexing", progress_pct=98)

    # Run ANALYZE outside the transaction (needs autocommit) — SQLAlchemy 2.0 style
    try:
        from bookrag.db.session import get_sync_engine
        engine = get_sync_engine()
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("ANALYZE child_chunks"))
            conn.execute(text("ANALYZE chapter_summaries"))
    except Exception as exc:
        log.warning("ANALYZE failed (non-fatal): %s", exc)


# ── Main worker loop ──────────────────────────────────────────────────────────

def process_job(job_id: str) -> None:
    """Process a single ingestion job end-to-end."""
    with sync_session() as db:
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).one()
        book = db.query(Book).filter(Book.id == job.book_id).one()

        try:
            _update_job(db, job, status="extracting", started_at=datetime.utcnow())
            book.status = "processing"
            db.flush()

            chapter_pairs = _phase_extract(db, job, book)
            _phase_chunk(db, job, book, chapter_pairs)
            _phase_embed(db, job, book)
            _phase_summarise(db, job, book)
            _phase_index(db, job)

            _update_job(db, job, status="done", progress_pct=100, finished_at=datetime.utcnow())
            book.status = "completed"
            log.info("[job %s] Ingestion complete for book '%s'", job.id[:8], book.title)

        except Exception as exc:
            log.exception("[job %s] Ingestion failed: %s", job.id[:8], exc)
            _update_job(db, job, status="failed", error_msg=str(exc), finished_at=datetime.utcnow())
            book.status = "failed"


def run_worker() -> None:
    """Continuously poll for queued jobs and process them."""
    log.info("BookRAG worker started. Polling every %ds.", POLL_INTERVAL)
    while True:
        try:
            with sync_session() as db:
                job = (
                    db.query(IngestionJob)
                    .filter(IngestionJob.status == "queued")
                    .order_by(IngestionJob.id)
                    .first()
                )
                if job:
                    log.info("Picked up job %s", job.id[:8])
                    job_id = job.id

            if job:
                process_job(job_id)
            else:
                time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            log.info("Worker stopping.")
            break
        except Exception as exc:
            log.exception("Worker error (will retry): %s", exc)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_worker()
