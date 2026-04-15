"""
Layer 1 — Background ingestion worker.

Polls ingestion_jobs for queued jobs and processes them in order.
Each phase commits its own transaction so locks are released between phases,
allowing concurrent re-ingest requests to proceed without hanging.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
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
from bookrag.ingestion.extractor import ChapterType, extract

log = logging.getLogger(__name__)
_settings = get_settings()

POLL_INTERVAL = 5   # seconds between polls when idle


# ── Helpers ───────────────────────────────────────────────────────────────────

def _update_job(db: Session, job: IngestionJob, **kwargs) -> None:
    for k, v in kwargs.items():
        setattr(job, k, v)
    db.flush()


def _load(db: Session, job_id: str) -> tuple[IngestionJob, Book]:
    """Re-load job and book from DB — used at the start of each phase session."""
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).one()
    book = db.query(Book).filter(Book.id == job.book_id).one()
    return job, book


# ── Job phases ────────────────────────────────────────────────────────────────

def _phase_extract(db: Session, job: IngestionJob, book: Book) -> list:
    """Extract and persist chapters. Returns list of (Chapter, RawChapter) pairs."""
    # Lock the book row for the duration of this transaction.  Without this,
    # a concurrent `bookrag add --force` or `bookrag remove` can DELETE the
    # book while extract() is running (PDF parsing takes minutes, during which
    # the book row is only SELECTed — not locked).  The subsequent chapter
    # INSERT then fails with a FK violation because the book no longer exists.
    # The CLI already handles lock-timeout gracefully when --force is used.
    book = db.query(Book).filter(Book.id == book.id).with_for_update().one()
    _update_job(db, job, status="extracting")
    log.info("[job %s] Extracting %s", job.id[:8], book.file_path)

    raw_chapters = extract(Path(book.file_path))
    book.total_pages = max((c.page_end for c in raw_chapters if c.page_end), default=None)
    book.total_chapters = len(raw_chapters)
    db.flush()

    # Delete any data already inserted for this book (idempotent retry).
    # Use 'fetch' so SQLAlchemy loads and marks deleted objects in the identity
    # map — prevents the ORM from re-flushing stale objects and causing
    # UniqueViolation on retry.
    db.query(ChapterSummary).filter(ChapterSummary.book_id == book.id).delete(synchronize_session="fetch")
    db.query(ChildChunk).filter(ChildChunk.book_id == book.id).delete(synchronize_session="fetch")
    db.query(ParentChunk).filter(ParentChunk.book_id == book.id).delete(synchronize_session="fetch")
    db.query(Chapter).filter(Chapter.book_id == book.id).delete(synchronize_session="fetch")
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
            chapter_type=raw.chapter_type.value,   # "content" | "front_matter" | …
        )
        db.add(ch)
        chapter_objs.append((ch, raw))

    db.flush()
    log.info("[job %s] Extracted %d chapters", job.id[:8], len(chapter_objs))
    return chapter_objs


def _phase_chunk(db: Session, job: IngestionJob, book: Book, chapter_pairs: list) -> int:
    """Chunk all chapters and persist parent + child chunks. Returns total child count."""
    _update_job(db, job, status="chunking")
    log.info("[job %s] Chunking %d chapters", job.id[:8], len(chapter_pairs))
    total_children = 0

    for ch_obj, raw_ch in chapter_pairs:
        # Front- and back-matter chapters (title page, copyright, index, etc.)
        # contain no retrievable content — skip chunking and embedding entirely.
        if raw_ch.chapter_type in (ChapterType.FRONT_MATTER, ChapterType.BACK_MATTER):
            log.info(
                "[job %s] Skipping chunking for %s chapter '%s'",
                job.id[:8], raw_ch.chapter_type.value, raw_ch.title,
            )
            continue

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


def _phase_embed(job_id: str) -> None:
    """
    Embed all un-embedded child chunks in batches.
    Each batch commits independently so locks are released continuously.
    Resumable: only processes chunks where embedded_at IS NULL.
    """
    batch_size = _settings.embed_batch_size
    start_time = time.monotonic()
    embedded_count = 0

    # Initialise count from DB in case we're resuming.
    with sync_session() as db:
        job, book = _load(db, job_id)
        _update_job(db, job, status="embedding")
        embedded_count = job.chunks_embedded or 0
        book_id = book.id
        chunks_total = job.chunks_total or 0

    while True:
        # Short read session — don't hold it during the slow embed call.
        with sync_session() as db:
            rows = (
                db.query(ChildChunk.id, ChildChunk.text)
                .filter(ChildChunk.book_id == book_id, ChildChunk.embedded_at.is_(None))
                .order_by(ChildChunk.created_at)
                .limit(batch_size)
                .all()
            )
        if not rows:
            break

        chunk_ids = [r.id for r in rows]
        texts = [r.text for r in rows]
        vectors = embed_batch(texts, batch_size=batch_size)

        # Short write session — rows may have been deleted by a concurrent
        # re-ingest; using synchronize_session=False avoids StaleDataError.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with sync_session() as db:
            job, _ = _load(db, job_id)
            for chunk_id, vec in zip(chunk_ids, vectors):
                db.query(ChildChunk).filter(ChildChunk.id == chunk_id).update(
                    {"embedding": vec, "embedded_at": now},
                    synchronize_session=False,
                )

            embedded_count += len(chunk_ids)
            elapsed = time.monotonic() - start_time
            rate = embedded_count / elapsed if elapsed > 0 else 1
            remaining = chunks_total - embedded_count
            eta = (
                datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=remaining / rate)
                if rate > 0 else None
            )
            pct = int(100 * embedded_count / max(chunks_total, 1))
            _update_job(
                db, job,
                chunks_embedded=embedded_count,
                progress_pct=min(pct, 95),
                estimated_finish=eta,
            )
        # batch committed, locks released

        if embedded_count % (batch_size * 10) == 0 or len(rows) < batch_size:
            elapsed = time.monotonic() - start_time
            rate = embedded_count / elapsed if elapsed > 0 else 1
            log.info("[job %s] Embedded %d/%d chunks (%.0f/s)",
                     job_id[:8], embedded_count, chunks_total, rate)

    log.info("[job %s] Embedding complete: %d chunks", job_id[:8], embedded_count)


def _phase_summarise(job_id: str) -> None:
    """
    Generate chapter summaries via Ollama, then embed them.
    Each chapter commits independently. Skips chapters already summarised.
    """
    import ollama
    import tiktoken
    from bookrag.ingestion.embedder import embed_documents

    with sync_session() as db:
        job, book = _load(db, job_id)
        _update_job(db, job, status="summarising")
        book_id = book.id
        chapter_ids = [
            ch.id for ch in
            db.query(Chapter).filter(Chapter.book_id == book_id).order_by(Chapter.chapter_index).all()
        ]

    log.info("[job %s] Summarising %d chapters via Ollama", job_id[:8], len(chapter_ids))
    client = ollama.Client(host=_settings.ollama_host)
    enc = tiktoken.get_encoding("cl100k_base")

    for i, chapter_id in enumerate(chapter_ids):
        with sync_session() as db:
            job, _ = _load(db, job_id)
            ch = db.query(Chapter).filter(Chapter.id == chapter_id).one()

            # Skip if already done
            if db.query(ChapterSummary).filter(ChapterSummary.chapter_id == ch.id).first():
                log.info("[job %s] Chapter %d/%d already summarised, skipping",
                         job_id[:8], i + 1, len(chapter_ids))
                continue

            log.info("[job %s] Summarising chapter %d/%d: %s [%s]",
                     job_id[:8], i + 1, len(chapter_ids),
                     ch.title or "Untitled", ch.chapter_type)

            raw = ch.raw_text.strip()

            # Front- and back-matter chapters (title page, copyright, index, etc.)
            # contain no meaningful content to summarise — store raw text directly
            # so we avoid wasting an Ollama round-trip on them.
            if ch.chapter_type in ("front_matter", "back_matter"):
                log.info(
                    "[job %s] Chapter %d is %s — skipping LLM summarisation",
                    job_id[:8], i + 1, ch.chapter_type,
                )
                summary_text = raw or f"[{ch.title or 'Untitled'} — {ch.chapter_type}]"
            elif len(raw) < 500:
                log.info("[job %s] Chapter %d too short (%d chars) — storing raw text as summary",
                         job_id[:8], i + 1, len(raw))
                summary_text = raw or f"[{ch.title or 'Untitled'} — no text content]"
            else:
                text_sample = raw[:12000]
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
                    log.warning("[job %s] Summary generation failed for chapter %d: %s",
                                job_id[:8], i, exc)
                    summary_text = raw[:2000]

            token_count = len(enc.encode(summary_text))
            embedding = embed_documents([summary_text])[0]

            summary = ChapterSummary(
                chapter_id=ch.id,
                book_id=book_id,
                summary_text=summary_text,
                token_count=token_count,
                embedding=embedding,
            )
            db.add(summary)
        # chapter summary committed, locks released

    log.info("[job %s] Chapter summaries done", job_id[:8])


def _phase_index(db: Session, job: IngestionJob) -> None:
    """
    Run VACUUM ANALYZE to keep statistics fresh, then mark the job complete.
    """
    _update_job(db, job, status="indexing", progress_pct=98)

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
    """
    Process a single ingestion job end-to-end.
    Each phase runs in its own transaction so database locks are released
    between phases — re-ingest requests never block for more than seconds.
    """
    job_short = job_id[:8]

    def _mark_failed(exc: Exception) -> None:
        try:
            with sync_session() as db:
                job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
                if not job:
                    return
                book = db.query(Book).filter(Book.id == job.book_id).first()
                job.status = "failed"
                job.error_msg = str(exc)[:1000]
                job.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
                if book:
                    book.status = "failed"
        except Exception:
            log.exception("[job %s] Failed to update job status after error", job_short)

    try:
        # Phase 0 — mark started
        with sync_session() as db:
            job, book = _load(db, job_id)
            _update_job(db, job, status="extracting",
                        started_at=datetime.now(timezone.utc).replace(tzinfo=None))
            book.status = "processing"

        # Phase 1 — extract (PDF parse + chapter insert); expire_on_commit=False keeps
        # chapter objects accessible after commit so _phase_chunk can read their .id
        with sync_session() as db:
            job, book = _load(db, job_id)
            chapter_pairs = _phase_extract(db, job, book)

        # Phase 2 — chunk
        with sync_session() as db:
            job, book = _load(db, job_id)
            _phase_chunk(db, job, book, chapter_pairs)

        # Phase 3 — embed (commits per batch internally)
        _phase_embed(job_id)

        # Phase 4 — summarise (commits per chapter internally)
        _phase_summarise(job_id)

        # Phase 5 — index + mark done
        with sync_session() as db:
            job, book = _load(db, job_id)
            _phase_index(db, job)
            _update_job(db, job, status="done", progress_pct=100,
                        finished_at=datetime.now(timezone.utc).replace(tzinfo=None))
            book.status = "completed"
            log.info("[job %s] Ingestion complete for book '%s'", job_short, book.title)

    except Exception as exc:
        log.exception("[job %s] Ingestion failed: %s", job_short, exc)
        _mark_failed(exc)
        raise


def _recover_stale_jobs() -> None:
    """
    On startup, reset any jobs left in an in-progress state back to 'queued'.
    These are jobs that were being processed when the worker last crashed or
    was restarted. _phase_extract is idempotent (deletes + rebuilds), so
    requeuing is always safe.
    """
    IN_PROGRESS = ("processing", "extracting", "chunking", "embedding", "summarising", "indexing")
    with sync_session() as db:
        stale = (
            db.query(IngestionJob)
            .filter(IngestionJob.status.in_(IN_PROGRESS))
            .all()
        )
        if stale:
            for job in stale:
                log.warning("Recovering stale job %s (was '%s') → queued", job.id[:8], job.status)
                job.status = "queued"
                job.progress_pct = 0
            log.info("Recovered %d stale job(s).", len(stale))


def run_worker() -> None:
    """Continuously poll for queued jobs and process them."""
    log.info("BookRAG worker started. Polling every %ds.", POLL_INTERVAL)
    _recover_stale_jobs()
    while True:
        try:
            job_id = None
            with sync_session() as db:
                job = (
                    db.query(IngestionJob)
                    .filter(IngestionJob.status == "queued")
                    .order_by(IngestionJob.id)
                    .with_for_update(skip_locked=True)
                    .first()
                )
                if job:
                    job.status = "processing"
                    db.flush()
                    job_id = job.id
                    log.info("Claimed job %s", job_id[:8])

            if job_id:
                try:
                    process_job(job_id)
                except Exception as exc:
                    log.error("Job %s failed: %s — continuing to next job", job_id[:8], exc)
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
