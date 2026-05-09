"""
bookrag/api/server.py

FastAPI HTTP server for BookRAG.

Endpoints:
  GET  /health           Liveness / readiness check
  GET  /books            List all ingested books
  POST /ask              Ask a question, returns streamed or batch answer

Start with:
  bookrag serve
  # or directly:
  uvicorn bookrag.api.server:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from bookrag.api.schemas import (
    AskRequest,
    AskResponse,
    BookItem,
    BooksResponse,
    HealthResponse,
    SourceChunk,
)

log = logging.getLogger(__name__)


# ── Lifespan: pre-warm BM25 index on startup ──────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm BM25 indexes for all 'ready' books at startup."""
    from dotenv import load_dotenv
    load_dotenv()

    from bookrag.config import get_settings
    from bookrag.db.session import get_db
    from bookrag.db.models import Book
    from bookrag.retrieval.bm25 import BM25IndexManager

    settings = get_settings()

    if settings.enable_bm25:
        try:
            with next(get_db()) as db:
                books = db.query(Book).filter(Book.status == "ready").all()
                if books:
                    index_dir = Path(settings.bm25_index_dir)
                    manager = BM25IndexManager(index_dir=index_dir)
                    for book in books:
                        try:
                            manager.build_index([book.id], db, force=False)
                            log.info(f"BM25 pre-warmed for book: {book.title}")
                        except Exception as e:
                            log.warning(f"BM25 pre-warm failed for {book.id}: {e}")
        except Exception as e:
            log.warning(f"BM25 pre-warm skipped: {e}")

    yield
    # Shutdown — nothing to clean up


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BookRAG API",
    description="Open-source book Q&A API powered by local LLMs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_db_session():
    """Yield a sync DB session."""
    from bookrag.db.session import get_db
    with next(get_db()) as db:
        yield db


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness + readiness check."""
    from bookrag.config import get_settings
    from bookrag.db.session import get_db
    from bookrag.db.models import Book
    from bookrag.retrieval.bm25 import BM25IndexManager
    from pathlib import Path
    from sqlalchemy import text as sql_text

    settings = get_settings()
    db_status = "connected"
    bm25_count = 0

    try:
        with next(get_db()) as db:
            db.execute(sql_text("SELECT 1"))
            if settings.enable_bm25:
                books = db.query(Book).filter(Book.status == "ready").all()
                index_dir = Path(settings.bm25_index_dir)
                manager = BM25IndexManager(index_dir=index_dir)
                bm25_count = sum(
                    1 for b in books
                    if manager.get_index_path([b.id]).exists()
                )
    except Exception as e:
        db_status = f"error: {e}"

    overall = "ok" if db_status == "connected" else "degraded"
    return HealthResponse(status=overall, db=db_status, bm25_indexes=bm25_count)


@app.get("/books", response_model=BooksResponse)
def list_books():
    """List all ingested books."""
    from bookrag.db.session import get_db
    from bookrag.db.models import Book

    try:
        with next(get_db()) as db:
            books = db.query(Book).all()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")

    items = [
        BookItem(
            id=b.id,
            title=b.title,
            author=getattr(b, "author", None),
            num_pages=getattr(b, "num_pages", None),
            status=b.status,
        )
        for b in books
    ]
    return BooksResponse(books=items, total=len(items))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Ask a question and get an answer from the ingested books."""
    from bookrag.db.session import get_db
    from bookrag.db.models import Book
    from bookrag.agent.core import agent_ask

    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    try:
        with next(get_db()) as db:
            # Resolve book scope
            if request.book_ids:
                books = db.query(Book).filter(Book.id.in_(request.book_ids)).all()
                if not books:
                    raise HTTPException(status_code=404, detail="No matching books found.")
            else:
                books = db.query(Book).filter(Book.status == "ready").all()

            if not books:
                raise HTTPException(status_code=404, detail="No ready books in the system.")

            book_ids = [b.id for b in books]

            # Run the agent
            result = agent_ask(
                question=request.question,
                book_ids=book_ids,
                session_id=request.session_id,
                db=db,
            )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("agent_ask failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Build source list
    sources = [
        SourceChunk(
            parent_id=c.parent_id,
            chapter_title=c.chapter_title,
            page_start=c.page_start,
            page_end=c.page_end,
            text=c.parent_text[:400],  # truncate for API response
            score=round(c.score, 4),
        )
        for c in getattr(result, "chunks", [])
    ]

    return AskResponse(
        answer=result.answer if hasattr(result, "answer") else str(result),
        session_id=request.session_id,
        sources=sources,
        book_ids_used=book_ids,
    )
