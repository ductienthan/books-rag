"""Database session factories — sync (for worker) and async (for CLI / API)."""
from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from bookrag.config import get_settings

_settings = get_settings()

# ── Sync engine (background worker) ──────────────────────────────────────────
_sync_engine = create_engine(
    _settings.postgres_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

_SyncSession = sessionmaker(bind=_sync_engine, expire_on_commit=False)


@contextmanager
def sync_session() -> Generator[Session, None, None]:
    session = _SyncSession()
    try:
        # Set HNSW ef_search for this connection
        session.execute(text(f"SET hnsw.ef_search = {_settings.hnsw_ef_search}"))
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Async engine (CLI / API) ──────────────────────────────────────────────────
_async_engine = create_async_engine(
    _settings.async_postgres_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

_AsyncSession = async_sessionmaker(bind=_async_engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    session = _AsyncSession()
    try:
        await session.execute(text(f"SET hnsw.ef_search = {_settings.hnsw_ef_search}"))
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def get_sync_engine():
    return _sync_engine


def get_async_engine():
    return _async_engine
