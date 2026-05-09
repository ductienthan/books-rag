"""
Layer 3 / 5 — Session memory.

Loads the rolling window of recent messages from the DB for context injection.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from sqlalchemy.orm import Session as DBSession

from bookrag.config import get_settings
from bookrag.db.models import Message, Session

log = logging.getLogger(__name__)
_settings = get_settings()


def get_history(session_id: str, db: DBSession) -> list[dict]:
    """Return the last N messages for a session as a list of {role, content} dicts."""
    n = _settings.session_memory_window
    msgs = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(n)
        .all()
    )
    msgs.reverse()   # chronological order
    return [{"role": m.role, "content": m.content} for m in msgs]


def save_message(
    session_id: str,
    role: str,
    content: str,
    db: DBSession,
    book_ids: list[str] | None = None,
    chunk_ids: list[str] | None = None,
) -> Message:
    """Persist a message and update session.last_active_at."""
    msg = Message(
        session_id=session_id,
        role=role,
        content=content,
        book_ids_referenced=json.dumps(book_ids) if book_ids else None,
        chunk_ids_used=json.dumps(chunk_ids) if chunk_ids else None,
    )
    db.add(msg)

    session = db.query(Session).filter(Session.id == session_id).one()
    session.last_active_at = datetime.utcnow()
    db.flush()
    return msg


def create_session(db: DBSession, book_ids: list[str] | None = None) -> Session:
    """Create a new session, optionally scoped to specific books."""
    session = Session(
        book_scope=json.dumps(book_ids) if book_ids else None,
    )
    db.add(session)
    db.flush()
    return session


def set_session_scope(session_id: str, book_ids: list[str], db: DBSession) -> None:
    """Update the book scope for an existing session. Pass empty list to clear scope."""
    session = db.query(Session).filter(Session.id == session_id).one()
    session.book_scope = json.dumps(book_ids) if book_ids else None
    db.flush()
