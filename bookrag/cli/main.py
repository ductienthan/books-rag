"""
Layer 4 — CLI / TUI (typer + rich).

Commands:
  bookrag setup               First-time setup wizard
  bookrag add <file>          Ingest a PDF or EPUB
  bookrag list                List all ingested books
  bookrag status [job_id]     Check ingestion job status
  bookrag ask '<question>'    Ask a question (interactive or one-shot)
  bookrag session new         Create a new session
  bookrag session scope       Set book scope for current session
  bookrag remove <book_id>    Remove a book and all its data
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
import typer
from rich import print as rprint
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="bookrag",
    help="Open-source book ingestion and Q&A system powered by local LLMs.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# ── Shared state (session_id persisted per-process in a temp file) ─────────────
_SESSION_FILE = Path.home() / ".bookrag_session"


def _get_session_id() -> str | None:
  if _SESSION_FILE.exists():
    sid = _SESSION_FILE.read_text().strip()
    return sid if sid else None
  return None


def _set_session_id(session_id: str) -> None:
  _SESSION_FILE.write_text(session_id)


def _get_db():
  """Return a sync DB session."""
  from bookrag.db.session import sync_session
  return sync_session()


def _reset_db_pool():
  """Dispose the connection pool to clear any stale idle-in-transaction connections."""
  from bookrag.db.session import get_sync_engine
  get_sync_engine().dispose()


# ── setup ─────────────────────────────────────────────────────────────────────

@app.command()
def setup():
  """First-time setup: initialise database and verify models."""
  console.rule("[bold cyan]BookRAG Setup[/bold cyan]")

  # 1. Run Alembic migrations
  rprint("[bold]1.[/bold] Initialising database schema…")
  try:
    import psycopg
    from sqlalchemy.engine import make_url
    from alembic.config import Config
    from alembic import command as alembic_cmd
    from bookrag.config import get_settings
    cfg = get_settings()

    # Create the database if it doesn't exist
    url = make_url(cfg.postgres_url)
    db_name = url.database
    admin_dsn = f"host={url.host} port={url.port or 5432} dbname=postgres user={url.username} password={url.password}"
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
      exists = conn.execute(
          "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
      ).fetchone()
      if not exists:
        conn.execute(f'CREATE DATABASE "{db_name}"')
        rprint(f"   [green]✓[/green] Created database '{db_name}'.")
      else:
        rprint(f"   [dim]Database '{db_name}' already exists.[/dim]")

    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", cfg.postgres_url)
    alembic_cmd.upgrade(alembic_cfg, "head")
    rprint("   [green]✓[/green] Database schema ready.")
  except Exception as exc:
    rprint(f"   [red]✗ Database init failed:[/red] {exc}")
    raise typer.Exit(1)

  # 2. Verify Ollama is reachable
  rprint("[bold]2.[/bold] Checking Ollama connection…")
  try:
    import ollama
    from bookrag.config import get_settings
    cfg = get_settings()
    client = ollama.Client(host=cfg.ollama_host)
    models = client.list()
    rprint(f"   [green]✓[/green] Ollama reachable at {cfg.ollama_host}")

    model_names = [m.model for m in models.models]
    if cfg.ollama_llm_model not in model_names:
      rprint(
          f"   [yellow]![/yellow] Pulling {cfg.ollama_llm_model} (this may take a few minutes)…")
      client.pull(cfg.ollama_llm_model)
      rprint(f"   [green]✓[/green] {cfg.ollama_llm_model} ready.")
    else:
      rprint(f"   [green]✓[/green] {cfg.ollama_llm_model} already present.")
  except Exception as exc:
    rprint(f"   [yellow]![/yellow] Ollama check failed: {exc}")
    rprint("     Make sure Ollama is running: https://ollama.com")

  # 3. Load embedding model
  rprint("[bold]3.[/bold] Loading embedding model (first run downloads ~270 MB)…")
  try:
    from bookrag.ingestion.embedder import _get_model
    _get_model()
    rprint("   [green]✓[/green] Embedding model ready.")
  except Exception as exc:
    rprint(f"   [red]✗ Embedding model failed:[/red] {exc}")
    raise typer.Exit(1)

  # 4. Create a default session
  try:
    with _get_db() as db:
      from bookrag.agent.memory import create_session
      s = create_session(db)
      _set_session_id(s.id)
    rprint(f"   [green]✓[/green] Default session created: {s.id[:8]}…")
  except Exception as exc:
    rprint(f"   [yellow]![/yellow] Session init warning: {exc}")

  console.rule()
  rprint("[bold green]Setup complete![/bold green] Run [bold]bookrag add <file>[/bold] to ingest your first book.")


# ── add ───────────────────────────────────────────────────────────────────────

@app.command()
def add(
    file: Path = typer.Argument(..., help="Path to PDF or EPUB file"),
    title: Optional[str] = typer.Option(
        None, "--title", "-t", help="Override book title"),
    author: Optional[str] = typer.Option(
        None, "--author", "-a", help="Override author name"),
    wait: bool = typer.Option(False, "--wait", "-w",
                              help="Wait for ingestion to complete"),
    local: bool = typer.Option(False, "--local", "-l",
                               help="Run ingestion in this process (no Docker worker needed)"),
    force: bool = typer.Option(False, "--force", "-f",
                               help="Re-ingest even if book already exists"),
):
  """Ingest a PDF or EPUB book into BookRAG."""
  if local:
    import logging
    import sys
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.root.setLevel(logging.INFO)
    logging.root.handlers = [handler]

  if not file.exists():
    rprint(f"[red]File not found:[/red] {file}")
    raise typer.Exit(1)

  suffix = file.suffix.lower()
  if suffix not in (".pdf", ".epub"):
    rprint(f"[red]Unsupported file type:[/red] {suffix} (use .pdf or .epub)")
    raise typer.Exit(1)

  import hashlib
  from bookrag.db.models import Book, IngestionJob

  # Compute file hash
  sha256 = hashlib.sha256(file.read_bytes()).hexdigest()

  import sys, traceback as _tb
  _reset_db_pool()  # clear any stale idle-in-transaction connections from previous runs
  try:
    with _get_db() as db:
      from bookrag.config import get_settings
      cfg = get_settings()

      existing = db.query(Book).filter(Book.file_hash == sha256).first()
      if existing:
        # Treat any incomplete/interrupted status as re-ingestable without --force.
        # "completed" is the only status that means the ingestion finished successfully.
        _INCOMPLETE = {"pending", "processing", "extracting", "chunking",
                       "embedding", "summarising", "indexing", "failed"}
        should_reingest = force or existing.status in _INCOMPLETE
        if should_reingest:
          if existing.status == "completed":
            rprint(f"[yellow]Re-ingesting '[bold]{existing.title}[/bold]' (--force)…[/yellow]")
          else:
            rprint(f"[yellow]Re-ingesting '[bold]{existing.title}[/bold]' "
                   f"(previous run was interrupted at status='{existing.status}')…[/yellow]")
          from bookrag.db.models import ChapterSummary, ChildChunk, ParentChunk, Chapter, IngestionJob, Message, Session as DBSession
          book_id = existing.id
          # Fail fast if the worker holds a lock on this book's rows instead of hanging.
          db.execute(text("SET LOCAL lock_timeout = '8s'"))
          # Expunge from identity map so ORM doesn't try to cascade on top of bulk deletes
          db.expunge(existing)
          # Delete only sessions scoped to this specific book; messages cascade per session.
          stale_sessions = (
              db.query(DBSession)
              .filter(DBSession.book_scope.like(f'%"{book_id}"%'))
              .all()
          )
          for s in stale_sessions:
              db.query(Message).filter(Message.session_id == s.id).delete(synchronize_session=False)
              db.delete(s)
          try:
            db.query(ChapterSummary).filter(ChapterSummary.book_id == book_id).delete(synchronize_session=False)
            db.query(ChildChunk).filter(ChildChunk.book_id == book_id).delete(synchronize_session=False)
            db.query(ParentChunk).filter(ParentChunk.book_id == book_id).delete(synchronize_session=False)
            db.query(Chapter).filter(Chapter.book_id == book_id).delete(synchronize_session=False)
            db.query(IngestionJob).filter(IngestionJob.book_id == book_id).delete(synchronize_session=False)
            db.query(Book).filter(Book.id == book_id).delete(synchronize_session=False)
            db.flush()
          except Exception as exc:
            if "lock_timeout" in str(exc).lower() or "LockNotAvailable" in type(exc).__name__:
              rprint("[red]✗ Cannot delete old data:[/red] the worker is still processing this book.")
              rprint("[dim]Wait for the worker to finish (or stop it), then re-run with --force.[/dim]")
              raise typer.Exit(1)
            raise
          if _SESSION_FILE.exists():
            _SESSION_FILE.unlink()
        else:
          rprint(f"[yellow]Book already ingested:[/yellow] '{existing.title}' (id: {existing.id[:8]}…)")
          rprint("[dim]Use --force to re-ingest.[/dim]")
          raise typer.Exit(0)
      else:
        count = db.query(Book).count()
        if count >= cfg.max_books:
          rprint(f"[red]Book limit reached:[/red] maximum {cfg.max_books} books allowed.")
          raise typer.Exit(1)

      print("DEBUG: detecting metadata...", flush=True)
      detected_title = title or _detect_title(file) or file.stem
      detected_author = author or _detect_author(file)
      print(f"DEBUG: title={detected_title}", flush=True)

      import shutil
      data_books_dir = Path(cfg.data_books_dir)
      data_books_dir.mkdir(parents=True, exist_ok=True)
      dest = data_books_dir / f"{sha256[:16]}{suffix}"
      if not dest.exists():
        shutil.copy2(file, dest)
      print(f"DEBUG: file at {dest}", flush=True)

      book = Book(
          title=detected_title,
          author=detected_author,
          file_path=str(dest),
          file_hash=sha256,
          file_type=suffix.lstrip("."),
          status="pending",
      )
      db.add(book)
      print("DEBUG: flushing book...", flush=True)
      db.flush()
      print(f"DEBUG: book created id={book.id}", flush=True)

      # When running locally, claim the job immediately so the Docker worker
      # doesn't race to pick it up between commit and process_job() starting.
      job = IngestionJob(book_id=book.id, status="processing" if local else "queued")
      db.add(job)
      db.flush()
      print(f"DEBUG: job created id={job.id}", flush=True)

      book_id = book.id
      job_id = job.id

  except BaseException as exc:
    print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stdout, flush=True)
    _tb.print_exc(file=sys.stdout)
    sys.stdout.flush()
    if isinstance(exc, typer.Exit):
      raise
    raise typer.Exit(1)

  if local:
    rprint(f"[green]✓[/green] Ingesting [bold]{detected_title}[/bold] locally…")
    _run_ingestion_local(job_id)
  else:
    rprint(f"[green]✓[/green] Queued [bold]{detected_title}[/bold]")
    rprint(f"   Job ID: [dim]{job_id}[/dim]")
    rprint(f"   Expected time: [bold]15–20 minutes[/bold] for large books.")
    if wait:
      _watch_job(job_id)
    else:
      rprint(
          f"\nRun [bold]bookrag status {job_id[:8]}[/bold] to check progress.")
      rprint("[dim]The worker processes jobs automatically in the background.[/dim]")


def _run_ingestion_local(job_id: str) -> None:
  """Run the full ingestion pipeline in the current process (no Docker worker needed)."""
  from bookrag.ingestion.worker import process_job
  try:
      process_job(job_id)
  except Exception as exc:
      rprint(f"[red]✗ Ingestion failed:[/red] {exc}")
      raise typer.Exit(1)

  # Check DB to confirm success (process_job swallows exceptions internally)
  from bookrag.db.models import IngestionJob
  with _get_db() as db:
      job = db.query(IngestionJob).filter(IngestionJob.id == job_id).one()
      if job.status == "failed":
          rprint(f"[red]✗ Ingestion failed:[/red] {job.error_msg}")
          raise typer.Exit(1)

  rprint("[green]✓[/green] Ingestion complete.")


def _detect_title(file: Path) -> str | None:
  try:
    if file.suffix.lower() == ".pdf":
      import pdfplumber
      with pdfplumber.open(str(file)) as pdf:
        return pdf.metadata.get("Title") or None
    elif file.suffix.lower() == ".epub":
      import ebooklib
      from ebooklib import epub
      book = epub.read_epub(str(file), options={"ignore_ncx": True})
      return book.get_metadata("DC", "title")[0][0] if book.get_metadata("DC", "title") else None
  except Exception:
    return None


def _detect_author(file: Path) -> str | None:
  try:
    if file.suffix.lower() == ".pdf":
      import pdfplumber
      with pdfplumber.open(str(file)) as pdf:
        return pdf.metadata.get("Author") or None
    elif file.suffix.lower() == ".epub":
      import ebooklib
      from ebooklib import epub
      book = epub.read_epub(str(file), options={"ignore_ncx": True})
      creators = book.get_metadata("DC", "creator")
      return creators[0][0] if creators else None
  except Exception:
    return None


# ── list ──────────────────────────────────────────────────────────────────────

@app.command(name="list")
def list_books():
  """List all ingested books."""
  from bookrag.db.models import Book, IngestionJob

  with _get_db() as db:
    books = db.query(Book).order_by(Book.created_at.desc()).all()

  if not books:
    rprint("[dim]No books ingested yet. Run [bold]bookrag add <file>[/bold] to get started.[/dim]")
    return

  table = Table(title="BookRAG Library", header_style="bold cyan")
  table.add_column("ID", style="dim", width=10)
  table.add_column("Title", style="bold")
  table.add_column("Author")
  table.add_column("Type", width=5)
  table.add_column("Pages", justify="right")
  table.add_column("Status")
  table.add_column("Added")

  STATUS_STYLE = {
      "completed": "[green]✓ ready[/green]",
      "processing": "[yellow]⟳ processing[/yellow]",
      "pending": "[dim]⏳ queued[/dim]",
      "failed": "[red]✗ failed[/red]",
  }

  for b in books:
    table.add_row(
        b.id[:8] + "…",
        b.title,
        b.author or "—",
        b.file_type.upper(),
        str(b.total_pages or "—"),
        STATUS_STYLE.get(b.status, b.status),
        b.created_at.strftime("%Y-%m-%d"),
    )

  console.print(table)


# ── status ────────────────────────────────────────────────────────────────────

@app.command()
def status(
    job_id: Optional[str] = typer.Argument(
        None, help="Job ID prefix (or omit for all active jobs)"),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Watch until complete"),
):
  """Check ingestion job status."""
  if watch and job_id:
    _watch_job(job_id)
    return

  from bookrag.db.models import IngestionJob, Book

  with _get_db() as db:
    q = db.query(IngestionJob)
    if job_id:
      q = q.filter(IngestionJob.id.like(f"{job_id}%"))
    else:
      q = q.filter(IngestionJob.status.in_(
          ["queued", "extracting", "chunking", "embedding", "indexing"]))
    jobs = q.order_by(IngestionJob.id).all()

  if not jobs:
    rprint("[dim]No active jobs found.[/dim]")
    return

  table = Table(header_style="bold cyan")
  table.add_column("Job ID", style="dim")
  table.add_column("Book")
  table.add_column("Status")
  table.add_column("Progress", justify="right")
  table.add_column("ETA")

  with _get_db() as db:
    for job in jobs:
      book = db.query(Book).filter(Book.id == job.book_id).first()
      eta = job.estimated_finish.strftime(
          "%H:%M:%S") if job.estimated_finish else "—"
      table.add_row(
          job.id[:8] + "…",
          book.title if book else "?",
          job.status,
          f"{job.progress_pct}%",
          eta,
      )

  console.print(table)


def _watch_job(job_id: str) -> None:
  """Poll job progress and display a live progress bar."""
  from bookrag.db.models import IngestionJob, Book

  with Progress(
      SpinnerColumn(),
      TextColumn("[bold]{task.description}"),
      BarColumn(),
      TaskProgressColumn(),
      TimeElapsedColumn(),
      console=console,
  ) as progress:
    task = progress.add_task("Ingesting…", total=100)

    while True:
      with _get_db() as db:
        job = db.query(IngestionJob).filter(
            IngestionJob.id.like(f"{job_id}%")).first()
        if not job:
          rprint(f"[red]Job not found:[/red] {job_id}")
          break
        book = db.query(Book).filter(Book.id == job.book_id).first()

      progress.update(
          task,
          completed=job.progress_pct,
          description=f"{book.title if book else '?'} — {job.status}",
      )

      if job.status in ("done", "failed"):
        if job.status == "done":
          rprint(
              f"\n[green]✓[/green] Ingestion complete: [bold]{book.title}[/bold]")
        else:
          rprint(f"\n[red]✗[/red] Ingestion failed: {job.error_msg}")
        break

      time.sleep(3)


# ── ask ───────────────────────────────────────────────────────────────────────

@app.command()
def ask(
    question: Optional[str] = typer.Argument(
        None, help="Question to ask (omit for interactive mode)"),
    session_id: Optional[str] = typer.Option(
        None, "--session", "-s", help="Session ID to use"),
    books: Optional[str] = typer.Option(
        None, "--books", "-b", help="Comma-separated book IDs to scope to"),
):
  """Ask a question against your ingested books."""
  from bookrag.db.models import Book, Session as DBSession
  from bookrag.agent.loop import ask as agent_ask
  from bookrag.agent.memory import create_session, set_session_scope

  # Resolve session
  sid = session_id or _get_session_id()
  with _get_db() as db:
    if not sid or not db.query(DBSession).filter(DBSession.id == sid).first():
      s = create_session(db)
      sid = s.id
      _set_session_id(sid)

    # Apply book scope if specified
    if books:
      book_id_list = [b.strip() for b in books.split(",")]
      set_session_scope(sid, book_id_list, db)

  if question:
    _run_query(question, sid)
  else:
    _interactive_loop(sid)


def _run_query(question: str, session_id: str) -> None:
  import time
  from bookrag.agent.loop import ask as agent_ask

  start = time.monotonic()
  with console.status("[bold cyan]Thinking…[/bold cyan]", spinner="dots"):
    with _get_db() as db:
      response = agent_ask(question, session_id, db)
  elapsed = time.monotonic() - start

  # Answer panel
  warning = "" if response.is_grounded else "\n⚠️  [dim]Low confidence — answer may not be fully supported by the books.[/dim]"
  console.print(Panel(
      Markdown(response.answer + warning),
      title="[bold cyan]Answer[/bold cyan]",
      border_style="cyan",
  ))

  # Sources
  if response.sources:
    table = Table(show_header=True, header_style="dim",
                  box=None, padding=(0, 2))
    table.add_column("Book", style="bold")
    table.add_column("Chapter")
    table.add_column("Pages")
    table.add_column("Score", justify="right", style="dim")
    for src in response.sources:
      table.add_row(src["book"], src["chapter"],
                    src["pages"], str(src["score"]))
    console.print(table)

  console.print(f"[dim]Answered in {elapsed:.1f}s[/dim]")


def _interactive_loop(session_id: str) -> None:
  rprint("[bold cyan]BookRAG[/bold cyan] — Interactive Mode")
  rprint("[dim]Type your question and press Enter. Type [bold]exit[/bold] or Ctrl+C to quit.[/dim]\n")

  while True:
    try:
      question = Prompt.ask("[bold]>[/bold]")
    except (KeyboardInterrupt, EOFError):
      rprint("\n[dim]Goodbye![/dim]")
      break

    if question.strip().lower() in ("exit", "quit", "q"):
      rprint("[dim]Goodbye![/dim]")
      break

    if not question.strip():
      continue

    _run_query(question.strip(), session_id)
    console.print()


# ── session ───────────────────────────────────────────────────────────────────

session_app = typer.Typer(help="Manage sessions.")
app.add_typer(session_app, name="session")


@session_app.command(name="new")
def session_new(
    books: Optional[str] = typer.Option(
        None, "--books", "-b", help="Comma-separated book IDs"),
):
  """Create a new session."""
  from bookrag.agent.memory import create_session

  book_ids = [b.strip() for b in books.split(",")] if books else None
  with _get_db() as db:
    s = create_session(db, book_ids=book_ids)
  _set_session_id(s.id)
  rprint(f"[green]✓[/green] New session: [bold]{s.id}[/bold]")


@session_app.command(name="scope")
def session_scope(
    books: str = typer.Argument(..., help="Comma-separated book ID prefixes"),
):
  """Set which books the current session searches."""
  from bookrag.db.models import Book
  from bookrag.agent.memory import set_session_scope

  sid = _get_session_id()
  if not sid:
    rprint("[red]No active session. Run [bold]bookrag session new[/bold] first.[/red]")
    raise typer.Exit(1)

  prefixes = [b.strip() for b in books.split(",")]
  with _get_db() as db:
    matched_ids = []
    for prefix in prefixes:
      book = db.query(Book).filter(Book.id.like(f"{prefix}%")).first()
      if book:
        matched_ids.append(book.id)
        rprint(f"  [green]✓[/green] {book.title} ({book.id[:8]}…)")
      else:
        rprint(f"  [yellow]![/yellow] No book found for prefix: {prefix}")

    if matched_ids:
      set_session_scope(sid, matched_ids, db)
      rprint(f"[green]✓[/green] Session scoped to {len(matched_ids)} book(s).")


# ── remove ────────────────────────────────────────────────────────────────────

@app.command()
def remove(
    book_id: str = typer.Argument(..., help="Book ID prefix"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
  """Remove a book and all its chunks/embeddings."""
  from bookrag.db.models import Book

  with _get_db() as db:
    book = db.query(Book).filter(Book.id.like(f"{book_id}%")).first()
    if not book:
      rprint(f"[red]Book not found:[/red] {book_id}")
      raise typer.Exit(1)

    if not yes:
      confirmed = Confirm.ask(
          f"Remove [bold]{book.title}[/bold] and ALL its data?")
      if not confirmed:
        rprint("[dim]Cancelled.[/dim]")
        return

    db.delete(book)
    rprint(f"[green]✓[/green] Removed [bold]{book.title}[/bold].")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
  app()
