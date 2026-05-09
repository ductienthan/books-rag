"""
Layer 4 — CLI / TUI (typer + rich).

Commands:
  bookrag setup                    First-time setup wizard
  bookrag add <file>               Ingest a PDF or EPUB
  bookrag list                     List all ingested books
  bookrag status [job_id]          Check ingestion job status
  bookrag ask '<question>'         Ask a question (interactive or one-shot)
  bookrag session new              Create a new session
  bookrag session scope            Set book scope for current session
  bookrag remove <book_id>         Remove a book and all its data
  bookrag cache-stats              Display cache statistics
  bookrag clear-cache              Clear query result cache
  bookrag build-bm25-index         Build BM25 keyword search indexes
  bookrag rebuild-bm25-index       Rebuild all BM25 indexes
  bookrag debug-retrieve '<q>'     Inspect retrieved chunks without generating an answer
  bookrag serve                    Start the FastAPI HTTP server
  bookrag eval                     Run retrieval evaluation on eval_questions.json
"""
from __future__ import annotations
from rich.text import Text
from rich.table import Table
from rich.prompt import Confirm, Prompt
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.console import Console
from rich import print as rprint
import typer
from sqlalchemy import text

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()


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

  import sys
  import traceback as _tb
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
            rprint(
                f"[yellow]Re-ingesting '[bold]{existing.title}[/bold]' (--force)…[/yellow]")
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
            db.query(Message).filter(Message.session_id ==
                                     s.id).delete(synchronize_session=False)
            db.delete(s)
          try:
            db.query(ChapterSummary).filter(ChapterSummary.book_id ==
                                            book_id).delete(synchronize_session=False)
            db.query(ChildChunk).filter(ChildChunk.book_id ==
                                        book_id).delete(synchronize_session=False)
            db.query(ParentChunk).filter(ParentChunk.book_id ==
                                         book_id).delete(synchronize_session=False)
            db.query(Chapter).filter(Chapter.book_id ==
                                     book_id).delete(synchronize_session=False)
            db.query(IngestionJob).filter(IngestionJob.book_id ==
                                          book_id).delete(synchronize_session=False)
            db.query(Book).filter(Book.id == book_id).delete(
                synchronize_session=False)
            db.flush()
          except Exception as exc:
            if "lock_timeout" in str(exc).lower() or "LockNotAvailable" in type(exc).__name__:
              rprint(
                  "[red]✗ Cannot delete old data:[/red] the worker is still processing this book.")
              rprint(
                  "[dim]Wait for the worker to finish (or stop it), then re-run with --force.[/dim]")
              raise typer.Exit(1)
            raise
          if _SESSION_FILE.exists():
            _SESSION_FILE.unlink()
        else:
          rprint(
              f"[yellow]Book already ingested:[/yellow] '{existing.title}' (id: {existing.id[:8]}…)")
          rprint("[dim]Use --force to re-ingest.[/dim]")
          raise typer.Exit(0)
      else:
        count = db.query(Book).count()
        if count >= cfg.max_books:
          rprint(
              f"[red]Book limit reached:[/red] maximum {cfg.max_books} books allowed.")
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
      job = IngestionJob(
          book_id=book.id, status="processing" if local else "queued")
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
    rprint(
        f"[green]✓[/green] Ingesting [bold]{detected_title}[/bold] locally…")
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


# ── book name resolver ────────────────────────────────────────────────────────

def _resolve_book_names_to_ids(books_input: str, db) -> list[str]:
  """Resolve comma-separated book names to UUIDs via fuzzy matching.

  Always treats input as book titles/names — users never need to know internal IDs.
  Misspellings are handled with suggestions.
  """
  from rapidfuzz import process as rfprocess, fuzz
  from bookrag.db.models import Book

  all_books = db.query(Book).filter(Book.status == "completed").all()
  if not all_books:
    console.print(
        "[red]No books have been ingested yet. Run 'bookrag add' first.[/red]")
    raise typer.Exit(1)

  title_to_book: dict[str, Book] = {b.title: b for b in all_books}
  for b in all_books:
    if b.author:
      title_to_book.setdefault(b.author, b)

  resolved: list[str] = []
  for raw in books_input.split(","):
    entry = raw.strip()
    if not entry:
      continue
    result = rfprocess.extractOne(
        entry, title_to_book.keys(), scorer=fuzz.partial_ratio)
    if result and result[1] >= 60:
      book = title_to_book[result[0]]
      if result[0].lower() != entry.lower():
        console.print(
            f'[cyan]Scoped to: "{result[0]}" (matched "{entry}", {result[1]:.0f}%)[/cyan]'
        )
      resolved.append(book.id)
    elif result and result[1] >= 40:
      console.print(
          f'[red]No book matched "{entry}". Did you mean: "{result[0]}" ({result[1]:.0f}%)?[/red]'
      )
      raise typer.Exit(1)
    else:
      available = ", ".join(f'"{b.title}"' for b in all_books)
      console.print(
          f'[red]No book matched "{entry}". Available books: {available}[/red]')
      raise typer.Exit(1)
  return resolved


# ── ask ───────────────────────────────────────────────────────────────────────

@app.command()
def ask(
    question: Optional[str] = typer.Argument(
        None, help="Question to ask (omit for interactive mode)"),
    session_id: Optional[str] = typer.Option(
        None, "--session", "-s", help="Session ID to use"),
    books: Optional[str] = typer.Option(
        None, "--books", "-b",
        help="Comma-separated book names to scope to (partial/misspelled names accepted)"),
    debug: bool = typer.Option(
        False, "--debug", help="Print retrieved chunks before the answer"),
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

    # Apply book scope if specified; clear any lingering scope when omitted
    if books:
      book_id_list = _resolve_book_names_to_ids(books, db)
      set_session_scope(sid, book_id_list, db)
    else:
      set_session_scope(sid, [], db)

  if question:
    # Check if streaming is enabled
    from bookrag.config import get_settings
    settings = get_settings()
    if settings.enable_streaming:
      _run_query_stream(question, sid, debug=debug)
    else:
      _run_query(question, sid, debug=debug)
  else:
    _interactive_loop(sid)


def _run_query_stream(question: str, session_id: str, debug: bool = False) -> None:
  """Run query with streaming display (Phase 3.3)."""
  import time
  from rich.live import Live
  from rich.spinner import Spinner
  from rich.text import Text
  from bookrag.agent.loop import ask_stream, StreamUpdate

  start = time.monotonic()

  # Variables to accumulate state
  answer_parts = []
  progress_messages = []
  final_response = None

  # Create initial display
  progress_text = Text("", style="cyan")
  answer_text = Text("", style="white")

  with Live("", console=console, refresh_per_second=10) as live:
    with _get_db() as db:
      for update in ask_stream(question, session_id, db):
        if update.type == "progress":
          # Update progress message
          progress_messages.append(update.data)
          progress_text = Text(f"⏳ {update.data}", style="cyan")
          live.update(progress_text)

        elif update.type == "chunk":
          # Append answer chunk and update display
          answer_parts.append(update.data)
          current_answer = "".join(answer_parts)

          # Show progress + partial answer
          display = Text()
          display.append("✓ Generating answer...\n\n", style="green")
          display.append(current_answer, style="white")
          live.update(display)

        elif update.type == "complete":
          # Store final response
          final_response = update.data

  elapsed = time.monotonic() - start

  if not final_response:
    console.print("[red]Error: No response received[/red]")
    return

  # Debug: print raw retrieved chunks before the answer
  if debug and final_response.chunks:
    console.print(f"\n[bold yellow]── Debug: {len(final_response.chunks)} retrieved chunk(s) ──[/bold yellow]\n")
    _print_debug_chunks(final_response.chunks)

  # Now display final formatted output (same as non-streaming)
  warning = "" if final_response.is_grounded else "\n⚠️  [dim]Low confidence — answer may not be fully supported by the books.[/dim]"
  console.print(Panel(
      Markdown(final_response.answer + warning),
      title="[bold cyan]Answer[/bold cyan]",
      border_style="cyan",
  ))

  # Text excerpts
  if final_response.chunks:
    console.print("\n[bold]Relevant Excerpts:[/bold]\n")
    for i, chunk in enumerate(final_response.chunks[:3], 1):
      excerpt = chunk.parent_text
      if len(excerpt) > 400:
        excerpt = excerpt[:400] + "..."

      source_info = f"[dim]{chunk.book_title}"
      if chunk.chapter_title:
        source_info += f" • {chunk.chapter_title}"
      if chunk.page_start:
        if chunk.page_start == chunk.page_end:
          source_info += f" • p. {chunk.page_start}"
        else:
          source_info += f" • pp. {chunk.page_start}–{chunk.page_end}"
      source_info += "[/dim]"

      console.print(Panel(
          f"{excerpt}\n\n{source_info}",
          title=f"[dim]Excerpt {i}[/dim]",
          border_style="dim",
          padding=(1, 2),
      ))
    console.print()

  # Sources table
  if final_response.sources:
    table = Table(show_header=True, header_style="dim",
                  box=None, padding=(0, 2))
    table.add_column("Book", style="bold")
    table.add_column("Chapter")
    table.add_column("Pages")
    table.add_column("Score", justify="right", style="dim")
    for src in final_response.sources:
      table.add_row(src["book"], src["chapter"],
                    src["pages"], str(src["score"]))
    console.print(table)

  console.print(f"[dim]Answered in {elapsed:.1f}s[/dim]")


def _run_query(question: str, session_id: str, debug: bool = False) -> None:
  import time
  from bookrag.agent.loop import ask as agent_ask

  start = time.monotonic()
  with console.status("[bold cyan]Thinking…[/bold cyan]", spinner="dots"):
    with _get_db() as db:
      response = agent_ask(question, session_id, db)
  elapsed = time.monotonic() - start

  # Debug: print raw retrieved chunks before the answer
  if debug and response.chunks:
    console.print(f"\n[bold yellow]── Debug: {len(response.chunks)} retrieved chunk(s) ──[/bold yellow]\n")
    _print_debug_chunks(response.chunks)

  # Answer panel
  warning = "" if response.is_grounded else "\n⚠️  [dim]Low confidence — answer may not be fully supported by the books.[/dim]"
  console.print(Panel(
      Markdown(response.answer + warning),
      title="[bold cyan]Answer[/bold cyan]",
      border_style="cyan",
  ))

  # Text excerpts from retrieved chunks
  if response.chunks:
    console.print("\n[bold]Relevant Excerpts:[/bold]\n")
    # Display up to 3 excerpts
    for i, chunk in enumerate(response.chunks[:3], 1):
      # Truncate long text to 400 characters for display
      excerpt = chunk.parent_text
      if len(excerpt) > 400:
        excerpt = excerpt[:400] + "..."

      # Build source citation
      source_info = f"[dim]{chunk.book_title}"
      if chunk.chapter_title:
        source_info += f" • {chunk.chapter_title}"
      if chunk.page_start:
        if chunk.page_start == chunk.page_end:
          source_info += f" • p. {chunk.page_start}"
        else:
          source_info += f" • pp. {chunk.page_start}–{chunk.page_end}"
      source_info += "[/dim]"

      console.print(Panel(
          f"{excerpt}\n\n{source_info}",
          title=f"[dim]Excerpt {i}[/dim]",
          border_style="dim",
          padding=(1, 2),
      ))
    console.print()

  # Sources table
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


# ── debug_retrieve ────────────────────────────────────────────────────────────

def _print_debug_chunks(chunks: list, show_parent: bool = False) -> None:
  """Print retrieved chunks in a human-readable debug format."""
  if not chunks:
    console.print("[yellow]No chunks retrieved.[/yellow]")
    return
  for i, chunk in enumerate(chunks, 1):
    console.rule(f"Chunk {i} | score: {chunk.score:.4f}")
    page_range = ""
    if chunk.page_start and chunk.page_end:
      page_range = f" | Pages: {chunk.page_start}–{chunk.page_end}"
    elif chunk.page_start:
      page_range = f" | Page: {chunk.page_start}"
    chapter_info = f"Ch {chunk.chapter_index}"
    if chunk.chapter_title:
      chapter_info += f": {chunk.chapter_title}"
    console.print(
        f"[dim]Book: {chunk.book_title} | {chapter_info}{page_range}[/dim]"
    )
    console.print(chunk.child_text)
    if show_parent:
      console.print("\n[dim italic]--- Parent context ---[/dim italic]")
      console.print(chunk.parent_text)
    console.print()


@app.command()
def debug_retrieve(
    query: str = typer.Argument(..., help="Query to retrieve chunks for"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of chunks to retrieve"),
    book_id: Optional[str] = typer.Option(None, "--book-id", help="Limit to a specific book ID"),
    show_parent: bool = typer.Option(False, "--show-parent", help="Also print parent context window"),
):
  """Inspect retrieved chunks for a query without generating an answer."""
  from bookrag.retrieval.searcher import search

  book_ids = [book_id] if book_id else []

  console.print(f"[bold cyan]Retrieving top {top_k} chunks for:[/bold cyan] {query}")
  if book_ids:
    console.print(f"[dim]Scoped to book: {book_id}[/dim]")
  console.print()

  with _get_db() as db:
    results = search([query], book_ids, db, top_k=top_k)

  console.print(f"[bold]Found {len(results)} chunk(s)[/bold]\n")
  _print_debug_chunks(results, show_parent=show_parent)


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


# ── Phase 3: Cache Management ─────────────────────────────────────────────────

@app.command(name="cache-stats")
def cache_stats():
  """Display cache statistics and performance metrics."""
  from pathlib import Path
  from bookrag.config import get_settings
  from bookrag.retrieval.cache import CacheManager

  settings = get_settings()
  cache_mgr = CacheManager(
      cache_dir=Path(settings.cache_dir),
      answer_ttl=settings.answer_cache_ttl,
      enable_embedding_cache=settings.enable_embedding_cache,
      enable_answer_cache=settings.enable_answer_cache,
  )

  console.rule("[bold cyan]Cache Statistics[/bold cyan]")

  stats = cache_mgr.stats()

  # Overall stats table
  table = Table(title="Cache Overview", show_header=True,
                header_style="bold cyan")
  table.add_column("Metric", style="dim")
  table.add_column("Value", justify="right")

  if "embedding_cache" in stats:
    table.add_row("Embedding Cache Entries", str(
        stats["embedding_cache"]["count"]))
    table.add_row("Embedding Cache Size",
                  f"{stats['embedding_cache']['size_mb']:.2f} MB")

  if "answer_cache" in stats:
    answer_stats = stats["answer_cache"]
    table.add_row("Answer Cache Entries (Total)",
                  str(answer_stats["total_entries"]))
    table.add_row("Answer Cache Entries (Valid)",
                  f"[green]{answer_stats['valid_entries']}[/green]")
    table.add_row("Answer Cache Entries (Expired)",
                  f"[yellow]{answer_stats['expired_entries']}[/yellow]")
    table.add_row("Answer Cache Size", f"{answer_stats['size_mb']:.2f} MB")
    table.add_row("Answer Cache TTL",
                  f"{answer_stats['ttl_seconds']}s ({answer_stats['ttl_seconds']//60}min)")

  table.add_row("──────────", "──────────")
  table.add_row("[bold]Total Cache Size[/bold]",
                f"[bold]{stats['total_size_mb']:.2f} MB[/bold]")

  console.print(table)

  # Cache status
  if "answer_cache" in stats and stats["answer_cache"]["expired_entries"] > 0:
    console.print(
        f"\n[yellow]💡 Tip:[/yellow] Run [bold]bookrag clear-cache --expired[/bold] "
        f"to remove {stats['answer_cache']['expired_entries']} expired entries."
    )

  # Performance estimates
  if "answer_cache" in stats and stats["answer_cache"]["valid_entries"] > 0:
    avg_time_saved = 120  # seconds (average query time without cache)
    # assume 2 uses per cache entry
    total_hits_estimated = stats["answer_cache"]["valid_entries"] * 2
    time_saved_total = avg_time_saved * total_hits_estimated / 60  # minutes

    console.print(
        f"\n[green]⚡ Performance:[/green] With {stats['answer_cache']['valid_entries']} "
        f"cached answers, you've potentially saved ~{time_saved_total:.0f} minutes of wait time!"
    )


@app.command(name="clear-cache")
def clear_cache(
    expired_only: bool = typer.Option(
        False, "--expired", "-e", help="Only clear expired cache entries"),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation"),
):
  """Clear the query result cache."""
  from pathlib import Path
  from bookrag.config import get_settings
  from bookrag.retrieval.cache import CacheManager

  settings = get_settings()
  cache_mgr = CacheManager(
      cache_dir=Path(settings.cache_dir),
      answer_ttl=settings.answer_cache_ttl,
      enable_embedding_cache=settings.enable_embedding_cache,
      enable_answer_cache=settings.enable_answer_cache,
  )

  if expired_only:
    # Clear only expired entries
    count = cache_mgr.clear_expired()
    rprint(f"[green]✓[/green] Cleared {count} expired cache entries.")
    return

  # Clear all caches
  if not yes:
    confirmed = Confirm.ask(
        "[yellow]Warning:[/yellow] This will clear ALL cached answers and embeddings. Continue?")
    if not confirmed:
      rprint("[dim]Cancelled.[/dim]")
      return

  stats = cache_mgr.clear_all()
  rprint(f"[green]✓[/green] Cache cleared:")
  rprint(f"  • Embeddings: {stats['embeddings_cleared']} entries")
  rprint(f"  • Answers: {stats['answers_cleared']} entries")
  rprint("\n[dim]Next queries will rebuild the cache.[/dim]")


# ── Phase 3: BM25 Index Management ────────────────────────────────────────────

@app.command(name="build-bm25-index")
def build_bm25_index(
    book_id: Optional[str] = typer.Argument(
        None, help="Build index for specific book ID (prefix). If omitted, builds for all books."),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force rebuild even if index exists"),
):
  """Build BM25 keyword search indexes for ingested books."""
  from pathlib import Path
  from bookrag.config import get_settings
  from bookrag.retrieval.bm25 import BM25IndexManager
  from bookrag.db.models import Book

  settings = get_settings()

  if not settings.enable_bm25:
    rprint("[yellow]⚠[/yellow]  BM25 is disabled in configuration (ENABLE_BM25=false).")
    rprint("[dim]Tip: Set ENABLE_BM25=true in .env to enable BM25 hybrid search.[/dim]")
    return

  console.rule("[bold cyan]BM25 Index Builder[/bold cyan]")

  with _get_db() as db:
    # Get books to index
    if book_id:
      # Build for specific book
      book = db.query(Book).filter(Book.id.like(f"{book_id}%")).first()
      if not book:
        rprint(f"[red]Book not found:[/red] {book_id}")
        raise typer.Exit(1)
      books = [book]
    else:
      # Build for all books
      books = db.query(Book).all()

    if not books:
      rprint("[yellow]No books found to index.[/yellow]")
      return

    rprint(f"Building BM25 index for {len(books)} book(s)...\n")

    index_dir = Path(settings.bm25_index_dir)
    manager = BM25IndexManager(index_dir=index_dir)

    # Build indexes
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
      task = progress.add_task("[cyan]Building indexes...", total=len(books))

      for book in books:
        progress.update(
            task, description=f"[cyan]Building: {book.title[:40]}...")

        try:
          index = manager.build_index([book.id], db, force=force)
          stats = index.stats()

          rprint(f"  [green]✓[/green] {book.title[:50]}")
          rprint(f"    [dim]└─ {stats['num_documents']} chunks, "
                 f"{stats['total_tokens']} tokens indexed[/dim]")

        except Exception as e:
          rprint(f"  [red]✗[/red] {book.title[:50]}")
          rprint(f"    [dim]└─ Error: {e}[/dim]")

        progress.advance(task)

    rprint(f"\n[green]✓[/green] BM25 index building complete!")
    rprint(f"[dim]Indexes saved to: {index_dir}[/dim]")


@app.command(name="rebuild-bm25-index")
def rebuild_bm25_index(
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation"),
):
  """Rebuild ALL BM25 indexes (force rebuild)."""
  from pathlib import Path
  from bookrag.config import get_settings
  from bookrag.retrieval.bm25 import BM25IndexManager

  settings = get_settings()

  if not settings.enable_bm25:
    rprint("[yellow]⚠[/yellow]  BM25 is disabled in configuration (ENABLE_BM25=false).")
    return

  # Confirm
  if not yes:
    confirmed = Confirm.ask(
        "[yellow]Warning:[/yellow] This will rebuild ALL BM25 indexes. Continue?")
    if not confirmed:
      rprint("[dim]Cancelled.[/dim]")
      return

  console.rule("[bold cyan]BM25 Index Rebuild[/bold cyan]")

  with _get_db() as db:
    index_dir = Path(settings.bm25_index_dir)
    manager = BM25IndexManager(index_dir=index_dir)

    rprint("Rebuilding all BM25 indexes...\n")

    try:
      count = manager.rebuild_all(db)
      rprint(f"\n[green]✓[/green] Rebuilt {count} BM25 indexes successfully!")
      rprint(f"[dim]Indexes saved to: {index_dir}[/dim]")

    except Exception as e:
      rprint(f"[red]Error:[/red] {e}")
      raise typer.Exit(1)


# ── HTTP API Server ───────────────────────────────────────────────────────────

@app.command(name="serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Bind address"),
    port: int = typer.Option(8080, "--port", "-p", help="Port number"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev only)"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of Uvicorn worker processes"),
):
  """Start the BookRAG FastAPI HTTP server.

  The server exposes:
    GET  /health   — liveness check (DB + BM25 index status)
    GET  /books    — list all ingested books
    POST /ask      — answer a question from the ingested books

  Example:
    bookrag serve --port 8080
    curl http://localhost:8080/health
    curl -X POST http://localhost:8080/ask \\
         -H 'Content-Type: application/json' \\
         -d '{"question": "What is motivational interviewing?"}'
  """
  try:
    import uvicorn
  except ImportError:
    rprint("[red]Error:[/red] uvicorn is not installed.")
    rprint("[dim]Run: pip install 'uvicorn[standard]'[/dim]")
    raise typer.Exit(1)

  console.rule(f"[bold cyan]BookRAG API Server[/bold cyan]")
  rprint(f"  [dim]Host:[/dim]    {host}")
  rprint(f"  [dim]Port:[/dim]    {port}")
  rprint(f"  [dim]Workers:[/dim] {workers}")
  rprint(f"  [dim]Docs:[/dim]    http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs\n")

  uvicorn.run(
    "bookrag.api.server:app",
    host=host,
    port=port,
    reload=reload,
    workers=workers if not reload else 1,
    log_level="info",
  )


# ── Retrieval Evaluation ──────────────────────────────────────────────────────

@app.command(name="eval")
def eval_retrieval(
    questions_file: Path = typer.Option(
        Path("eval_questions.json"),
        "--questions", "-q",
        help="Path to JSON file with evaluation questions",
        exists=True,
    ),
    book_id: Optional[str] = typer.Option(
        None, "--book", "-b", help="Book ID (prefix) to evaluate against. Defaults to all books."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of chunks to retrieve per query"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save results JSON to this path"),
):
  """Run retrieval evaluation against eval_questions.json.

  Measures how well the hybrid retriever surfaces relevant chunks for each
  question using keyword-coverage as a proxy relevance signal.

  Metrics reported:
    • Keyword Coverage   — fraction of question keywords found in top-K chunks
    • Keyword Hit Rate   — fraction of questions where ≥1 keyword is covered
    • Avg chunks returned

  Example:
    bookrag eval --questions eval_questions.json --book bb1c7eb5 --top-k 10
  """
  import json
  from bookrag.config import get_settings
  from bookrag.db.models import Book
  from bookrag.retrieval.searcher import search as retrieval_search

  settings = get_settings()

  console.rule("[bold cyan]BookRAG Retrieval Evaluation[/bold cyan]")
  rprint(f"  [dim]Questions:[/dim] {questions_file}")
  rprint(f"  [dim]Top-K:[/dim]    {top_k}\n")

  # Load questions
  try:
    with open(questions_file) as f:
      questions = json.load(f)
  except Exception as e:
    rprint(f"[red]Error loading questions:[/red] {e}")
    raise typer.Exit(1)

  with _get_db() as db:
    # Resolve books
    if book_id:
      book = db.query(Book).filter(Book.id.like(f"{book_id}%")).first()
      if not book:
        rprint(f"[red]Book not found:[/red] {book_id}")
        raise typer.Exit(1)
      book_ids = [book.id]
      rprint(f"  [dim]Book:[/dim]     {book.title}\n")
    else:
      books = db.query(Book).filter(Book.status == "ready").all()
      book_ids = [b.id for b in books]
      rprint(f"  [dim]Books:[/dim]    {len(book_ids)} ready book(s)\n")

    if not book_ids:
      rprint("[yellow]No ready books found.[/yellow]")
      raise typer.Exit(1)

    # Run evaluation
    total = len(questions)
    keyword_coverages: list[float] = []
    hit_count = 0
    results_log: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
      task = progress.add_task(f"[cyan]Evaluating {total} questions...", total=total)

      for q in questions:
        qid = q.get("id", "?")
        question = q.get("question", "")
        keywords = [kw.lower() for kw in q.get("keywords", [])]

        try:
          chunks = retrieval_search([question], book_ids, db, top_k=top_k)
        except Exception as e:
          log_entry = {"id": qid, "question": question, "error": str(e), "coverage": 0.0}
          results_log.append(log_entry)
          keyword_coverages.append(0.0)
          progress.advance(task)
          continue

        # Compute keyword coverage: what fraction of keywords appear in top-K
        combined_text = " ".join(c.parent_text.lower() for c in chunks)
        if keywords:
          covered = sum(1 for kw in keywords if kw in combined_text)
          coverage = covered / len(keywords)
        else:
          coverage = 1.0

        keyword_coverages.append(coverage)
        if coverage > 0:
          hit_count += 1

        results_log.append({
          "id": qid,
          "category": q.get("category", ""),
          "question": question,
          "keywords": keywords,
          "chunks_returned": len(chunks),
          "keyword_coverage": round(coverage, 3),
        })

        progress.advance(task)

  # ── Summary ───────────────────────────────────────────────────────────────
  avg_coverage = sum(keyword_coverages) / len(keyword_coverages) if keyword_coverages else 0
  hit_rate = hit_count / total if total else 0
  avg_chunks = sum(r.get("chunks_returned", 0) for r in results_log) / total if total else 0

  console.rule("[bold green]Results[/bold green]")

  table = Table(show_header=True, header_style="bold cyan")
  table.add_column("Metric", style="dim", width=28)
  table.add_column("Value", justify="right")
  table.add_row("Questions evaluated", str(total))
  table.add_row("Avg keyword coverage", f"{avg_coverage:.1%}")
  table.add_row("Keyword hit rate", f"{hit_rate:.1%}")
  table.add_row("Avg chunks returned", f"{avg_chunks:.1f}")
  console.print(table)

  # Grade
  if avg_coverage >= 0.75:
    grade = "[green]GOOD[/green]"
  elif avg_coverage >= 0.50:
    grade = "[yellow]NEEDS IMPROVEMENT[/yellow]"
  else:
    grade = "[red]POOR[/red]"

  rprint(f"\nRetrieval quality: {grade}")
  rprint(f"[dim](Keyword coverage ≥ 75% = GOOD, 50–74% = NEEDS IMPROVEMENT, <50% = POOR)[/dim]\n")

  # Save results if requested
  if output:
    summary = {
      "total_questions": total,
      "avg_keyword_coverage": round(avg_coverage, 4),
      "keyword_hit_rate": round(hit_rate, 4),
      "avg_chunks_returned": round(avg_chunks, 2),
      "top_k": top_k,
      "book_ids": book_ids,
      "results": results_log,
    }
    with open(output, "w") as f:
      json.dump(summary, f, indent=2)
    rprint(f"[green]✓[/green] Results saved to {output}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
  app()
