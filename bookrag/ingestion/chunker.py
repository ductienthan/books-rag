"""
Layer 1 — Hierarchical chunking.

Produces three levels from raw chapter text:
  child  :  80–120 tokens  (embedded; used for ANN search)
  parent :  300–400 tokens (fetched at query time; sent to LLM)
  summary:  the full chapter text (later summarised by Gemma 4)
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import tiktoken

from bookrag.config import get_settings
from bookrag.ingestion.extractor import RawChapter

_settings = get_settings()
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class ChildChunkData:
    chunk_index: int
    text: str
    token_count: int
    text_hash: str
    char_start: int
    char_end: int


@dataclass
class ParentChunkData:
    chunk_index: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    children: list[ChildChunkData] = field(default_factory=list)


@dataclass
class ChunkResult:
    chapter_index: int
    chapter_title: str | None
    page_start: int | None
    page_end: int | None
    raw_text: str
    parents: list[ParentChunkData] = field(default_factory=list)


# ── Sentence splitter ─────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r'(?<=[.!?…])\s+(?=[A-Z"\'])')


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences. Falls back to word-boundary splits for very
    long sequences that have no sentence-ending punctuation.
    """
    parts = _SENTENCE_END.split(text)
    result: list[str] = []
    for part in parts:
        if _count_tokens(part) > _settings.child_chunk_max_tokens * 3:
            # Hard split by words to avoid token overflow
            words = part.split()
            chunk, current_tokens = [], 0
            for word in words:
                wt = _count_tokens(word)
                if current_tokens + wt > _settings.child_chunk_max_tokens:
                    result.append(" ".join(chunk))
                    chunk, current_tokens = [word], wt
                else:
                    chunk.append(word)
                    current_tokens += wt
            if chunk:
                result.append(" ".join(chunk))
        else:
            result.append(part)
    return [s.strip() for s in result if s.strip()]


# ── Page boundary helpers ─────────────────────────────────────────────────────

def _build_page_boundaries(pages: list) -> list[tuple[int, int, int]]:
    """
    Build a list of (char_start, char_end, page_number) triples that mirror
    the character offsets produced by RawChapter.raw_text — i.e. pages whose
    text is non-empty joined with '\\n\\n'.

    Used to map a parent chunk's char_start/char_end back to a page range.
    """
    boundaries: list[tuple[int, int, int]] = []
    offset = 0
    sep = "\n\n"
    first = True
    for page in pages:
        if not page.text.strip():
            continue
        if not first:
            offset += len(sep)
        text_len = len(page.text)
        boundaries.append((offset, offset + text_len, page.page_number))
        offset += text_len
        first = False
    return boundaries


def _page_range_for_chars(
    char_start: int,
    char_end: int,
    boundaries: list[tuple[int, int, int]],
    fallback_start: int | None,
    fallback_end: int | None,
) -> tuple[int | None, int | None]:
    """
    Return the (first_page, last_page) that overlap with [char_start, char_end].
    Falls back to the chapter-level page range when no boundary matches.
    """
    first_page: int | None = None
    last_page:  int | None = None
    for (ps, pe, pnum) in boundaries:
        # Overlap condition: the page and the chunk share at least one character
        if ps < char_end and pe > char_start:
            if first_page is None:
                first_page = pnum
            last_page = pnum
    return (first_page or fallback_start), (last_page or fallback_end)


# ── Core chunking logic ───────────────────────────────────────────────────────

def _make_child_chunks(text: str, char_offset: int) -> list[ChildChunkData]:
    """
    Slide a window of 80–120 tokens with a 20-token overlap over sentences.
    """
    sentences = _split_sentences(text)
    min_t = _settings.child_chunk_min_tokens
    max_t = _settings.child_chunk_max_tokens
    overlap_t = _settings.child_chunk_overlap_tokens

    chunks: list[ChildChunkData] = []
    window: list[str] = []
    window_tokens = 0
    idx = 0

    for sent in sentences:
        st = _count_tokens(sent)
        if window_tokens + st > max_t and window:
            chunk_text = " ".join(window)
            chunks.append(ChildChunkData(
                chunk_index=idx,
                text=chunk_text,
                token_count=_count_tokens(chunk_text),
                text_hash=_sha256(chunk_text),
                char_start=char_offset,
                char_end=char_offset + len(chunk_text),
            ))
            char_offset += len(chunk_text)
            idx += 1
            # Overlap: keep last sentences that fit within overlap budget
            overlap_buf, overlap_tok = [], 0
            for s in reversed(window):
                t = _count_tokens(s)
                if overlap_tok + t <= overlap_t:
                    overlap_buf.insert(0, s)
                    overlap_tok += t
                else:
                    break
            window = overlap_buf
            window_tokens = overlap_tok

        window.append(sent)
        window_tokens += st

    # Flush remaining
    if window:
        chunk_text = " ".join(window)
        if _count_tokens(chunk_text) >= min_t // 2:   # skip tiny trailing fragments
            chunks.append(ChildChunkData(
                chunk_index=idx,
                text=chunk_text,
                token_count=_count_tokens(chunk_text),
                text_hash=_sha256(chunk_text),
                char_start=char_offset,
                char_end=char_offset + len(chunk_text),
            ))

    return chunks


def _make_parent_chunks(
    text: str,
    page_start: int | None,
    page_end: int | None,
    page_boundaries: list[tuple[int, int, int]] | None = None,
) -> list[ParentChunkData]:
    """
    Slide a window of 300–400 tokens with 40-token overlap over paragraphs.
    Each parent chunk is sub-divided into child chunks.

    When page_boundaries is provided (built from the chapter's RawPage list),
    each parent chunk receives its own precise page_start/page_end rather than
    inheriting the whole chapter's range.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    min_t = _settings.parent_chunk_min_tokens
    max_t = _settings.parent_chunk_max_tokens
    overlap_t = _settings.parent_chunk_overlap_tokens

    parents: list[ParentChunkData] = []
    window: list[str] = []
    window_tokens = 0
    char_offset = 0
    idx = 0

    def _flush():
        nonlocal idx, char_offset
        chunk_text = "\n\n".join(window)
        cs = char_offset
        ce = char_offset + len(chunk_text)

        if page_boundaries:
            cp_start, cp_end = _page_range_for_chars(
                cs, ce, page_boundaries, page_start, page_end
            )
        else:
            cp_start, cp_end = page_start, page_end

        children = _make_child_chunks(chunk_text, cs)
        parents.append(ParentChunkData(
            chunk_index=idx,
            text=chunk_text,
            token_count=_count_tokens(chunk_text),
            char_start=cs,
            char_end=ce,
            page_start=cp_start,
            page_end=cp_end,
            children=children,
        ))
        char_offset += len(chunk_text)
        idx += 1

    for para in paragraphs:
        pt = _count_tokens(para)
        if window_tokens + pt > max_t and window:
            _flush()
            # Overlap
            overlap_buf, overlap_tok = [], 0
            for p in reversed(window):
                t = _count_tokens(p)
                if overlap_tok + t <= overlap_t:
                    overlap_buf.insert(0, p)
                    overlap_tok += t
                else:
                    break
            window = overlap_buf
            window_tokens = overlap_tok

        window.append(para)
        window_tokens += pt

    if window:
        _flush()

    return parents


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_chapter(chapter: RawChapter) -> ChunkResult:
    """
    Take a RawChapter and produce the full hierarchical chunk structure.

    Builds a page boundary map from the chapter's RawPage list so that each
    parent chunk gets its own precise page_start/page_end rather than the
    chapter's full range.
    """
    text = chapter.raw_text
    page_boundaries = _build_page_boundaries(chapter.pages)
    parents = _make_parent_chunks(
        text,
        chapter.page_start,
        chapter.page_end,
        page_boundaries,
    )

    return ChunkResult(
        chapter_index=chapter.chapter_index,
        chapter_title=chapter.title,
        page_start=chapter.page_start,
        page_end=chapter.page_end,
        raw_text=text,
        parents=parents,
    )
