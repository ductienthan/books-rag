"""
Layer 1 — File extraction.

PDF  : pdfplumber (page-generator, low RAM) + EasyOCR fallback for sparse pages.
EPUB : ebooklib + BeautifulSoup4 for HTML stripping.

Chapter detection improvements
───────────────────────────────
Fix 1  TOC-level filtering + front/back matter title matching
       – Only the shallowest TOC level is used as chapter boundaries (configurable).
       – Known front-matter and back-matter titles are classified at extraction time.

Fix 2  Post-extraction consolidation pass (_consolidate_chapters)
       – All leading front-matter TOC entries are merged into a single
         "Front Matter" chapter so they don't each spawn an Ollama call.
       – All trailing back-matter entries are merged into "Back Matter".
       – Content chapters below min_chapter_chars are absorbed into their
         neighbours so tiny section-header stubs don't become separate chapters.

Fix 3  Improved heading regex for the no-TOC fallback path
       – Now also matches numbered chapters ("1. Title") in addition to the
         existing keyword-based patterns.

Fix 4  ChapterType enum on RawChapter
       – Propagated to the Chapter DB row so the worker can gate LLM calls.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from bookrag.config import get_settings

log = logging.getLogger(__name__)
_settings = get_settings()


# ── Chapter type ──────────────────────────────────────────────────────────────

class ChapterType(str, Enum):
    """Role of a chapter within the book."""
    FRONT_MATTER = "front_matter"   # cover, title page, copyright, dedication…
    BACK_MATTER  = "back_matter"    # index, bibliography, glossary…
    CONTENT      = "content"        # main body chapters
    UNKNOWN      = "unknown"        # unclassified; treated as content downstream


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RawPage:
    """A single extracted page or EPUB section."""
    page_number: int          # 1-based
    text: str
    used_ocr: bool = False


@dataclass
class RawChapter:
    """A detected chapter spanning one or more pages."""
    chapter_index: int
    title: str | None
    pages: list[RawPage] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    chapter_type: ChapterType = ChapterType.UNKNOWN
    _text_override: str | None = field(default=None, repr=False)

    @property
    def raw_text(self) -> str:
        if self._text_override is not None:
            return self._text_override
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


# ── Chapter classification patterns ──────────────────────────────────────────
#
# Both regexes use full-string matching (^ … $) with optional qualifiers
# ("of", "to", "for", "by") so that:
#   "Preface"                              → FRONT_MATTER  ✓
#   "Preface to the Fourth Edition"        → FRONT_MATTER  ✓
#   "Glossary of Motivational Interviewing Concepts" → BACK_MATTER ✓
#   "Index Funds and How They Work"        → NOT matched   ✓
#   "Part I: Helping People Change"        → NOT matched   ✓  (content)

_FRONT_MATTER_RE = re.compile(
    r"^\s*("
    r"half[\s\-]?title(\s+page?)?"
    r"|series\s+page?"
    r"|title\s+page?"
    r"|copyright(\s+page?)?"
    r"|dedication"
    r"|epigraph"
    r"|blank\s+page?"
    r"|cover(\s+page?)?"
    r"|foreword(\s+(by|to|of)\s+[^\n]*)?"
    r"|contents"
    r"|table\s+of\s+contents"
    r"|also\s+by(\s+[^\n]*)?"
    r"|series\s+information"
    r"|about\s+the\s+(author|authors|publisher)"
    r"|acknowledg(e?ments?)(\s+(to|of)\s+[^\n]*)?"
    r"|permissions?"
    r"|preface(\s+(to|for|of)\s+[^\n]*)?"
    r"|a\s+note\s+(on|about|to)\s*[^\n]*"
    r"|author'?s?\s+note"
    r")\s*$",
    re.IGNORECASE,
)

_BACK_MATTER_RE = re.compile(
    r"^\s*("
    r"index(\s+(of|to)\s+[^\n]*)?"
    r"|bibliography(\s+(of|to|for)\s+[^\n]*)?"
    r"|references(\s+(of|to|for)\s+[^\n]*)?"
    r"|appendix\w*(\s+[^\n]*)?"
    r"|appendices(\s+[^\n]*)?"
    r"|glossary(\s+(of|to|for)\s+[^\n]*)?"
    r"|notes(\s+(on|to|of)\s+[^\n]*)?"
    r"|endnotes(\s+[^\n]*)?"
    r"|colophon"
    r"|about\s+the\s+(author|authors|publisher)"
    r"|further\s+reading"
    r"|selected\s+bibliography"
    r")\s*$",
    re.IGNORECASE,
)


def _classify_chapter(title: str | None) -> ChapterType:
    """Classify a chapter as front matter, back matter, or content by title."""
    if not title:
        return ChapterType.UNKNOWN
    if _FRONT_MATTER_RE.match(title):
        return ChapterType.FRONT_MATTER
    if _BACK_MATTER_RE.match(title):
        return ChapterType.BACK_MATTER
    return ChapterType.CONTENT


# ── Heading regex for the no-TOC fallback path ───────────────────────────────
#
# Fix 3: extended to also match numbered chapters such as "1. The Mind and
# Heart When Helping" in addition to keyword-based headings.

_CHAPTER_RE = re.compile(
    r"^\s*("
    # Named keywords: "Chapter 3", "Part IV", "Section One", "Prologue", …
    r"(chapter|part|section|prologue|epilogue|introduction|preface|appendix|"
    r"foreword|interlude|afterword|conclusion|coda|overview)"
    r"[\s\.\-–:]*[\w]*"
    r"|"
    # Numbered chapters: "1. Title" or "12) Title" — 1-3 digits, period/paren,
    # then an uppercase letter followed by 4-60 more chars (avoids page numbers)
    r"(\d{1,3}[\.\)]\s+[A-Z][^\n]{4,60})"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Remove excessive whitespace while preserving paragraph breaks."""
    text = text.replace("\x00", "")           # PostgreSQL rejects NUL bytes
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse 3+ newlines → 2
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


# ── PDF extraction ────────────────────────────────────────────────────────────

def _extract_pdf_toc(pdf) -> list[tuple[int, str, int]] | None:
    """
    Extract (page_number, title, level) triples from the PDF outline.
    Level 0 = top-level; higher values = sub-sections.
    Returns None when no usable outline is found.
    """
    try:
        outlines = list(pdf.doc.get_outlines())
    except Exception:
        return None

    if not outlines:
        return None

    pageid_to_num = {page.page_obj.pageid: page.page_number for page in pdf.pages}
    toc: list[tuple[int, str, int]] = []

    for level, title, _dest, ref, _ in outlines:
        try:
            obj = ref.resolve()
            page_num = pageid_to_num.get(obj["D"][0].objid)
            if page_num and title:
                toc.append((page_num, title.replace("\x00", "").strip(), level))
        except Exception:
            continue

    return toc if toc else None


def extract_pdf(file_path: Path) -> list[RawChapter]:
    """
    Extract text from a PDF file, then apply the consolidation pass.

    Strategy:
    - With TOC  → _extract_chapters_by_toc  (Fix 1: level filter + classification)
    - Without   → _detect_chapters_from_pages (Fix 3: improved regex)
    Both paths feed into _consolidate_chapters (Fix 2).
    """
    import pdfplumber

    log.info("Extracting PDF: %s", file_path)

    with pdfplumber.open(str(file_path)) as pdf:
        toc = _extract_pdf_toc(pdf)

        if toc:
            log.info("Using PDF TOC (%d entries) for chapter structure", len(toc))
            chapters = _extract_chapters_by_toc(pdf, toc)
        else:
            pages = _extract_all_pages(pdf)
            log.info("No PDF TOC found — falling back to heading detection")
            chapters = _detect_chapters_from_pages(pages)

    # Fix 2 — post-extraction consolidation (merge front/back matter + short stubs)
    chapters = _consolidate_chapters(chapters, _settings.min_chapter_chars)
    return chapters


def _extract_all_pages(pdf) -> list[RawPage]:
    """Extract all pages with pdfplumber (no OCR). Used by heading-detection fallback."""
    pages: list[RawPage] = []
    for i, pdf_page in enumerate(pdf.pages):
        page_num = i + 1
        text = _clean_text(pdf_page.extract_text() or "")
        pages.append(RawPage(page_number=page_num, text=text))
    log.info("Extracted %d pages", len(pages))
    return pages


def _extract_chapters_by_toc(pdf, toc: list[tuple[int, str, int]]) -> list[RawChapter]:
    """
    Extract chapter text at chapter level using TOC page ranges.

    Fix 1 improvements:
    - Top-level-only filter: when toc_top_level_only=True (default), only the
      shallowest TOC level is used as chapter boundaries.  Books that nest
      chapters under parts (Level 0 = parts, Level 1 = chapters) are split at
      the part level — preventing dozens of tiny sub-section chapters.
    - Each chapter is classified (front / back / content) at extraction time so
      the consolidation pass and the worker can act on that information.
    - Chapter-level OCR fallback is retained from the original implementation.
    """
    total_pages = len(pdf.pages)

    # ── Fix 1a: top-level-only filter ────────────────────────────────────────
    if _settings.toc_top_level_only and toc:
        min_level = min(lvl for _, _, lvl in toc)
        filtered = [(p, t, l) for p, t, l in toc if l == min_level]
        if len(filtered) < len(toc):
            log.info(
                "TOC level filter: kept %d top-level entries (level=%d), "
                "dropped %d sub-section entries",
                len(filtered), min_level, len(toc) - len(filtered),
            )
        toc = filtered

    # Deduplicate entries that share a start page (keep first title)
    seen: set[int] = set()
    deduped: list[tuple[int, str, int]] = []
    for page_num, title, level in sorted(toc, key=lambda x: x[0]):
        if page_num not in seen:
            seen.add(page_num)
            deduped.append((page_num, title, level))

    chapters: list[RawChapter] = []
    for i, (start_page, title, _level) in enumerate(deduped):
        end_page = deduped[i + 1][0] - 1 if i + 1 < len(deduped) else total_pages

        page_objs = pdf.pages[start_page - 1: end_page]   # pdfplumber is 0-indexed
        raw_pages: list[RawPage] = []
        for pdf_page in page_objs:
            text = _clean_text(pdf_page.extract_text() or "")
            raw_pages.append(RawPage(page_number=pdf_page.page_number, text=text))

        chapter_text = "\n\n".join(p.text for p in raw_pages if p.text.strip())

        # Chapter-level OCR: if the full chapter text is sparse, OCR all pages
        if len(chapter_text.strip()) < _settings.ocr_char_threshold:
            ocr_parts: list[str] = []
            for pdf_page in page_objs:
                ocr_text = _ocr_page(pdf_page, pdf_page.page_number)
                if ocr_text.strip():
                    ocr_parts.append(ocr_text)
            ocr_combined = "\n\n".join(ocr_parts)
            if len(ocr_combined.strip()) > len(chapter_text.strip()):
                chapter_text = ocr_combined
                log.debug(
                    "Chapter '%s' (pages %d-%d): OCR improved %d→%d chars",
                    title, start_page, end_page,
                    len("\n\n".join(p.text for p in raw_pages if p.text.strip())),
                    len(chapter_text),
                )

        # ── Fix 1b: classify chapter type by title ────────────────────────────
        chapter_type = _classify_chapter(title)

        log.info(
            "Chapter %d '%s' [%s]: pages %d-%d, %d chars",
            len(chapters) + 1, title, chapter_type.value,
            start_page, end_page, len(chapter_text.strip()),
        )

        chapters.append(RawChapter(
            chapter_index=len(chapters),
            title=title,
            pages=raw_pages,
            page_start=start_page,
            page_end=end_page,
            chapter_type=chapter_type,
            _text_override=chapter_text,
        ))

    return chapters


# ── Fix 2: post-extraction consolidation pass ────────────────────────────────

def _consolidate_chapters(
    chapters: list[RawChapter],
    min_chars: int,
) -> list[RawChapter]:
    """
    Merge front/back matter stubs and short content chapters.

    Steps:
    1. Peel all leading FRONT_MATTER chapters → merge into one "Front Matter".
    2. Peel all trailing BACK_MATTER chapters → merge into one "Back Matter".
    3. Walk remaining content chapters: if a chapter's text is < min_chars,
       absorb it into the NEXT chapter (or the previous if it's the last).
    4. Reassemble: Front Matter + content + Back Matter.
    5. Re-index chapter_index so it is contiguous from 0.

    All pages from merged chapters are retained so page-level attribution
    (page_start / page_end) is always correct.
    """
    if not chapters:
        return chapters

    total_raw = len(chapters)
    remaining = list(chapters)   # work on a copy; never mutate the input

    def _merge(group: list[RawChapter], title: str, ctype: ChapterType) -> RawChapter:
        all_pages = [p for ch in group for p in ch.pages]
        all_text  = "\n\n".join(ch.raw_text for ch in group if ch.raw_text.strip())
        return RawChapter(
            chapter_index=0,   # re-indexed at step 5
            title=title,
            pages=all_pages,
            page_start=group[0].page_start,
            page_end=group[-1].page_end,
            chapter_type=ctype,
            _text_override=all_text or None,
        )

    # Step 1 — peel leading front matter
    front_group: list[RawChapter] = []
    while remaining and remaining[0].chapter_type == ChapterType.FRONT_MATTER:
        front_group.append(remaining.pop(0))

    # Step 2 — peel trailing back matter
    back_group: list[RawChapter] = []
    while remaining and remaining[-1].chapter_type == ChapterType.BACK_MATTER:
        back_group.insert(0, remaining.pop())

    # Step 3 — merge short content chapters forward
    merged_content: list[RawChapter] = []
    pending: RawChapter | None = None

    for ch in remaining:
        if pending is not None:
            # Absorb the pending short chapter into the current one
            combined_pages = pending.pages + ch.pages
            combined_text  = "\n\n".join(
                t for t in [pending.raw_text, ch.raw_text] if t.strip()
            )
            ch = RawChapter(
                chapter_index=0,
                title=ch.title or pending.title,
                pages=combined_pages,
                page_start=pending.page_start,
                page_end=ch.page_end,
                chapter_type=ch.chapter_type,
                _text_override=combined_text or None,
            )
            pending = None

        if (
            len(ch.raw_text.strip()) < min_chars
            and ch.chapter_type in (ChapterType.CONTENT, ChapterType.UNKNOWN)
        ):
            pending = ch   # hold back; will be merged into the next chapter
        else:
            merged_content.append(ch)

    # If the last chapter was short and there's nothing ahead to absorb it,
    # merge it backward into the previous chapter (or keep it if no previous).
    if pending is not None:
        if merged_content:
            last = merged_content[-1]
            combined_pages = last.pages + pending.pages
            combined_text  = "\n\n".join(
                t for t in [last.raw_text, pending.raw_text] if t.strip()
            )
            merged_content[-1] = RawChapter(
                chapter_index=0,
                title=last.title,
                pages=combined_pages,
                page_start=last.page_start,
                page_end=pending.page_end,
                chapter_type=last.chapter_type,
                _text_override=combined_text or None,
            )
        else:
            merged_content.append(pending)

    # Step 4 — reassemble
    result: list[RawChapter] = []

    if front_group:
        merged_front = _merge(front_group, "Front Matter", ChapterType.FRONT_MATTER)
        log.info(
            "Consolidated %d front-matter entries → 'Front Matter' (%d chars)",
            len(front_group), len(merged_front.raw_text),
        )
        result.append(merged_front)

    result.extend(merged_content)

    if back_group:
        merged_back = _merge(back_group, "Back Matter", ChapterType.BACK_MATTER)
        log.info(
            "Consolidated %d back-matter entries → 'Back Matter' (%d chars)",
            len(back_group), len(merged_back.raw_text),
        )
        result.append(merged_back)

    # Step 5 — re-index
    for i, ch in enumerate(result):
        ch.chapter_index = i

    log.info(
        "Chapter consolidation: %d raw entries → %d final chapters "
        "(%s front-matter, %d content, %s back-matter)",
        total_raw,
        len(result),
        f"{len(front_group)} merged" if front_group else "none",
        len(merged_content),
        f"{len(back_group)} merged" if back_group else "none",
    )

    return result


# ── OCR helper ────────────────────────────────────────────────────────────────

def _ocr_page(pdf_page, page_num: int) -> str:
    """Rasterize a pdfplumber page and run EasyOCR on it."""
    try:
        import easyocr
        import numpy as np

        if not hasattr(_ocr_page, "_reader"):
            log.info("Initialising EasyOCR reader (first run may take a moment)…")
            _ocr_page._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        img = pdf_page.to_image(resolution=200).original  # PIL Image
        img_array = np.array(img)
        results = _ocr_page._reader.readtext(img_array, detail=0, paragraph=True)
        return "\n".join(results)
    except Exception as exc:
        log.warning("OCR failed on page %d: %s", page_num, exc)
        return ""


# ── Fallback: detect chapters from pages (no TOC) ────────────────────────────

def _detect_chapters_from_pages(pages: list[RawPage]) -> list[RawChapter]:
    """
    Group pages into chapters by detecting headings (Fix 3 regex).

    A new chapter is started when the first 500 chars of a page match:
    - Named keywords: chapter / part / section / prologue / epilogue / …
    - Numbered chapters: "1. Title", "12. Title" (new in Fix 3)
    """
    chapters: list[RawChapter] = []
    current: list[RawPage] = []
    current_title: str | None = None

    def _flush(idx: int, title: str | None, page_list: list[RawPage]) -> RawChapter:
        return RawChapter(
            chapter_index=idx,
            title=title,
            pages=page_list,
            page_start=page_list[0].page_number if page_list else None,
            page_end=page_list[-1].page_number if page_list else None,
            chapter_type=_classify_chapter(title),
        )

    for page in pages:
        match = _CHAPTER_RE.search(page.text[:500])
        if match and current:
            chapters.append(_flush(len(chapters), current_title, current))
            current = []
            current_title = match.group(0).strip() or None

        current.append(page)

    if current:
        chapters.append(_flush(len(chapters), current_title, current))

    if not chapters:
        chapters = [RawChapter(
            chapter_index=0,
            title="Full Text",
            pages=pages,
            page_start=pages[0].page_number if pages else None,
            page_end=pages[-1].page_number if pages else None,
            chapter_type=ChapterType.CONTENT,
        )]

    log.info("Detected %d chapters (pre-consolidation)", len(chapters))
    return chapters


# ── EPUB extraction ───────────────────────────────────────────────────────────

def extract_epub(file_path: Path) -> list[RawChapter]:
    """
    Extract text from an EPUB file.
    - ebooklib iterates spine items (chapters).
    - BeautifulSoup4 strips HTML; paragraph breaks are preserved.
    - Consolidation pass is applied identically to the PDF path.
    """
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    log.info("Extracting EPUB: %s", file_path)
    book = epub.read_epub(str(file_path), options={"ignore_ncx": True})
    chapters: list[RawChapter] = []

    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    for idx, item in enumerate(items):
        soup = BeautifulSoup(item.get_content(), "lxml")

        heading = soup.find(re.compile(r"^h[1-3]$"))
        title = heading.get_text(strip=True) if heading else None

        for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br"]):
            tag.insert_after("\n")

        text = _clean_text(soup.get_text())
        if not text:
            continue

        page = RawPage(page_number=idx + 1, text=text)
        chapters.append(RawChapter(
            chapter_index=len(chapters),
            title=title,
            pages=[page],
            page_start=idx + 1,
            page_end=idx + 1,
            chapter_type=_classify_chapter(title),
        ))

    if not chapters:
        raise ValueError(f"No readable content found in EPUB: {file_path}")

    log.info("Extracted %d EPUB sections (pre-consolidation) from %s",
             len(chapters), file_path.name)

    chapters = _consolidate_chapters(chapters, _settings.min_chapter_chars)
    return chapters


# ── Dispatcher ────────────────────────────────────────────────────────────────

def extract(file_path: Path) -> list[RawChapter]:
    """Extract chapters from PDF or EPUB. Dispatches by file extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(file_path)
    elif suffix == ".epub":
        return extract_epub(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. Only .pdf and .epub are supported."
        )
