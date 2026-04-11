"""
Layer 1 — File extraction.

PDF  : pdfplumber (page-generator, low RAM) + EasyOCR fallback for sparse pages.
EPUB : ebooklib + BeautifulSoup4 for HTML stripping.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from bookrag.config import get_settings

log = logging.getLogger(__name__)
_settings = get_settings()


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

    @property
    def raw_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


# ── Heading patterns for chapter detection ────────────────────────────────────
_CHAPTER_RE = re.compile(
    r"^\s*(chapter|part|section|prologue|epilogue|introduction|preface|appendix)\b\s*[\d\w]*[:\s\-–]*([^\n]*)",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_text(text: str) -> str:
    """Remove excessive whitespace while preserving paragraph breaks."""
    # Strip NUL bytes (PostgreSQL rejects them)
    text = text.replace("\x00", "")
    # Collapse runs of 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


# ── PDF extraction ────────────────────────────────────────────────────────────

def _extract_pdf_toc(pdf) -> list[tuple[int, str]] | None:
    """
    Extract (page_number, title) pairs from the PDF's built-in TOC/outline.
    Returns None if no usable outline is found.
    """
    try:
        outlines = list(pdf.doc.get_outlines())
    except Exception:
        return None

    if not outlines:
        return None

    pageid_to_num = {page.page_obj.pageid: page.page_number for page in pdf.pages}
    toc: list[tuple[int, str]] = []

    for _level, title, _dest, ref, _ in outlines:
        try:
            obj = ref.resolve()
            page_num = pageid_to_num.get(obj["D"][0].objid)
            if page_num and title:
                toc.append((page_num, title.replace("\x00", "").strip()))
        except Exception:
            continue

    return toc if toc else None


def _chapters_from_toc(pages: list[RawPage], toc: list[tuple[int, str]]) -> list[RawChapter]:
    """Group pages into chapters using a pre-built TOC."""
    if not pages:
        return []
    # Sort by page number; keep first title when multiple entries share a page
    seen_pages: set[int] = set()
    deduped: list[tuple[int, str]] = []
    for page_num, title in sorted(toc, key=lambda x: x[0]):
        if page_num not in seen_pages:
            seen_pages.add(page_num)
            deduped.append((page_num, title))
    toc = deduped

    chapters: list[RawChapter] = []
    for i, (start_page, title) in enumerate(toc):
        end_page = toc[i + 1][0] - 1 if i + 1 < len(toc) else pages[-1].page_number
        chapter_pages = [p for p in pages if start_page <= p.page_number <= end_page]
        if not chapter_pages:
            continue
        chapters.append(RawChapter(
            chapter_index=len(chapters),
            title=title,
            pages=chapter_pages,
            page_start=chapter_pages[0].page_number,
            page_end=chapter_pages[-1].page_number,
        ))

    return chapters


def extract_pdf(file_path: Path) -> list[RawChapter]:
    """
    Extract text from a PDF file.
    - Uses the PDF's built-in TOC/outline for chapter structure when available.
    - Falls back to regex-based heading detection when no TOC exists.
    - Falls back to EasyOCR for pages with < OCR_CHAR_THRESHOLD characters.
    """
    import pdfplumber

    log.info("Extracting PDF: %s", file_path)
    pages: list[RawPage] = []

    with pdfplumber.open(str(file_path)) as pdf:
        for i, pdf_page in enumerate(pdf.pages):
            page_num = i + 1
            text = pdf_page.extract_text() or ""
            used_ocr = False

            if len(text.strip()) < _settings.ocr_char_threshold:
                log.debug("Page %d is sparse (%d chars) — running OCR", page_num, len(text))
                text = _ocr_page(pdf_page, page_num)
                used_ocr = True

            pages.append(RawPage(page_number=page_num, text=_clean_text(text), used_ocr=used_ocr))

        log.info("Extracted %d pages from %s", len(pages), file_path.name)

        toc = _extract_pdf_toc(pdf)

    if toc:
        log.info("Using PDF TOC (%d entries) for chapter structure", len(toc))
        return _chapters_from_toc(pages, toc)

    log.info("No PDF TOC found — falling back to heading detection")
    return _detect_chapters_from_pages(pages)


def _ocr_page(pdf_page, page_num: int) -> str:
    """Rasterize a pdfplumber page and run EasyOCR on it."""
    try:
        import easyocr
        import numpy as np

        # Lazy-init reader (cached in module scope)
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


def _detect_chapters_from_pages(pages: list[RawPage]) -> list[RawChapter]:
    """Group pages into chapters by detecting headings."""
    chapters: list[RawChapter] = []
    current: list[RawPage] = []
    current_title: str | None = None

    def _flush(chapter_idx: int, title: str | None, page_list: list[RawPage]) -> RawChapter:
        return RawChapter(
            chapter_index=chapter_idx,
            title=title,
            pages=page_list,
            page_start=page_list[0].page_number if page_list else None,
            page_end=page_list[-1].page_number if page_list else None,
        )

    for page in pages:
        match = _CHAPTER_RE.search(page.text[:500])   # only check first 500 chars
        if match and current:
            chapters.append(_flush(len(chapters), current_title, current))
            current = []
            current_title = (match.group(0).strip()) or None

        current.append(page)

    if current:
        chapters.append(_flush(len(chapters), current_title, current))

    if not chapters:
        # No chapters detected — treat the whole book as one chapter
        chapters = [RawChapter(chapter_index=0, title="Full Text", pages=pages,
                               page_start=pages[0].page_number if pages else None,
                               page_end=pages[-1].page_number if pages else None)]

    log.info("Detected %d chapters", len(chapters))
    return chapters


# ── EPUB extraction ───────────────────────────────────────────────────────────

def extract_epub(file_path: Path) -> list[RawChapter]:
    """
    Extract text from an EPUB file.
    - ebooklib iterates spine items (chapters).
    - BeautifulSoup4 strips HTML; paragraph breaks are preserved.
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

        # Detect title from heading tags
        heading = soup.find(re.compile(r"^h[1-3]$"))
        title = heading.get_text(strip=True) if heading else None

        # Inject newlines around block-level tags before stripping
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
        ))

    if not chapters:
        raise ValueError(f"No readable content found in EPUB: {file_path}")

    log.info("Extracted %d chapters from EPUB %s", len(chapters), file_path.name)
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
        raise ValueError(f"Unsupported file type: {suffix}. Only .pdf and .epub are supported.")
