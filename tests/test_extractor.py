"""Tests for the file extractor (no real files needed — mocked)."""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from bookrag.ingestion.extractor import _detect_chapters_from_pages, _clean_text, RawPage


def test_clean_text_removes_excess_newlines():
    raw = "Hello\n\n\n\n\nWorld\n\n  \n\nFoo"
    cleaned = _clean_text(raw)
    assert "\n\n\n" not in cleaned
    assert "Hello" in cleaned
    assert "World" in cleaned


def test_chapter_detection_single_chapter():
    pages = [
        RawPage(page_number=1, text="Introduction to the topic."),
        RawPage(page_number=2, text="More details about the topic."),
    ]
    chapters = _detect_chapters_from_pages(pages)
    assert len(chapters) >= 1
    assert chapters[0].chapter_index == 0


def test_chapter_detection_multiple_chapters():
    pages = [
        RawPage(page_number=1, text="Chapter 1: The Beginning\nSome text here."),
        RawPage(page_number=2, text="More text in chapter one."),
        RawPage(page_number=3, text="Chapter 2: The Middle\nAnother section starts."),
        RawPage(page_number=4, text="Content of chapter two."),
    ]
    chapters = _detect_chapters_from_pages(pages)
    assert len(chapters) == 2
    assert chapters[0].page_start == 1
    assert chapters[1].page_start == 3


def test_chapter_raw_text_concatenates_pages():
    pages = [
        RawPage(page_number=1, text="First page content."),
        RawPage(page_number=2, text="Second page content."),
    ]
    chapters = _detect_chapters_from_pages(pages)
    raw = chapters[0].raw_text
    assert "First page content." in raw
    assert "Second page content." in raw


def test_unsupported_file_type_raises():
    from bookrag.ingestion.extractor import extract
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract(Path("some_file.txt"))
