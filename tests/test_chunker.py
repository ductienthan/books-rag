"""Tests for the hierarchical chunker."""
import pytest
from bookrag.ingestion.chunker import _make_child_chunks, _make_parent_chunks, chunk_chapter
from bookrag.ingestion.extractor import RawChapter, RawPage


def _make_chapter(text: str, idx: int = 0) -> RawChapter:
    page = RawPage(page_number=1, text=text)
    return RawChapter(chapter_index=idx, title=f"Chapter {idx+1}", pages=[page], page_start=1, page_end=10)


def test_child_chunks_token_range():
    """Child chunks should stay within the 80-120 token target."""
    # 50 sentences of ~10 words each = ~500 tokens
    text = " ".join(["The quick brown fox jumps over the lazy dog."] * 50)
    chunks = _make_child_chunks(text, char_offset=0)
    assert len(chunks) > 1, "Should produce multiple child chunks"
    for c in chunks:
        assert c.token_count <= 150, f"Child chunk too large: {c.token_count} tokens"


def test_child_chunks_have_hashes():
    text = "Hello world. " * 20
    chunks = _make_child_chunks(text, char_offset=0)
    for c in chunks:
        assert len(c.text_hash) == 64


def test_parent_chunks_contain_children():
    text = "\n\n".join(["This is a paragraph about nothing in particular. " * 5] * 20)
    parents = _make_parent_chunks(text, page_start=1, page_end=5)
    assert len(parents) >= 1
    for p in parents:
        assert len(p.children) >= 1
        assert p.token_count <= 450   # some slack over the 400-token target


def test_chunk_chapter_full():
    text = "\n\n".join([
        "This is the opening paragraph of this chapter. It sets up the themes.",
        "The second paragraph develops the key idea further with more detail.",
        "The third paragraph introduces a counterargument to consider.",
        "In conclusion, we have examined the main points of this chapter.",
    ] * 10)
    chapter = _make_chapter(text)
    result = chunk_chapter(chapter)
    assert result.chapter_index == 0
    assert len(result.parents) >= 1
    total_children = sum(len(p.children) for p in result.parents)
    assert total_children >= 1


def test_empty_chapter_graceful():
    """An empty chapter should not raise — returns zero chunks."""
    chapter = _make_chapter("   ", idx=0)
    result = chunk_chapter(chapter)
    assert result is not None
