"""
bookrag/retrieval/bm25.py

Phase 3 Component 2: BM25 Keyword-based Search

PURPOSE:
- Build BM25 index for keyword/lexical retrieval
- Complement vector search with exact keyword matching
- Improve retrieval for technical terms, abbreviations, and exact phrases

FEATURES:
- BM25Okapi scoring algorithm
- Persistent index storage (pickle)
- Token-level matching
- Efficient retrieval

BENEFITS:
- Better recall for specific terms (e.g., "OARS", "MI")
- Handles abbreviations and acronyms
- Exact phrase matching
- +15% retrieval quality improvement (expected)
"""

import pickle
import re
from pathlib import Path
from typing import Optional
import logging

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session as DBSession

from bookrag.db.models import ChildChunk, ParentChunk

log = logging.getLogger(__name__)


def simple_tokenize(text: str) -> list[str]:
    """
    Tokenization for BM25.

    - Preserves ALL-CAPS tokens (2+ chars) before lowercasing so that
      acronyms are indexed as both their uppercase and lowercase forms
      without duplication.  Example: "The WHO and UN agreed" →
      tokens include 'who' and 'un' alongside the regular lowercase tokens.
    - Lowercase everything
    - Remove special characters (keep alphanumeric and hyphens)
    - Split on whitespace
    - Keep tokens of length > 1

    This is fully book-agnostic: any ALL-CAPS abbreviation in any book is
    handled automatically via regex — no domain-specific word lists needed.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens (acronyms appear once as lowercase)
    """
    # Extract ALL-CAPS tokens (2+ chars) before lowercasing
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)

    # Lowercase and strip punctuation
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)

    # Split and filter short/empty tokens
    tokens = [t for t in text.split() if t and len(t) > 1]

    # Re-inject acronyms as lowercase; set() deduplicates against existing tokens
    existing = set(tokens)
    for a in acronyms:
        lc = a.lower()
        if lc not in existing:
            tokens.append(lc)
            existing.add(lc)

    return tokens


class BM25Index:
    """
    BM25 index for keyword-based retrieval.

    Builds and maintains a BM25 index over parent chunks for fast keyword search.
    Complements vector search with exact keyword matching.

    Usage:
        # Build index
        index = BM25Index.build(book_ids, db)
        index.save(Path(".indexes/bm25/book_123.pkl"))

        # Load and search
        index = BM25Index.load(Path(".indexes/bm25/book_123.pkl"))
        results = index.search("some query", top_k=20)
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: list[str] = []
        self.documents: list[list[str]] = []  # Tokenized documents

    @classmethod
    def build(
        cls,
        book_ids: list[str],
        db: DBSession,
        progress_callback: Optional[callable] = None
    ) -> 'BM25Index':
        """
        Build BM25 index from child chunks (storing parent_chunk_id).

        Indexing child chunks (80-120 tokens) instead of parent chunks gives
        the BM25 scorer higher precision — each scored unit is smaller and more
        focused — while chunk_ids still stores the *parent* chunk ID so that
        results are directly compatible with RRF fusion against vector search.

        Args:
            book_ids: List of book IDs to index
            db: Database session
            progress_callback: Optional callback(current, total)

        Returns:
            BM25Index instance
        """
        index = cls()

        # Fetch all child chunks for the books
        chunks = (
            db.query(ChildChunk)
            .filter(ChildChunk.book_id.in_(book_ids))
            .all()
        )

        log.info(f"Building BM25 index for {len(chunks)} child chunks from {len(book_ids)} books")

        # Tokenize documents; store parent_chunk_id so RRF can match vector results
        for i, chunk in enumerate(chunks):
            tokens = simple_tokenize(chunk.text)

            index.documents.append(tokens)
            index.chunk_ids.append(chunk.parent_chunk_id)  # RRF-compatible parent ID

            if progress_callback and i % 100 == 0:
                progress_callback(i, len(chunks))

        # Build BM25 index
        if index.documents:
            index.bm25 = BM25Okapi(index.documents)
            log.info(f"BM25 index built successfully: {len(index.chunk_ids)} child chunks indexed")
        else:
            log.warning("No documents to index")

        return index

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        Search using BM25 scoring.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        if self.bm25 is None or not self.chunk_ids:
            log.warning("BM25 index is empty")
            return []

        # Tokenize query
        query_tokens = simple_tokenize(query)

        if not query_tokens:
            log.warning(f"No tokens found in query: {query}")
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results (only include positive scores)
        results = [
            (self.chunk_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

        log.debug(f"BM25 search for '{query}': {len(results)} results (top score: {results[0][1] if results else 0:.2f})")

        return results

    def save(self, path: Path) -> None:
        """
        Save index to disk using pickle.

        Args:
            path: Path to save the index
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save as pickle
        data = {
            'bm25': self.bm25,
            'chunk_ids': self.chunk_ids,
            'documents': self.documents,
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        log.info(f"BM25 index saved to {path} ({len(self.chunk_ids)} chunks)")

    @classmethod
    def load(cls, path: Path) -> 'BM25Index':
        """
        Load index from disk.

        Args:
            path: Path to the index file

        Returns:
            BM25Index instance
        """
        index = cls()

        with open(path, 'rb') as f:
            data = pickle.load(f)

        index.bm25 = data['bm25']
        index.chunk_ids = data['chunk_ids']
        index.documents = data['documents']

        log.info(f"BM25 index loaded from {path} ({len(index.chunk_ids)} chunks)")

        return index

    def stats(self) -> dict:
        """Get index statistics."""
        return {
            'num_documents': len(self.chunk_ids),
            'avg_doc_length': np.mean([len(doc) for doc in self.documents]) if self.documents else 0,
            'total_tokens': sum(len(doc) for doc in self.documents),
            'is_built': self.bm25 is not None,
        }


class BM25IndexManager:
    """
    Manage BM25 indexes for multiple books.

    - Creates one index per book (or combined index)
    - Handles loading and caching
    - Provides unified search interface
    """

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, BM25Index] = {}

    def get_index_path(self, book_ids: list[str]) -> Path:
        """Get index file path for a set of books."""
        # Sort book IDs for consistent naming
        book_key = "_".join(sorted(book_ids))
        return self.index_dir / f"{book_key}.pkl"

    def build_index(self, book_ids: list[str], db: DBSession, force: bool = False) -> BM25Index:
        """
        Build or load BM25 index for given books.

        Args:
            book_ids: List of book IDs
            db: Database session
            force: Force rebuild even if index exists

        Returns:
            BM25Index instance
        """
        index_path = self.get_index_path(book_ids)

        # Check if index exists and not forcing rebuild
        if index_path.exists() and not force:
            log.info(f"Loading existing BM25 index from {index_path}")
            return BM25Index.load(index_path)

        # Build new index
        log.info(f"Building new BM25 index for {len(book_ids)} books")
        index = BM25Index.build(book_ids, db)

        # Save for future use
        index.save(index_path)

        return index

    def search(self, book_ids: list[str], query: str, db: DBSession, top_k: int = 20) -> list[tuple[str, float]]:
        """
        Search BM25 index for given books.

        Args:
            book_ids: List of book IDs to search
            query: Search query
            db: Database session
            top_k: Number of results

        Returns:
            List of (chunk_id, score) tuples
        """
        # Get or build index
        cache_key = "_".join(sorted(book_ids))

        if cache_key not in self._cache:
            self._cache[cache_key] = self.build_index(book_ids, db)

        index = self._cache[cache_key]

        # Search
        return index.search(query, top_k=top_k)

    def clear_cache(self) -> None:
        """Clear in-memory index cache."""
        self._cache.clear()
        log.info("BM25 index cache cleared")

    def rebuild_all(self, db: DBSession) -> int:
        """
        Rebuild all existing indexes.

        Args:
            db: Database session

        Returns:
            Number of indexes rebuilt
        """
        from bookrag.db.models import Book

        # Get all books
        books = db.query(Book).all()

        count = 0
        for book in books:
            index = self.build_index([book.id], db, force=True)
            count += 1
            log.info(f"Rebuilt BM25 index for book: {book.title}")

        return count
