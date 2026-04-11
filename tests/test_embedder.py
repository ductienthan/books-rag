"""Tests for the embedding engine."""
import pytest
from bookrag.ingestion.embedder import embed_documents, embed_query


def test_embed_query_shape():
    vec = embed_query("What is the main theme of this book?")
    assert isinstance(vec, list)
    assert len(vec) == 768


def test_embed_documents_batch():
    texts = ["Hello world.", "The cat sat on the mat.", "Machine learning is fascinating."]
    vecs = embed_documents(texts)
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 768


def test_embeddings_are_normalized():
    """nomic-embed-text-v1.5 with normalize=True should return unit vectors."""
    import math
    vec = embed_query("test")
    magnitude = math.sqrt(sum(x**2 for x in vec))
    assert abs(magnitude - 1.0) < 0.01, f"Vector not normalized: magnitude={magnitude}"


def test_document_query_similarity():
    """A query and a matching document should have high cosine similarity."""
    import numpy as np
    query_vec = embed_query("What is photosynthesis?")
    doc_vec = embed_documents(["Photosynthesis is the process by which plants convert sunlight into energy."])[0]
    unrelated_vec = embed_documents(["The stock market crashed in 1929 during the Great Depression."])[0]

    sim_match = float(np.dot(query_vec, doc_vec))
    sim_unrelated = float(np.dot(query_vec, unrelated_vec))
    assert sim_match > sim_unrelated, "Matching document should score higher than unrelated"
