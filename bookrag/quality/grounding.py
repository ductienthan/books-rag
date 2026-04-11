"""
Layer 5 — Grounding check.

Sends a tiny Gemma 4 call to verify the answer is supported by the retrieved chunks.
Returns True if grounded, False if the answer appears to hallucinate.
"""
from __future__ import annotations

import logging
import re

import ollama

from bookrag.config import get_settings
from bookrag.retrieval.searcher import RetrievedChunk

log = logging.getLogger(__name__)
_settings = get_settings()

_GROUNDING_PROMPT = """You are a fact-checker. Given a question, an answer, and source excerpts, determine if the answer is supported by the excerpts.

Respond with exactly one word: "GROUNDED" if the answer is based on the excerpts, or "UNGROUNDED" if the answer contains claims not found in the excerpts.

Question: {question}

Answer: {answer}

Source excerpts:
{context}

Response (one word only):"""


def check_grounding(
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
) -> bool:
    """
    Returns True if the answer is grounded in the retrieved chunks.
    Falls back to True on any error (non-blocking).
    """
    if not chunks:
        return False

    context = "\n\n".join(f"[{i+1}] {c.parent_text[:500]}" for i, c in enumerate(chunks[:3]))
    prompt = _GROUNDING_PROMPT.format(
        question=question[:300],
        answer=answer[:600],
        context=context,
    )

    try:
        client = ollama.Client(host=_settings.ollama_host)
        response = client.generate(
            model=_settings.ollama_llm_model,
            prompt=prompt,
            options={"num_predict": 16, "temperature": 0.0},
        )
        verdict = response.response.strip().upper()
        is_grounded = "GROUNDED" in verdict and "UNGROUNDED" not in verdict
        log.debug("Grounding check: %s", verdict)
        return is_grounded

    except Exception as exc:
        log.warning("Grounding check failed (defaulting to True): %s", exc)
        return True   # fail open — don't block the answer
