"""
Layer 3 — Answer generation via Gemma 4 (Ollama).

Builds a structured prompt from:
  - System instructions + citation format
  - Session memory (rolling window)
  - Retrieved parent chunks (tagged with source info)
  - User question
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import ollama

from bookrag.config import get_settings
from bookrag.retrieval.searcher import RetrievedChunk

log = logging.getLogger(__name__)
_settings = get_settings()

_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions exclusively from the provided book excerpts.

Rules:
1. Base your answer ONLY on the provided excerpts. Do not use outside knowledge.
2. Always cite your sources inline using the format: [Book: {title}, Chapter {N}, p. {page}]
3. If the excerpts do not contain enough information to answer, say:
   "I cannot find this in your books. The question may fall outside the ingested content."
4. Be concise and precise. Avoid padding or filler.
5. If multiple books are relevant, synthesize the information and cite each source."""

_CONTEXT_TEMPLATE = """[Source {idx}: Book: "{title}" | Chapter {chapter_idx}{chapter_title_part} | Pages {pages}]
{text}"""

_OUT_OF_SCOPE_ADDENDUM = """
NOTE: The retrieved excerpts have low confidence scores. If you cannot answer from the context below, say so clearly rather than guessing."""


@dataclass
class GeneratedAnswer:
    answer: str
    book_ids: list[str]
    chunk_ids: list[str]
    is_grounded: bool = True


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        chapter_title_part = f" ({chunk.chapter_title})" if chunk.chapter_title else ""
        pages = f"{chunk.page_start}–{chunk.page_end}" if chunk.page_start else "N/A"
        parts.append(_CONTEXT_TEMPLATE.format(
            idx=i,
            title=chunk.book_title,
            chapter_idx=chunk.chapter_index + 1,
            chapter_title_part=chapter_title_part,
            pages=pages,
            text=chunk.parent_text,
        ))
    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict],
    low_confidence: bool = False,
) -> GeneratedAnswer:
    """
    Generate an answer using Gemma 4 via Ollama.
    history: list of {role, content} dicts (session memory).
    """
    client = ollama.Client(host=_settings.ollama_host)

    context_block = build_context_block(chunks)
    system = _SYSTEM_PROMPT
    if low_confidence:
        system += _OUT_OF_SCOPE_ADDENDUM

    # Build message list for chat API
    messages = [{"role": "system", "content": system}]

    # Inject session history
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Inject context + question
    user_content = f"Book excerpts:\n\n{context_block}\n\n---\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat(
            model=_settings.ollama_llm_model,
            messages=messages,
            options={
                "num_predict": _settings.llm_max_tokens,
                "temperature": 0.1,
            },
        )
        answer = response.message.content.strip()
    except Exception as exc:
        log.error("LLM generation failed: %s", exc)
        answer = "Sorry, I encountered an error generating an answer. Please try again."

    book_ids = list({c.book_id for c in chunks})
    chunk_ids = [c.child_id for c in chunks]

    return GeneratedAnswer(
        answer=answer,
        book_ids=book_ids,
        chunk_ids=chunk_ids,
        is_grounded=not low_confidence,
    )
