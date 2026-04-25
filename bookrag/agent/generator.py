"""
Layer 3 — Answer generation via Gemma 4 (Ollama).

Builds a structured prompt from:
  - System instructions + citation format
  - Session memory (rolling window)
  - Retrieved parent chunks (tagged with source info)
  - User question

Phase 2 Enhancements:
  - Dynamic context building with token budget
  - Enhanced prompts for comprehensive answers
  - Token counting and budget management

Phase 3.3 Enhancements:
  - Streaming LLM output for better UX
  - Progress indicators during generation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import ollama
import tiktoken

from bookrag.config import get_settings
from bookrag.retrieval.searcher import RetrievedChunk

log = logging.getLogger(__name__)
_settings = get_settings()
_enc = tiktoken.get_encoding("cl100k_base")

# Phase 2: Enhanced system prompt for better quality
_SYSTEM_PROMPT = """You are an expert literary assistant specializing in detailed, comprehensive analysis of book content.

Your role:
1. Provide thorough, well-structured answers using ONLY the provided excerpts
2. Include direct quotes and specific details from the text when answering
3. Synthesize information across multiple excerpts when relevant
4. Always cite sources inline using: [Book: {title}, Chapter {N}, p. {page}]

Quality guidelines:
- Prefer detailed explanations over brief summaries
- Use specific examples and quotes from the text
- When multiple excerpts discuss the same topic, synthesize them coherently
- If excerpts contain partial information, acknowledge what's covered and what's not
- For conceptual questions, explain using the book's own language and examples

Critical rules:
- NEVER use outside knowledge or make assumptions beyond the text
- If information is not in the excerpts, state clearly: "I cannot find this in your books."
- Do not paraphrase excessively; preserve the author's language when appropriate
- Cite every major point with its source"""

# Phase 2: Enhanced context template with better structure
_CONTEXT_TEMPLATE = """═══ Source {idx} ═══════════════════════════════════════════
Book:    {title}
Chapter: {chapter_idx}. {chapter_title}
Pages:   {pages}

{text}
═══════════════════════════════════════════════════════════
"""

_OUT_OF_SCOPE_ADDENDUM = """
NOTE: The retrieved excerpts have low confidence scores. If you cannot answer from the context below, say so clearly rather than guessing."""


@dataclass
class GeneratedAnswer:
    answer: str
    book_ids: list[str]
    chunk_ids: list[str]
    is_grounded: bool = True


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_enc.encode(text))


def build_context_block(
    chunks: list[RetrievedChunk],
    max_tokens: int | None = None
) -> str:
    """
    Format retrieved chunks into a numbered context block.

    Phase 2 Enhancement: Respects token budget and includes as many chunks
    as possible without exceeding max_tokens.

    Args:
        chunks: List of retrieved chunks to include
        max_tokens: Maximum tokens to include (uses config if not specified)

    Returns:
        Formatted context string with source citations
    """
    if max_tokens is None:
        max_tokens = _settings.max_context_tokens

    parts = []
    total_tokens = 0

    for i, chunk in enumerate(chunks, 1):
        # Format this chunk
        chapter_title = chunk.chapter_title or "Untitled"
        pages = f"{chunk.page_start}–{chunk.page_end}" if chunk.page_start else "N/A"

        formatted_chunk = _CONTEXT_TEMPLATE.format(
            idx=i,
            title=chunk.book_title,
            chapter_idx=chunk.chapter_index + 1,
            chapter_title=chapter_title,
            pages=pages,
            text=chunk.parent_text,
        )

        # Count tokens for this chunk
        chunk_tokens = count_tokens(formatted_chunk)

        # Check if adding this chunk would exceed budget
        if total_tokens + chunk_tokens > max_tokens and parts:
            # We have at least one chunk, stop here
            log.info(
                f"Context budget reached: {total_tokens} tokens. "
                f"Including {len(parts)}/{len(chunks)} chunks"
            )
            break

        # Add this chunk
        parts.append(formatted_chunk)
        total_tokens += chunk_tokens

    log.info(f"Built context with {len(parts)} chunks, {total_tokens} tokens")
    return "\n\n".join(parts)


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


def generate_answer_stream(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict],
    low_confidence: bool = False,
) -> Iterator[str]:
    """
    Generate an answer using Gemma 4 via Ollama with streaming.

    Phase 3.3: Streaming version that yields answer chunks as they're generated.

    Args:
        question: User's question
        chunks: Retrieved chunks for context
        history: Session message history
        low_confidence: Whether retrieval had low confidence

    Yields:
        Answer text chunks as they're generated by the LLM

    Note:
        The complete answer should be reconstructed by joining all yielded chunks.
        Book IDs and chunk IDs are not available until streaming completes.
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
        # Enable streaming
        # Note: Removed num_predict limit for streaming to avoid Ollama issues
        stream = client.chat(
            model=_settings.ollama_llm_model,
            messages=messages,
            stream=True,  # Enable streaming!
            options={
                "temperature": 0.1,
            },
        )

        # Yield chunks as they arrive
        for chunk in stream:
            content = chunk.message.content
            if content:  # Only yield non-empty chunks
                yield content

    except Exception as exc:
        log.error("LLM streaming generation failed: %s", exc)
        yield "Sorry, I encountered an error generating an answer. Please try again."
