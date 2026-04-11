"""
Layer 3 — Query rewriter.

Uses Gemma 4 (via Ollama) to:
1. Expand pronouns / implicit references using recent session context.
2. Decompose multi-part questions into sub-queries.
3. Enrich very short queries with domain keywords.

Returns a list of rewritten query strings (usually 1, may be 2-3 for complex questions).
"""
from __future__ import annotations

import json
import logging
import re

import ollama

from bookrag.config import get_settings

log = logging.getLogger(__name__)
_settings = get_settings()

_REWRITE_PROMPT = """\
You are a search query optimiser for a book Q&A system.

Given the conversation history and the user's latest question, return a JSON object with:
  "queries": list of 1-3 refined search queries (strings)

Rules:
- Resolve pronouns and implicit references using the history.
- If the question has multiple distinct parts, split into separate queries.
- Keep each query concise (5-15 words).
- Do NOT answer the question — only rewrite it.
- Return ONLY valid JSON, no commentary.

Conversation history (last 3 turns):
{history}

User question: {question}

JSON:"""


def rewrite_query(question: str, history: list[dict]) -> list[str]:
    """
    Rewrite a query using recent conversation context.
    Falls back to the original question on any error.
    """
    if not history and len(question.split()) >= 5:
        # Short-circuit: no history, question is already decent
        return [question]

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}" for m in history[-6:]
    )
    prompt = _REWRITE_PROMPT.format(history=history_text or "(none)", question=question)

    try:
        client = ollama.Client(host=_settings.ollama_host)
        response = client.generate(
            model=_settings.ollama_llm_model,
            prompt=prompt,
            options={"num_predict": 256, "temperature": 0.0},
        )
        raw = response.response.strip()

        # Extract JSON even if the model added surrounding text
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            queries = data.get("queries", [])
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                queries = [q.strip() for q in queries if q.strip()]
                if queries:
                    log.debug("Rewritten queries: %s", queries)
                    return queries

    except Exception as exc:
        log.warning("Query rewrite failed (using original): %s", exc)

    return [question]
