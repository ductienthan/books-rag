FROM python:3.11-slim

WORKDIR /app

# System dependencies — cached until base image changes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv — cached until uv image changes
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy ONLY dependency files first — cached until pyproject.toml or uv.lock changes
COPY pyproject.toml uv.lock README.md ./

# Install dependencies — cached unless pyproject.toml or uv.lock changes
RUN uv sync --no-dev

# Pre-download the embedding model — cached unless dependencies change
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"

# Copy source code LAST — a code change only re-runs from here, everything above stays cached
COPY bookrag/ ./bookrag/
COPY alembic/ ./alembic/
COPY alembic.ini .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["python", "-m", "bookrag.worker"]
