FROM ghcr.io/astral-sh/uv:latest AS uv-stage

FROM python:3.10-slim AS builder

COPY --from=uv-stage /uv /usr/local/bin/uv

WORKDIR /app

# Copy both dependency files – uv needs the lockfile for --frozen
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen

# ------------------------------------------------------------
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY src/ src/
COPY app/ app/
COPY artifacts/ artifacts/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501 8000
ENV PYTHONPATH=/app

CMD ["sh", "-c", "PYTHONPATH=/app streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]