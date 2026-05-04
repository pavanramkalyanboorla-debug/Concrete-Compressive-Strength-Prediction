FROM ghcr.io/astral-sh/uv:latest AS uv-stage

FROM python:3.10-slim AS builder

COPY --from=uv-stage /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files and readme (hatchling needs readme)
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
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]