FROM ghcr.io/astral-sh/uv:latest AS uv-stage

FROM python:3.10-slim AS builder

COPY --from=uv-stage /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files and readme
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen

# ------------------------------------------------------------
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src/ src/
COPY app/ app/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# 1. Download the raw dataset (from the UCI‑derived source)
ADD https://figshare.unimelb.edu.au/ndownloader/files/13603310 data/Dataset2.xlsx

# 2. Train the model (this creates artifacts/model.pkl and preprocessor.pkl)
RUN python src/training.py

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]