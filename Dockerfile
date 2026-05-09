# Concrete Mix Optimizer — NEW Dockerfile (pre‑trained model approach)
FROM ghcr.io/astral-sh/uv:latest AS uv-stage

FROM python:3.10-slim AS deps
COPY --from=uv-stage /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen

# ── Single serving stage (no training) ──
FROM python:3.10-slim AS runtime
COPY --from=deps /app/.venv /app/.venv
WORKDIR /app

# Copy pre‑trained artifacts (uploaded via hf upload)
COPY artifacts/model.pkl artifacts/model.pkl
COPY artifacts/preprocessor.pkl artifacts/preprocessor.pkl
COPY artifacts/uncertainty_params.pkl artifacts/uncertainty_params.pkl

COPY src/ src/
COPY app/ app/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]