FROM ghcr.io/astral-sh/uv:latest AS uv-stage

# ------------------------------------------------------------
# Stage 1: install Python dependencies (cached unless pyproject.toml/uv.lock changes)
# ------------------------------------------------------------
FROM python:3.10-slim AS deps
COPY --from=uv-stage /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen

# ------------------------------------------------------------
# Stage 2: train the model (only re‑runs when data or training code changes)
# ------------------------------------------------------------
FROM python:3.10-slim AS trainer
COPY --from=deps /app/.venv /app/.venv
WORKDIR /app

# Copy only the files required for training

COPY src/constants.py src/constants.py
COPY src/components/ src/components/
COPY src/pipeline/ src/pipeline/
COPY src/utils/ src/utils/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# Download dataset needed by training pipeline
RUN python -c "import pandas as pd; df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'); df.to_csv('data/concrete.csv', index=False)"

# Train the model
RUN python src/pipeline/training_pipeline.py

# Compute uncertainty parameters
RUN python src/train_uncertainty.py
# ------------------------------------------------------------
# Stage 3: final app image (fast rebuild on UI changes)
# ------------------------------------------------------------
FROM python:3.10-slim AS app
COPY --from=deps /app/.venv /app/.venv
WORKDIR /app

# Take the trained model from the trainer stage
COPY --from=trainer /app/artifacts/ artifacts/

# Copy the full source code (overwrites trainer files, which is fine)
COPY src/ src/
COPY app/ app/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]