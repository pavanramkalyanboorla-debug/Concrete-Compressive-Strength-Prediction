FROM python:3.10-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency metadata first (better caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment using uv
RUN uv sync --frozen --no-dev --python 3.10

# --- Final stage (smaller) ---
FROM python:3.10-slim

WORKDIR /app

# Copy the whole virtualenv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Ensure the virtual env is used
ENV PATH="/app/.venv/bin:$PATH"

# Expose both FastAPI and Streamlit (if you still run both)
EXPOSE 8000 8501

# Default command (FastAPI; adjust if needed)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]