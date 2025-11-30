# Multi-stage build for worker service
FROM astral/uv:0.9.9-python3.13-bookworm-slim AS builder

WORKDIR /app

# Build argument to control CUDA installation
ARG ENABLE_CUDA=false

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Install CUDA-enabled PyTorch if requested (for production GPU deployment)
RUN if [ "$ENABLE_CUDA" = "true" ]; then \
    uv pip install torch --index-url https://download.pytorch.org/whl/cu126; \
    fi

# Production stage
FROM python:3.13-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy uv and virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Copy application code
COPY app ./app

# Expose port
EXPOSE 8082

# Run the application using uv
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082", "--reload"]