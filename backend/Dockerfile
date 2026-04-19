# ---- Backend Dockerfile ----
# Python 3.12 + uv for fast dependency management
# Runs FastAPI via uvicorn on port 8000

FROM python:3.12-slim AS base

# System dependencies for OpenCV, DeepFace, and Whisper
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv via pip (avoids pulling from ghcr.io)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Use CPU-only PyTorch (~200MB instead of ~2GB CUDA version)
# The container has no GPU, so CUDA wheels are wasted space and bandwidth.
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# Install dependencies
RUN uv sync --no-install-project

# Copy application source
COPY . .

# Create volume mount points for persistent data
VOLUME ["/app/chroma_data"]

EXPOSE 8000

# Run FastAPI with uvicorn — bind to 0.0.0.0 so Docker networking works
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
