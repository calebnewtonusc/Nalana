# ─────────────────────────────────────────────────────────────────────────────
# Nalana API Server — Multi-stage Dockerfile
#
# Base:    nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# Python:  3.11 (via deadsnakes PPA)
# Model:   Loaded from /model (mounted volume or COPY at build time)
# Port:    8000
#
# Build:
#   docker build -f deploy/Dockerfile -t nalana-api:latest .
#
# Run (no GPU, dev):
#   docker run -p 8000:8000 -e NALANA_MODEL_PATH=/model -v ./checkpoints:/model nalana-api
#
# Run (4xA100, production):
#   docker run --gpus '"device=0,1,2,3"' -p 8000:8000 \
#     -e NALANA_MODEL_PATH=/model -e NALANA_TP_SIZE=4 \
#     -v /mnt/models/nalana-v1:/model:ro \
#     nalana-api:latest
# ─────────────────────────────────────────────────────────────────────────────

# ─── Stage 1: Builder — install Python packages ───────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS builder

# Prevent apt interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        wget \
        git \
        ca-certificates \
        build-essential \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy and install Python dependencies in two layers:
# 1. Heavy ML/CUDA packages (cached unless requirements change)
# 2. Application packages
COPY requirements.txt .

# hadolint ignore=DL3013
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 first (largest package)
# hadolint ignore=DL3013
RUN pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM (CUDA-matched build)
# hadolint ignore=DL3013
RUN pip install vllm==0.4.2

# Install remaining requirements
# hadolint ignore=DL3013
RUN pip install \
    transformers>=4.45.0 \
    datasets>=2.20.0 \
    peft>=0.12.0 \
    trl>=0.11.0 \
    accelerate>=0.34.0 \
    anthropic>=0.40.0 \
    httpx>=0.27.0 \
    tqdm>=4.66.0

# Install API server dependencies
# hadolint ignore=DL3013
RUN pip install \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    uvloop>=0.19.0 \
    pydantic>=2.6.0 \
    redis[hiredis]>=5.0.0 \
    python-dotenv>=1.0.0


# ─── Stage 2: Runtime — lean production image ─────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-distutils \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# ─── Application code ─────────────────────────────────────────────────────────
WORKDIR /app

# Copy only what's needed for runtime (not training data, not notebooks)
COPY deploy/api_server.py  deploy/api_server.py
COPY core/                 core/

# Create non-root user for security
RUN groupadd -r nalana && useradd -r -g nalana -d /app nalana \
 && chown -R nalana:nalana /app

# Model directory — populated at runtime via volume mount
RUN mkdir -p /model && chown nalana:nalana /model

# HuggingFace cache directory
RUN mkdir -p /app/.cache/huggingface && chown -R nalana:nalana /app/.cache

# ─── Environment defaults ─────────────────────────────────────────────────────
ENV NALANA_MODEL_PATH=/model
ENV NALANA_TP_SIZE=4
ENV NALANA_MAX_MODEL_LEN=8192
ENV REDIS_URL=redis://redis:6379
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV CUDA_VISIBLE_DEVICES=all

# ─── Expose ───────────────────────────────────────────────────────────────────
EXPOSE 8000

# ─── Healthcheck ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# ─── Entrypoint ───────────────────────────────────────────────────────────────
USER nalana

CMD ["uvicorn", "deploy.api_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--log-level", "warning"]
