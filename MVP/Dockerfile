# OSKAR v0.7 ML API Dockerfile
# Optimized for heavy NLP models (SentenceTransformers, PyTorch, FAISS)

# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CPU first to save space (no CUDA for standard inference edge nodes)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---
# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install runtime binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for security
RUN useradd -m -s /bin/bash oskar_user
USER oskar_user

# Copy application code
COPY --chown=oskar_user:oskar_user . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TESSERACT_CMD="tesseract"
ENV NEO4J_URI="bolt://neo4j:7687"
# Redis URI for Phase 24
ENV REDIS_URL="redis://redis:6379/0"

EXPOSE 8000

# Run Uvicorn backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
