# ==============================================================================
# YogaSutra — Multi-Stage Production Containerization
# ==============================================================================
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build dependencies for C/C++ extensions & OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel/build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core runtime dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.30.0 \
    pydantic>=2.0.0 \
    python-dotenv>=1.0.0 \
    rich>=13.0.0 \
    google-genai>=1.0.0 \
    google-adk>=1.0.0 \
    opencv-python-headless>=4.8.0 \
    mediapipe>=0.10.0 \
    numpy>=1.26.0 \
    httpx>=0.27.0 \
    jinja2>=3.1.0 \
    websockets>=12.0

# ------------------------------------------------------------------------------
# Stage 2: Final Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Label metadata
LABEL org.opencontainers.image.title="YogaSutra" \
      org.opencontainers.image.description="Transparent Multi-Agent AI System for Personalized Yoga Practice" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.authors="YogaSutra Architecture Team"

WORKDIR /app

# Install minimal runtime shared libraries for OpenCV, audio & graphics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create unprivileged user for security compliance (Principle of Least Privilege)
RUN groupadd -g 1001 yogasutra && \
    useradd -u 1001 -g yogasutra -s /bin/bash -m yogasutra

# Copy application code, assets, data, and scripts
COPY --chown=yogasutra:yogasutra app /app/app
COPY --chown=yogasutra:yogasutra data /app/data
COPY --chown=yogasutra:yogasutra scripts /app/scripts
COPY --chown=yogasutra:yogasutra main.py pyproject.toml README.md /app/

# Ensure runtime directories exist with proper write permissions for non-root user
RUN mkdir -p /app/data/traces /app/data/audio_cache /app/data/evaluation /app/data/models && \
    chown -R yogasutra:yogasutra /app/data

# Environment configuration defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONTDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    YOGASUTRA_MODEL=gemini-2.5-flash \
    YOGASUTRA_TRACE_DIR=/app/data/traces \
    YOGASUTRA_LOG_LEVEL=INFO \
    MPLCONFIGDIR=/tmp/matplotlib

# Switch to unprivileged user
USER yogasutra

# Expose HTTP API port
EXPOSE 8000

# Health check to ensure API responsiveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Default entrypoint starts the FastAPI unified server
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
