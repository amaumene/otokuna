# Multi-stage Dockerfile for Otokuna Flask web application
# Stage 1: Builder - Install dependencies
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements files
COPY requirements.txt requirements.txt

# Copy libs package for installation
#COPY libs/ libs/

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Stage 2: Runtime - Minimal production image
FROM python:3.12-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # SQLite for redislite
    libsqlite3-0 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . /app/
#COPY --chown=appuser:appuser libs/ /libs/

# Create necessary directories with proper permissions
#RUN mkdir -p /tmp/otokuna_dtale_state && \
#    chown -R appuser:appuser /tmp/otokuna_dtale_state

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OTOKUNA_CONFIG_FILE=/app/config/config.yml

# Switch to non-root user
USER appuser

# Expose port (matching fly.toml internal_port)
EXPOSE 8000

ENV OTOKUNA_CONFIG_FILE=/app/config/config.yml

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/').read()" || exit 1

# Run gunicorn
CMD ["gunicorn", \
     "--workers", "1", \
     "--bind", "[::]:8000", \
     "--timeout", "120", \
     "--worker-tmp-dir", "/dev/shm", \
     "--no-control-socket", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "app:app"]
