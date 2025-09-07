# Edge-optimized Dockerfile for Raspberry Pi and edge devices
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install with optimizations
COPY requirements.txt .

# Install Python dependencies with optimizations for edge devices
RUN pip install --no-cache-dir --compile \
    --global-option="-j4" \
    -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Set environment variables for edge optimization
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV NUMEXPR_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2

# Create non-root user
RUN useradd -m -u 1000 cyberai && \
    chown -R cyberai:cyberai /app
USER cyberai

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=60s --timeout=15s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "src.main"]