# Use Ubuntu 22.04 ARM64 as base image
# Note: Docker will automatically select the target architecture base image during build
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
# We use --system to install directly into the system python or create a venv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-install-project --no-dev

# Copy project code
COPY . .

# Expose API port
EXPOSE 8080

# Environment variables
ENV PATH="/app/.venv/bin:$PATH"
# Default library path (expects user to mount file here)
ENV RKLLM_LIB_PATH=/app/lib/librkllmrt.so
# Default logs directory
ENV LOG_DIR=/app/logs

# Create necessary directories
RUN mkdir -p /app/lib /app/logs /app/models

# Command to run the application
CMD ["uv", "run", "python", "main.py"]
