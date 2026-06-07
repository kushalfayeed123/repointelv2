# Use an explicit, stable Python base image
FROM python:3.12-slim-bookworm

# Install system dependencies (including git for repository syncing and curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Astral 'uv' natively into the image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up working directory boundaries
WORKDIR /app

# Copy dependency configuration frames first to utilize Docker layer caching
COPY pyproject.toml uv.lock ./

# Synchronize the production virtual environment
RUN uv sync --frozen --no-dev

# Copy the remaining codebase modules into the container image
COPY . .

# Ensure the sandbox workspace cache path exists and is writeable
RUN mkdir -p workspace_cache db && chmod -R 777 workspace_cache db

# Expose potential runtime network ports
EXPOSE 8000
EXPOSE 5001

# The execution fallback command will be overridden by the cloud service configuration
CMD ["uv", "run", "src/mcp_server_sse.py"]