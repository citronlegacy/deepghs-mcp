# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────
# deepghs-mcp — Dockerfile
# Multi-stage build: slim final image using Python 3.12 slim
# ──────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies into an isolated prefix
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: final runtime image ──────────────────────────────
FROM python:3.12-slim AS final

LABEL org.opencontainers.image.title="DeepGHS MCP"
LABEL org.opencontainers.image.description="MCP server for the DeepGHS anime AI ecosystem — dataset discovery, tag search, character dataset finder, and training pipeline code generation."
LABEL org.opencontainers.image.url="https://github.com/citronlegacy/deepghs-mcp"
LABEL org.opencontainers.image.source="https://github.com/citronlegacy/deepghs-mcp"
LABEL org.opencontainers.image.licenses="MIT"
LABEL io.modelcontextprotocol.server.name="io.github.citronlegacy/deepghs-mcp"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy server source
COPY deepghs_mcp.py .

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false mcpuser
USER mcpuser

# MCP stdio transport — no ports needed
# HF_TOKEN is passed at runtime via the MCP client config env block
ENV HF_TOKEN=""

# Health check: verify the server imports and starts cleanly
# Uses a 30s start period to allow model init on first run
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import deepghs_mcp" || exit 1

ENTRYPOINT ["python", "deepghs_mcp.py"]
