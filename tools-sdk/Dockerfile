# =============================================================================
# DeepFabric Tools-SDK Spin Server
# Multi-stage Dockerfile for building and running the Spin WASM components
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build Rust WASM components
# -----------------------------------------------------------------------------
FROM rust:1.83-bookworm AS rust-builder

# Install wasm32-wasip1 target
RUN rustup target add wasm32-wasip1

WORKDIR /build

# Copy Rust component sources
COPY components/vfs/Cargo.toml components/vfs/Cargo.toml
COPY components/vfs/src components/vfs/src
COPY components/mock/Cargo.toml components/mock/Cargo.toml
COPY components/mock/src components/mock/src

# Build VFS component
WORKDIR /build/components/vfs
RUN cargo build --target wasm32-wasip1 --release

# Build Mock component
WORKDIR /build/components/mock
RUN cargo build --target wasm32-wasip1 --release

# -----------------------------------------------------------------------------
# Stage 2: Build Python WASM components (optional - currently disabled)
# -----------------------------------------------------------------------------
FROM python:3.12-bookworm AS python-builder

# Install uv for fast dependency management
RUN pip install uv

WORKDIR /build

# Copy Python component sources
COPY components/github/pyproject.toml components/github/pyproject.toml
COPY components/github/app.py components/github/app.py

# Install dependencies and build
WORKDIR /build/components/github
RUN uv venv && \
    uv pip install -r pyproject.toml && \
    uv run componentize-py -w spin-http componentize app -o github.wasm || true

# -----------------------------------------------------------------------------
# Stage 3: Runtime - Spin server
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Spin CLI
ARG SPIN_VERSION=3.0.0
RUN curl -fsSL https://developer.fermyon.com/downloads/install.sh | bash -s -- -v v${SPIN_VERSION} && \
    mv spin /usr/local/bin/

WORKDIR /app

# Copy spin configuration
COPY spin.toml .
COPY runtime-config.toml .

# Create directory structure for components
RUN mkdir -p components/vfs/target/wasm32-wasip1/release \
             components/mock/target/wasm32-wasip1/release \
             components/github \
             .spin/logs

# Copy built WASM components from builder stages
COPY --from=rust-builder /build/components/vfs/target/wasm32-wasip1/release/vfs.wasm \
     components/vfs/target/wasm32-wasip1/release/
COPY --from=rust-builder /build/components/mock/target/wasm32-wasip1/release/mock.wasm \
     components/mock/target/wasm32-wasip1/release/

# Optionally copy Python component if built successfully
COPY --from=python-builder /build/components/github/github.wasm \
     components/github/ 2>/dev/null || true

# Environment variables for configuration
# Override these at runtime as needed
ENV SPIN_VARIABLE_GITHUB_TOKEN=""
ENV SPIN_VARIABLE_ALLOWED_REPOS=""
ENV SPIN_VARIABLE_ALLOW_WRITES="false"

# Expose the default Spin port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/vfs/health || exit 1

# Run Spin server
ENTRYPOINT ["spin", "up"]
CMD ["--listen", "0.0.0.0:3000", "--runtime-config-file", "runtime-config.toml"]
