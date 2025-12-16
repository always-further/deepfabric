# =============================================================================
# DeepFabric Tools-SDK Spin Server
# Multi-stage Dockerfile for building and running the Spin WASM components
# Uses Chainguard images for runtime security
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build Rust WASM components
# Note: Using official Rust image for build stage as it includes rustup
# needed for adding wasm32-wasip1 target. Build artifacts are copied to
# Chainguard runtime image.
# -----------------------------------------------------------------------------
FROM rust:1.87-bookworm AS rust-builder

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
# Stage 2: Runtime - Spin server
# -----------------------------------------------------------------------------
FROM cgr.dev/chainguard/wolfi-base:latest AS runtime

# Install minimal runtime dependencies
# libstdc++ is required for Spin binary
RUN apk add --no-cache \
    ca-certificates \
    curl \
    libstdc++

# Disable Spin telemetry
ENV SPIN_TELEMETRY_DISABLED=1

# Install Spin CLI - download binary directly for multi-arch support
ARG SPIN_VERSION=3.0.0
ARG TARGETARCH
RUN mkdir -p /usr/local/bin && \
    case "${TARGETARCH}" in \
        arm64) ARCH="aarch64" ;; \
        amd64) ARCH="amd64" ;; \
        *) ARCH="amd64" ;; \
    esac && \
    echo "Downloading Spin for architecture: ${ARCH}" && \
    curl -fsSL "https://github.com/spinframework/spin/releases/download/v${SPIN_VERSION}/spin-v${SPIN_VERSION}-linux-${ARCH}.tar.gz" -o /tmp/spin.tar.gz && \
    tar -xzf /tmp/spin.tar.gz -C /usr/local/bin spin && \
    rm /tmp/spin.tar.gz && \
    chmod +x /usr/local/bin/spin && \
    ls -la /usr/local/bin/spin

WORKDIR /app

# Copy spin configuration
COPY spin.toml .
COPY runtime-config.toml .

# Create directory structure for components
RUN mkdir -p components/vfs/target/wasm32-wasip1/release \
             components/mock/target/wasm32-wasip1/release \
             .spin/logs

# Copy built WASM components from builder stages
COPY --from=rust-builder /build/components/vfs/target/wasm32-wasip1/release/vfs.wasm \
     components/vfs/target/wasm32-wasip1/release/
COPY --from=rust-builder /build/components/mock/target/wasm32-wasip1/release/mock.wasm \
     components/mock/target/wasm32-wasip1/release/

# Expose the default Spin port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/vfs/health || exit 1

# Run Spin server
ENTRYPOINT ["spin", "up"]
CMD ["--listen", "0.0.0.0:3000", "--runtime-config-file", "runtime-config.toml"]
