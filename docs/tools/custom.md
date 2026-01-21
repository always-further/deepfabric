# Custom Tools

Build your own Spin components when the built-in VFS tools and Mock component don't cover your use case. Custom components let you add domain-specific tools with real execution logic - database queries, specialized file formats, custom APIs, or any functionality your training data requires.

**Why build custom tools?** VFS handles file operations, Mock simulates external APIs with configurable responses. But sometimes you need tools that actually execute custom logic - parsing specific formats, running calculations, or integrating with internal systems.

## Prerequisites

- [Spin CLI](spin.md) installed
- Rust toolchain with `wasm32-wasip1` target:
  ```bash
  rustup target add wasm32-wasip1
  ```

## Creating a Component

### 1. Initialize Project

```bash
cd tools-sdk/components
mkdir my-tools
cd my-tools
cargo init --lib
```

### 2. Configure Cargo.toml

```toml title="Cargo.toml"
[package]
name = "my-tools"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
anyhow = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
spin-sdk = "3.0"
```

### 3. Implement the Handler

```rust title="src/lib.rs"
use anyhow::Result;
use serde::{Deserialize, Serialize};
use spin_sdk::{
    http::{Request, Response},
    http_component,
};

#[derive(Deserialize)]
struct ExecuteRequest {
    session_id: String,
    tool: String,
    args: serde_json::Value,
}

#[derive(Serialize)]
struct ExecuteResponse {
    success: bool,
    result: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_type: Option<String>,
}

#[http_component]
fn handle_request(req: Request) -> Result<Response> {
    let path = req.path();

    match path {
        p if p.ends_with("/execute") => handle_execute(req),
        p if p.ends_with("/health") => handle_health(),
        _ => not_found(),
    }
}

fn handle_execute(req: Request) -> Result<Response> {
    let request: ExecuteRequest = serde_json::from_slice(req.body())?;

    let response = match request.tool.as_str() {
        "my_tool" => execute_my_tool(&request),
        _ => ExecuteResponse {
            success: false,
            result: format!("Unknown tool: {}", request.tool),
            error_type: Some("UnknownTool".to_string()),
        },
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&response)?)
        .build())
}

fn execute_my_tool(req: &ExecuteRequest) -> ExecuteResponse {
    // Your tool logic here
    let input = req.args.get("input")
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    ExecuteResponse {
        success: true,
        result: format!("Processed: {}", input),
        error_type: None,
    }
}

fn handle_health() -> Result<Response> {
    Ok(Response::builder()
        .status(200)
        .body(r#"{"status":"healthy"}"#)
        .build())
}

fn not_found() -> Result<Response> {
    Ok(Response::builder()
        .status(404)
        .body(r#"{"error":"not found"}"#)
        .build())
}
```

### 4. Register in spin.toml

Add to `tools-sdk/spin.toml`:

```toml title="spin.toml"
[[trigger.http]]
route = "/my-tools/..."
component = "my-tools"

[component.my-tools]
source = "components/my-tools/target/wasm32-wasip1/release/my_tools.wasm"
allowed_outbound_hosts = []

[component.my-tools.build]
command = "cargo build --target wasm32-wasip1 --release"
workdir = "components/my-tools"
```

### 5. Build and Run

```bash
spin build
spin up
```

Test your component:

```bash title="Test custom tool"
curl -X POST http://localhost:3000/my-tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "tool": "my_tool",
    "args": {"input": "hello"}
  }'
```

## Using Custom Tools

### Register Tool Definition

Custom tools require a complete config with agent mode enabled. Here's a minimal working example:

```yaml title="config.yaml"
topics:
  prompt: "Tasks requiring custom tool processing"
  mode: tree
  depth: 2
  degree: 3

generation:
  system_prompt: "Generate examples using the available tools."

  # Agent mode is required to use tools
  conversation:
    type: chain_of_thought
    reasoning_style: agent

  # Tool configuration
  tools:
    spin_endpoint: "http://localhost:3000"
    custom:
      - name: my_tool
        description: "Process input and return result"
        parameters:
          - name: input
            type: str
            description: "The input to process"
            required: true
        returns: "Processed result"
        component: my-tools  # Routes to /my-tools/execute
    max_per_query: 3
    max_agent_steps: 5

  llm:
    provider: "openai"
    model: "gpt-4o"

output:
  num_samples: 10
  batch_size: 2
  save_as: "custom-tools-dataset.jsonl"
```

!!! info "How It Works"
    1. DeepFabric loads custom tool definitions from the config
    2. The LLM sees the tool's name, description, and parameters
    3. When the LLM generates a tool call, it's sent to `{spin_endpoint}/{component}/execute`
    4. Your Spin component handles the request and returns results

## External APIs

To call external APIs, add allowed hosts:

```toml title="spin.toml"
[component.my-tools]
allowed_outbound_hosts = ["https://api.example.com"]
```

Then use HTTP in your handler:

```rust title="External API call"
use spin_sdk::http::{send, Method, Request as OutRequest};

fn call_external_api() -> Result<String> {
    let req = OutRequest::builder()
        .method(Method::Get)
        .uri("https://api.example.com/data")
        .build();

    let response = send(req)?;
    Ok(String::from_utf8_lossy(response.body()).to_string())
}
```

## Python Components

!!! tip "Python Support"
    Spin also supports Python via `componentize-py`. See the GitHub component in `tools-sdk/components/github/` for an example.

## Packaging with Docker

Package your custom tools as a Docker image for deployment and CI/CD workflows.

### Dockerfile

Create a multi-stage Dockerfile that builds your WASM components and packages them with Spin:

```dockerfile title="Dockerfile"
# Stage 1: Build Rust WASM components
FROM rust:1.87-bookworm AS rust-builder

RUN rustup target add wasm32-wasip1

WORKDIR /build

# Copy your component source
COPY components/my-tools/Cargo.toml components/my-tools/Cargo.toml
COPY components/my-tools/src components/my-tools/src

# Build component
WORKDIR /build/components/my-tools
RUN cargo build --target wasm32-wasip1 --release

# Stage 2: Runtime
FROM cgr.dev/chainguard/wolfi-base:latest AS runtime

RUN apk add --no-cache ca-certificates curl libstdc++

# Install Spin CLI
ARG SPIN_VERSION=3.0.0
ARG TARGETARCH
RUN mkdir -p /usr/local/bin && \
    case "${TARGETARCH}" in \
        arm64) ARCH="aarch64" ;; \
        amd64) ARCH="amd64" ;; \
        *) ARCH="amd64" ;; \
    esac && \
    curl -fsSL "https://github.com/spinframework/spin/releases/download/v${SPIN_VERSION}/spin-v${SPIN_VERSION}-linux-${ARCH}.tar.gz" \
      -o /tmp/spin.tar.gz && \
    tar -xzf /tmp/spin.tar.gz -C /usr/local/bin spin && \
    rm /tmp/spin.tar.gz

WORKDIR /app

COPY spin.toml .
RUN mkdir -p components/my-tools/target/wasm32-wasip1/release

COPY --from=rust-builder \
    /build/components/my-tools/target/wasm32-wasip1/release/my_tools.wasm \
    components/my-tools/target/wasm32-wasip1/release/

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/my-tools/health || exit 1

ENTRYPOINT ["spin", "up"]
CMD ["--listen", "0.0.0.0:3000"]
```

### Build and Run

```bash
# Build image
docker build -t my-tools:latest .

# Run locally
docker run -d -p 3000:3000 my-tools:latest

# Test
curl http://localhost:3000/my-tools/health
```

## CI/CD with GitHub Actions

Automate building and publishing your tools image using GitHub Actions.

### Workflow File

```yaml title=".github/workflows/tools-docker.yml"
name: Build and Publish Tools

on:
  push:
    branches: [main]
    paths:
      - 'components/**'
      - 'spin.toml'
      - 'Dockerfile'
  release:
    types: [published]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/my-tools

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### GitLab CI/CD

```yaml title=".gitlab-ci.yml"
stages:
  - build
  - publish

variables:
  IMAGE_NAME: registry.gitlab.com/$CI_PROJECT_PATH/my-tools

build:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  script:
    - docker build -t $IMAGE_NAME:$CI_COMMIT_SHA .
    - docker tag $IMAGE_NAME:$CI_COMMIT_SHA $IMAGE_NAME:latest
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
      changes:
        - components/**/*
        - spin.toml
        - Dockerfile

publish:
  stage: publish
  image: docker:24
  services:
    - docker:24-dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker push $IMAGE_NAME:$CI_COMMIT_SHA
    - docker push $IMAGE_NAME:latest
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
```

### Consuming Your Custom Image

Once published, run your custom tools container and configure DeepFabric to use it. See [Register Tool Definition](#register-tool-definition) for a complete config example.

Run with Docker Compose:

```yaml title="docker-compose.yml"
services:
  tools:
    image: ghcr.io/your-org/my-tools:latest
    ports:
      - "3000:3000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/my-tools/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```
