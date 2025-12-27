# Custom Tools

Create custom Spin components to add new tools for dataset generation.

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

Add to your DeepFabric config:

```yaml title="config.yaml"
generation:
  tools:
    spin_endpoint: "http://localhost:3000"
    custom:
      - name: my_tool
        description: "Process input and return result"
        parameters:
          - name: input
            type: str
            required: true
        returns: "Processed result"
        component: my-tools
```

!!! info "Component Routing"
    The `component` field specifies which Spin component handles this tool.

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

## Best Practices

!!! tip "Session Isolation"
    Use `session_id` to scope any state your tool maintains.

!!! tip "Error Handling"
    Return structured errors with `error_type` for debugging.

!!! tip "Idempotency"
    Design tools to be safely retryable when possible.

!!! tip "Documentation"
    Include clear descriptions in tool definitions for better LLM reasoning.
