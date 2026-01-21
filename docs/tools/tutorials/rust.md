# Rust Custom Tools Tutorial

Build custom tools as a Rust Spin component.

For full Rust component documentation, see [Spin Rust Components](https://spinframework.dev/v2/rust-components).

## Prerequisites

- [Spin CLI](../spin.md) installed
- Rust toolchain with `wasm32-wasip1` target:

```bash
rustup target add wasm32-wasip1
```

## Step 1: Create Your Project

Create a new directory for your custom tools:

```bash
mkdir weather-tools
cd weather-tools
cargo init --lib
```

Update `Cargo.toml`:

```toml title="Cargo.toml"
[package]
name = "weather-tools"
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

Create `src/lib.rs`:

```rust title="src/lib.rs"
use anyhow::Result;
use serde::{Deserialize, Serialize};
use spin_sdk::{
    http::{IntoResponse, Request, Response},
    http_component,
};

#[derive(Deserialize)]
struct ExecuteRequest {
    #[allow(dead_code)]
    session_id: Option<String>,
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
fn handle_request(req: Request) -> Result<impl IntoResponse> {
    let path = req.path();

    match path {
        p if p.ends_with("/execute") => handle_execute(req),
        p if p.ends_with("/health") => handle_health(),
        _ => not_found(),
    }
}

fn handle_execute(req: Request) -> Result<Response> {
    let body = req.body();
    let request: ExecuteRequest = serde_json::from_slice(body)?;

    let response = match request.tool.as_str() {
        "get_weather" => execute_get_weather(&request.args),
        "get_forecast" => execute_get_forecast(&request.args),
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

fn execute_get_weather(args: &serde_json::Value) -> ExecuteResponse {
    let location = args.get("location").and_then(|v| v.as_str()).unwrap_or("");

    if location.is_empty() {
        return ExecuteResponse {
            success: false,
            result: "Missing required: location".to_string(),
            error_type: Some("InvalidArguments".to_string()),
        };
    }

    // Mock weather data - replace with real API call if needed
    let result = serde_json::json!({
        "location": location,
        "temperature": 72,
        "condition": "sunny",
        "humidity": 45
    });

    ExecuteResponse {
        success: true,
        result: result.to_string(),
        error_type: None,
    }
}

fn execute_get_forecast(args: &serde_json::Value) -> ExecuteResponse {
    let location = args.get("location").and_then(|v| v.as_str()).unwrap_or("");
    let days = args.get("days").and_then(|v| v.as_i64()).unwrap_or(3) as usize;

    if location.is_empty() {
        return ExecuteResponse {
            success: false,
            result: "Missing required: location".to_string(),
            error_type: Some("InvalidArguments".to_string()),
        };
    }

    // Mock forecast data - replace with real API call if needed
    let all_days = vec![
        serde_json::json!({"day": "Monday", "high": 75, "low": 60, "condition": "sunny"}),
        serde_json::json!({"day": "Tuesday", "high": 72, "low": 58, "condition": "partly cloudy"}),
        serde_json::json!({"day": "Wednesday", "high": 68, "low": 55, "condition": "cloudy"}),
    ];

    let forecast_days: Vec<_> = all_days.into_iter().take(days).collect();

    let result = serde_json::json!({
        "location": location,
        "days": forecast_days
    });

    ExecuteResponse {
        success: true,
        result: result.to_string(),
        error_type: None,
    }
}

fn handle_health() -> Result<Response> {
    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(r#"{"status":"healthy","component":"weather"}"#)
        .build())
}

fn not_found() -> Result<Response> {
    Ok(Response::builder()
        .status(404)
        .header("content-type", "application/json")
        .body(r#"{"error":"not found"}"#)
        .build())
}
```

## Step 2: Create spin.toml

Create `spin.toml` in your project root:

```toml title="spin.toml"
spin_manifest_version = 2

[application]
name = "weather-tools"
version = "0.1.0"

[[trigger.http]]
route = "/weather/..."
component = "weather"

[component.weather]
source = "target/wasm32-wasip1/release/weather_tools.wasm"
allowed_outbound_hosts = []

[component.weather.build]
command = "cargo build --target wasm32-wasip1 --release"
```

## Step 3: Build and Run

```bash
spin build
spin up
```

Test your component:

```bash
# Health check
curl http://localhost:3000/weather/health

# Execute tool
curl -X POST http://localhost:3000/weather/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "tool": "get_weather",
    "args": {"location": "Seattle"}
  }'
```

Expected output:

```json
{
  "success": true,
  "result": "{\"condition\":\"sunny\",\"humidity\":45,\"location\":\"Seattle\",\"temperature\":72}"
}
```

## Step 4: Configure DeepFabric

Create `config.yaml`:

```yaml title="config.yaml"
topics:
  prompt: "Weather-related assistant tasks"
  mode: tree
  depth: 2
  degree: 3

generation:
  system_prompt: "You are a weather assistant that helps users with weather information."

  conversation:
    type: chain_of_thought
    reasoning_style: agent

  tools:
    spin_endpoint: "http://localhost:3000"
    custom:
      - name: get_weather
        description: "Get current weather for a location"
        parameters:
          - name: location
            type: str
            description: "City name"
            required: true
        returns: "Weather data including temperature and conditions"
        component: weather

      - name: get_forecast
        description: "Get weather forecast for upcoming days"
        parameters:
          - name: location
            type: str
            description: "City name"
            required: true
          - name: days
            type: int
            description: "Number of days (1-7)"
            required: false
        returns: "Forecast data for requested days"
        component: weather

    max_per_query: 2
    max_agent_steps: 3

  llm:
    provider: "openai"
    model: "gpt-4o"

output:
  num_samples: 20
  batch_size: 5
  save_as: "weather-tools-dataset.jsonl"
```

## Step 5: Generate Dataset

```bash
deepfabric start config.yaml
```

## Calling External APIs

To call real external APIs instead of returning mock data, you'll need an API key. You can get a free API key from [OpenWeatherMap](https://openweathermap.org/api).

Once you have your API key, make these three changes:

### 1. Update spin.toml

Add `allowed_outbound_hosts` to permit outbound requests and define the API key variable:

```toml title="spin.toml"
spin_manifest_version = 2

[application]
name = "weather-tools"
version = "0.1.0"

[variables]
openweathermap_api_key = { required = true }

[[trigger.http]]
route = "/weather/..."
component = "weather"

[component.weather]
source = "target/wasm32-wasip1/release/weather_tools.wasm"
allowed_outbound_hosts = ["https://api.openweathermap.org"]

[component.weather.variables]
api_key = "{{ openweathermap_api_key }}"

[component.weather.build]
command = "cargo build --target wasm32-wasip1 --release"
```

### 2. Update src/lib.rs

Replace your `src/lib.rs` with this version that calls the OpenWeatherMap API:

```rust title="src/lib.rs"
use anyhow::Result;
use serde::{Deserialize, Serialize};
use spin_sdk::{
    http::{IntoResponse, Method, Request, Response},
    http_component,
    variables,
};

#[derive(Deserialize)]
struct ExecuteRequest {
    #[allow(dead_code)]
    session_id: Option<String>,
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
fn handle_request(req: Request) -> Result<impl IntoResponse> {
    let path = req.path();

    match path {
        p if p.ends_with("/execute") => handle_execute(req),
        p if p.ends_with("/health") => handle_health(),
        _ => not_found(),
    }
}

fn handle_execute(req: Request) -> Result<Response> {
    let body = req.body();
    let request: ExecuteRequest = serde_json::from_slice(body)?;

    let response = match request.tool.as_str() {
        "get_weather" => execute_get_weather(&request.args),
        "get_forecast" => execute_get_forecast(&request.args),
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

fn execute_get_weather(args: &serde_json::Value) -> ExecuteResponse {
    let location = args.get("location").and_then(|v| v.as_str()).unwrap_or("");

    if location.is_empty() {
        return ExecuteResponse {
            success: false,
            result: "Missing required: location".to_string(),
            error_type: Some("InvalidArguments".to_string()),
        };
    }

    let api_key = match variables::get("api_key") {
        Ok(key) => key,
        Err(_) => {
            return ExecuteResponse {
                success: false,
                result: "Failed to get API key".to_string(),
                error_type: Some("ConfigError".to_string()),
            };
        }
    };

    let encoded_location = urlencoding::encode(location);
    let url = format!(
        "https://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=metric",
        encoded_location, api_key
    );

    let request = Request::builder()
        .method(Method::Get)
        .uri(&url)
        .build();

    match spin_sdk::http::send(request) {
        Ok(response) => {
            if response.status() != 200 {
                return ExecuteResponse {
                    success: false,
                    result: format!("API error: {}", response.status()),
                    error_type: Some("APIError".to_string()),
                };
            }
            let body = String::from_utf8_lossy(response.body()).to_string();
            ExecuteResponse {
                success: true,
                result: body,
                error_type: None,
            }
        }
        Err(e) => ExecuteResponse {
            success: false,
            result: format!("Request failed: {}", e),
            error_type: Some("APIError".to_string()),
        },
    }
}

fn execute_get_forecast(args: &serde_json::Value) -> ExecuteResponse {
    let location = args.get("location").and_then(|v| v.as_str()).unwrap_or("");
    let days = args.get("days").and_then(|v| v.as_i64()).unwrap_or(3) as i32;

    if location.is_empty() {
        return ExecuteResponse {
            success: false,
            result: "Missing required: location".to_string(),
            error_type: Some("InvalidArguments".to_string()),
        };
    }

    let api_key = match variables::get("api_key") {
        Ok(key) => key,
        Err(_) => {
            return ExecuteResponse {
                success: false,
                result: "Failed to get API key".to_string(),
                error_type: Some("ConfigError".to_string()),
            };
        }
    };

    let encoded_location = urlencoding::encode(location);
    // cnt parameter limits the number of 3-hour forecast periods (8 per day)
    let cnt = std::cmp::min(days * 8, 40); // Max 40 periods (5 days)
    let url = format!(
        "https://api.openweathermap.org/data/2.5/forecast?q={}&appid={}&units=metric&cnt={}",
        encoded_location, api_key, cnt
    );

    let request = Request::builder()
        .method(Method::Get)
        .uri(&url)
        .build();

    match spin_sdk::http::send(request) {
        Ok(response) => {
            if response.status() != 200 {
                return ExecuteResponse {
                    success: false,
                    result: format!("API error: {}", response.status()),
                    error_type: Some("APIError".to_string()),
                };
            }
            let body = String::from_utf8_lossy(response.body()).to_string();
            ExecuteResponse {
                success: true,
                result: body,
                error_type: None,
            }
        }
        Err(e) => ExecuteResponse {
            success: false,
            result: format!("Request failed: {}", e),
            error_type: Some("APIError".to_string()),
        },
    }
}

fn handle_health() -> Result<Response> {
    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(r#"{"status":"healthy","component":"weather"}"#)
        .build())
}

fn not_found() -> Result<Response> {
    Ok(Response::builder()
        .status(404)
        .header("content-type", "application/json")
        .body(r#"{"error":"not found"}"#)
        .build())
}
```

### 3. Update Cargo.toml

Add the `urlencoding` dependency:

```toml title="Cargo.toml"
[package]
name = "weather-tools"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
anyhow = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
spin-sdk = "3.0"
urlencoding = "2"
```

### 4. Rebuild and run

After making these changes, rebuild the WASM component and run with your API key:

```bash
spin build
SPIN_VARIABLE_OPENWEATHERMAP_API_KEY=your-api-key spin up
```

Spin variables use the `SPIN_VARIABLE_` prefix with the variable name in uppercase.

## Next Steps

- Package as Docker image - see [Custom Tools](../custom.md#packaging-with-docker)
