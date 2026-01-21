# Python Custom Tools Tutorial

Build custom tools as a Python Spin component.

For full Python component documentation, see [Spin Python Components](https://spinframework.dev/v2/python-components).

## Prerequisites

- [Spin CLI](../spin.md) installed
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for Python package management

## Step 1: Create Your Project

Create a new directory for your custom tools:

```bash
mkdir weather-tools
cd weather-tools
uv init
```

Install the required dependencies:

```bash
uv add spin-sdk==3.4.1 componentize-py==0.17.2
```

Create `app.py`:

```python title="app.py"
"""Weather Tools Spin Component."""

import json

from spin_sdk import http


# Tool handlers
def handle_get_weather(args: dict) -> dict:
    """Get current weather for a location."""
    location = args.get("location")
    if not location:
        return {
            "success": False,
            "result": "Missing required: location",
            "error_type": "InvalidArguments",
        }

    # Your tool logic here - this example returns mock data
    # In production, you might call a real weather API
    weather_data = {
        "location": location,
        "temperature": 72,
        "condition": "sunny",
        "humidity": 45,
    }

    return {"success": True, "result": json.dumps(weather_data)}


def handle_get_forecast(args: dict) -> dict:
    """Get weather forecast for upcoming days."""
    location = args.get("location")
    days = args.get("days", 3)

    if not location:
        return {
            "success": False,
            "result": "Missing required: location",
            "error_type": "InvalidArguments",
        }

    forecast = {
        "location": location,
        "days": [
            {"day": "Monday", "high": 75, "low": 60, "condition": "sunny"},
            {"day": "Tuesday", "high": 72, "low": 58, "condition": "partly cloudy"},
            {"day": "Wednesday", "high": 68, "low": 55, "condition": "cloudy"},
        ][:days],
    }

    return {"success": True, "result": json.dumps(forecast)}


# Tool router
TOOL_HANDLERS = {
    "get_weather": handle_get_weather,
    "get_forecast": handle_get_forecast,
}


class IncomingHandler(http.IncomingHandler):
    def handle_request(self, request: http.Request) -> http.Response:
        """Handle incoming HTTP requests."""
        path = request.uri
        if path.startswith("/weather"):
            path = path[8:]  # Remove "/weather" prefix
        if not path:
            path = "/"
        method = request.method

        # Health check
        if path == "/health" and method == "GET":
            return http.Response(
                200,
                {"content-type": "application/json"},
                bytes(json.dumps({"status": "healthy", "component": "weather"}), "utf-8"),
            )

        # Execute tool
        if path == "/execute" and method == "POST":
            try:
                body = json.loads(request.body)
                tool = body.get("tool", "")
                args = body.get("args", {})

                if tool in TOOL_HANDLERS:
                    result = TOOL_HANDLERS[tool](args)
                else:
                    result = {
                        "success": False,
                        "result": f"Unknown tool: {tool}",
                        "error_type": "UnknownTool",
                    }

                return http.Response(
                    200,
                    {"content-type": "application/json"},
                    bytes(json.dumps(result), "utf-8"),
                )

            except json.JSONDecodeError as e:
                return http.Response(
                    400,
                    {"content-type": "application/json"},
                    bytes(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}), "utf-8"),
                )

        # 404 for unknown routes
        return http.Response(
            404,
            {"content-type": "application/json"},
            bytes(json.dumps({"error": "Not found"}), "utf-8"),
        )
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
source = "weather.wasm"
allowed_outbound_hosts = []

[component.weather.build]
command = "uv run componentize-py -w spin-http componentize app -o weather.wasm"
```

## Step 3: Build and Run

Build and start your Spin application:

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
  "result": "{\"location\": \"Seattle\", \"temperature\": 72, \"condition\": \"sunny\", \"humidity\": 45}"
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

![DeepFabric generating dataset with Python Spin tools](../../images/python-spin.png)

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
source = "weather.wasm"
allowed_outbound_hosts = ["https://api.openweathermap.org"]

[component.weather.variables]
api_key = "{{ openweathermap_api_key }}"

[component.weather.build]
command = "uv run componentize-py -w spin-http componentize app -o weather.wasm"
```

### 2. Update app.py

Replace your `app.py` with this version that calls the OpenWeatherMap API:

```python title="app.py"
"""Weather Tools Spin Component with External API."""

import json
from urllib.parse import quote

from spin_sdk import http, variables
from spin_sdk.http import Request, send


def handle_get_weather(args: dict) -> dict:
    """Get current weather from OpenWeatherMap API."""
    location = args.get("location")
    if not location:
        return {
            "success": False,
            "result": "Missing required: location",
            "error_type": "InvalidArguments",
        }

    api_key = variables.get("api_key")
    encoded_location = quote(location)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={encoded_location}&appid={api_key}&units=metric"
    response = send(Request("GET", url, {}, None))

    if response.status != 200:
        return {
            "success": False,
            "result": f"API error: {response.status}",
            "error_type": "APIError",
        }

    return {"success": True, "result": response.body.decode("utf-8")}


def handle_get_forecast(args: dict) -> dict:
    """Get weather forecast from OpenWeatherMap API."""
    location = args.get("location")
    days = args.get("days", 3)

    if not location:
        return {
            "success": False,
            "result": "Missing required: location",
            "error_type": "InvalidArguments",
        }

    api_key = variables.get("api_key")
    encoded_location = quote(location)
    # cnt parameter limits the number of 3-hour forecast periods (8 per day)
    cnt = min(days * 8, 40)  # Max 40 periods (5 days)
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={encoded_location}&appid={api_key}&units=metric&cnt={cnt}"
    response = send(Request("GET", url, {}, None))

    if response.status != 200:
        return {
            "success": False,
            "result": f"API error: {response.status}",
            "error_type": "APIError",
        }

    return {"success": True, "result": response.body.decode("utf-8")}


TOOL_HANDLERS = {
    "get_weather": handle_get_weather,
    "get_forecast": handle_get_forecast,
}


class IncomingHandler(http.IncomingHandler):
    def handle_request(self, request: http.Request) -> http.Response:
        """Handle incoming HTTP requests."""
        path = request.uri
        if path.startswith("/weather"):
            path = path[8:]
        if not path:
            path = "/"
        method = request.method

        if path == "/health" and method == "GET":
            return http.Response(
                200,
                {"content-type": "application/json"},
                bytes(json.dumps({"status": "healthy", "component": "weather"}), "utf-8"),
            )

        if path == "/execute" and method == "POST":
            try:
                body = json.loads(request.body)
                tool = body.get("tool", "")
                args = body.get("args", {})

                if tool in TOOL_HANDLERS:
                    result = TOOL_HANDLERS[tool](args)
                else:
                    result = {
                        "success": False,
                        "result": f"Unknown tool: {tool}",
                        "error_type": "UnknownTool",
                    }

                return http.Response(
                    200,
                    {"content-type": "application/json"},
                    bytes(json.dumps(result), "utf-8"),
                )

            except json.JSONDecodeError as e:
                return http.Response(
                    400,
                    {"content-type": "application/json"},
                    bytes(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}), "utf-8"),
                )

        return http.Response(
            404,
            {"content-type": "application/json"},
            bytes(json.dumps({"error": "Not found"}), "utf-8"),
        )
```

### 3. Rebuild and run

After making these changes, rebuild the WASM component and run with your API key:

```bash
spin build
SPIN_VARIABLE_OPENWEATHERMAP_API_KEY=your-api-key spin up
```

Spin variables use the `SPIN_VARIABLE_` prefix with the variable name in uppercase.

## Next Steps

- Package as Docker image - see [Custom Tools](../custom.md#packaging-with-docker)
