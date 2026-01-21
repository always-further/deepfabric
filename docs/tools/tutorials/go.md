# Go Custom Tools Tutorial

Build custom tools as a Go Spin component.

For full Go component documentation, see [Spin Go Components](https://spinframework.dev/v2/go-components).

## Prerequisites

- [Spin CLI](../spin.md) installed
- Go 1.21+
- TinyGo for WASM compilation:

```bash
# macOS
brew install tinygo

# Linux
wget https://github.com/tinygo-org/tinygo/releases/download/v0.31.0/tinygo_0.31.0_amd64.deb
sudo dpkg -i tinygo_0.31.0_amd64.deb
```

## Step 1: Create Your Project

Create a new directory for your custom tools:

```bash
mkdir weather-tools
cd weather-tools
go mod init weather-tools
```

Install the Spin SDK:

```bash
go get github.com/fermyon/spin/sdk/go/v2
```

Create `main.go`:

```go title="main.go"
package main

import (
	"encoding/json"
	"net/http"
	"strings"

	spinhttp "github.com/fermyon/spin/sdk/go/v2/http"
)

type ExecuteRequest struct {
	SessionID string                 `json:"session_id"`
	Tool      string                 `json:"tool"`
	Args      map[string]interface{} `json:"args"`
}

type ExecuteResponse struct {
	Success   bool    `json:"success"`
	Result    string  `json:"result"`
	ErrorType *string `json:"error_type,omitempty"`
}

func init() {
	spinhttp.Handle(func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/weather")
		if path == "" {
			path = "/"
		}

		w.Header().Set("Content-Type", "application/json")

		switch {
		case path == "/health" && r.Method == "GET":
			handleHealth(w)
		case path == "/execute" && r.Method == "POST":
			handleExecute(w, r)
		default:
			w.WriteHeader(http.StatusNotFound)
			json.NewEncoder(w).Encode(map[string]string{"error": "not found"})
		}
	})
}

func handleHealth(w http.ResponseWriter) {
	json.NewEncoder(w).Encode(map[string]string{
		"status":    "healthy",
		"component": "weather",
	})
}

func handleExecute(w http.ResponseWriter, r *http.Request) {
	var req ExecuteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	var resp ExecuteResponse
	switch req.Tool {
	case "get_weather":
		resp = handleGetWeather(req.Args)
	case "get_forecast":
		resp = handleGetForecast(req.Args)
	default:
		errType := "UnknownTool"
		resp = ExecuteResponse{
			Success:   false,
			Result:    "Unknown tool: " + req.Tool,
			ErrorType: &errType,
		}
	}

	json.NewEncoder(w).Encode(resp)
}

func handleGetWeather(args map[string]interface{}) ExecuteResponse {
	location, _ := args["location"].(string)

	if location == "" {
		errType := "InvalidArguments"
		return ExecuteResponse{
			Success:   false,
			Result:    "Missing required: location",
			ErrorType: &errType,
		}
	}

	// Mock weather data - replace with real API call if needed
	result := map[string]interface{}{
		"location":    location,
		"temperature": 72,
		"condition":   "sunny",
		"humidity":    45,
	}
	resultJSON, _ := json.Marshal(result)

	return ExecuteResponse{
		Success: true,
		Result:  string(resultJSON),
	}
}

func handleGetForecast(args map[string]interface{}) ExecuteResponse {
	location, _ := args["location"].(string)
	days := 3
	if d, ok := args["days"].(float64); ok {
		days = int(d)
	}

	if location == "" {
		errType := "InvalidArguments"
		return ExecuteResponse{
			Success:   false,
			Result:    "Missing required: location",
			ErrorType: &errType,
		}
	}

	// Mock forecast data - replace with real API call if needed
	allDays := []map[string]interface{}{
		{"day": "Monday", "high": 75, "low": 60, "condition": "sunny"},
		{"day": "Tuesday", "high": 72, "low": 58, "condition": "partly cloudy"},
		{"day": "Wednesday", "high": 68, "low": 55, "condition": "cloudy"},
	}

	if days > len(allDays) {
		days = len(allDays)
	}

	result := map[string]interface{}{
		"location": location,
		"days":     allDays[:days],
	}
	resultJSON, _ := json.Marshal(result)

	return ExecuteResponse{
		Success: true,
		Result:  string(resultJSON),
	}
}

func main() {}
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
source = "main.wasm"
allowed_outbound_hosts = []

[component.weather.build]
command = "tinygo build -target=wasi -gc=leaking -o main.wasm main.go"
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
source = "main.wasm"
allowed_outbound_hosts = ["https://api.openweathermap.org"]

[component.weather.variables]
api_key = "{{ openweathermap_api_key }}"

[component.weather.build]
command = "tinygo build -target=wasi -gc=leaking -o main.wasm main.go"
```

### 2. Update main.go

Replace your `main.go` with this version that calls the OpenWeatherMap API:

```go title="main.go"
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	spinhttp "github.com/fermyon/spin/sdk/go/v2/http"
	"github.com/fermyon/spin/sdk/go/v2/variables"
)

type ExecuteRequest struct {
	SessionID string                 `json:"session_id"`
	Tool      string                 `json:"tool"`
	Args      map[string]interface{} `json:"args"`
}

type ExecuteResponse struct {
	Success   bool    `json:"success"`
	Result    string  `json:"result"`
	ErrorType *string `json:"error_type,omitempty"`
}

func init() {
	spinhttp.Handle(func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/weather")
		if path == "" {
			path = "/"
		}

		w.Header().Set("Content-Type", "application/json")

		switch {
		case path == "/health" && r.Method == "GET":
			handleHealth(w)
		case path == "/execute" && r.Method == "POST":
			handleExecute(w, r)
		default:
			w.WriteHeader(http.StatusNotFound)
			json.NewEncoder(w).Encode(map[string]string{"error": "not found"})
		}
	})
}

func handleHealth(w http.ResponseWriter) {
	json.NewEncoder(w).Encode(map[string]string{
		"status":    "healthy",
		"component": "weather",
	})
}

func handleExecute(w http.ResponseWriter, r *http.Request) {
	var req ExecuteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	var resp ExecuteResponse
	switch req.Tool {
	case "get_weather":
		resp = handleGetWeather(req.Args)
	case "get_forecast":
		resp = handleGetForecast(req.Args)
	default:
		errType := "UnknownTool"
		resp = ExecuteResponse{
			Success:   false,
			Result:    "Unknown tool: " + req.Tool,
			ErrorType: &errType,
		}
	}

	json.NewEncoder(w).Encode(resp)
}

func handleGetWeather(args map[string]interface{}) ExecuteResponse {
	location, _ := args["location"].(string)

	if location == "" {
		errType := "InvalidArguments"
		return ExecuteResponse{
			Success:   false,
			Result:    "Missing required: location",
			ErrorType: &errType,
		}
	}

	apiKey, err := variables.Get("api_key")
	if err != nil {
		errType := "ConfigError"
		return ExecuteResponse{
			Success:   false,
			Result:    "Failed to get API key",
			ErrorType: &errType,
		}
	}

	encodedLocation := url.QueryEscape(location)
	apiURL := fmt.Sprintf("https://api.openweathermap.org/data/2.5/weather?q=%s&appid=%s&units=metric", encodedLocation, apiKey)

	resp, err := spinhttp.Get(apiURL)
	if err != nil {
		errType := "APIError"
		return ExecuteResponse{
			Success:   false,
			Result:    fmt.Sprintf("API request failed: %v", err),
			ErrorType: &errType,
		}
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		errType := "APIError"
		return ExecuteResponse{
			Success:   false,
			Result:    fmt.Sprintf("API error: %d", resp.StatusCode),
			ErrorType: &errType,
		}
	}

	return ExecuteResponse{
		Success: true,
		Result:  string(body),
	}
}

func handleGetForecast(args map[string]interface{}) ExecuteResponse {
	location, _ := args["location"].(string)
	days := 3
	if d, ok := args["days"].(float64); ok {
		days = int(d)
	}

	if location == "" {
		errType := "InvalidArguments"
		return ExecuteResponse{
			Success:   false,
			Result:    "Missing required: location",
			ErrorType: &errType,
		}
	}

	apiKey, err := variables.Get("api_key")
	if err != nil {
		errType := "ConfigError"
		return ExecuteResponse{
			Success:   false,
			Result:    "Failed to get API key",
			ErrorType: &errType,
		}
	}

	encodedLocation := url.QueryEscape(location)
	// cnt parameter limits the number of 3-hour forecast periods (8 per day)
	cnt := days * 8
	if cnt > 40 {
		cnt = 40 // Max 40 periods (5 days)
	}
	apiURL := fmt.Sprintf("https://api.openweathermap.org/data/2.5/forecast?q=%s&appid=%s&units=metric&cnt=%d", encodedLocation, apiKey, cnt)

	resp, err := spinhttp.Get(apiURL)
	if err != nil {
		errType := "APIError"
		return ExecuteResponse{
			Success:   false,
			Result:    fmt.Sprintf("API request failed: %v", err),
			ErrorType: &errType,
		}
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		errType := "APIError"
		return ExecuteResponse{
			Success:   false,
			Result:    fmt.Sprintf("API error: %d", resp.StatusCode),
			ErrorType: &errType,
		}
	}

	return ExecuteResponse{
		Success: true,
		Result:  string(body),
	}
}

func main() {}
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
