# Mock Component

The Mock component executes tools with configurable responses. Load any tool schema and define mock responses, fixtures, or let the system generate default outputs.

## Use Cases

- Test agent workflows without real API access
- Create training data for external services (GitHub, Slack, databases)
- Define deterministic responses for reproducible datasets

## Loading Tool Schemas

### From JSON/YAML

```bash
curl -X POST http://localhost:3000/mock/load-schema \
  -H "Content-Type: application/json" \
  -d '[
    {
      "name": "get_weather",
      "description": "Get weather for a location",
      "inputSchema": {
        "type": "object",
        "properties": {
          "location": {"type": "string"}
        },
        "required": ["location"]
      },
      "mock_response": {"temperature": 72, "condition": "sunny"}
    }
  ]'
```

### From MCP Server

Pull tools from a Model Context Protocol server:

```bash
curl -X POST http://localhost:3000/mock/pull \
  -H "Content-Type: application/json" \
  -d '{"url": "http://localhost:8000/mcp"}'
```

## Configuration

```yaml
generation:
  tools:
    spin_endpoint: "http://localhost:3000"
    tools_endpoint: "http://localhost:3000/mock/list-tools"
    components:
      mock:  # Routes to /mock/execute
        - get_weather
        - search_code
```

The `mock` component routes to `/mock/execute`. Tool definitions are loaded from `tools_endpoint` and filtered by the tool names listed under `components.mock`.

## Mock Responses

### Default Behavior

Without a `mock_response`, the component echoes the tool call:

```json
{
  "tool": "get_weather",
  "arguments": {"location": "Seattle"},
  "mock": true
}
```

### Custom Response

Define a static response in the schema:

```json
{
  "name": "get_weather",
  "mock_response": {
    "temperature": 72,
    "condition": "sunny"
  }
}
```

### Template Interpolation

Use `{{argument_name}}` to include call arguments in responses:

```json
{
  "name": "get_weather",
  "mock_response": {
    "location": "{{location}}",
    "temperature": 72
  }
}
```

Calling `get_weather(location="Seattle")` returns:
```json
{"location": "Seattle", "temperature": 72}
```

## Fixtures

Fixtures return specific responses based on argument matching:

```bash
curl -X POST http://localhost:3000/mock/add-fixture \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_weather",
    "match": {"location": "Seattle"},
    "response": {"temperature": 55, "condition": "rainy"}
  }'
```

Now `get_weather(location="Seattle")` returns the rainy fixture, while other locations use the default mock response.

### Multiple Fixtures

Add fixtures for different argument combinations:

```bash
# Seattle fixture
curl -X POST .../add-fixture -d '{
  "name": "search_code",
  "match": {"repo": "main", "query": "TODO"},
  "response": {"matches": ["file1.py:10", "file2.py:25"]}
}'

# Different repo
curl -X POST .../add-fixture -d '{
  "name": "search_code",
  "match": {"repo": "testing"},
  "response": {"matches": []}
}'
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mock/load-schema` | POST | Load tool definitions |
| `/mock/pull` | POST | Pull from MCP server |
| `/mock/execute` | POST | Execute a tool |
| `/mock/update-response` | POST | Update tool's mock response |
| `/mock/add-fixture` | POST | Add argument-specific fixture |
| `/mock/list-tools` | GET | List loaded tools |
| `/mock/clear` | POST | Clear all tools |

### Execute Request

```json
{
  "name": "tool_name",
  "arguments": {"arg1": "value1"}
}
```

Note: Mock execute uses `name` and `arguments`, unlike VFS which uses `tool` and `args`.

## Example: GitHub Tools

```yaml
# Load GitHub tool schemas via mock component
generation:
  tools:
    spin_endpoint: "http://localhost:3000"
    tools_endpoint: "http://localhost:3000/mock/list-tools"
    components:
      mock:
        - create_issue
        - list_issues
        - get_file_contents
```

Then load schemas and fixtures:

```bash
# Load GitHub tools
curl -X POST http://localhost:3000/mock/load-schema \
  -d '[{"name": "create_issue", "description": "Create GitHub issue", ...}]'

# Add fixture for specific repo
curl -X POST http://localhost:3000/mock/add-fixture \
  -d '{
    "name": "create_issue",
    "match": {"repo": "myorg/myrepo"},
    "response": {"issue_number": 42, "url": "https://github.com/myorg/myrepo/issues/42"}
  }'
```
