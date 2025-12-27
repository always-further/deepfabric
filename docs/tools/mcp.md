# MCP Integration

DeepFabric supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for tool definitions. Load tools from MCP servers or use MCP-format schemas directly.

## What is MCP?

MCP is a standard for defining tools that AI models can use. It specifies:

- Tool names and descriptions
- Input schemas (JSON Schema format)
- Annotations for tool behavior (read-only, destructive, etc.)

## Loading from MCP Server

Pull tools directly from an MCP server:

```bash title="Pull tools"
curl -X POST http://localhost:3000/mock/pull \
  -H "Content-Type: application/json" \
  -d '{"url": "http://your-mcp-server:8000"}'
```

!!! info "JSON-RPC"
    The Mock component sends a `tools/list` JSON-RPC request and loads the returned tools.

## MCP Tool Format

```json title="MCP tool schema"
{
  "name": "search_code",
  "description": "Search code in a repository",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repository": {
        "type": "string",
        "description": "Repository name"
      },
      "query": {
        "type": "string",
        "description": "Search query"
      }
    },
    "required": ["repository", "query"]
  },
  "annotations": {
    "readOnlyHint": true
  }
}
```

### Annotations

MCP annotations describe tool behavior:

| Annotation | Description |
|------------|-------------|
| `readOnlyHint` | Tool doesn't modify state |
| `destructiveHint` | Tool may be destructive |
| `idempotentHint` | Safe to retry |
| `openWorldHint` | Interacts with external systems |

## DeepFabric Tool Format

DeepFabric also uses its own format internally:

```yaml title="DeepFabric format"
- name: search_code
  description: Search code in a repository
  parameters:
    - name: repository
      type: str
      description: Repository name
      required: true
    - name: query
      type: str
      description: Search query
      required: true
  returns: "List of matching files and line numbers"
```

!!! tip "Automatic Conversion"
    DeepFabric automatically converts between formats:

    - MCP schemas are converted for internal use
    - Output datasets use OpenAI tool format

## Configuration

### Using tools_endpoint

Load tools from an HTTP endpoint returning MCP format:

```yaml title="config.yaml"
generation:
  tools:
    spin_endpoint: "http://localhost:3000"
    tools_endpoint: "http://localhost:3000/mock/list-tools"
    tool_execute_path: "/mock/execute"
```

### Custom Tools in Config

Define tools directly in YAML:

```yaml title="config.yaml"
generation:
  tools:
    custom:
      - name: send_email
        description: Send an email
        parameters:
          - name: to
            type: str
            required: true
          - name: subject
            type: str
            required: true
          - name: body
            type: str
            required: true
        returns: "Confirmation message"
```

## Example Workflow

```mermaid
sequenceDiagram
    participant MCP as MCP Server
    participant Mock as Mock Component
    participant DF as DeepFabric
    participant LLM

    DF->>Mock: POST /mock/pull
    Mock->>MCP: tools/list (JSON-RPC)
    MCP-->>Mock: Tool definitions
    Mock-->>DF: OK
    DF->>LLM: Generate with tools
    LLM-->>DF: Tool call
    DF->>Mock: Execute tool
    Mock-->>DF: Mock response
```

1. **Start MCP server** with your tool definitions
2. **Load into Mock component**:
   ```bash
   curl -X POST http://localhost:3000/mock/pull \
     -d '{"url": "http://localhost:8000"}'
   ```
3. **Configure DeepFabric**:
   ```yaml
   tools:
     spin_endpoint: "http://localhost:3000"
     tools_endpoint: "http://localhost:3000/mock/list-tools"
   ```
4. **Generate dataset** with your MCP tools

## Output Format

Generated datasets include OpenAI-format tool definitions:

```json title="Output tool format"
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_code",
        "description": "Search code in a repository",
        "parameters": {
          "type": "object",
          "properties": {
            "repository": {"type": "string"},
            "query": {"type": "string"}
          },
          "required": ["repository", "query"]
        }
      }
    }
  ]
}
```

!!! success "Framework Compatibility"
    This format is compatible with OpenAI, TRL, and most training frameworks.
