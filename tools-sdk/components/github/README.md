# GitHub Spin Component Example

A Python-based Spin component that provides real GitHub API access for DeepFabric dataset generation.

## Overview

This component wraps the GitHub REST API and exposes it as tool endpoints for DeepFabric's agent-based dataset generation. It includes safety features like repository allowlisting and write protection.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- [Spin CLI](https://developer.fermyon.com/spin/v2/install) v2.0+

## Project Structure

```
components/github/
├── app.py              # Main component code
├── pyproject.toml      # Python dependencies (for uv/componentize-py)
├── requirements.txt    # Legacy requirements file
└── README.md           # This file
```

## Building

From the `tools-sdk` directory:

```bash
# Build all components (including this one)
spin build

# Or build just this component
cd components/github
uv run componentize-py -w spin-http componentize app -o github.wasm
```

The build produces `github.wasm` (~37MB) which Spin loads at runtime.

## Configuration

The component uses Spin variables for configuration. Set these as environment variables with the `SPIN_VARIABLE_` prefix:

| Variable | Description | Default |
|----------|-------------|---------|
| `SPIN_VARIABLE_GITHUB_TOKEN` | GitHub personal access token | `""` (required) |
| `SPIN_VARIABLE_ALLOWED_REPOS` | Comma-separated repo allowlist | `""` (all allowed) |
| `SPIN_VARIABLE_ALLOW_WRITES` | Enable write operations | `"false"` |

### Example

```bash
SPIN_VARIABLE_GITHUB_TOKEN=ghp_xxxx \
SPIN_VARIABLE_ALLOWED_REPOS="myorg/repo1,myorg/repo2" \
spin up
```

## API Endpoints

All endpoints are prefixed with `/github/`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/github/health` | GET | Health check with config status |
| `/github/components` | GET | List available tools |
| `/github/execute` | POST | Execute a tool |

### Execute Request Format

```json
{
  "session_id": "uuid-here",
  "tool": "gh_list_issues",
  "args": {
    "owner": "myorg",
    "repo": "myrepo",
    "state": "open"
  }
}
```

### Response Format

```json
{
  "success": true,
  "result": "[{\"number\": 1, \"title\": \"Issue title\", ...}]"
}
```

Or on error:

```json
{
  "success": false,
  "result": "Repository 'other/repo' is not in the allowed list. Allowed repositories: myorg/repo1, myorg/repo2",
  "error_type": "RepoNotAllowed"
}
```

## Available Tools

### Read Operations

| Tool | Description | Required Args |
|------|-------------|---------------|
| `gh_get_file_contents` | Get file/directory contents | `owner`, `repo` |
| `gh_search_code` | Search code across GitHub | `query` |
| `gh_search_repositories` | Search for repositories | `query` |
| `gh_list_issues` | List repository issues | `owner`, `repo` |
| `gh_get_issue` | Get issue details | `owner`, `repo`, `issue_number` |
| `gh_list_pull_requests` | List repository PRs | `owner`, `repo` |
| `gh_get_pull_request` | Get PR details | `owner`, `repo`, `pullNumber` |
| `gh_list_commits` | List repository commits | `owner`, `repo` |
| `gh_get_commit` | Get commit details | `owner`, `repo`, `sha` |
| `gh_list_branches` | List repository branches | `owner`, `repo` |

### Write Operations (require `ALLOW_WRITES=true`)

| Tool | Description | Required Args |
|------|-------------|---------------|
| `gh_add_issue_comment` | Add comment to issue | `owner`, `repo`, `issue_number`, `body` |

## Safety Features

### Repository Allowlisting

When `ALLOWED_REPOS` is set, only those repositories can be accessed. Requests to other repos return a helpful error:

```
Repository 'owner/repo' is not in the allowed list. Allowed repositories: repo1, repo2
```

This prevents the LLM from accidentally querying arbitrary repositories during dataset generation.

### Write Protection

Write operations are disabled by default. Set `ALLOW_WRITES=true` to enable them. This prevents accidental modifications to repositories.

## Creating Your Own Component

Use this component as a template for creating new Spin components for DeepFabric.

### 1. Create the directory structure

```bash
mkdir -p components/mycomponent
cd components/mycomponent
```

### 2. Create `pyproject.toml`

```toml
[project]
name = "mycomponent-spin-component"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "tools-sdk==3.4.1",
    "componentize-py==0.17.2",
]

[tool.hatch.build.targets.wheel]
packages = ["."]
```

### 3. Create `app.py`

```python
import json
from spin_sdk import http, variables
from spin_sdk.http import send


class IncomingHandler(http.IncomingHandler):
    def handle_request(self, request: http.Request) -> http.Response:
        # Strip component prefix from path
        path = request.uri
        if path.startswith("/mycomponent"):
            path = path[len("/mycomponent"):]
        if not path:
            path = "/"

        # Health check
        if path == "/health" and request.method == "GET":
            return http.Response(
                200,
                {"content-type": "application/json"},
                b'{"status": "healthy", "component": "mycomponent"}',
            )

        # Tool execution
        if path == "/execute" and request.method == "POST":
            try:
                body = json.loads(request.body)
                tool = body.get("tool", "")
                args = body.get("args", {})

                # Dispatch to tool handlers
                if tool == "my_tool":
                    result = handle_my_tool(args)
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
            except Exception as e:
                return http.Response(
                    500,
                    {"content-type": "application/json"},
                    bytes(json.dumps({"success": False, "error": str(e)}), "utf-8"),
                )

        return http.Response(404, {}, b"Not found")


def handle_my_tool(args: dict) -> dict:
    """Handle my_tool execution."""
    # Your tool logic here
    return {"success": True, "result": "Tool executed successfully"}
```

### 4. Add to `spin.toml`

```toml
[[trigger.http]]
route = "/mycomponent/..."
component = "mycomponent"

[component.mycomponent]
source = "components/mycomponent/mycomponent.wasm"
allowed_outbound_hosts = ["https://api.example.com"]  # If making HTTP calls

[component.mycomponent.build]
command = "uv run componentize-py -w spin-http componentize app -o mycomponent.wasm"
workdir = "components/mycomponent"
watch = ["components/mycomponent/**/*.py"]
```

### 5. Register tools in DeepFabric

Add tool definitions to `deepfabric/tools/defaults.py`:

```python
MY_TOOL = ToolDefinition(
    name="my_tool",
    description="Description of what my tool does",
    parameters=[
        ToolParameter(
            name="param1",
            type="str",
            description="Parameter description",
            required=True,
        ),
    ],
    returns="Description of return value",
    category="mycomponent",
    component="mycomponent",  # Must match spin.toml component name
)
```

### 6. Build and test

```bash
spin build
spin up

# Test health endpoint
curl http://localhost:3000/mycomponent/health

# Test tool execution
curl -X POST http://localhost:3000/mycomponent/execute \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "tool": "my_tool", "args": {"param1": "value"}}'
```

## Making Outbound HTTP Requests

Spin components run in WebAssembly and cannot use Python's `urllib`. Use Spin's native HTTP client instead:

```python
from spin_sdk import http
from spin_sdk.http import send

def make_api_request(url: str, headers: dict) -> dict:
    """Make an outbound HTTP request."""
    # Headers must be a dict of string -> string
    response = send(http.Request(
        "GET",
        url,
        headers,
        b"",  # Request body (empty for GET)
    ))

    if 200 <= response.status < 300:
        return json.loads(response.body.decode("utf-8"))
    else:
        raise ValueError(f"HTTP {response.status}: {response.body.decode('utf-8')}")
```

Remember to add allowed hosts to `spin.toml`:

```toml
[component.mycomponent]
allowed_outbound_hosts = ["https://api.example.com"]
```

## Debugging

### View component logs

```bash
spin up
# Logs are written to .spin/logs/
```

### Common issues

1. **`unknown url type: https`** - You're using `urllib` instead of Spin's HTTP client
2. **`unknown variable`** - Variable not defined in `[variables]` section of `spin.toml`
3. **`no WIT files found`** - Wrong `source` path in `spin.toml` (should point to `.wasm`, not `.py`)
4. **Component not responding** - Check that route prefix matches what you're stripping in `handle_request`

## Testing with DeepFabric

```yaml
# config.yaml
generation:
  tools:
    spin_endpoint: "http://localhost:3000"
    available:
      - my_tool
```

```bash
# Terminal 1: Start Spin
spin up

# Terminal 2: Run DeepFabric
deepfabric start config.yaml
```
