# Tools

DeepFabric uses [Spin](https://spinframework.dev), a WebAssembly framework, to execute tools during dataset generation. Tools run in isolated sandboxes, producing authentic training data based on real execution results.

## Why Real Execution Matters

Traditional synthetic data generators simulate tool outputs, which creates unrealistic training data. With Spin, tools execute against real state:

```
# Simulated (unrealistic)
Agent: read_file("config.json")
Result: {"setting": "value"}  # LLM hallucinated this

# Real execution (accurate)
Agent: read_file("config.json")
Result: FileNotFound  # Actual state
Agent: write_file("config.json", content)
Result: Written 42 bytes  # Real operation
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   DeepFabric    │────▶│   Spin Service  │
│   (Python)      │     │   (WASM Host)   │
└─────────────────┘     └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        ┌──────────┐      ┌──────────┐      ┌──────────┐
        │   VFS    │      │   Mock   │      │  GitHub  │
        │Component │      │Component │      │Component │
        └──────────┘      └──────────┘      └──────────┘
```

Components are WebAssembly modules that handle specific tool categories:

| Component | Purpose | Tools |
|-----------|---------|-------|
| **VFS** | Virtual filesystem | read_file, write_file, list_files, delete_file |
| **Mock** | Dynamic mock execution | Any tool loaded via MCP |
| **GitHub** | GitHub API (experimental) | Issues, PRs, commits |

## Quick Start

### Install Spin

#### Docker

We provide a prepacked Docker image:

```bash
docker run -d -p 3000:3000 ghcr.io/always-further/deepfabric/tools-sdk:latest
```

This will now be accessible at `http://localhost:3000`.

#### Local Installation

Follow the [Spin installation guide](https://spinframework.dev/v3/install) for your OS.

and from the `tools-sdk/` directory:

```bash
spin build
spin up
```

The service runs at `http://localhost:3000`.

Configure DeepFabric to use it:

```yaml
generation:
  tools:
    spin_endpoint: "http://localhost:3000"
    components:
      builtin:  # VFS tools -> /vfs/execute
        - read_file
        - write_file
        - list_files
```

Each component routes to its own endpoint (`/{component}/execute`).

## Session Isolation

Each dataset sample gets an isolated session. Files created during one sample don't affect others:

```python
# Session A: Creates config.json
# Session B: config.json doesn't exist

# After sample generation, session is cleaned up
```

## Next Steps

- [Spin Setup](spin.md) - Installation and running
- [VFS Component](vfs.md) - Virtual filesystem tools
- [Mock Component](mock.md) - Dynamic tool mocking
- [MCP Integration](mcp.md) - Loading tools from MCP servers
- [Custom Tools](custom.md) - Creating your own components
