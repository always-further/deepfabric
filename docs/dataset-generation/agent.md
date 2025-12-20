# Agent Datasets

Agent datasets train models to use tools. DeepFabric supports two modes: single-turn (one-shot tool calls) and multi-turn (extended conversations with multiple tool interactions).

**Prerequisites**: Agent datasets require the Spin tool service. See [Tools](../tools/index.md) for setup instructions.

## When to Use

- Training tool-calling capabilities
- Building agents that interact with APIs or systems
- ReAct-style reasoning with action-observation loops

## Single-Turn Agent

Single-turn mode generates complete tool workflows in one assistant response.

```yaml
generation:
  system_prompt: "Generate tool usage examples with reasoning."
  instructions: "Create realistic scenarios requiring tools."

  conversation:
    type: chain_of_thought
    reasoning_style: agent
    agent_mode: single_turn

  tools:
    spin_endpoint: "http://localhost:3000"
    components:
      builtin:
        - read_file
        - write_file
        - list_files
    max_per_query: 3
    max_agent_steps: 5

output:
  system_prompt: |
    You are an AI with access to tools. Analyze tasks, execute tools, and interpret results.
  num_samples: 10
  save_as: "agent-dataset.jsonl"
```

### Single-Turn Output

```json
{
  "messages": [
    {"role": "system", "content": "You are an AI with access to tools."},
    {"role": "user", "content": "Find all while loops in main.py"},
    {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call_0", "type": "function", "function": {"name": "search_file", "arguments": "{\"file_path\": \"main.py\", \"keyword\": \"while\"}"}}
    ]},
    {"role": "tool", "content": "[15, 42, 101]", "tool_call_id": "call_0"},
    {"role": "assistant", "content": "Found 3 while loops on lines 15, 42, and 101."}
  ],
  "reasoning": {
    "style": "agent",
    "content": [
      {"step_number": 1, "thought": "Need to search for 'while' keyword in main.py", "action": "search_file(file_path='main.py', keyword='while')"}
    ]
  },
  "tools": [...]
}
```

## Multi-Turn Agent

Multi-turn mode creates extended conversations with multiple tool interactions, following a ReAct pattern.

```yaml
generation:
  conversation:
    type: chain_of_thought
    reasoning_style: agent
    agent_mode: multi_turn
    min_turns: 2
    max_turns: 4
    min_tool_calls: 2

  tools:
    spin_endpoint: "http://localhost:3000"
    components:
      builtin:
        - read_file
        - write_file
        - list_files
    max_agent_steps: 5
```

Multi-turn datasets include:
- Multiple tool call rounds
- Observation-based decisions
- Extended reasoning traces

## Tool Options

```yaml
tools:
  spin_endpoint: "http://localhost:3000"  # Spin service URL
  components:                              # Component-based tool routing
    builtin:                               # Built-in VFS tools -> /vfs/execute
      - read_file
      - write_file
    mock:                                  # Mock tools -> /mock/execute
      - get_weather
  tools_endpoint: "http://localhost:3000/mock/list-tools"  # For non-builtin tools
  max_per_query: 3                         # Max tools per sample
  max_agent_steps: 5                       # Max reasoning iterations
  scenario_seed:                           # Pre-populate files
    files:
      "config.json": '{"debug": true}'
```

## CLI Usage

```bash
# Single-turn agent (requires Spin running)
deepfabric generate config.yaml \
  --conversation-type chain_of_thought \
  --reasoning-style agent \
  --agent-mode single_turn

# Multi-turn agent
deepfabric generate config.yaml \
  --agent-mode multi_turn \
  --min-turns 2 \
  --max-turns 4
```

## Next Steps

See [Tools](../tools/index.md) for:
- Installing and running Spin
- Available VFS tools (read_file, write_file, etc.)
- Creating custom tools
- Mock tool execution for testing
