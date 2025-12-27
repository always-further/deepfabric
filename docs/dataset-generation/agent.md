# Agent Datasets

Agent datasets train models to use tools. DeepFabric supports two modes: single-turn (one-shot tool calls) and multi-turn (extended conversations with multiple tool interactions).

!!! tip "Start Simple"
    If you're unsure which to use, start with single-turn. Multi-turn is more brittle and requires careful configuration.

!!! warning "Prerequisites"
    Agent datasets require the Spin tool service. See [Tools](../tools/index.md) for setup, or just use Docker:

    ```bash
    docker run -d -p 3000:3000 ghcr.io/always-further/deepfabric/tools-sdk:latest
    ```

## When to Use

- Training tool-calling capabilities
- Building agents that interact with APIs or systems
- ReAct-style reasoning with action-observation loops

## Single-Turn Agent

Single-turn mode generates complete tool workflows in one assistant response.

```yaml title="config.yaml"
topics:
  prompt: "Python Programming"
  graph: tree
  depth: 2
  degree: 3

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
  num_samples: 4
  batch_size: 2
  save_as: "agent-dataset.jsonl"
```

### Single-Turn Output

??? example "Sample Output"

    ```json title="agent-dataset.jsonl"
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

```yaml title="config.yaml"
topics:
  prompt: "Platform engineering tasks requiring tool usage"
  mode: graph
  depth: 2
  degree: 3

generation:
  system_prompt: "Generate tool usage examples with reasoning."
  instructions: "Create realistic scenarios requiring tools."

  conversation:
    type: chain_of_thought
    reasoning_style: agent
    agent_mode: multi_turn

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
  num_samples: 4
  batch_size: 2
  save_as: "agent-platform-dataset.jsonl"
```

Multi-turn datasets include:

- Multiple tool call rounds
- Observation-based decisions
- Extended reasoning traces
- Planning over several steps

??? example "Multi-Turn Sample Output (Long)"

    ```json title="agent-platform-dataset.jsonl"
    {
        "messages": [
            {
                "content": "You are an AI with access to tools...",
                "role": "system"
            },
            {
                "content": "I need to update the CI/CD pipeline configuration...",
                "role": "user"
            },
            {
                "content": "",
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "arguments": "{}",
                            "name": "list_files"
                        },
                        "id": "5AgSQoude",
                        "type": "function"
                    }
                ]
            },
            {
                "content": "[]",
                "role": "tool",
                "tool_call_id": "5AgSQoude"
            },
            {
                "content": "It seems there are no configuration files available...",
                "role": "assistant"
            }
        ],
        "reasoning": {
            "content": [
                {
                    "action": "list_files({})",
                    "step_number": 1,
                    "thought": "To provide a list of configuration files..."
                }
            ],
            "style": "agent"
        }
    }
    ```

!!! note "Errors Are Expected"
    You will sometimes see the model attempt to call files or APIs that do not exist in the tool context. This is expected behavior and even wanted behavior, as the model is then forced to deal with its mistakes, just like in real-world scenarios. A `FileNotFound` error is a valid tool response that the model must handle appropriately and do so in a ReAct manner.

## Tool Options

```yaml title="Tool configuration"
tools:
  spin_endpoint: "http://localhost:3000"  # Spin service URL
  components:                              # Component-based tool routing
    builtin:                               # Built-in VFS tools -> /vfs/execute
      - read_file
      - write_file
      - list_files
  max_per_query: 3                         # Max tools per sample
  max_agent_steps: 5                       # Max reasoning iterations
  scenario_seed:                           # Pre-populate files
    files:
      "config.json": '{"debug": true}'
```

### Mock Files

With `scenario_seed`, you can pre-create files in the agent's virtual filesystem:

```yaml title="Pre-populated files"
scenario_seed:
  files:
    "main.py": |
      for i in range(10):
          print(i)
    "Dockerfile": |
      FROM python:3.9-slim
      COPY . /app
      RUN pip install -r /app/requirements.txt
```

!!! tip "Realistic Scenarios"
    Pre-populating files creates more realistic training scenarios where the agent can read and modify existing content.

## CLI Usage

=== "Single-Turn"

    ```bash title="Single-turn agent"
    deepfabric generate config.yaml \
      --conversation-type chain_of_thought \
      --reasoning-style agent \
      --agent-mode single_turn
    ```

=== "Multi-Turn"

    ```bash title="Multi-turn agent"
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
