# Configuration Reference

DeepFabric uses YAML configuration with four main sections: `llm`, `topics`, `generation`, and `output`.

## Complete Example

```yaml title="config.yaml"
# Shared LLM defaults (optional)
llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.7

# Topic generation
topics:
  prompt: "Python programming fundamentals"
  mode: graph             # tree | graph
  prompt_style: anchored  # default | isolated | anchored (graph mode only)
  depth: 2
  degree: 3
  save_as: "topics.json"
  llm:                    # Override shared LLM
    model: "gpt-4o-mini"

# Sample generation
generation:
  system_prompt: |
    Generate clear, educational examples.
  instructions: "Create diverse, practical scenarios."

  # Agent mode is implicit when tools are configured
  conversation:
    type: cot
    reasoning_style: agent

  tools:
    spin_endpoint: "http://localhost:3000"
    components:
      builtin:
        - read_file
        - write_file

  max_retries: 3
  llm:
    temperature: 0.5

# Output configuration
output:
  system_prompt: |
    You are a helpful assistant with tool access.
  include_system_message: true
  num_samples: 4
  batch_size: 2
  save_as: "dataset.jsonl"

  # Optional: Checkpoint for resumable generation
  checkpoint:
    interval: 500       # Save every 500 samples
    retry_failed: false

# Optional: Upload to HuggingFace
huggingface:
  repository: "org/dataset-name"
  tags: ["python", "agents"]
```

!!! note "HuggingFace Upload"
    The `huggingface` section is optional and used to upload the dataset after generation. It requires a token exported as `HF_TOKEN` or pre-authentication via `huggingface-cli`.

## Section Reference

### llm (Optional)

Shared LLM defaults inherited by `topics` and `generation`.

| Field | Type | Description |
|-------|------|-------------|
| `provider` | string | LLM provider: openai, anthropic, gemini, ollama |
| `model` | string | Model name |
| `temperature` | float | Sampling temperature (0.0-2.0) |
| `base_url` | string | Custom API endpoint |

### topics

Controls topic tree/graph generation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | required | Root topic for generation |
| `mode` | string | "tree" | Generation mode: tree or graph |
| `depth` | int | 2 | Hierarchy depth (1-10) |
| `degree` | int | 3 | Subtopics per node (1-50) |
| `max_concurrent` | int | 4 | Max concurrent LLM calls (graph mode only, 1-20) |
| `prompt_style` | string | "default" | Graph expansion prompt style (graph mode only, see below) |
| `system_prompt` | string | "" | Custom instructions for topic LLM |
| `save_as` | string | - | Path to save topics JSONL |
| `llm` | object | - | Override shared LLM settings |

#### topics.prompt_style (Graph Mode Only)

Controls how subtopics are generated during graph expansion:

| Style | Cross-connections | Examples | Use Case |
|-------|-------------------|----------|----------|
| `default` | Yes | Generic | General-purpose topic graphs with cross-links |
| `isolated` | No | Generic | Independent subtopics without cross-connections |
| `anchored` | No | Domain-aware | Focused generation with domain-specific examples |

**`anchored`** is recommended for specialized domains (security, technical) where you want subtopics to stay tightly focused on the parent topic. It automatically detects the domain from your `system_prompt` and provides relevant examples to guide generation.

```yaml title="Example: Security-focused topic generation"
topics:
  prompt: "Credential access attack scenarios"
  mode: graph
  prompt_style: anchored   # Uses security-domain examples
  depth: 3
  degree: 8
  system_prompt: |
    Generate adversarial security test cases for AI assistant hardening.
```

### generation

Controls sample generation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | string | - | Instructions for generation LLM |
| `instructions` | string | - | Additional guidance |
| `conversation` | object | - | Conversation type settings |
| `tools` | object | - | Tool configuration |
| `max_retries` | int | 3 | Retries on API failures |
| `sample_retries` | int | 2 | Retries on validation failures |
| `max_tokens` | int | 2000 | Max tokens per generation |
| `llm` | object | - | Override shared LLM settings |

#### generation.conversation

| Field | Type | Options | Description |
|-------|------|---------|-------------|
| `type` | string | basic, cot | Conversation format |
| `reasoning_style` | string | freetext, agent | For cot only |

!!! note "Agent Mode"
    Agent mode is automatically enabled when tools are configured. No explicit `agent_mode` setting is required.

#### generation.tools

| Field | Type | Description |
|-------|------|-------------|
| `spin_endpoint` | string | Spin service URL |
| `tools_endpoint` | string | Endpoint to load tool definitions (for non-builtin components) |
| `components` | object | Map of component name to tool names (see below) |
| `custom` | list | Inline tool definitions |
| `max_per_query` | int | Max tools per sample |
| `max_agent_steps` | int | Max ReAct iterations |
| `scenario_seed` | object | Initial file state |

##### components

The `components` field maps component names to lists of tool names. Each component routes to `/{component}/execute`:

```yaml title="Component routing"
components:
  builtin:              # Routes to /vfs/execute (built-in tools)
    - read_file
    - write_file
  mock:                 # Routes to /mock/execute
    - get_weather
  github:               # Routes to /github/execute
    - list_issues
```

!!! info "Component Types"
    - `builtin`: Uses built-in VFS tools (read_file, write_file, list_files, delete_file)
    - Other components: Load tool definitions from `tools_endpoint`

### output

Controls final dataset.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | string | - | System message in training data |
| `include_system_message` | bool | true | Include system message |
| `num_samples` | int \| string | required | Total samples: integer, `"auto"`, or percentage like `"50%"` |
| `batch_size` | int | 1 | Parallel generation batch size |
| `save_as` | string | required | Output file path |
| `checkpoint` | object | - | Checkpoint configuration (see below) |

!!! tip "Auto and Percentage Samples"
    `num_samples` supports special values:

    - **Integer** (e.g., `100`): Generate exactly this many samples
    - **`"auto"`**: Generate one sample per topic path (100% coverage)
    - **Percentage** (e.g., `"50%"`, `"200%"`): Generate samples relative to topic path count

    When `num_samples` exceeds the number of topic paths, topics cycle for even coverage. An integer larger than the path count is equivalent to a percentage—for example, with 50 paths, `num_samples: 100` behaves the same as `num_samples: "200%"`.

#### output.checkpoint (Optional)

Configuration for checkpoint-based resume capability. Checkpoints allow pausing and resuming long-running dataset generation without losing progress.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `interval` | int | required | Save checkpoint every N samples |
| `path` | string | XDG data dir | Directory to store checkpoint files (auto-managed) |
| `retry_failed` | bool | false | When resuming, retry previously failed samples |

```yaml title="Checkpoint configuration"
output:
  save_as: "dataset.jsonl"
  num_samples: 5000
  batch_size: 5
  checkpoint:
    interval: 500     # Save every 500 samples
    retry_failed: false
```

Checkpointing creates three files in the checkpoint directory:

- `{name}.checkpoint.json` - Metadata (progress, IDs processed)
- `{name}.checkpoint.jsonl` - Samples saved so far
- `{name}.checkpoint.failures.jsonl` - Failed samples for debugging

!!! tip "Choosing Checkpoint Interval"
    The checkpoint `interval` specifies how many samples to generate between saves. Checkpoints save after each **step** completes (a step generates `batch_size` samples), so:

    - If `interval < batch_size`, checkpoints save after every step
    - If `interval > batch_size`, checkpoints save when cumulative samples reach the threshold

    For example, with `batch_size: 20` and `interval: 8`, checkpoints actually save every 20 samples (after each step) since 20 > 8. To checkpoint less frequently, set `interval` larger than `batch_size`—e.g., `interval: 100` with `batch_size: 20` saves every 5 steps (100 samples).

    Choose an interval that balances recovery granularity (smaller = less work lost) against I/O overhead (larger = fewer disk writes).

!!! tip "Memory Optimization"
    When checkpointing is enabled, samples are flushed to disk periodically, keeping memory usage constant regardless of dataset size.

### huggingface (Optional)

| Field | Type | Description |
|-------|------|-------------|
| `repository` | string | HuggingFace repo (user/name) |
| `tags` | list | Dataset tags |

## CLI Overrides

Most config options can be overridden via CLI:

```bash title="CLI overrides"
deepfabric generate config.yaml \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --num-samples 100 \
  --batch-size 10 \
  --temperature 0.5
```

!!! tip "Full Options"
    Run `deepfabric generate --help` for all available options.
