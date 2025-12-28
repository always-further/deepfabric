# generate

The `generate` command executes the complete synthetic data generation pipeline from YAML configuration to finished dataset. This command represents the primary interface for transforming domain concepts into structured training data through topic modeling and content generation.

```mermaid
graph LR
    A[YAML Config] --> B[Topic Generation]
    B --> C[Dataset Creation]
    C --> D[Output Files]
```

The generation process operates through multiple stages that can be monitored in real-time, providing visibility into topic expansion, content creation, and quality control measures.

## Basic Usage

Generate a complete dataset from a configuration file:

```bash title="Basic generation"
deepfabric generate config.yaml
```

This command reads your configuration, generates the topic structure, creates training examples, and saves all outputs to the specified locations.

## Configuration Override

Override specific configuration parameters without modifying the configuration file:

```bash title="Override parameters"
deepfabric generate config.yaml \
  --provider anthropic \
  --model claude-sonnet-4-5 \
  --temperature 0.8 \
  --num-samples 100 \
  --batch-size 5
```

!!! tip "Experimentation"
    Configuration overrides apply to all stages, enabling experimentation with different settings while maintaining the base configuration.

## File Management Options

Control where intermediate and final outputs are saved:

```bash title="Custom output paths"
deepfabric generate config.yaml \
  --topics-save-as custom_topics.jsonl \
  --output-save-as custom_dataset.jsonl
```

!!! note "Output Organization"
    These options override the file paths specified in your configuration, useful for organizing outputs by experiment or preventing accidental overwrites.

## Loading Existing Topic Structures

Skip topic generation by loading previously generated topic trees or graphs:

```bash title="Load existing topics"
deepfabric generate config.yaml --topics-load existing_topics.jsonl
```

!!! tip "Faster Iteration"
    This approach accelerates iteration when experimenting with dataset generation parameters while keeping the topic structure constant.

## Topic-Only Generation

Generate and save only the topic structure without proceeding to dataset creation:

=== "Tree Mode"

    ```bash
    deepfabric generate config.yaml --topic-only
    ```

=== "Graph Mode"

    ```bash
    deepfabric generate config.yaml --mode graph --topic-only
    ```

The `--topic-only` flag stops the pipeline after topic generation and saves the topic structure to the configured location.

## Topic Modeling Parameters

Fine-tune topic generation behavior through command-line parameters:

```bash title="Topic parameters"
deepfabric generate config.yaml \
  --degree 5 \
  --depth 4 \
  --temperature 0.7
```

| Parameter | Effect |
|-----------|--------|
| `--degree` | More subtopics per node (broader exploration) |
| `--depth` | More levels of detail (deeper exploration) |
| `--temperature` | Higher values create more diverse topics |

## Dataset Generation Controls

Adjust dataset creation parameters for different scales and quality requirements:

```bash title="Generation controls"
deepfabric generate config.yaml \
  --num-samples 500 \
  --batch-size 10 \
  --no-system-message
```

| Parameter | Description |
|-----------|-------------|
| `--num-samples` | Controls dataset size |
| `--batch-size` | Affects generation speed and resource usage |
| `--include-system-message` | Include system prompts in training examples |
| `--no-system-message` | Exclude system prompts from training examples |

## Conversation Type Options

Control the type of conversations generated:

```bash title="Conversation options"
deepfabric generate config.yaml \
  --conversation-type cot \
  --reasoning-style freetext
```

| Option | Values | Description |
|--------|--------|-------------|
| `--conversation-type` | `basic`, `cot` | Base conversation type |
| `--reasoning-style` | `freetext`, `agent` | Reasoning style for cot |
| `--agent-mode` | `single_turn`, `multi_turn` | Agent mode (requires tools) |
| `--min-turns` | INT | Minimum turns for multi_turn mode |
| `--max-turns` | INT | Maximum turns for multi_turn mode |
| `--min-tool-calls` | INT | Minimum tool calls before conclusion |

## TUI Options

Control the terminal user interface:

=== "Rich TUI"

    ```bash
    deepfabric generate config.yaml --tui rich
    ```

    Two-pane interface with real-time progress (default).

=== "Simple Output"

    ```bash
    deepfabric generate config.yaml --tui simple
    ```

    Headless-friendly plain text output.

## Provider and Model Selection

Use different providers or models for different components:

```bash title="Provider selection"
deepfabric generate config.yaml \
  --provider openai \
  --model gpt-4 \
  --temperature 0.9
```

!!! info "Provider Scope"
    Provider changes apply to all components unless overridden in the configuration file.

## Complete Example

??? example "Comprehensive generation command"

    ```bash title="Full example"
    deepfabric generate research-dataset.yaml \
      --topics-save-as research_topics.jsonl \
      --output-save-as research_examples.jsonl \
      --provider anthropic \
      --model claude-sonnet-4-5 \
      --degree 4 \
      --depth 3 \
      --num-samples 200 \
      --batch-size 8 \
      --temperature 0.8 \
      --include-system-message
    ```

    This creates a research dataset with comprehensive topic coverage and high-quality content generation.

## Progress Monitoring

The generation process provides real-time feedback:

- :material-file-tree: Topic tree construction progress with node counts
- :material-percent: Dataset generation status with completion percentages
- :material-alert: Error reporting with retry attempts and failure categorization
- :material-chart-bar: Final statistics including success rates and output file locations

## Error Recovery

!!! warning "Partial Failures"
    When generation fails partway through, the system saves intermediate results where possible. Topic trees are saved incrementally, enabling recovery by loading partial results and continuing from the dataset generation stage.

??? tip "Optimizing Generation Performance"
    Balance `batch-size` with your API rate limits and system resources. Larger batches increase throughput but consume more memory and may trigger rate limiting. Start with smaller batches and increase based on your provider's capabilities.

## Output Validation

After generation completes, verify your outputs:

```bash title="Validation commands"
# Check dataset format
head -n 5 your_dataset.jsonl

# Validate JSON structure
python -m json.tool your_dataset.jsonl > /dev/null

# Count generated examples
wc -l your_dataset.jsonl
```
