# generate

The `generate` command executes the complete synthetic data generation pipeline from YAML configuration to finished dataset. This command represents the primary interface for transforming domain concepts into structured training data through topic modeling and content generation.

The generation process operates through multiple stages that can be monitored in real-time, providing visibility into topic expansion, content creation, and quality control measures. Each stage builds upon previous work, enabling sophisticated workflows where different components use optimized parameters for their specific functions.

## Basic Usage

Generate a complete dataset from a configuration file:

```bash
deepfabric generate config.yaml
```

This command reads your configuration, generates the topic structure, creates training examples, and saves all outputs to the specified locations. The process displays real-time progress information and completion statistics.

## Configuration Override

Override specific configuration parameters without modifying the configuration file:

```bash
deepfabric generate config.yaml \
  --provider anthropic \
  --model claude-sonnet-4-5 \
  --temperature 0.8 \
  --num-samples 100 \
  --batch-size 5
```

Configuration overrides apply to all stages that use the specified parameters, enabling experimentation with different settings while maintaining the base configuration.

## File Management Options

Control where intermediate and final outputs are saved:

```bash
deepfabric generate config.yaml \
  --topics-save-as custom_topics.jsonl \
  --output-save-as custom_dataset.jsonl
```

These options override the file paths specified in your configuration, useful for organizing outputs by experiment or preventing accidental overwrites during iterative development.

## Loading Existing Topic Structures

Skip topic generation by loading previously generated topic trees or graphs:

```bash
# Load existing topics (JSONL for tree, JSON for graph)
deepfabric generate config.yaml --topics-load existing_topics.jsonl
```

This approach accelerates iteration when experimenting with dataset generation parameters while keeping the topic structure constant.

## Topic-Only Generation

Generate and save only the topic structure without proceeding to dataset creation:

```bash
# Generate topic tree only
deepfabric generate config.yaml --topic-only

# Generate topic graph only
deepfabric generate config.yaml --mode graph --topic-only
```

The `--topic-only` flag stops the pipeline after topic generation and saves the topic structure to the configured location. This enables rapid iteration on topic modeling parameters, review of topic structures before committing computational resources to dataset generation, and separation of topic exploration from content creation workflows.

## Topic Modeling Parameters

Fine-tune topic generation behavior through command-line parameters:

```bash
deepfabric generate config.yaml \
  --degree 5 \
  --depth 4 \
  --temperature 0.7
```

These parameters control the breadth and depth of topic exploration. Higher degree values create more subtopics per node, while greater depth values enable more detailed exploration of each subtopic area.

## Dataset Generation Controls

Adjust dataset creation parameters for different scales and quality requirements:

```bash
deepfabric generate config.yaml \
  --num-samples 500 \
  --batch-size 10 \
  --no-system-message
```

The `--num-samples` parameter controls dataset size, `--batch-size` affects generation speed and resource usage, and `--include-system-message/--no-system-message` determines whether system prompts are included in the final training examples.

## Conversation Type Options

Control the type of conversations generated:

```bash
deepfabric generate config.yaml \
  --conversation-type chain_of_thought \
  --reasoning-style freetext
```

| Option | Values | Description |
|--------|--------|-------------|
| `--conversation-type` | `basic`, `chain_of_thought` | Base conversation type |
| `--reasoning-style` | `freetext`, `agent` | Reasoning style for chain_of_thought |
| `--agent-mode` | `single_turn`, `multi_turn` | Agent mode (requires tools) |
| `--min-turns` | INT | Minimum turns for multi_turn mode |
| `--max-turns` | INT | Maximum turns for multi_turn mode |
| `--min-tool-calls` | INT | Minimum tool calls before conclusion |

## TUI Options

Control the terminal user interface:

```bash
# Rich two-pane interface (default)
deepfabric generate config.yaml --tui rich

# Simple headless-friendly output
deepfabric generate config.yaml --tui simple
```

## Provider and Model Selection

Use different providers or models for different components:

```bash
deepfabric generate config.yaml \
  --provider openai \
  --model gpt-4 \
  --temperature 0.9
```

Provider changes apply to all components unless overridden in the configuration file. This enables quick experiments with different model capabilities or cost structures.

## Complete Example

A comprehensive generation command with multiple overrides:

```bash
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

This command creates a research dataset with comprehensive topic coverage, high-quality content generation, and systematic organization of all outputs.

## Progress Monitoring

The generation process provides real-time feedback including:

- Topic tree construction progress with node counts
- Dataset generation status with completion percentages  
- Error reporting with retry attempts and failure categorization
- Final statistics including success rates and output file locations

Monitor the process to understand generation bottlenecks and optimize parameters for your specific use case and infrastructure constraints.

## Error Recovery

When generation fails partway through, the system saves intermediate results where possible. Topic trees are saved incrementally, enabling recovery by loading partial results and continuing from the dataset generation stage.

??? tip "Optimizing Generation Performance"
    Balance `batch-size` with your API rate limits and system resources. Larger batches increase throughput but consume more memory and may trigger rate limiting. Start with smaller batches and increase based on your provider's capabilities and your system's performance characteristics.

## Output Validation

After generation completes, verify your outputs:

```bash
# Check dataset format
head -n 5 your_dataset.jsonl

# Validate JSON structure
python -m json.tool your_dataset.jsonl > /dev/null

# Count generated examples
wc -l your_dataset.jsonl
```

This validation ensures the generation process completed successfully and produced properly formatted output ready for use in training or evaluation pipelines.
