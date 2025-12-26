# CLI Reference

DeepFabric's command-line interface provides a modular set of tools that support complex workflows through focused, single-purpose commands. Each command handles a specific aspect of the synthetic data generation pipeline, enabling both simple one-shot operations and sophisticated multi-stage workflows.

The CLI design follows Unix philosophy principles where each command does one thing well and commands compose together naturally. This approach supports iterative development, error recovery, and pipeline optimization through selective execution of generation stages.

## Command Overview

The DeepFabric CLI provides the following commands:

**`generate`** transforms YAML configurations into synthetic datasets through the complete pipeline of topic modeling, dataset creation, and output formatting.

**`validate`** checks configuration files for common issues, parameter compatibility problems, and authentication requirements before running expensive generation processes.

**`visualize`** creates SVG representations of topic graphs, enabling visual exploration of domain structures and relationship patterns.

**`upload-hf`** publishes datasets to Hugging Face Hub.

**`upload-kaggle`** publishes datasets to Kaggle.

**`evaluate`** evaluates fine-tuned models on tool-calling tasks using transformers or Ollama backends.

**`info`** displays version information, available commands, and environment configuration status.

## Global Options

All commands support common options for help and version information:

```bash
deepfabric --help     # Show command overview
deepfabric --version  # Display version information
```

Individual commands provide detailed help through the standard help flag:

```bash
deepfabric generate --help
deepfabric validate --help
```

## Command Composition

The modular design enables sophisticated workflows through command composition. For example, a complete dataset development cycle might involve:

```bash
# Validate configuration
deepfabric validate config.yaml

# Generate the dataset
deepfabric generate config.yaml

# Visualize topic structure (if using graphs)
deepfabric visualize topic_graph.json --output structure.svg

# Upload to Hugging Face
deepfabric upload-hf dataset.jsonl --repo username/dataset-name
```

Each command operates independently, allowing selective re-execution when iterating on specific aspects of your generation process.

## Error Handling

All commands provide detailed error messages with actionable guidance for resolution. Error categories include configuration problems, authentication issues, API failures, and file system problems.

The commands use consistent exit codes where success returns 0 and various error conditions return non-zero values, enabling reliable scripting and automated workflows.

## Configuration Override Patterns

Many commands accept configuration file arguments along with parameter overrides, enabling experimentation without modifying configuration files:

```bash
deepfabric generate config.yaml \
  --temperature 0.9 \
  --num-samples 50 \
  --provider anthropic \
  --model claude-sonnet-4-5
```

This pattern supports rapid iteration during development while maintaining reproducible baseline configurations.

## Command Sections

Detailed documentation for each command covers syntax, options, examples, and common usage patterns:

- [**generate**](generate.md) - Complete dataset generation from YAML configuration
- [**validate**](validate.md) - Configuration validation and problem detection
- [**visualize**](visualize.md) - Topic graph visualization and analysis
- [**upload-hf**](upload-hf.md) - Hugging Face Hub integration and publishing
- [**upload-kaggle**](upload-kaggle.md) - Kaggle integration and publishing
- [**evaluate**](evaluate.md) - Model evaluation on tool-calling tasks
- [**info**](info.md) - Version and environment information

Each command section includes practical examples, common usage patterns, and troubleshooting guidance for typical issues encountered during synthetic data generation workflows.
