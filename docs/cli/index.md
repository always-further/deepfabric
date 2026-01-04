# CLI Reference

DeepFabric's command-line interface provides a modular set of tools that support complex workflows through focused, single-purpose commands.

## Command Overview

<div class="grid cards" markdown>

-   :material-database-plus: **generate**

    ---

    Complete dataset generation from YAML configuration

    [:octicons-arrow-right-24: Reference](generate.md)

-   :material-check-circle: **validate**

    ---

    Configuration validation and problem detection

    [:octicons-arrow-right-24: Reference](validate.md)

-   :material-graph: **visualize**

    ---

    Topic graph visualization and analysis

    [:octicons-arrow-right-24: Reference](visualize.md)

-   :material-cloud-upload: **upload-hf**

    ---

    Hugging Face Hub integration and publishing

    [:octicons-arrow-right-24: Reference](upload-hf.md)

-   :material-upload: **upload-kaggle**

    ---

    Kaggle integration and publishing

    [:octicons-arrow-right-24: Reference](upload-kaggle.md)

-   :material-cloud-upload-outline: **upload**

    ---

    DeepFabric Cloud integration and publishing (experimental)

    [:octicons-arrow-right-24: Reference](upload.md)

-   :material-import: **import-tools**

    ---

    Import tool definitions from MCP servers

    [:octicons-arrow-right-24: Reference](import-tools.md)

-   :material-test-tube: **evaluate**

    ---

    Model evaluation on tool-calling tasks

    [:octicons-arrow-right-24: Reference](evaluate.md)

-   :material-information: **info**

    ---

    Version and environment information

    [:octicons-arrow-right-24: Reference](info.md)

</div>

## Global Options

All commands support common options for help and version information:

```bash
deepfabric --help     # Show command overview
deepfabric --version  # Display version information
```

Individual commands provide detailed help:

```bash
deepfabric generate --help
deepfabric validate --help
```

## Command Composition

!!! tip "Modular Workflow"
    The modular design enables sophisticated workflows through command composition:

```bash title="Complete workflow"
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

All commands provide detailed error messages with actionable guidance for resolution. Error categories include:

- Configuration problems
- Authentication issues
- API failures
- File system problems

!!! info "Exit Codes"
    Commands use consistent exit codes where success returns 0 and various error conditions return non-zero values, enabling reliable scripting and automated workflows.

## Configuration Override Patterns

Many commands accept configuration file arguments along with parameter overrides:

```bash title="Override config options"
deepfabric generate config.yaml \
  --temperature 0.9 \
  --num-samples 50 \
  --provider anthropic \
  --model claude-sonnet-4-5
```

This pattern supports rapid iteration during development while maintaining reproducible baseline configurations.
