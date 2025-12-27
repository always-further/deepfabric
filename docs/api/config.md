# Configuration API

The DeepFabricConfig class provides programmatic access to YAML configuration loading and parameter management. This API enables configuration loading, validation, and parameter extraction for use with Tree, Graph, and DataSetGenerator classes.

## DeepFabricConfig Class

The configuration system loads YAML files and provides structured access to all generation parameters.

```python title="Basic usage"
from deepfabric import DeepFabricConfig

# Load configuration from YAML
config = DeepFabricConfig.from_yaml("config.yaml")

# Access configuration sections
topics_params = config.get_topics_params()
generation_params = config.get_generation_params()
output_config = config.get_output_config()
```

### Configuration Structure

DeepFabric uses a structured YAML configuration format:

```yaml title="config.yaml"
# Optional shared LLM defaults
llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7

# Topic generation configuration (required)
topics:
  prompt: "Machine learning concepts"
  system_prompt: "You are creating educational topic structures."
  mode: "tree"  # or "graph"
  depth: 3
  degree: 4

# Sample generation configuration (required)
generation:
  system_prompt: "You are an expert instructor."
  instructions: "Create detailed explanations with practical examples."
  conversation:
    type: "basic"  # or "chain_of_thought"
    reasoning_style: "freetext"  # or "agent"

# Output configuration (required)
output:
  save_as: "dataset.jsonl"
  include_system_message: true
  num_samples: 100
  batch_size: 5

# Optional integrations
huggingface:
  repository: "username/dataset-name"

kaggle:
  handle: "username/dataset-name"
```

!!! note "Required Sections"
    The `topics`, `generation`, and `output` sections are required. All other sections are optional.

### Loading Configurations

#### from_yaml(filepath: str)

Class method for loading configurations from YAML files:

```python title="Load config"
config = DeepFabricConfig.from_yaml("production_config.yaml")
```

!!! warning "Errors"
    Raises `ConfigurationError` if the file is not found, contains invalid YAML, or uses the old configuration format.

### Parameter Extraction Methods

#### get_topics_params(**overrides)

Extract topic parameters with optional overrides for use with Tree or Graph constructors:

=== "Basic Usage"

    ```python title="Extract params"
    topics_params = config.get_topics_params()
    tree = Tree(**topics_params)
    ```

=== "With Overrides"

    ```python title="Override params"
    topics_params = config.get_topics_params(
        degree=5,
        temperature=0.9,
        provider="anthropic",
        model="claude-sonnet-4-5"
    )
    ```

**Returns:** Dictionary with keys:

| Key | Description |
|-----|-------------|
| `topic_prompt` | The seed topic |
| `topic_system_prompt` | System prompt for topic generation |
| `provider` | LLM provider name |
| `model_name` | Model name |
| `temperature` | Generation temperature |
| `base_url` | Optional API base URL |
| `depth` | Tree/graph depth |
| `degree` | Branching factor |
| `max_concurrent` | Maximum concurrent LLM calls |

#### get_generation_params(**overrides)

Extract generator parameters for use with DataSetGenerator constructor:

```python title="Extract generation params"
generation_params = config.get_generation_params(
    temperature=0.8,
    provider="openai",
    model="gpt-4"
)
generator = DataSetGenerator(**generation_params)
```

**Returns:** Dictionary with keys including:

| Key | Description |
|-----|-------------|
| `generation_system_prompt` | System prompt for generation |
| `instructions` | Content generation instructions |
| `provider`, `model_name` | LLM settings |
| `max_retries`, `sample_retries` | Request configuration |
| `rate_limit` | Rate limiting configuration |
| `conversation_type`, `reasoning_style` | Conversation settings |
| `sys_msg`, `dataset_system_prompt` | Output settings |

#### get_output_config()

Access output configuration:

```python title="Get output config"
output_config = config.get_output_config()

save_path = output_config["save_as"]
num_samples = output_config["num_samples"]
batch_size = output_config["batch_size"]
```

#### get_huggingface_config()

Extract Hugging Face Hub integration settings:

```python title="Get HF config"
hf_config = config.get_huggingface_config()

if hf_config:
    repository = hf_config.get("repository")
    token = hf_config.get("token")
```

!!! note
    Returns empty dictionary if Hugging Face integration is not configured.

#### get_kaggle_config()

Extract Kaggle integration settings:

```python title="Get Kaggle config"
kaggle_config = config.get_kaggle_config()

if kaggle_config:
    handle = kaggle_config.get("handle")
```

!!! note
    Returns empty dictionary if Kaggle integration is not configured.

#### get_configured_providers()

Get the set of LLM providers used in this configuration:

```python title="Get providers"
providers = config.get_configured_providers()
# Returns: {"openai", "anthropic"} for example
```

### LLM Configuration Inheritance

The configuration system supports LLM setting inheritance:

| Priority | Source | Description |
|----------|--------|-------------|
| 1 (highest) | Section-specific (`topics.llm`, `generation.llm`) | Overrides for specific sections |
| 2 | Top-level shared (`llm`) | Used if section-specific not set |
| 3 (lowest) | Built-in defaults | Used if neither is set |

```yaml title="Inheritance example"
# Shared defaults
llm:
  provider: "openai"
  model: "gpt-4"

topics:
  prompt: "..."
  # Uses openai/gpt-4 from shared llm config

generation:
  llm:
    provider: "anthropic"
    model: "claude-sonnet-4-5"
  # Uses anthropic/claude-sonnet-4-5 (overrides shared)
```

### Error Handling

```python title="Exception handling"
from deepfabric import ConfigurationError

try:
    config = DeepFabricConfig.from_yaml("config.yaml")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

!!! warning "Common Error Scenarios"
    - File not found
    - Invalid YAML syntax
    - Old configuration format (migration required)
    - Invalid structure or missing required fields

### Migration from Old Format

!!! info "Format Migration"
    If you have configuration files using the old format, DeepFabric will provide a migration message showing the mapping.

| Old Format | New Format |
|------------|------------|
| `dataset_system_prompt` | `output.system_prompt` |
| `topic_tree` / `topic_graph` | `topics` (with `mode: tree\|graph`) |
| `topic_tree.topic_prompt` | `topics.prompt` |
| `data_engine` | `generation` |
| `data_engine.generation_system_prompt` | `generation.system_prompt` |
| `dataset.creation.num_steps` | `output.num_samples` |
| `dataset.save_as` | `output.save_as` |

### Integration Example

??? example "Complete workflow using configuration"

    ```python title="Full workflow"
    import asyncio
    from deepfabric import DeepFabricConfig, Tree, DataSetGenerator

    # Load configuration
    config = DeepFabricConfig.from_yaml("config.yaml")

    # Create topic model
    topics_params = config.get_topics_params()
    tree = Tree(**topics_params)

    async def build():
        async for _ in tree.build_async():
            pass

    asyncio.run(build())

    # Create generator
    generation_params = config.get_generation_params()
    generator = DataSetGenerator(**generation_params)

    # Get output settings
    output = config.get_output_config()

    # Generate dataset
    dataset = asyncio.run(generator.create_data_async(
        num_steps=output["num_samples"],
        batch_size=output["batch_size"],
        topic_model=tree
    ))

    # Save
    dataset.save(output["save_as"])
    ```
