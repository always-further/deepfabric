# upload

The `upload` command publishes datasets and topic graphs to DeepFabric Cloud.

!!! note "Experimental Feature"
    This command requires the `EXPERIMENTAL_DF` environment variable to be set:
    ```bash
    export EXPERIMENTAL_DF=1
    ```

## Basic Usage

Upload a dataset or topic graph to DeepFabric Cloud:

```bash title="Upload dataset"
deepfabric upload dataset my-dataset.jsonl --handle username/my-dataset
```

```bash title="Upload topic graph"
deepfabric upload graph topic-graph.json --handle username/my-graph
```

## Authentication Methods

The upload command supports multiple authentication approaches:

=== "Environment Variable (Recommended)"

    Most secure approach for production environments and CI/CD:

    ```bash
    export DEEPFABRIC_API_KEY="df_your_api_key_here"
    deepfabric upload dataset dataset.jsonl --handle username/dataset-name
    ```

=== "Interactive Login"

    If not authenticated, you'll be automatically prompted to log in via browser:

    ```bash
    deepfabric upload dataset dataset.jsonl --handle username/dataset-name
    # You will be prompted: "Would you like to log in now? [Y/n]"
    ```

    You can also log in separately beforehand:

    ```bash
    deepfabric auth login
    ```

## Subcommands

### upload dataset

Upload a JSONL dataset file to DeepFabric Cloud.

```bash
deepfabric upload dataset FILE [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `FILE` | Path to the JSONL dataset file (required) |

**Options:**

| Option | Description |
|--------|-------------|
| `--handle TEXT` | Dataset handle (e.g., `username/dataset-name`) |
| `--name TEXT` | Display name for the dataset |
| `--description TEXT` | Description for the dataset |
| `--tags TEXT` | Tags for the dataset (can be specified multiple times) |
| `--config PATH` | Config file with upload settings |

**Example:**

```bash
deepfabric upload dataset training-data.jsonl \
  --handle myuser/python-training \
  --name "Python Training Dataset" \
  --description "Synthetic Python programming Q&A pairs" \
  --tags python \
  --tags programming
```

### upload graph

Upload a JSON topic graph file to DeepFabric Cloud.

```bash
deepfabric upload graph FILE [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `FILE` | Path to the JSON topic graph file (required) |

**Options:**

| Option | Description |
|--------|-------------|
| `--handle TEXT` | Graph handle (e.g., `username/graph-name`) |
| `--name TEXT` | Display name for the graph |
| `--description TEXT` | Description for the graph |
| `--config PATH` | Config file with upload settings |

**Example:**

```bash
deepfabric upload graph ml-topics.json \
  --handle myuser/machine-learning-topics \
  --name "Machine Learning Topics" \
  --description "Comprehensive ML topic hierarchy"
```

## Configuration File

You can specify upload settings in your YAML configuration file:

```yaml title="config.yaml"
topics:
  prompt: "Machine learning algorithms"
  mode: graph
  prompt_style: anchored

generation:
  system_prompt: "You are a helpful assistant..."

output:
  save_as: "dataset.jsonl"

# DeepFabric Cloud upload settings
deepfabric_cloud:
  dataset: "myuser/ml-training-data"
  graph: "myuser/ml-topics"
  description: "Machine learning training dataset"
  tags:
    - "machine-learning"
    - "synthetic"
```

Then upload using the config file:

```bash
deepfabric upload dataset dataset.jsonl --config config.yaml
deepfabric upload graph topic_graph.json --config config.yaml
```

!!! tip "CLI Override"
    Command-line options override values from the config file. This allows you to use a base config while customizing individual uploads.

## Handle Format

The handle follows the format `username/resource-name`:

- **username**: Your DeepFabric Cloud username
- **resource-name**: A URL-friendly slug for the resource

If you don't provide a handle, one will be derived from the filename.

## Error Handling

### Duplicate Names

If a resource with the same slug already exists:

```
A dataset with slug 'my-dataset' already exists. Use a different --handle value.
```

### Authentication Required

If not authenticated:

```
Authentication required. Run 'deepfabric auth login' first.
```

Or set the `DEEPFABRIC_API_KEY` environment variable.

## Complete Workflow

```bash title="Generate and upload workflow"
# Set up authentication
export EXPERIMENTAL_DF=1
export DEEPFABRIC_API_KEY="df_your_api_key"

# Generate dataset
deepfabric generate config.yaml

# Upload both graph and dataset
deepfabric upload graph topic_graph.json --handle myuser/my-topics
deepfabric upload dataset dataset.jsonl --handle myuser/my-dataset
```
