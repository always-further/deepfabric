# upload-kaggle

The `upload-kaggle` command publishes datasets to Kaggle.

## Basic Usage

Upload a dataset file to Kaggle:

```bash title="Basic upload"
deepfabric upload-kaggle dataset.jsonl --handle username/dataset-name
```

## Authentication Methods

The `upload-kaggle` command supports multiple authentication approaches:

=== "Environment Variables"

    Most secure approach for production environments:

    ```bash
    export KAGGLE_USERNAME="your-kaggle-username"
    export KAGGLE_KEY="your-kaggle-api-key"
    deepfabric upload-kaggle dataset.jsonl --handle username/dataset-name
    ```

=== "Command Line"

    Credential specification directly in the command:

    ```bash
    deepfabric upload-kaggle dataset.jsonl --handle username/dataset-name --username your-username --key your-api-key
    ```

## Options

| Option | Description |
|--------|-------------|
| `--handle` | Kaggle dataset handle (required) |
| `--username` | Kaggle username (or set `KAGGLE_USERNAME` env var) |
| `--key` | Kaggle API key (or set `KAGGLE_KEY` env var) |
| `--tags` | Tags for the dataset (can be specified multiple times) |
| `--version-notes` | Version notes for the dataset update |
| `--description` | Description for the dataset |

## Dataset Tagging

Customize dataset discoverability through tag specification:

```bash title="Add tags"
deepfabric upload-kaggle dataset.jsonl \
  --handle username/educational-content \
  --tags educational \
  --tags programming \
  --tags synthetic
```

## Version Notes

When updating an existing dataset, provide version notes to document changes:

```bash title="Version notes"
deepfabric upload-kaggle dataset.jsonl \
  --handle username/my-dataset \
  --version-notes "Added 500 new samples for edge cases"
```

## Dataset Description

Provide a description for new datasets:

```bash title="With description"
deepfabric upload-kaggle dataset.jsonl \
  --handle username/new-dataset \
  --description "Synthetic dataset for training code generation models"
```

## Batch Upload Operations

Upload multiple related datasets:

```bash title="Multiple uploads"
# Upload training and validation sets
deepfabric upload-kaggle train_dataset.jsonl --handle myorg/comprehensive-dataset --tags training
deepfabric upload-kaggle val_dataset.jsonl --handle myorg/comprehensive-dataset --tags validation
```
