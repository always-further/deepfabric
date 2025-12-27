# upload-hf

The `upload-hf` command publishes datasets to Hugging Face Hub.

## Basic Usage

Upload a dataset file to Hugging Face Hub:

```bash title="Basic upload"
deepfabric upload-hf dataset.jsonl --repo username/dataset-name
```

This command uploads the dataset file and creates a dataset card with automatically generated metadata.

## Authentication Methods

The upload command supports multiple authentication approaches:

=== "Environment Variable"

    Most secure approach for production environments:

    ```bash
    export HF_TOKEN="your-huggingface-token"
    deepfabric upload-hf dataset.jsonl --repo username/dataset-name
    ```

=== "Command Line"

    Token specification directly in the command:

    ```bash
    deepfabric upload-hf dataset.jsonl --repo username/dataset-name --token your-token
    ```

=== "Hugging Face CLI"

    Works automatically if you've previously authenticated:

    ```bash
    huggingface-cli login
    deepfabric upload-hf dataset.jsonl --repo username/dataset-name
    ```

## Repository Management

The `upload-hf` command handles repository creation and updates automatically:

:material-plus-circle: **New Repositories**
:   Created automatically when uploading to non-existent repositories.

:material-sync: **Existing Repositories**
:   Receive updates to both dataset files and dataset card.

:material-tag: **Repository Naming**
:   Follows Hugging Face conventions: `username/dataset-name` or `organization/dataset-name`.

## Dataset Tagging

Customize dataset discoverability through tag specification:

```bash title="Add tags"
deepfabric upload-hf dataset.jsonl \
  --repo username/educational-content \
  --tags educational \
  --tags programming \
  --tags synthetic
```

## Generated Documentation

!!! info "Dataset Card"
    The upload process creates a basic dataset card if one doesn't already exist.

## File Organization

The upload process organizes files according to Hugging Face Hub conventions:

```
repository-name/
├── README.md          # Generated dataset card
├── dataset.jsonl      # Your uploaded dataset
└── .gitattributes     # LFS configuration for large files
```

!!! note "Large Files"
    Large dataset files are automatically configured for Git LFS to ensure efficient storage and retrieval.

## Batch Upload Operations

Upload multiple related datasets to maintain organized dataset collections:

```bash title="Multiple uploads"
# Upload training and validation sets
deepfabric upload-hf train_dataset.jsonl --repo myorg/comprehensive-dataset --tags training
deepfabric upload-hf val_dataset.jsonl --repo myorg/comprehensive-dataset --tags validation
```

This approach creates dataset repositories with multiple related files and appropriate metadata for each component.
