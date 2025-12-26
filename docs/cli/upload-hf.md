# upload-hf

The `upload-hf` command publishes datasets to Hugging Face Hub.

## Basic Usage

Upload a dataset file to Hugging Face Hub:

```bash
deepfabric upload-hf dataset.jsonl --repo username/dataset-name
```

This command uploads the dataset file and creates a dataset card with automatically generated metadata about the content and generation process.

## Authentication Methods

The upload command supports multiple authentication approaches for flexibility across different environments:

**Environment Variable** provides the most secure approach for production environments:

```bash
export HF_TOKEN="your-huggingface-token"
deepfabric upload-hf dataset.jsonl --repo username/dataset-name
```

**Command Line Option** enables token specification directly in the command:

```bash
deepfabric upload-hf dataset.jsonl --repo username/dataset-name --token your-token
```

**Hugging Face CLI** authentication works automatically if you've previously authenticated with the Hugging Face CLI:

```bash
huggingface-cli login
deepfabric upload-hf dataset.jsonl --repo username/dataset-name
```

## Repository Management

The upload command handles repository creation and updates automatically:

**New Repositories** are created automatically when uploading to non-existent repositories, using the dataset filename as the initial commit message.

**Existing Repositories** receive updates to both the dataset files and the dataset card, with commit messages indicating the update source.

**Repository Naming** follows Hugging Face conventions with the format `username/dataset-name` or `organization/dataset-name`.

## Dataset Tagging

Customize dataset discoverability through tag specification:

```bash
deepfabric upload-hf dataset.jsonl \
  --repo username/educational-content \
  --tags educational \
  --tags programming \
  --tags synthetic
```

## Generated Documentation

The upload process creates a basic dataset card if one doesn't already exist. 


## File Organization

The upload process organizes files according to Hugging Face Hub conventions:

```
repository-name/
├── README.md          # Generated dataset card
├── dataset.jsonl      # Your uploaded dataset
└── .gitattributes     # LFS configuration for large files
```

Large dataset files are automatically configured for Git LFS to ensure efficient storage and retrieval.

## Batch Upload Operations

Upload multiple related datasets to maintain organized dataset collections:

```bash
# Upload training and validation sets
deepfabric upload-hf train_dataset.jsonl --repo myorg/comprehensive-dataset --tags training
deepfabric upload-hf val_dataset.jsonl --repo myorg/comprehensive-dataset --tags validation
```

This approach creates dataset repositories with multiple related files and appropriate metadata for each component.
