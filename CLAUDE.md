# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepFabric is a Python library for generating synthetic datasets using LLM providers. It consists of three main components that work together in a pipeline:

1. **Tree**: Generates hierarchical topic structures from a root prompt
2. **DataSetGenerator**: Creates training examples based on topics 
3. **Dataset**: Manages and exports the final dataset

## Development Commands

### Package Management
- `uv sync --all-extras` - Install all dependencies including dev tools
- `make install` - Same as above via Makefile

### Code Quality
- `make format` - Format code with black and ruff auto-fix
- `make lint` - Run ruff linting checks
- `make security` - Run bandit security analysis on deepfabric/ directory

### Testing
- `make test` - Run pytest test suite
- `uv run pytest` - Direct pytest execution
- Test files are in `tests/` directory

### Build and Release
- `make build` - Clean, test, and build package
- `make clean` - Remove build artifacts and cache files
- `make all` - Complete workflow: install, format, lint, test, security, build

## Core Architecture

### Configuration System
- YAML-based configuration with direct system prompt specification
- CLI supports extensive parameter overrides

### Data Flow
1. Tree generates topic hierarchy from root prompt
2. DataSetGenerator uses topics to create question/answer pairs
3. Dataset validates, stores, and exports training examples
4. Optional HuggingFace Hub upload with auto-generated dataset cards

### Key Classes and Their Relationships
- `DeepFabricConfig`: Loads YAML and provides argument objects
- `TreeArguments`/`DataSetGeneratorArguments`: Pydantic-style dataclasses for parameters
- `Dataset`: Handles JSONL import/export and validation
- `HFUploader`: Manages Hugging Face Hub integration

## Important Implementation Details

### JSON Validation
The engine includes robust JSON parsing with regex extraction and retry logic for handling LLM response inconsistencies.

### Error Handling
- Max retries configurable for failed LLM calls
- Failed samples tracked separately from successful ones
- Comprehensive error reporting in final summary

### System Message Control
The `sys_msg` parameter controls whether system messages are included in the final dataset format - this affects training data structure.

## Configuration Patterns

### YAML Structure
```yaml
dataset_system_prompt: "..."
topic_tree:
  args: {...}
  save_as: "file.jsonl"
data_engine:
  args: {...}
dataset:
  creation: {...}
  save_as: "file.jsonl"
huggingface: {...}  # optional
```

### Provider Configuration
Specify provider and model separately in config, or use combined format in code:
- Config: `provider: "ollama"`, `model: "mistral:latest"`
- Code: `model_name: "ollama/mistral:latest"`

## CLI Usage Patterns

The CLI supports both YAML configuration and parameter overrides:
```bash
deepfabric start config.yaml --model gpt-4 --temperature 0.8 --hf-repo user/dataset
```

## Code Style Notes
- Uses uv for dependency management
- Ruff for linting with extensive rule set
- Black for formatting
- Bandit for security analysis
- Python 3.11+ required
- Google-style docstrings preferred
- do not place imports anywhere but the top of the file
- When updating `docs/` documentation, if new Markdown files are added or removed, consider updating `mkdocs.yml`.