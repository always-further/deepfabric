# info

The `info` command displays version information, available commands, and environment configuration status.

## Usage

```bash
deepfabric info
```

## Output

The command shows:

- DeepFabric version number
- List of available commands with descriptions
- Required environment variables for LLM providers

## Example Output

```
DeepFabric v0.1.0
Large Scale Topic based Synthetic Data Generation

Available Commands:
  generate - Generate training data from configuration
  validate - Validate a configuration file
  visualize - Create SVG visualization of a topic graph
  upload - Upload dataset to Hugging Face Hub
  upload-kaggle - Upload dataset to Kaggle
  info - Show this information

Environment Variables:
  OPENAI_API_KEY - OpenAI API key
  ANTHROPIC_API_KEY - Anthropic API key
  HF_TOKEN - Hugging Face API token
```
