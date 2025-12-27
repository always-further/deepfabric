# Getting Started

## Installation

=== "pip"

    ```bash
    pip install deepfabric
    ```

=== "uv"

    ```bash
    uv add deepfabric
    ```

=== "Development"

    ```bash
    git clone https://github.com/always-further/deepfabric.git
    cd deepfabric
    uv sync --all-extras
    ```

## Provider Setup

Set your API key for your chosen provider:

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

=== "Anthropic"

    ```bash
    export ANTHROPIC_API_KEY="sk-ant-..."
    ```

=== "Google Gemini"

    ```bash
    export GEMINI_API_KEY="..."
    ```

=== "Ollama (Local)"

    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull mistral
    ollama serve
    ```

    !!! info "No API Key Required"
        Ollama runs locally, so no API key is needed.

## Verify Installation

```bash
deepfabric --help
deepfabric info
```

## Generate Your First Dataset

```bash title="Generate 10 samples"
deepfabric generate \
  --topic-prompt "Python programming basics" \
  --depth 2 \
  --degree 2 \
  --provider openai \
  --model gpt-4o \
  --num-samples 1 \
  --output-save-as dataset.jsonl
```

This creates a JSONL file with 1 training sample.
## Using a Config File

For more control, create a configuration file:

```yaml title="config.yaml"
topics:
  prompt: "Machine learning fundamentals"
  mode: tree
  depth: 2
  degree: 3

generation:
  system_prompt: "Generate educational Q&A pairs."
  conversation:
    type: basic
  llm:
    provider: openai
    model: gpt-4o

output:
  system_prompt: "You are a helpful ML tutor."
  num_samples: 5
  batch_size: 1
  save_as: "ml-dataset.jsonl"
```

Then run:

```bash
deepfabric generate config.yaml
```

!!! tip "Config vs CLI"
    Use configuration files for reproducible dataset generation. CLI flags are great for quick experiments.

## Next Steps

<div class="grid cards" markdown>

-   :material-database-outline: **Dataset Generation**

    ---

    Learn about different dataset types and configuration options

    [:octicons-arrow-right-24: Explore](../dataset-generation/index.md)

-   :material-tools: **Tools**

    ---

    Real tool execution for agent datasets using Spin/WASM

    [:octicons-arrow-right-24: Learn more](../tools/index.md)

-   :material-school-outline: **Training**

    ---

    Use your datasets with TRL, Unsloth, and other frameworks

    [:octicons-arrow-right-24: Get started](../training/index.md)

-   :material-chart-line: **Evaluation**

    ---

    Test and evaluate your fine-tuned models

    [:octicons-arrow-right-24: Evaluate](../evaluation/index.md)

</div>
