# Basic Datasets

Basic datasets generate simple question-answer pairs without reasoning traces or tool calls.

## When to Use

- General instruction-following tasks
- Domain-specific Q&A (e.g., customer support, FAQs)
- Models that don't need to show reasoning
- Quick dataset generation with minimal configuration

## Configuration

```yaml title="config.yaml"
topics:
  prompt: "Python programming fundamentals"
  mode: tree
  depth: 2
  degree: 2

generation:
  system_prompt: "Generate clear, educational Q&A pairs."
  instructions: "Create diverse questions with detailed answers."

  conversation:
    type: basic

  llm:
    provider: "openai"
    model: "gpt-4o"

output:
  system_prompt: |
    You are a helpful assistant.
  num_samples: 2
  batch_size: 1
  save_as: "dataset.jsonl"
```

!!! note "Key Setting"
    The key setting is `conversation.type: basic`.

## Output Format

Basic datasets produce standard chat-format JSONL:

```json title="dataset.jsonl"
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What are Python's numeric data types?"
    },
    {
      "role": "assistant",
      "content": "Python has three built-in numeric types: integers (int), floating-point numbers (float), and complex numbers (complex)..."
    }
  ]
}
```

## CLI Usage

Generate a basic dataset from the command line:

```bash
deepfabric generate config.yaml
```

Or with inline options:

```bash title="CLI generation"
deepfabric generate \
  --topic-prompt "Machine learning basics" \
  --conversation-type basic \
  --num-samples 2 \
  --batch-size 1 \
  --provider openai \
  --model gpt-4o \
  --output-save-as ml-dataset.jsonl
```

## Tips

!!! tip "Topic Depth and Degree"
    Topic depth and degree control dataset diversity. A tree with `depth: 3` and `degree: 3` produces 27 unique paths (`3^3 = 27` leaf nodes).

!!! warning "System Prompt Confusion"
    System prompts differ between generation and output:

    - `generation.system_prompt` - Instructions for the LLM generating examples
    - `output.system_prompt` - The system message included in training data

!!! info "Sample Size"
    `num_samples` specifies the total number of samples to generate:

    - `num_samples` - The target number of samples to generate
    - `batch_size` - How many samples to generate per step (affects parallelism)
    - Steps = ceil(`num_samples` / `batch_size`)

    For example, `num_samples: 10` with `batch_size: 2` runs 5 steps, generating 2 samples each.

    **Special values:**

    - `"auto"` - Generate exactly one sample per topic path (100% coverage)
    - `"50%"` - Generate samples for 50% of topic paths
    - `"200%"` - Generate 2× the number of topic paths (cycles through topics twice)

## Graph to Sample Ratio

When configuring topic generation with a tree or graph, the total number of unique topics is determined by the structure:

- **Tree**: Total Paths = degree^depth (leaf nodes only)
- **Graph**: Total Paths = degree^depth (approximate, varies due to cross-connections)

For example, a tree with `depth: 2` and `degree: 2` yields 4 unique paths (`2^2 = 4`).

!!! info "Topic Cycling"
    When `num_samples` exceeds the number of unique topic paths, DeepFabric automatically cycles through topics to ensure even coverage. For example, with 4 paths and `num_samples: 8`, each topic is used twice.

    This works with both integer values and percentages:

    - `num_samples: 8` with 4 paths → 2x cycling
    - `num_samples: "200%"` → 2x cycling (explicit)

    An integer value larger than the number of paths is equivalent to using a percentage. For example, with 4 paths, `num_samples: 8` behaves identically to `num_samples: "200%"`.
