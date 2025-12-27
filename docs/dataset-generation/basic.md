# Basic Datasets

Basic datasets generate simple question-answer pairs without reasoning traces or tool calls.

## When to Use

- General instruction-following tasks
- Domain-specific Q&A (e.g., customer support, FAQs)
- Models that don't need to show reasoning
- Quick dataset generation with minimal configuration

## Configuration

```yaml
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

The key setting is `conversation.type: basic`.

## Output Format

Basic datasets produce standard chat-format JSONL:

```json
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

```bash
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

**Topic depth and degree** control dataset diversity. A tree with `depth: 3` and `degree: 3` produces 27 unique paths (`3^3 = 27` leaf nodes).

**System prompts** differ between generation and output:
- `generation.system_prompt` - Instructions for the LLM generating examples
- `output.system_prompt` - The system message included in training data

**Sample size** controls the number of generation steps.
- `num_samples` is the number of generation steps to run.
- `batch_size` is how many samples to generate per step.
- Total samples = `num_samples` × `batch_size`.

For example, `num_samples: 5` with `batch_size: 2` runs 5 steps, generating 2 samples each, for a total of 10 samples.

## Graph to sample ratio

When configuring topic generation with a tree or graph, the total number of unique topics is determined by the structure:

- **Tree**: Total Paths = degree^depth (leaf nodes only)
- **Graph**: Total Paths = degree^depth (approximate, varies due to cross-connections)

For example, a tree with `depth: 2` and `degree: 2` yields 4 unique paths (`2^2 = 4`).

The number of samples generated is dependent on the total unique paths and the `num_samples` setting. If the number of samples exceeds the number of unique paths, DeepFabric will warn and flag the discrepancy.

For example, with a tree of `depth: 2` and `degree: 2`, there are 4 unique paths. If `num_samples` is set to 5, DeepFabric will generate a warning:

```bash
Path validation failed - stopping before topic generation
Error: Insufficient expected paths for dataset generation:
  - Expected tree paths: ~4 (depth=2, degree=2)
  - Requested samples: 5 (5 steps x 1 batch size)
  - Shortfall: ~1 samples

Recommendations:
  - Use one of these combinations to utilize the 4 paths:
    --num-samples 1 --batch-size 4  (generates 4 samples)
    --num-samples 2 --batch-size 2  (generates 4 samples)
    --num-samples 4 --batch-size 1  (generates 4 samples)
  - Or increase --depth (currently 2) or --degree (currently 2)
```
  