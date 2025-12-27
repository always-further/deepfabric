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

**Topic depth and degree** control dataset diversity. A tree with `depth: 3` and `degree: 3` produces 40 unique topics (1 + 3 + 9 + 27).

**System prompts** differ between generation and output:
- `generation.system_prompt` - Instructions for the LLM generating examples
- `output.system_prompt` - The system message included in training data

**Sample size** affects generation speed and amount. 
- `num_samples: 10` creates 10 examples.
- `batch_size` controls parallel requests to the LLM.

So `num_samples: 5` with `batch_size: 5` sends 5 parallel requests, each generating 5 examples, to give a total of 25 samples.

## Graph to sample ratio

When configuring topic generation with a tree or graph, the total number of unique topics is determined by the structure:

- **Tree**: Total Topics = (degree^(depth + 1) - 1) / (degree - 1)
- **Graph**: Total Topics = degree * depth + 1

For example, a tree with `depth: 2` and `degree: 2` yields 4 unique topics.

The amount of samples generated is dependent on the total unique topics and the `num_samples` setting. If the number of samples exceeds the number of unique topics, DeepFabric will warn and flag the discrepancy.

For example, with a tree of `depth: 2` and `degree: 2`, there are 7 unique topics. If `num_samples` is set to 5, DeepFabric will generate a warning

```bash
❌  Path validation failed - stopping before topic generation
❌ Error: Insufficient expected paths for dataset generation:
  • Expected tree paths: ~4 (depth=2, degree=2)
  • Requested samples: 5 (5 steps × 1 batch size)
  • Shortfall: ~1 samples

Recommendations:
  • Use one of these combinations to utilize the 4 paths:
    --num-steps 1 --batch-size 4  (generates 4 samples)
    --num-steps 2 --batch-size 2  (generates 4 samples)
    --num-steps 3 --batch-size 1  (generates 3 samples)
  • Or increase --depth (currently 2) or --degree (currently 2)
  ```
  