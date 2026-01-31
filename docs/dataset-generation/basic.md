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

!!! info "Sample Size and Generation Model"
    DeepFabric uses a cycle-based generation model:

    - **Unique topics**: Deduplicated count from topic tree/graph (by UUID)
    - **Cycles**: Number of iterations through all unique topics
    - **Concurrency**: `batch_size` controls parallel LLM calls

    For example, with 4 unique topics and `num_samples: 10`:

    - Cycles needed: 3 (ceil(10/4))
    - Cycle 1: 4 samples, Cycle 2: 4 samples, Cycle 3: 2 samples (partial)

    **Special values for `num_samples`:**

    - `"auto"` - Generate exactly one sample per unique topic (1 cycle)
    - `"50%"` - Generate samples for 50% of unique topics
    - `"200%"` - Generate 2× the number of unique topics (2 full cycles)

## Topic Count and Cycles

When configuring topic generation with a tree or graph, the total number of unique topics is determined by the structure:

- **Tree**: Unique topics = degree^depth (each leaf node has a unique UUID)
- **Graph**: Unique topics ≤ degree^depth (deduplicated by node UUID, may be fewer due to cross-connections)

For example, a tree with `depth: 2` and `degree: 2` yields 4 unique topics (`2^2 = 4`).

!!! info "Cycle-Based Generation"
    When `num_samples` exceeds the number of unique topics, DeepFabric iterates through multiple **cycles**:

    - Each unique topic (identified by UUID) is processed once per cycle
    - Cycles continue until `num_samples` is reached
    - The final cycle may be partial (fewer topics than a full cycle)

    For example, with 4 unique topics and `num_samples: 10`:

    - Cycle 1: Topics 1-4 (4 samples)
    - Cycle 2: Topics 1-4 (4 samples)
    - Cycle 3: Topics 1-2 (2 samples, partial)

    Checkpoints track progress as `(topic_uuid, cycle)` tuples, allowing precise resume from any point in any cycle.

!!! tip "Graphs with Shared Nodes"
    In graph mode, multiple paths can lead to the same topic node. DeepFabric deduplicates by node UUID, so each unique topic generates exactly one sample per cycle—regardless of how many paths lead to it.
