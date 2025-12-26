# DataSetGenerator API

The DataSetGenerator class transforms topic structures into practical training examples through configurable templates, quality control mechanisms, and batch processing. This API provides comprehensive control over the dataset creation process from topic selection through content validation.

## DataSetGenerator Configuration

Dataset generation configuration is passed directly to the DataSetGenerator constructor:

```python
from deepfabric import DataSetGenerator

generator = DataSetGenerator(
    instructions="Create detailed explanations with practical examples for intermediate learners.",
    generation_system_prompt="You are an expert instructor creating educational content.",
    provider="openai",
    model_name="gpt-4",
    temperature=0.8,
    max_retries=3,
    request_timeout=30,
    default_batch_size=5,
    default_num_examples=3
)
```

### Parameters

**instructions** (str, optional): Core guidance for content generation specifying format, complexity, target audience, and quality expectations. Default: empty string.

**generation_system_prompt** (str, required): System prompt providing behavioral context for the generation model.

**provider** (str, required): LLM provider name, e.g., `openai`, `anthropic`, `gemini`, `ollama`.

**model_name** (str, required): Model name specific to the provider, e.g., `gpt-4`, `claude-sonnet-4-5`.

**temperature** (float, optional): Controls creativity and diversity in content generation. Range 0.0-2.0, typically 0.7-0.9. Default: 0.7.

**max_retries** (int, optional): Number of retry attempts for failed generation requests. Default: 3.

**request_timeout** (int, optional): Maximum seconds to wait for API responses. Default: 30.

**default_batch_size** (int, optional): Default number of examples to generate per API call. Default: 5.

**default_num_examples** (int, optional): Default number of example demonstrations to include. Default: 3.

**conversation_type** (str, optional): Type of conversation format. One of: `basic`, `chain_of_thought`. Default: `basic`.

**reasoning_style** (str, optional): Style of reasoning for chain_of_thought type. One of: `freetext` (natural language reasoning), `agent` (structured for tool-calling). Default: None.

**sys_msg** (bool, optional): Whether to include system messages in the dataset. Default: True.

**rate_limit** (dict, optional): Rate limiting configuration. See [Rate Limiting Guide](../dataset-generation/rate-limiting.md) for details.

## DataSetGenerator Class

The DataSetGenerator class orchestrates the conversion from topics to training examples:

```python
from deepfabric import DataSetGenerator, Tree

# Create generator
generator = DataSetGenerator(
    instructions="Create detailed educational content",
    generation_system_prompt="You are an expert instructor",
    provider="openai",
    model_name="gpt-4",
    temperature=0.8
)

# Generate dataset from topic model
dataset = asyncio.run(generator.create_data_async(
    num_steps=100,
    batch_size=5,
    topic_model=tree,
    sys_msg=True
))
```

### Core Methods

#### create_data_async()

Primary coroutine for generating complete datasets (use `asyncio.run` or await within an event loop):

```python
dataset = asyncio.run(generator.create_data_async(
    num_steps=100,              # Total examples to generate
    batch_size=5,               # Examples per API call
    topic_model=topic_model,    # Tree or Graph instance
    model_name=None,            # Override model (optional)
    sys_msg=True                # Include system messages
))
```

**Parameters:**

- **num_steps** (int): Total number of training examples to generate
- **batch_size** (int): Number of examples processed in each API call
- **topic_model** (Tree | Graph): Source of topics for generation
- **model_name** (str, optional): Override the configured model
- **sys_msg** (bool): Include system prompts in output format

**Returns:** Dataset instance containing generated training examples

> **Note:** The synchronous `create_data()` wrapper remains available for convenience and calls `asyncio.run` internally. Use `create_data_async()` directly when composing within existing event loops.


#### create_batch()

Generate a single batch of examples for fine-grained control:

```python
batch = generator.create_batch(
    topics=selected_topics,
    batch_size=3,
    model_name="openai/gpt-4-turbo"
)
```

Enables custom topic selection and incremental dataset building.

> **Note:** The `create_batch()` method is not currently implemented. Use `create_data_async()` or `create_data_with_events_async()` for dataset generation.

### Conversation Types and Templates

The generator uses different conversation types to control the structure and format of generated content.

#### Available Conversation Types

Configure the conversation type during initialization:

```python
from deepfabric import DataSetGenerator

# Basic conversational format (default)
generator = DataSetGenerator(
    instructions="Create educational content",
    generation_system_prompt="You are an expert instructor",
    provider="openai",
    model_name="gpt-4",
    conversation_type="basic"  # Default
)

# Chain of Thought with reasoning
generator = DataSetGenerator(
    instructions="Create reasoning examples",
    generation_system_prompt="You are a reasoning expert",
    provider="openai",
    model_name="gpt-4",
    conversation_type="chain_of_thought",
    reasoning_style="freetext"  # freetext or agent
)
```

**Supported conversation types:**
- **basic**: Standard conversational format (question/answer pairs)
- **chain_of_thought**: Includes reasoning traces in responses

#### Reasoning Styles

For `chain_of_thought` conversation type, specify the reasoning style:

```python
generator = DataSetGenerator(
    instructions="Create problem solutions with reasoning",
    generation_system_prompt="You are an expert problem solver",
    provider="openai",
    model_name="gpt-4",
    conversation_type="chain_of_thought",
    reasoning_style="freetext"  # Natural language reasoning
)

# For agent/tool-calling scenarios
generator = DataSetGenerator(
    instructions="Create tool-calling examples",
    generation_system_prompt="You are an AI assistant with tools",
    provider="openai",
    model_name="gpt-4",
    conversation_type="chain_of_thought",
    reasoning_style="agent"  # Structured step-by-step for tool-calling
)
```

**Supported reasoning styles:**
- **freetext**: Natural language reasoning traces
- **agent**: Structured step-by-step reasoning for tool-calling scenarios

#### Customizing Content Generation

While you cannot set custom templates directly, you can control generation through configuration parameters:

**1. Instructions Parameter:**

Provide detailed guidance for content structure and style:

```python
generator = DataSetGenerator(
    instructions="""
    Create detailed explanations with:
    - Clear definitions
    - Practical code examples
    - Common pitfalls to avoid
    - Best practices for production use
    Target audience: intermediate developers
    """,
    generation_system_prompt="You are a senior software engineer",
    provider="openai",
    model_name="gpt-4"
)
```

**2. Generation System Prompt:**

Define the persona and behavior of the content generator:

```python
generator = DataSetGenerator(
    instructions="Create tutorials",
    generation_system_prompt="""
    You are an expert educator specializing in data science.
    Create comprehensive tutorials that balance theory and practice.
    Use clear examples and explain complex concepts in simple terms.
    """,
    provider="openai",
    model_name="gpt-4"
)
```

**3. Example Data for Few-Shot Learning:**

Provide examples to guide the generation style using a HuggingFace Dataset:

```python
from datasets import load_dataset

example_dataset = load_dataset("json", data_files="examples.jsonl")["train"]

generator = DataSetGenerator(
    instructions="Follow the style of the provided examples",
    generation_system_prompt="You are an expert instructor",
    provider="openai",
    model_name="gpt-4",
    example_data=example_dataset
)
```

### Quality Control and Monitoring

The generator includes built-in quality control and monitoring mechanisms:

#### Retry Configuration

Configure retry behavior during initialization:

```python
generator = DataSetGenerator(
    instructions="Create educational content",
    generation_system_prompt="You are an expert instructor",
    provider="openai",
    model_name="gpt-4",
    max_retries=5,  # Number of retry attempts
    request_timeout=60  # Timeout in seconds
)
```

#### Rate Limiting Configuration

Control API rate limits and retry behavior:

```python
generator = DataSetGenerator(
    instructions="Create educational content",
    generation_system_prompt="You are an expert instructor",
    provider="openai",
    model_name="gpt-4",
    rate_limit={
        "max_requests_per_minute": 50,
        "max_tokens_per_minute": 100000,
        "max_retries": 5,
        "initial_retry_delay": 1.0,
        "max_retry_delay": 60.0
    }
)
```

See the [Rate Limiting Guide](../dataset-generation/rate-limiting.md) for detailed configuration options.

#### Monitoring Failed Samples

Access failed samples and analyze failures:

```python
# Generate dataset
dataset = asyncio.run(generator.create_data_async(
    num_steps=100,
    batch_size=5,
    topic_model=tree
))

# Check for failures
if generator.failed_samples:
    print(f"Failed samples: {len(generator.failed_samples)}")

    # Get detailed failure analysis
    summary = generator.summarize_failures()
    print(f"Total failures: {summary['total_failures']}")
    print(f"Failure types: {summary['failure_types']}")

    # Print detailed summary
    generator.print_failure_summary()
```

### Advanced Usage

#### Multi-Provider Generation

Use different models for different types of content:

```python
# High-quality generator for complex topics
complex_generator = DataSetGenerator(
    instructions="Create advanced technical content",
    generation_system_prompt="You are an expert technical writer",
    provider="anthropic",
    model_name="claude-sonnet-4-5",
    temperature=0.7
)

# Faster generator for simple topics
simple_generator = DataSetGenerator(
    instructions="Create basic explanations",
    generation_system_prompt="You are a teacher for beginners",
    provider="openai",
    model_name="gpt-4-turbo",
    temperature=0.8
)

# Generate with different generators
complex_dataset = asyncio.run(complex_generator.create_data_async(
    num_steps=50,
    batch_size=5,
    topic_model=tree
))

simple_dataset = asyncio.run(simple_generator.create_data_async(
    num_steps=100,
    batch_size=10,
    topic_model=tree
))
```

#### Progress Monitoring with Events

Track generation progress in real-time:

```python
async def generate_with_progress():
    generator = DataSetGenerator(
        instructions="Create educational content",
        generation_system_prompt="You are an expert instructor",
        provider="openai",
        model_name="gpt-4"
    )

    async for event in generator.create_data_with_events_async(
        num_steps=100,
        batch_size=5,
        topic_model=tree
    ):
        if isinstance(event, dict):
            # Handle progress events
            if event["event"] == "generation_start":
                print(f"Starting generation: {event['total_samples']} samples")
            elif event["event"] == "step_complete":
                print(f"Step {event['step']}: {event['samples_generated']} samples")
                if event["failed_in_step"] > 0:
                    print(f"  Failures in step: {event['failed_in_step']}")
            elif event["event"] == "generation_complete":
                print(f"Complete: {event['total_samples']} generated, {event['failed_samples']} failed")
        else:
            # Final dataset
            dataset = event
            return dataset

dataset = asyncio.run(generate_with_progress())
```

#### Saving Datasets

Save generated datasets to disk:

```python
# Generate dataset
dataset = asyncio.run(generator.create_data_async(
    num_steps=100,
    batch_size=5,
    topic_model=tree
))

# Save to file
generator.save_dataset("training_data.jsonl")

# Or use the dataset directly
dataset.save("training_data.jsonl")
```

### Error Handling

Comprehensive error handling for robust operation:

```python
from deepfabric import DataSetGeneratorError

try:
    generator = DataSetGenerator(
        instructions="Create educational content",
        generation_system_prompt="You are an expert instructor",
        provider="openai",
        model_name="gpt-4"
    )
    dataset = asyncio.run(generator.create_data_async(
        topic_model=tree,
        num_steps=100,
        batch_size=5
    ))
except DataSetGeneratorError as e:
    print(f"Generation error: {e}")
    # Handle configuration or API errors
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Performance Optimization

Optimize generation performance through configuration parameters:

```python
# Optimize for throughput
generator = DataSetGenerator(
    instructions="Create educational content",
    generation_system_prompt="You are an expert instructor",
    provider="openai",
    model_name="gpt-4",
    temperature=0.8,
    default_batch_size=10,  # Larger batches
    request_timeout=60,     # Longer timeout
    max_retries=3           # Fewer retries
)

# Optimize for reliability
generator = DataSetGenerator(
    instructions="Create educational content",
    generation_system_prompt="You are an expert instructor",
    provider="openai",
    model_name="gpt-4",
    temperature=0.7,
    default_batch_size=3,   # Smaller batches
    request_timeout=120,    # Extended timeout
    max_retries=5,          # More retries
    rate_limit={
        "max_retries": 10,
        "initial_retry_delay": 2.0,
        "max_retry_delay": 120.0
    }
)
```