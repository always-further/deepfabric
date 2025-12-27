# DataSetGenerator API

The DataSetGenerator class transforms topic structures into practical training examples through configurable templates, quality control mechanisms, and batch processing.

```mermaid
graph LR
    A[Topic Model] --> B[DataSetGenerator]
    B --> C[Batch Processing]
    C --> D[Quality Control]
    D --> E[Dataset]
```

## DataSetGenerator Configuration

Dataset generation configuration is passed directly to the DataSetGenerator constructor:

```python title="Basic configuration"
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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instructions` | str | `""` | Core guidance for content generation |
| `generation_system_prompt` | str | required | System prompt for the generation model |
| `provider` | str | required | LLM provider name (`openai`, `anthropic`, etc.) |
| `model_name` | str | required | Model name specific to the provider |
| `temperature` | float | `0.7` | Controls creativity (0.0-2.0) |
| `max_retries` | int | `3` | Retry attempts for failed requests |
| `request_timeout` | int | `30` | Maximum seconds to wait for API responses |
| `default_batch_size` | int | `5` | Examples to generate per API call |
| `default_num_examples` | int | `3` | Example demonstrations to include |
| `conversation_type` | str | `"basic"` | Format type: `basic` or `chain_of_thought` |
| `reasoning_style` | str | `None` | For chain_of_thought: `freetext` or `agent` |
| `sys_msg` | bool | `True` | Include system messages in dataset |
| `rate_limit` | dict | `None` | Rate limiting configuration |

!!! info "Rate Limiting"
    See [Rate Limiting Guide](../dataset-generation/rate-limiting.md) for detailed configuration options.

## DataSetGenerator Class

The DataSetGenerator class orchestrates the conversion from topics to training examples.

```python title="Basic usage"
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

Primary coroutine for generating complete datasets:

```python title="Generate dataset"
dataset = asyncio.run(generator.create_data_async(
    num_steps=100,              # Total examples to generate
    batch_size=5,               # Examples per API call
    topic_model=topic_model,    # Tree or Graph instance
    model_name=None,            # Override model (optional)
    sys_msg=True                # Include system messages
))
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_steps` | int | Total number of training examples to generate |
| `batch_size` | int | Number of examples processed in each API call |
| `topic_model` | Tree \| Graph | Source of topics for generation |
| `model_name` | str | Optional model override |
| `sys_msg` | bool | Include system prompts in output format |

**Returns:** Dataset instance containing generated training examples

!!! note "Sync Wrapper"
    The synchronous `create_data()` wrapper remains available for convenience. Use `create_data_async()` directly when composing within existing event loops.

#### create_batch()

!!! warning "Not Implemented"
    The `create_batch()` method is not currently implemented. Use `create_data_async()` or `create_data_with_events_async()` for dataset generation.

### Conversation Types and Templates

The generator uses different conversation types to control the structure and format of generated content.

#### Available Conversation Types

=== "Basic Format"

    Standard conversational format (question/answer pairs):

    ```python title="Basic conversation"
    generator = DataSetGenerator(
        instructions="Create educational content",
        generation_system_prompt="You are an expert instructor",
        provider="openai",
        model_name="gpt-4",
        conversation_type="basic"  # Default
    )
    ```

=== "Chain of Thought"

    Includes reasoning traces in responses:

    ```python title="Chain of thought"
    generator = DataSetGenerator(
        instructions="Create reasoning examples",
        generation_system_prompt="You are a reasoning expert",
        provider="openai",
        model_name="gpt-4",
        conversation_type="chain_of_thought",
        reasoning_style="freetext"  # freetext or agent
    )
    ```

#### Reasoning Styles

For `chain_of_thought` conversation type, specify the reasoning style:

=== "Freetext"

    Natural language reasoning traces:

    ```python title="Freetext reasoning"
    generator = DataSetGenerator(
        instructions="Create problem solutions with reasoning",
        generation_system_prompt="You are an expert problem solver",
        provider="openai",
        model_name="gpt-4",
        conversation_type="chain_of_thought",
        reasoning_style="freetext"
    )
    ```

=== "Agent"

    Structured step-by-step reasoning for tool-calling:

    ```python title="Agent reasoning"
    generator = DataSetGenerator(
        instructions="Create tool-calling examples",
        generation_system_prompt="You are an AI assistant with tools",
        provider="openai",
        model_name="gpt-4",
        conversation_type="chain_of_thought",
        reasoning_style="agent"
    )
    ```

#### Customizing Content Generation

Control generation through configuration parameters:

??? example "1. Instructions Parameter"

    Provide detailed guidance for content structure and style:

    ```python title="Detailed instructions"
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

??? example "2. Generation System Prompt"

    Define the persona and behavior of the content generator:

    ```python title="Custom persona"
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

??? example "3. Example Data for Few-Shot Learning"

    Provide examples to guide the generation style using a HuggingFace Dataset:

    ```python title="Few-shot learning"
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

The generator includes built-in quality control and monitoring mechanisms.

#### Retry Configuration

```python title="Retry settings"
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

```python title="Rate limiting"
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

!!! info "More Details"
    See the [Rate Limiting Guide](../dataset-generation/rate-limiting.md) for detailed configuration options.

#### Monitoring Failed Samples

```python title="Monitor failures"
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

=== "High-Quality Generator"

    ```python title="Complex topics"
    complex_generator = DataSetGenerator(
        instructions="Create advanced technical content",
        generation_system_prompt="You are an expert technical writer",
        provider="anthropic",
        model_name="claude-sonnet-4-5",
        temperature=0.7
    )

    complex_dataset = asyncio.run(complex_generator.create_data_async(
        num_steps=50,
        batch_size=5,
        topic_model=tree
    ))
    ```

=== "Fast Generator"

    ```python title="Simple topics"
    simple_generator = DataSetGenerator(
        instructions="Create basic explanations",
        generation_system_prompt="You are a teacher for beginners",
        provider="openai",
        model_name="gpt-4-turbo",
        temperature=0.8
    )

    simple_dataset = asyncio.run(simple_generator.create_data_async(
        num_steps=100,
        batch_size=10,
        topic_model=tree
    ))
    ```

#### Progress Monitoring with Events

Track generation progress in real-time:

```python title="Progress monitoring"
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

```python title="Save dataset"
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

```python title="Exception handling"
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
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Performance Optimization

=== "Optimize for Throughput"

    ```python title="High throughput"
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
    ```

=== "Optimize for Reliability"

    ```python title="High reliability"
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