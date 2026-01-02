# Running Evaluation

Configure and run model evaluation on tool-calling datasets.

## Configuration

```python title="Evaluator configuration"
from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

config = EvaluatorConfig(
    # Model configuration
    inference_config=InferenceConfig(
        model="./output/checkpoint-final",  # Local path, HuggingFace ID, or model name
        backend="transformers",              # transformers, ollama, or llm
        temperature=0.7,
        max_tokens=2048,
    ),

    # Evaluation settings
    batch_size=1,
    max_samples=100,           # Limit samples (None for all)
    save_predictions=True,
    output_path="eval_results.json",
)
```

## Basic Usage

=== "From DeepFabric Dataset"

    ```python title="Load with DeepFabric"
    from deepfabric import load_dataset
    from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

    # Load and split dataset
    dataset = load_dataset("your-username/my-dataset")
    splits = dataset.split(test_size=0.1, seed=42)
    eval_ds = splits["test"]

    # Configure and run
    config = EvaluatorConfig(
        inference_config=InferenceConfig(
            model="./fine-tuned-model",
            backend="transformers",
        ),
    )

    evaluator = Evaluator(config)
    results = evaluator.evaluate(dataset=eval_ds)
    ```

=== "From HuggingFace Dataset"

    ```python title="Load from HuggingFace"
    from datasets import load_dataset
    from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

    # Load eval split
    dataset = load_dataset("your-username/my-dataset", split="train")
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    eval_ds = splits["test"]

    # Configure and run
    config = EvaluatorConfig(
        inference_config=InferenceConfig(
            model="./fine-tuned-model",
            backend="transformers",
        ),
    )

    evaluator = Evaluator(config)
    results = evaluator.evaluate(dataset=eval_ds)
    ```

=== "From JSONL File"

    ```python title="Load from file"
    config = EvaluatorConfig(
        dataset_path="eval_dataset.jsonl",
        inference_config=InferenceConfig(
            model="./fine-tuned-model",
            backend="transformers",
        ),
    )

    evaluator = Evaluator(config)
    results = evaluator.evaluate()
    ```

## Inference Backends

=== "Transformers"

    Local inference with HuggingFace transformers:

    ```python title="Transformers backend"
    InferenceConfig(
        model="./fine-tuned-model",
        backend="transformers",
        device=None,              # Auto-detect (cuda, mps, cpu)
        temperature=0.7,
        max_tokens=2048,
    )
    ```

=== "With LoRA Adapter"

    ```python title="LoRA adapter"
    InferenceConfig(
        model="Qwen/Qwen2.5-7B-Instruct",
        adapter_path="./lora-adapter",
        backend="transformers",
    )
    ```

=== "With Unsloth"

    For adapters trained with Unsloth, we automatically sense the adapter format:

    ```python title="Unsloth backend"
    InferenceConfig(
        model="unsloth/Qwen2.5-7B-Instruct",
        adapter_path="./unsloth-adapter",
        backend="transformers",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    ```

=== "Ollama"

    For models served via Ollama:

    ```python title="Ollama backend"
    InferenceConfig(
        model="qwen2.5:7b",
        backend="ollama",
    )
    ```

### Using In-Memory Models

!!! tip "Avoid OOM and Speed Up Evaluation"
    After training, you can pass the model directly to the evaluator without saving and reloading from disk.

```python title="In-memory evaluation"
from unsloth import FastLanguageModel
from trl import SFTTrainer
from deepfabric import load_dataset
from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

# Load and split dataset
dataset = load_dataset("your-username/my-dataset")
splits = dataset.split(test_size=0.1, seed=42)
eval_ds = splits["test"]

# Train your model
trainer = SFTTrainer(model=model, tokenizer=tokenizer, ...)
trainer.train()

# Prepare for inference (important for Unsloth models)
FastLanguageModel.for_inference(model)

# Pass the in-memory model directly - no reloading needed
config = EvaluatorConfig(
    inference_config=InferenceConfig(
        model=model,            # Pass model object directly
        tokenizer=tokenizer,    # Required when using in-memory model
        temperature=0.1,
        max_tokens=512,
    ),
    max_samples=100,
    save_predictions=True,
    output_path="eval_results.json",
)

evaluator = Evaluator(config)
results = evaluator.evaluate(dataset=eval_ds)
```

This approach:

- **Avoids OOM errors** by not loading a second copy of the model
- **Saves time** by skipping disk I/O for model weights
- **Enables rapid iteration** in notebook environments like Colab

!!! warning "Required: Tokenizer"
    The `tokenizer` parameter is required when passing an in-memory model object.

## Cloud LLM Providers

Evaluate against cloud LLM APIs with native tool calling support:

=== "OpenAI"

    ```python title="OpenAI"
    InferenceConfig(
        model="gpt-4o",
        backend="llm",
        provider="openai",
        # api_key="sk-...",  # Or set OPENAI_API_KEY env var
    )
    ```

=== "Anthropic"

    ```python title="Anthropic"
    InferenceConfig(
        model="claude-3-5-sonnet-20241022",
        backend="llm",
        provider="anthropic",
        # api_key="...",  # Or set ANTHROPIC_API_KEY env var
    )
    ```

=== "Google Gemini"

    ```python title="Gemini"
    InferenceConfig(
        model="gemini-2.0-flash",
        backend="llm",
        provider="gemini",
        # api_key="...",  # Or set GOOGLE_API_KEY or GEMINI_API_KEY env var
    )
    ```

=== "OpenRouter"

    ```python title="OpenRouter"
    InferenceConfig(
        model="openai/gpt-4o",  # Or any OpenRouter model ID
        backend="llm",
        provider="openrouter",
        # api_key="...",  # Or set OPENROUTER_API_KEY env var
    )
    ```

### Rate Limiting

Cloud backends include built-in retry with exponential backoff:

```python title="Rate limiting config"
InferenceConfig(
    model="gpt-4o",
    backend="llm",
    provider="openai",
    rate_limit_config={
        "max_retries": 5,
        "base_delay": 1.0,
        "max_delay": 60.0,
    },
)
```

## Results

The `evaluate()` method returns an `EvaluationResult`:

```python title="Access results"
results = evaluator.evaluate(dataset=eval_ds)

# Aggregate metrics
print(results.metrics.tool_selection_accuracy)
print(results.metrics.parameter_accuracy)
print(results.metrics.overall_score)
print(results.metrics.samples_evaluated)
print(results.metrics.processing_errors)

# Individual predictions
for pred in results.predictions:
    print(f"Sample {pred.sample_id}:")
    print(f"  Expected: {pred.expected_tool}")
    print(f"  Predicted: {pred.predicted_tool}")
    print(f"  Correct: {pred.tool_selection_correct}")
```

## Saving Results

Results are saved automatically when `save_predictions=True`:

```python title="Save configuration"
config = EvaluatorConfig(
    ...,
    save_predictions=True,
    output_path="eval_results.json",
)
```

The output file contains:

- Configuration used
- All metrics
- Per-sample predictions and scores

## Print Summary

```python title="Print summary"
evaluator.print_summary(results.metrics)
```

??? example "Example Output"

    ```
    Evaluation Summary
    Samples Evaluated: 100
    Processed Successfully: 98
    Processing Errors: 2

    Metrics
    Tool Selection Accuracy: 85.00%
    Parameter Accuracy: 78.00%
    Execution Success Rate: 82.00%
    Response Quality: 0.00%

    Overall Score: 81.15%
    ```
!!! note "Response Quality Metric"
    The Response Quality metric is not used for tool evaluation and will always be 0%, it is a placeholder for future use.
