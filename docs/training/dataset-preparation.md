# Dataset Preparation

DeepFabric provides utilities for optimizing datasets before training. These optimizations can significantly reduce sequence lengths and memory usage, especially for tool-calling datasets.

## The Tool Overhead Problem

Tool-calling datasets often include all available tool definitions in every sample, even if only a few tools are actually used. This leads to:

- **Large sequence lengths** - Tool schemas can add thousands of tokens per sample
- **Memory issues** - Long sequences require more GPU memory (scales with sequence_length^2)
- **Slower training** - More tokens to process per sample

!!! example "Real Example"
    A dataset with 21 tools might have:

    - ~22,500 characters of tool JSON per sample
    - Average sequence length of 7,000+ tokens
    - Only 1-3 tools actually used per conversation

## Using prepare_dataset_for_training

The `prepare_dataset_for_training` function optimizes your dataset:

```python title="Prepare dataset"
from deepfabric import load_dataset
from deepfabric.training import prepare_dataset_for_training

# Load dataset
dataset = load_dataset("your-username/dataset")

# Prepare with optimizations
prepared = prepare_dataset_for_training(
    dataset,
    tool_strategy="used_only",  # Only include tools actually called
    clean_tool_schemas=True,    # Remove null values from schemas
    num_proc=16,                # Parallel processing
)

# Check the reduction
print(f"Samples: {len(prepared)}")
```

### Tool Inclusion Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `"used_only"` | Only tools called in the conversation | Best for memory efficiency |
| `"all"` | Keep all tools (no filtering) | When model needs to see full catalog |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tool_strategy` | `"used_only"` | How to filter tools |
| `clean_tool_schemas` | `True` | Remove null values from schemas |
| `min_tools` | `1` | Minimum tools to keep per sample |
| `num_proc` | - | Number of processes for parallel processing |

## Complete Training Pipeline

```python title="Full training pipeline"
from deepfabric import load_dataset
from deepfabric.training import prepare_dataset_for_training
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

# 1. Load and prepare dataset
dataset = load_dataset("your-username/tool-calling-dataset")
prepared = prepare_dataset_for_training(dataset, tool_strategy="used_only")

# 2. Split into train/val/test using native split()
train_temp = prepared.split(test_size=0.2, seed=42)
train_ds = train_temp["train"]

val_test = train_temp["test"].split(test_size=0.5, seed=42)
val_ds = val_test["train"]
test_ds = val_test["test"]  # Hold out for final evaluation

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# 3. Format with chat template
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def format_sample(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tools=example.get("tools"),
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_formatted = train_ds.map(format_sample)
val_formatted = val_ds.map(format_sample)

# 4. Convert to HuggingFace Dataset for TRL
train_hf = train_formatted.to_hf()
val_hf = val_formatted.to_hf()

# 5. Check sequence lengths
def get_length(example):
    return {"length": len(tokenizer(example["text"])["input_ids"])}

lengths = train_hf.map(get_length)
print(f"Max length: {max(lengths['length'])}")
print(f"Mean length: {sum(lengths['length'])/len(lengths['length']):.0f}")

# 6. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_hf,
    eval_dataset=val_hf,
    args=SFTConfig(
        output_dir="./output",
        max_seq_length=4096,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=100,
    ),
)
trainer.train()
```

## Low-Level Utilities

For more control, we provide the following low-level functions, to help you refine datasets as needed. This is especially useful for memory optimization.

### filter_tools_for_sample

Filter tools in a single sample:

```python title="Filter single sample"
from deepfabric.training import filter_tools_for_sample

sample = dataset[0]
filtered = filter_tools_for_sample(
    sample,
    strategy="used_only",
    min_tools=1,
    clean_schemas=True,
)
print(f"Tools: {len(sample['tools'])} -> {len(filtered['tools'])}")
```

### get_used_tool_names

Extract tool names that are actually called:

```python title="Get used tools"
from deepfabric.training import get_used_tool_names

messages = sample["messages"]
used = get_used_tool_names(messages)
print(f"Tools used: {used}")
# {'get_file_content', 'list_directory'}
```

### clean_tool_schema

Remove null values from tool schemas:

```python title="Clean schema"
from deepfabric.training import clean_tool_schema

tool = sample["tools"][0]
cleaned = clean_tool_schema(tool)
# Removes all None/null values recursively
```

## Memory Optimization Tips

If you're getting CUDA out-of-memory errors during training, consider these strategies:

!!! tip "If You're Still Running Out of Memory"

    === "Reduce Sequence Length"

        ```python
        args=SFTConfig(max_seq_length=2048)
        ```

    === "Filter Long Samples"

        ```python
        # Using DeepFabric Dataset
        short_samples = prepared.filter(lambda x: len(x["text"]) < 4096)
        ```

    === "Smaller Batches"

        ```python
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
        )
        ```

    === "Gradient Checkpointing"

        ```python
        args=SFTConfig(
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        ```

---

## Alternative: HuggingFace Datasets

If you prefer to use HuggingFace datasets directly, the preparation utilities work with both:

```python title="With HuggingFace datasets"
from datasets import load_dataset
from deepfabric.training import prepare_dataset_for_training

# Load with HuggingFace
dataset = load_dataset("your-username/dataset", split="train")

# Prepare works the same way
prepared = prepare_dataset_for_training(dataset, tool_strategy="used_only")

# Split with HuggingFace method
train_temp = prepared.train_test_split(test_size=0.2, seed=42)
train_ds = train_temp["train"]

val_test = train_temp["test"].train_test_split(test_size=0.5, seed=42)
val_ds = val_test["train"]
test_ds = val_test["test"]
```
