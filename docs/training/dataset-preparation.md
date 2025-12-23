# Dataset Preparation

DeepFabric provides utilities for optimizing datasets before training. These optimizations can significantly reduce sequence lengths and memory usage, especially for tool-calling datasets.

## The Tool Overhead Problem

Tool-calling datasets often include all available tool definitions in every sample, even if only a few tools are actually used. This leads to:

- **Large sequence lengths** - Tool schemas can add thousands of tokens per sample
- **Memory issues** - Long sequences require more GPU memory (scales with sequence_length^2)
- **Slower training** - More tokens to process per sample

For example, a dataset with 21 tools might have:
- ~22,500 characters of tool JSON per sample
- Average sequence length of 7,000+ tokens
- Only 1-3 tools actually used per conversation

## Using prepare_dataset_for_training

The `prepare_dataset_for_training` function optimizes your dataset:

```python
from datasets import load_dataset
from deepfabric.training import prepare_dataset_for_training

# Load dataset
dataset = load_dataset("your/dataset", split="train")

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

- `tool_strategy` - How to filter tools (default: `"used_only"`)
- `clean_tool_schemas` - Remove null values from schemas (default: `True`)
- `min_tools` - Minimum tools to keep per sample (default: `1`)
- `num_proc` - Number of processes for parallel processing

## Complete Training Pipeline

```python
from datasets import load_dataset
from deepfabric.training import prepare_dataset_for_training
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

# 1. Load and prepare dataset
dataset = load_dataset("your/tool-calling-dataset", split="train")
prepared = prepare_dataset_for_training(dataset, tool_strategy="used_only")

# 2. Split into train/val/test
train_temp = prepared.train_test_split(test_size=0.2, seed=42)
train_ds = train_temp["train"]

val_test = train_temp["test"].train_test_split(test_size=0.5, seed=42)
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

# 4. Check sequence lengths
def get_length(example):
    return {"length": len(tokenizer(example["text"])["input_ids"])}

lengths = train_formatted.map(get_length)
print(f"Max length: {max(lengths['length'])}")
print(f"Mean length: {sum(lengths['length'])/len(lengths['length']):.0f}")

# 5. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_formatted,
    eval_dataset=val_formatted,
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

For more control, use the individual functions:

### filter_tools_for_sample

Filter tools in a single sample:

```python
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

```python
from deepfabric.training import get_used_tool_names

messages = sample["messages"]
used = get_used_tool_names(messages)
print(f"Tools used: {used}")
# {'get_file_content', 'list_directory'}
```

### clean_tool_schema

Remove null values from tool schemas:

```python
from deepfabric.training import clean_tool_schema

tool = sample["tools"][0]
cleaned = clean_tool_schema(tool)
# Removes all None/null values recursively
```

## Memory Optimization Tips

If you're still running out of memory after filtering tools:

1. **Reduce max sequence length**
   ```python
   args=SFTConfig(max_seq_length=2048)
   ```

2. **Filter long samples**
   ```python
   prepared = prepared.filter(lambda x: len(x["text"]) < 4096)
   ```

3. **Reduce batch size, increase gradient accumulation**
   ```python
   args=SFTConfig(
       per_device_train_batch_size=1,
       gradient_accumulation_steps=8,
   )
   ```

4. **Enable gradient checkpointing**
   ```python
   args=SFTConfig(
       gradient_checkpointing=True,
       gradient_checkpointing_kwargs={"use_reentrant": False},
   )
   ```
