# Loading Datasets

DeepFabric provides a native `load_dataset` function for loading datasets from local files or DeepFabric Cloud. The API is designed to be familiar to HuggingFace users while avoiding external dependencies.

## From Local JSONL File

```python title="Load from local JSONL"
from deepfabric import load_dataset

dataset = load_dataset("dataset.jsonl")

print(f"Loaded {len(dataset)} samples")
print(f"Columns: {dataset.column_names}")
```

## From DeepFabric Cloud

Load datasets directly from DeepFabric Cloud using the `namespace/slug` format:

```python title="Load from DeepFabric Cloud"
from deepfabric import load_dataset

# Load from cloud (automatically cached to ~/.cache/deepfabric/datasets/)
dataset = load_dataset("your-username/my-dataset")

# Disable caching for fresh data
dataset = load_dataset("your-username/my-dataset", use_cache=False)
```

!!! tip "Cloud Caching"
    Cloud datasets are cached locally by default. The cache is stored in `~/.cache/deepfabric/datasets/`.

## From Text Files

Load text files with different sampling modes:

```python title="Load text files"
from deepfabric import load_dataset

# By line (default)
dataset = load_dataset("text", data_files="corpus.txt", sample_by="line")

# By paragraph (double newline separated)
dataset = load_dataset("text", data_files="corpus.txt", sample_by="paragraph")

# Entire document as one sample
dataset = load_dataset("text", data_files="corpus.txt", sample_by="document")
```

## Train/Validation/Test Split

Use the native `split()` method for reproducible train/test splits:

=== "Simple Split"

    ```python title="Two-way split"
    splits = dataset.split(test_size=0.1, seed=42)
    train_ds = splits["train"]
    eval_ds = splits["test"]
    ```

=== "Three-Way Split"

    ```python title="Train/val/test split (recommended)"
    # First split: 80% train, 20% temp
    train_temp = dataset.split(test_size=0.2, seed=42)
    train_ds = train_temp["train"]  # 80% for training

    # Second split: 50/50 of the 20% temp
    val_test = train_temp["test"].split(test_size=0.5, seed=42)
    val_ds = val_test["train"]   # 10% for validation during training
    test_ds = val_test["test"]   # 10% for final evaluation (hold out!)
    ```

!!! tip "Three-Way Split Recommended"
    For tool-calling evaluation, use a three-way split to have a held-out test set for final evaluation.

## Accessing Fields

```python title="Access sample fields"
# Column access - get all values for a field
messages = dataset["messages"]  # List of all message arrays

# Index access - get single sample
sample = dataset[0]  # Dict with all fields

# Slice access - get subset as new Dataset
subset = dataset[0:10]

# Iteration
for sample in dataset:
    messages = sample["messages"]
    reasoning = sample.get("reasoning")
    tools = sample.get("tools")

    # Messages structure
    for msg in messages:
        role = msg["role"]      # system, user, assistant, tool
        content = msg["content"]
        tool_calls = msg.get("tool_calls")
```

## Transformations

```python title="Map and filter"
# Transform samples
def add_length(sample):
    sample["length"] = len(sample["messages"])
    return sample

transformed = dataset.map(add_length)

# Filter samples
with_tools = dataset.filter(lambda x: x.get("tools") is not None)

# Filter by reasoning style
agent_samples = dataset.filter(
    lambda x: x.get("reasoning", {}).get("style") == "agent"
)
```

## Shuffling

```python title="Shuffle dataset"
shuffled = dataset.shuffle(seed=42)
```

## Integration with TRL

DeepFabric datasets integrate seamlessly with HuggingFace TRL for training:

```python title="Use with TRL"
from deepfabric import load_dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Load with DeepFabric
dataset = load_dataset("your-username/my-dataset")
splits = dataset.split(test_size=0.1, seed=42)

# Format with tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def format_sample(sample):
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tools=sample.get("tools"),
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_formatted = splits["train"].map(format_sample)

# Convert to HuggingFace Dataset for TRL
train_hf = train_formatted.to_hf()

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_hf,
    args=SFTConfig(output_dir="./output"),
)
trainer.train()
```

## Optimizing Tool-Calling Datasets

!!! warning "Large Sequence Lengths"
    Tool-calling datasets can have large sequence lengths due to tool schemas.

Use `prepare_dataset_for_training` to reduce overhead:

```python title="Optimize tool datasets"
from deepfabric import load_dataset
from deepfabric.training import prepare_dataset_for_training

dataset = load_dataset("your-username/tool-dataset")

# Filter to only tools actually used in each conversation
prepared = prepare_dataset_for_training(
    dataset,
    tool_strategy="used_only",
    clean_tool_schemas=True,
)
```

See [Dataset Preparation](dataset-preparation.md) for details.

## Saving Datasets

```python title="Save to JSONL"
dataset.to_jsonl("output.jsonl")
```

## Inspection

```python title="Inspect dataset"
# Dataset info
print(dataset)
# Dataset(num_rows=1000, columns=['messages', 'reasoning', 'tools', ...])

# First sample
print(dataset[0])

# Column names
print(dataset.column_names)
# ['messages', 'reasoning', 'tools', 'metadata', ...]

# Number of samples
print(len(dataset))
```

---

## Alternative: HuggingFace Datasets

If you prefer to use HuggingFace datasets directly, DeepFabric outputs are fully compatible:

```python title="Load with HuggingFace datasets"
from datasets import load_dataset

# From local file
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# From HuggingFace Hub
dataset = load_dataset("your-username/my-dataset", split="train")

# Split with HuggingFace
splits = dataset.train_test_split(test_size=0.1, seed=42)
```

### Upload to HuggingFace Hub

```bash title="Upload to HuggingFace"
deepfabric upload-hf dataset.jsonl --repo your-username/my-dataset
```

Then load from the Hub:

```python title="Load from Hub"
from datasets import load_dataset

dataset = load_dataset("your-username/my-dataset", split="train")
```

### Combining Datasets with HuggingFace

```python title="Combine datasets"
from datasets import load_dataset, concatenate_datasets

basic = load_dataset("user/basic-ds", split="train")
agent = load_dataset("user/agent-ds", split="train")

combined = concatenate_datasets([basic, agent])
combined = combined.shuffle(seed=42)
```

### Streaming Large Datasets

For datasets that don't fit in memory, use HuggingFace's streaming:

```python title="Stream large datasets"
from datasets import load_dataset

dataset = load_dataset(
    "your-username/large-dataset",
    split="train",
    streaming=True
)

for sample in dataset:
    process(sample)
```
