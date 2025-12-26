# Loading Datasets

DeepFabric outputs standard JSONL files compatible with HuggingFace datasets.

## From Local File

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
```

## From HuggingFace Hub

Upload first:

```bash
deepfabric upload-hf dataset.jsonl --repo your-username/my-dataset
```

Then load:

```python
dataset = load_dataset("your-username/my-dataset", split="train")
```

## Train/Validation/Test Split

For proper evaluation, split into three sets:

```python
# Two-way split (simple)
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
eval_ds = splits["test"]

# Three-way split (recommended for tool-calling evaluation)
train_temp = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = train_temp["train"]  # 80% for training

val_test = train_temp["test"].train_test_split(test_size=0.5, seed=42)
val_ds = val_test["train"]   # 10% for validation during training
test_ds = val_test["test"]   # 10% for final evaluation (hold out!)
```

## Accessing Fields

```python
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

## Filtering

Keep only samples with tool calls:

```python
with_tools = dataset.filter(lambda x: x.get("tools") is not None)
```

Filter by reasoning style:

```python
agent_samples = dataset.filter(
    lambda x: x.get("reasoning", {}).get("style") == "agent"
)
```

## Optimizing Tool-Calling Datasets

Tool-calling datasets can have large sequence lengths due to tool schemas. Use `prepare_dataset_for_training` to reduce overhead:

```python
from deepfabric.training import prepare_dataset_for_training

# Filter to only tools actually used in each conversation
prepared = prepare_dataset_for_training(
    dataset,
    tool_strategy="used_only",
    clean_tool_schemas=True,
)
```

See [Dataset Preparation](dataset-preparation.md) for details.

## Shuffling

```python
shuffled = dataset.shuffle(seed=42)
```

## Streaming Large Datasets

For datasets that don't fit in memory:

```python
dataset = load_dataset(
    "your-username/large-dataset",
    split="train",
    streaming=True
)

for sample in dataset:
    # Process one at a time
    process(sample)
```

## Combining Datasets

Merge multiple DeepFabric datasets:

```python
from datasets import concatenate_datasets

basic = load_dataset("user/basic-ds", split="train")
agent = load_dataset("user/agent-ds", split="train")

combined = concatenate_datasets([basic, agent])
combined = combined.shuffle(seed=42)
```

## Inspection

```python
# Dataset info
print(dataset)
# Dataset({features: ['messages', 'reasoning', 'tools', ...], num_rows: 1000})

# First sample
print(dataset[0])

# Column names
print(dataset.column_names)
# ['messages', 'reasoning', 'tools', 'metadata', ...]
```
