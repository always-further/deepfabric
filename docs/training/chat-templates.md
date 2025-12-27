# Chat Templates

Chat templates convert message arrays into the format each model expects. Using the correct template is critical for training quality.

## Basic Usage

```python title="Apply chat template"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
```

## Formatting Datasets

```python title="Format dataset"
def format_sample(example):
    # Clean messages - remove None values
    messages = [
        {k: v for k, v in msg.items() if v is not None}
        for msg in example["messages"]
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

formatted = dataset.map(format_sample)
```

## Tool Calling

Models handle tool definitions differently.

=== "Qwen"

    Qwen expects tools as a parameter:

    ```python title="Qwen tool calling"
    text = tokenizer.apply_chat_template(
        messages,
        tools=example.get("tools"),
        tokenize=False,
        add_generation_prompt=False
    )
    ```

=== "Llama / Mistral"

    Some models expect tools in the system message. Check your model's documentation.

### Handling tool_calls

Assistant messages with `tool_calls` need the nested structure:

```python title="Tool call structure"
messages = [
    {"role": "user", "content": "Read config.json"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"file_path": "config.json"}'
                }
            }
        ]
    },
    {"role": "tool", "content": '{"debug": true}', "tool_call_id": "call_0"},
    {"role": "assistant", "content": "The config has debug enabled."}
]
```

## Including Reasoning

DeepFabric stores reasoning separately. To include it in training:

```python title="Include reasoning traces"
def format_with_reasoning(example):
    messages = example["messages"].copy()
    reasoning = example.get("reasoning")

    if reasoning and reasoning.get("content"):
        # Prepend reasoning to first assistant message
        for msg in messages:
            if msg["role"] == "assistant" and msg.get("content"):
                thinking = reasoning["content"]
                if isinstance(thinking, list):
                    thinking = "\n".join(
                        f"Step {s['step_number']}: {s['thought']}"
                        for s in thinking
                    )
                msg["content"] = f"<thinking>{thinking}</thinking>\n{msg['content']}"
                break

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}
```

## Model-Specific Examples

=== "Qwen 2.5"

    ```python title="Qwen 2.5"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # Template: <|im_start|>system\n{content}<|im_end|>\n...
    ```

=== "Llama 3"

    ```python title="Llama 3"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    # Template: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...
    ```

=== "Mistral"

    ```python title="Mistral"
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        fix_mistral_regex=True  # Required for correct tokenization
    )
    # Template: [INST] {content} [/INST]
    ```

    !!! warning "Mistral Regex Fix"
        The `fix_mistral_regex=True` parameter is required for Mistral models to handle the chat template regex correctly.

## Debugging Templates

Print the template to understand the format:

```python title="Debug template"
print(tokenizer.chat_template)
```

Test with a simple example:

```python title="Test template"
test_messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
]
print(tokenizer.apply_chat_template(test_messages, tokenize=False))
```
