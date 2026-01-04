# Training Frameworks

Integration patterns for TRL and Unsloth, including training configuration, callbacks, and trainer control.

## TRL (Transformers Reinforcement Learning)

### Basic SFT

```python title="Basic SFT training"
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Load and format dataset
dataset = load_dataset("your-username/my-dataset", split="train")

def format_sample(example):
    messages = [{k: v for k, v in m.items() if v is not None}
                for m in example["messages"]]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

formatted = dataset.map(format_sample)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted,
    args=SFTConfig(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
    ),
)
trainer.train()
```

### With Tool Calling

Include tools in the chat template:

```python title="Tool calling format"
def format_with_tools(example):
    messages = [{k: v for k, v in m.items() if v is not None}
                for m in example["messages"]]
    tools = example.get("tools")

    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}
```

## Unsloth

Unsloth provides faster training with lower memory usage.

### Basic Setup

```python title="Unsloth setup"
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

### Training

```python title="Unsloth training"
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted,
    args=SFTConfig(
        output_dir="./output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
    ),
)
trainer.train()
```

### Saving

```python title="Save models"
# Save LoRA adapter
model.save_pretrained("./lora-adapter")

# Merge and save full model
model.save_pretrained_merged("./merged-model", tokenizer)
```

## SFTConfig Reference

`SFTConfig` (from TRL) extends HuggingFace's `TrainingArguments` with SFT-specific options.

### Core Training Parameters

```python title="Core parameters"
from trl import SFTConfig

config = SFTConfig(
    # Output
    output_dir="./output",              # Where to save checkpoints

    # Training duration
    num_train_epochs=3,                 # Number of epochs (use this OR max_steps)
    max_steps=-1,                       # Max training steps (-1 = use epochs)

    # Batch size
    per_device_train_batch_size=4,      # Batch size per GPU
    per_device_eval_batch_size=8,       # Eval batch size per GPU
    gradient_accumulation_steps=4,      # Effective batch = batch_size * accumulation * num_gpus

    # Learning rate
    learning_rate=2e-5,                 # Initial learning rate
    lr_scheduler_type="cosine",         # linear, cosine, constant, etc.
    warmup_steps=100,                   # Steps before reaching full LR
    warmup_ratio=0.0,                   # Alternative: warmup as ratio of total steps

    # Optimizer
    optim="adamw_torch",                # adamw_torch, adamw_8bit, paged_adamw_8bit
    weight_decay=0.01,                  # L2 regularization
    max_grad_norm=1.0,                  # Gradient clipping

    # Precision
    fp16=False,                         # Use FP16 mixed precision
    bf16=True,                          # Use BF16 mixed precision (preferred if supported)

    # Logging
    logging_steps=10,                   # Log metrics every N steps
    report_to="none",                   # tensorboard, wandb, or none

    # Checkpointing
    save_steps=500,                     # Save checkpoint every N steps
    save_total_limit=3,                 # Keep only N most recent checkpoints

    # Evaluation
    eval_strategy="steps",              # steps, epoch, or no
    eval_steps=100,                     # Evaluate every N steps

    # Memory optimization
    gradient_checkpointing=True,        # Trade compute for memory
)
```

### SFT-Specific Parameters

```python title="SFT-specific parameters"
config = SFTConfig(
    ...,
    # Sequence length
    max_seq_length=2048,                # Maximum sequence length

    # Packing (combine short samples into one sequence)
    packing=False,                      # Enable sequence packing

    # Dataset formatting
    dataset_text_field="text",          # Field containing formatted text
)
```

### Memory-Efficient Configurations

=== "24GB GPU (RTX 3090/4090)"

    ```python title="24GB config"
    config = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        bf16=True,
        optim="adamw_8bit",
    )
    ```

=== "16GB GPU (RTX 4080/A4000)"

    ```python title="16GB config"
    config = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        bf16=True,
        optim="paged_adamw_8bit",
    )
    ```

## TrainerState and TrainerControl

The HuggingFace Trainer uses `TrainerState` and `TrainerControl` to manage training flow.

### TrainerState

`TrainerState` contains the current training state, passed to all callbacks:

```python title="TrainerState fields"
state.global_step      # Current training step
state.epoch            # Current epoch (float, e.g., 1.5)
state.max_steps        # Total training steps
state.num_train_epochs # Total epochs
state.log_history      # List of logged metrics
state.best_metric      # Best evaluation metric
state.best_model_checkpoint  # Path to best checkpoint
state.is_world_process_zero  # True on main process (for distributed)
```

### TrainerControl

`TrainerControl` lets callbacks control training behavior:

```python title="TrainerControl example"
from transformers import TrainerCallback, TrainerControl

class MyCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Stop training early
        if state.global_step >= 1000:
            control.should_training_stop = True

        # Force logging
        control.should_log = True

        # Force evaluation
        control.should_evaluate = True

        # Force checkpoint save
        control.should_save = True

        return control
```

### Control Flags

| Flag | Effect |
|------|--------|
| `should_training_stop` | End training after current step |
| `should_epoch_stop` | End current epoch |
| `should_log` | Trigger logging |
| `should_evaluate` | Trigger evaluation |
| `should_save` | Trigger checkpoint save |

### Early Stopping Example

```python title="Early stopping"
from transformers import EarlyStoppingCallback

trainer = SFTTrainer(
    ...,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,    # Stop after 3 evals without improvement
            early_stopping_threshold=0.01 # Minimum improvement required
        )
    ],
    args=SFTConfig(
        ...,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ),
)
```

## DeepFabric Callback

DeepFabric provides a callback for logging training metrics to the DeepFabric platform.

### Basic Usage

```python title="DeepFabric callback"
from deepfabric.training import DeepFabricCallback
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(output_dir="./output"),
)

# Add callback after trainer creation
trainer.add_callback(DeepFabricCallback(trainer))
trainer.train()
```

### Configuration

```python title="Callback configuration"
callback = DeepFabricCallback(
    trainer=trainer,              # Optional: Trainer instance for model info
    api_key="your-api-key",       # Or set DEEPFABRIC_API_KEY env var
    endpoint="https://...",       # Or set DEEPFABRIC_API_URL env var
    enabled=True,                 # Disable without removing callback
)
```

### What Gets Logged

| Event | Metrics |
|-------|---------|
| `on_train_begin` | Model name, training config, max_steps, epochs |
| `on_log` | loss, learning_rate, epoch, global_step, throughput |
| `on_evaluate` | eval_loss, eval metrics |
| `on_save` | Checkpoint step |
| `on_train_end` | Final step, total_flos, best_metric, best_checkpoint |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPFABRIC_API_KEY` | API key for authentication | None |
| `DEEPFABRIC_API_URL` | Backend endpoint URL | `https://api.deepfabric.cloud` |

## Custom Callbacks

Create custom callbacks by extending `TrainerCallback`:

```python title="Custom callback"
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"Starting training for {state.num_train_epochs} epochs")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            print(f"Step {state.global_step}/{state.max_steps}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"Loss: {logs['loss']:.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"Eval loss: {metrics.get('eval_loss', 'N/A')}")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"Training complete at step {state.global_step}")

# Use the callback
trainer = SFTTrainer(
    ...,
    callbacks=[CustomCallback()],
)
```

### Available Callback Methods

| Method | When Called |
|--------|-------------|
| `on_init_end` | After trainer initialization |
| `on_train_begin` | Start of training |
| `on_train_end` | End of training |
| `on_epoch_begin` | Start of each epoch |
| `on_epoch_end` | End of each epoch |
| `on_step_begin` | Before each training step |
| `on_step_end` | After each training step |
| `on_log` | When metrics are logged |
| `on_evaluate` | After evaluation |
| `on_save` | When checkpoint is saved |

## Training Tips

!!! tip "Batch Size"
    Start small (2-4) and increase if memory allows. Use gradient accumulation for effective larger batches.

!!! tip "Learning Rate"
    2e-5 for full fine-tuning, 2e-4 for LoRA.

!!! tip "Epochs"
    1-3 epochs is usually sufficient. More can cause overfitting on small datasets.

!!! tip "Evaluation"
    Hold out 10% of data for validation. Monitor loss to detect overfitting.

!!! tip "Mixed Precision"
    Use bf16 if supported, otherwise fp16.

## Evaluation During Training

```python title="Evaluation config"
trainer = SFTTrainer(
    ...,
    eval_dataset=eval_ds,
    args=SFTConfig(
        ...,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    ),
)
```
