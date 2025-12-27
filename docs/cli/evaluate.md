# evaluate

The `evaluate` command evaluates a fine-tuned model on tool-calling tasks using either local transformers inference or Ollama.

## Usage

```bash title="Basic syntax"
deepfabric evaluate MODEL_PATH DATASET_PATH [OPTIONS]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `MODEL_PATH` | Path to base model or fine-tuned model (local directory or HuggingFace Hub ID) |
| `DATASET_PATH` | Path to evaluation dataset (JSONL format) |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --output` | PATH | None | Path to save evaluation results (JSON) |
| `--adapter-path` | PATH | None | Path to PEFT/LoRA adapter |
| `--batch-size` | INT | 1 | Batch size for evaluation |
| `--max-samples` | INT | All | Maximum number of samples to evaluate |
| `--temperature` | FLOAT | 0.7 | Sampling temperature |
| `--max-tokens` | INT | 2048 | Maximum tokens to generate |
| `--top-p` | FLOAT | 0.9 | Nucleus sampling top-p |
| `--backend` | CHOICE | transformers | Inference backend: `transformers` or `ollama` |
| `--device` | TEXT | Auto | Device to use (cuda, cpu, mps) - transformers only |
| `--no-save-predictions` | FLAG | False | Don't save individual predictions to output |

## Examples

=== "Checkpoint Evaluation"

    ```bash title="Evaluate a checkpoint"
    deepfabric evaluate ./checkpoints/final ./eval.jsonl --output results.json
    ```

=== "With LoRA Adapter"

    ```bash title="Evaluate with LoRA"
    deepfabric evaluate unsloth/Qwen3-4B-Instruct ./eval.jsonl \
        --adapter-path ./lora_model \
        --output results.json
    ```

=== "Quick Dev Evaluation"

    ```bash title="Quick evaluation"
    deepfabric evaluate ./my-model ./eval.jsonl --max-samples 50
    ```

=== "HuggingFace Model"

    ```bash title="Evaluate HF model"
    deepfabric evaluate username/model-name ./eval.jsonl \
        --temperature 0.5 \
        --device cuda
    ```

!!! tip "Quick Iteration"
    Use `--max-samples 50` during development for faster feedback loops.
