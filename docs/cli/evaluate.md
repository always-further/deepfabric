# evaluate

The `evaluate` command evaluates a fine-tuned model on tool-calling tasks using either local transformers inference or Ollama.

## Usage

```bash
deepfabric evaluate MODEL_PATH DATASET_PATH [OPTIONS]
```

## Arguments

- `MODEL_PATH` - Path to base model or fine-tuned model (local directory or HuggingFace Hub ID)
- `DATASET_PATH` - Path to evaluation dataset (JSONL format)

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

### Evaluate a checkpoint

```bash
deepfabric evaluate ./checkpoints/final ./eval.jsonl --output results.json
```

### Evaluate with LoRA adapter

```bash
deepfabric evaluate unsloth/Qwen3-4B-Instruct ./eval.jsonl \
    --adapter-path ./lora_model \
    --output results.json
```

### Quick evaluation during development

```bash
deepfabric evaluate ./my-model ./eval.jsonl --max-samples 50
```

### Evaluate HuggingFace model

```bash
deepfabric evaluate username/model-name ./eval.jsonl \
    --temperature 0.5 \
    --device cuda
```
