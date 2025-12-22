# Evaluation

DeepFabric includes an evaluation system for testing fine-tuned models on tool-calling tasks.

## Workflow

```
1. Generate dataset with train/eval split
2. Train model on training split
3. Evaluate on held-out eval split
4. Review metrics and improve
```

## Quick Example

```python
from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

config = EvaluatorConfig(
    inference_config=InferenceConfig(
        model="./output/checkpoint-final",
        backend="transformers",
    ),
)

evaluator = Evaluator(config)
results = evaluator.evaluate(dataset=eval_dataset)

print(f"Tool Selection: {results.metrics.tool_selection_accuracy:.2%}")
print(f"Parameter Accuracy: {results.metrics.parameter_accuracy:.2%}")
print(f"Overall Score: {results.metrics.overall_score:.2%}")
```

## What Gets Evaluated

For each sample in the evaluation dataset:

1. **Extract ground truth** from the sample's expected tool calls
2. **Run inference** with the user message and available tools
3. **Compare** predicted tool selection and parameters against ground truth
4. **Score** the result

## Key Metrics

| Metric | Description | Weight |
|--------|-------------|--------|
| Tool Selection Accuracy | Did the model pick the right tool? | 40% |
| Parameter Accuracy | Are the parameter types correct? | 35% |
| Execution Success Rate | Would the call execute successfully? | 25% |

## Next Steps

- [Running Evaluation](running.md) - Configuration and usage
- [Metrics](metrics.md) - Understanding the metrics
