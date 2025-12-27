# Metrics

Understanding DeepFabric's evaluation metrics.

## Overview

| Metric | Weight | Description |
|--------|--------|-------------|
| Tool Selection Accuracy | 40% | Correct tool chosen |
| Parameter Accuracy | 35% | Correct parameter types |
| Execution Success Rate | 25% | Valid executable call |
| Response Quality | 0% | Not used for tool evaluation |

## Tool Selection Accuracy

Measures whether the model selected the correct tool.

**Calculation**: `(correct selections) / (total samples)`

??? example "Example"
    - Expected: `read_file`
    - Predicted: `read_file`
    - Result: Correct (1.0)

!!! warning "Common Issues"
    - Model selects wrong tool for task
    - Model doesn't make a tool call when expected
    - Model hallucinates non-existent tools

## Parameter Accuracy

Measures whether the model provided correct parameter types.

**Calculation**: Validates that:

1. All required parameters are present
2. Parameter types match the schema

??? example "Example"
    ```python
    # Tool schema
    {
        "name": "read_file",
        "parameters": [
            {"name": "file_path", "type": "str", "required": True}
        ]
    }

    # Predicted call
    {"file_path": "config.json"}  # Correct - string provided

    {"file_path": 123}  # Wrong - integer instead of string
    ```

!!! note "Type Checking Only"
    Values aren't compared exactly. The evaluation checks types, not whether the specific value matches.

## Execution Success Rate

Measures whether the tool call could execute successfully.

A call is valid if:

- Correct tool is selected
- All required parameters are present
- Parameter types are correct

## Overall Score

Weighted combination of metrics:

```python title="Score calculation"
overall = (
    tool_selection * 0.40 +
    parameter_accuracy * 0.35 +
    execution_success * 0.25
)
```

### Custom Weights

```python title="Custom metric weights"
config = EvaluatorConfig(
    ...,
    metric_weights={
        "tool_selection": 0.50,
        "parameter_accuracy": 0.30,
        "execution_success": 0.20,
        "response_quality": 0.00,
    },
)
```

## Interpreting Results

| Score Range | Interpretation |
|-------------|----------------|
| 90-100% | Excellent - model is production-ready |
| 75-90% | Good - may need more training data |
| 50-75% | Fair - review failure cases |
| <50% | Poor - training issues likely |

## Debugging Low Scores

### Low Tool Selection

!!! tip "Check These"
    - Training data has clear tool usage patterns
    - Tools have distinct use cases
    - System prompt explains when to use each tool

### Low Parameter Accuracy

!!! tip "Check These"
    - Training examples show correct parameter formats
    - Required vs optional parameters are clear
    - Complex parameter types (lists, dicts) are handled

### High Processing Errors

!!! tip "Check These"
    - Model output format matches expected chat format
    - Model is generating valid JSON for tool calls
    - Inference configuration (temperature, max_tokens) is appropriate

## Sample Evaluation Details

Access individual sample results:

```python title="Inspect failures"
for pred in results.predictions:
    if not pred.tool_selection_correct:
        print(f"Sample {pred.sample_id}:")
        print(f"  Query: {pred.query}")
        print(f"  Expected: {pred.expected_tool}")
        print(f"  Predicted: {pred.predicted_tool}")
        if pred.error:
            print(f"  Error: {pred.error}")
```
