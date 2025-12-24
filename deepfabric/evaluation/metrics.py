"""Metrics computation for model evaluation."""

from typing import Any

from pydantic import BaseModel, Field

from ..schemas import ToolDefinition

# Tolerance for numeric comparison
NUMERIC_TOLERANCE = 1e-6

# Type validation dispatch table
_TYPE_CHECKS = {
    "str": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "list": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
}


def _is_valid_type(schema_type: str, value: Any) -> bool:
    """Check if value matches schema type.

    Args:
        schema_type: Schema type string ("str", "int", "float", "bool", "list", "dict")
        value: Value to check

    Returns:
        True if value matches type, False otherwise
    """
    check = _TYPE_CHECKS.get(schema_type)
    return check(value) if check else False


def _validate_parameter_types(
    predicted_params: dict[str, Any],
    tool_def: ToolDefinition,
) -> bool:
    """Validate parameter types against tool schema.

    Checks that:
    1. All required parameters are present
    2. Parameter types match schema (with type coercion)
    3. Ignores actual values - only validates structure

    Args:
        predicted_params: Parameters to validate
        tool_def: Tool definition with schema

    Returns:
        True if types are valid, False otherwise
    """
    # Create lookup for parameters by name
    schema_params = {p.name: p for p in tool_def.parameters}

    # Check all required parameters are present
    for param_name, param_schema in schema_params.items():
        if param_schema.required and param_name not in predicted_params:
            return False

    # Check types for each predicted parameter
    for param_name, predicted_value in predicted_params.items():
        # Skip extra parameters not in schema (allow for flexibility)
        if param_name not in schema_params:
            continue

        schema_param = schema_params[param_name]
        if not _is_valid_type(schema_param.type, predicted_value):
            return False

    return True


class EvaluationMetrics(BaseModel):
    """Computed evaluation metrics."""

    tool_selection_accuracy: float = Field(
        description="Accuracy of tool selection (0.0-1.0)",
    )
    parameter_accuracy: float = Field(
        description="Accuracy of parameter extraction (0.0-1.0)",
    )
    execution_success_rate: float = Field(
        description="Rate of valid tool calls (0.0-1.0)",
    )
    response_quality: float = Field(
        description="Quality of final response (0.0-1.0)",
    )
    overall_score: float = Field(
        description="Weighted overall score (0.0-1.0)",
    )
    samples_evaluated: int = Field(
        description="Total number of samples evaluated",
    )
    samples_processed: int = Field(
        description="Number of samples processed without system errors",
    )
    processing_errors: int = Field(
        description="Number of samples that failed to process (system errors, timeouts)",
    )


class SampleEvaluation(BaseModel):
    """Evaluation result for a single sample."""

    sample_id: int = Field(description="Sample index")
    query: str = Field(description="Input query")
    available_tools: list[str] = Field(
        default_factory=list,
        description="List of tool names available for this sample",
    )
    expected_tool: str | None = Field(
        default=None,
        description="Expected tool name",
    )
    predicted_tool: str | None = Field(
        default=None,
        description="Predicted tool name",
    )
    expected_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Expected parameters",
    )
    predicted_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Predicted parameters",
    )
    expected_answer: str | None = Field(
        default=None,
        description="Expected final answer",
    )
    predicted_answer: str | None = Field(
        default=None,
        description="Predicted final answer",
    )
    tool_selection_correct: bool = Field(
        description="Whether tool selection was correct",
    )
    parameters_correct: bool = Field(
        description="Whether parameters were correct",
    )
    execution_valid: bool = Field(
        description="Whether the tool call could be executed",
    )
    response_score: float = Field(
        description="Response quality score (0.0-1.0)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if prediction failed",
    )


def compute_tool_selection_accuracy(
    evaluations: list[SampleEvaluation],
) -> float:
    """Compute tool selection accuracy.

    Args:
        evaluations: List of sample evaluations

    Returns:
        Accuracy score (0.0-1.0)
    """
    if not evaluations:
        return 0.0

    correct = sum(1 for e in evaluations if e.tool_selection_correct)
    return correct / len(evaluations)


def compute_parameter_accuracy(
    evaluations: list[SampleEvaluation],
) -> float:
    """Compute parameter extraction accuracy.

    Args:
        evaluations: List of sample evaluations

    Returns:
        Accuracy score (0.0-1.0)
    """
    if not evaluations:
        return 0.0

    correct = sum(1 for e in evaluations if e.parameters_correct)
    return correct / len(evaluations)


def compute_execution_success_rate(
    evaluations: list[SampleEvaluation],
) -> float:
    """Compute execution success rate.

    Args:
        evaluations: List of sample evaluations

    Returns:
        Success rate (0.0-1.0)
    """
    if not evaluations:
        return 0.0

    valid = sum(1 for e in evaluations if e.execution_valid)
    return valid / len(evaluations)


def compute_response_quality(
    evaluations: list[SampleEvaluation],
) -> float:
    """Compute average response quality.

    Args:
        evaluations: List of sample evaluations

    Returns:
        Average quality score (0.0-1.0)
    """
    if not evaluations:
        return 0.0

    total_score = sum(e.response_score for e in evaluations)
    return total_score / len(evaluations)


def compute_overall_score(
    tool_accuracy: float,
    param_accuracy: float,
    exec_success: float,
    response_quality: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted overall score.

    Args:
        tool_accuracy: Tool selection accuracy
        param_accuracy: Parameter accuracy
        exec_success: Execution success rate
        response_quality: Response quality score
        weights: Custom weights for each metric (defaults used if None)

    Returns:
        Weighted overall score (0.0-1.0)
    """
    # Default weights (response_quality excluded for tool-calling mode)
    if weights is None:
        weights = {
            "tool_selection": 0.40,
            "parameter_accuracy": 0.35,
            "execution_success": 0.25,
            "response_quality": 0.00,  # Not used for tool-calling evaluation
        }

    return (
        tool_accuracy * weights.get("tool_selection", 0.0)
        + param_accuracy * weights.get("parameter_accuracy", 0.0)
        + exec_success * weights.get("execution_success", 0.0)
        + response_quality * weights.get("response_quality", 0.0)
    )


def compute_metrics(
    evaluations: list[SampleEvaluation],
    weights: dict[str, float] | None = None,
) -> EvaluationMetrics:
    """Compute all evaluation metrics from sample evaluations.

    Args:
        evaluations: List of sample evaluations
        weights: Custom weights for overall score computation

    Returns:
        EvaluationMetrics with all computed scores
    """
    if not evaluations:
        return EvaluationMetrics(
            tool_selection_accuracy=0.0,
            parameter_accuracy=0.0,
            execution_success_rate=0.0,
            response_quality=0.0,
            overall_score=0.0,
            samples_evaluated=0,
            samples_processed=0,
            processing_errors=0,
        )

    tool_acc = compute_tool_selection_accuracy(evaluations)
    param_acc = compute_parameter_accuracy(evaluations)
    exec_success = compute_execution_success_rate(evaluations)
    resp_quality = compute_response_quality(evaluations)

    overall = compute_overall_score(
        tool_acc,
        param_acc,
        exec_success,
        resp_quality,
        weights,
    )

    # Count processing status (system errors vs successfully processed)
    processed = sum(1 for e in evaluations if e.error is None)
    errors = len(evaluations) - processed

    return EvaluationMetrics(
        tool_selection_accuracy=tool_acc,
        parameter_accuracy=param_acc,
        execution_success_rate=exec_success,
        response_quality=resp_quality,
        overall_score=overall,
        samples_evaluated=len(evaluations),
        samples_processed=processed,
        processing_errors=errors,
    )


def compare_parameters(  # noqa: PLR0911
    expected: dict[str, Any],
    predicted: dict[str, Any],
    tool_name: str | None = None,
    tool_definitions: list[ToolDefinition] | None = None,
) -> bool:
    """Compare expected and predicted parameters.

    If tool schema is provided, validates parameter types and presence of required params.
    Otherwise, performs value-based comparison (legacy behavior for backward compatibility).

    Args:
        expected: Expected parameters
        predicted: Predicted parameters
        tool_name: Name of the tool being called (for schema lookup)
        tool_definitions: List of tool definitions with schemas

    Returns:
        True if parameters match (schema-aware) or values match (legacy), False otherwise
    """
    if not expected and not predicted:
        return True

    # Schema-aware validation if tool definition available
    if tool_name and tool_definitions:
        tool_def = next((t for t in tool_definitions if t.name == tool_name), None)
        if tool_def:
            return _validate_parameter_types(predicted, tool_def)

    # Legacy value-based comparison (backward compatibility)
    # Check if all expected keys are present
    if set(expected.keys()) != set(predicted.keys()):
        return False

    # Compare values
    for key, expected_val in expected.items():
        predicted_val = predicted.get(key)

        # Handle different types
        if isinstance(expected_val, str) and isinstance(predicted_val, str):
            # Case-insensitive string comparison
            if expected_val.lower().strip() != predicted_val.lower().strip():
                return False
        elif isinstance(expected_val, int | float) and isinstance(predicted_val, int | float):
            # Numeric comparison with small tolerance
            if abs(float(expected_val) - float(predicted_val)) > NUMERIC_TOLERANCE:
                return False
        elif expected_val != predicted_val:
            # Exact match for other types
            return False

    return True


def compute_response_similarity(
    expected: str | None,
    predicted: str | None,
) -> float:
    """Compute similarity between expected and predicted responses.

    Uses simple word overlap for now. Can be enhanced with semantic similarity.

    Args:
        expected: Expected response
        predicted: Predicted response

    Returns:
        Similarity score (0.0-1.0)
    """
    if not expected or not predicted:
        return 0.0 if expected != predicted else 1.0

    # Tokenize and normalize
    expected_words = set(expected.lower().split())
    predicted_words = set(predicted.lower().split())

    # Compute Jaccard similarity
    if not expected_words and not predicted_words:
        return 1.0

    intersection = expected_words & predicted_words
    union = expected_words | predicted_words

    return len(intersection) / len(union) if union else 0.0
