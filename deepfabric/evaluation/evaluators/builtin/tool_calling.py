"""Tool calling evaluator for assessing function calling accuracy."""

from ...inference import ModelResponse
from ...metrics import compare_parameters
from ...parser import GroundTruth
from ..base import BaseEvaluator, EvaluationContext, EvaluatorResult


class ToolCallingEvaluator(BaseEvaluator):
    """Evaluates tool selection and parameter extraction accuracy.

    This evaluator checks if the model:
    1. Selects the correct tool
    2. Extracts parameters correctly (with fuzzy matching)
    3. Can execute the tool successfully (tool + params both correct)

    Only applicable to samples with tool calls (skips samples without tools).
    """

    def get_name(self) -> str:
        """Return evaluator identifier."""
        return "tool_calling"

    def get_metrics(self) -> list[str]:
        """Return list of metrics this evaluator produces."""
        return [
            "tool_selection_accuracy",
            "parameter_accuracy",
            "execution_valid",
        ]

    def applicable_to(self, ground_truth: GroundTruth) -> bool:
        """Only apply to samples with expected tool calls."""
        return ground_truth.expected_tool is not None

    def evaluate(
        self,
        ground_truth: GroundTruth,
        prediction: ModelResponse,
        context: EvaluationContext,
    ) -> EvaluatorResult | None:
        """Evaluate tool calling accuracy.

        Args:
            ground_truth: Expected tool and parameters
            prediction: Model's generated response
            context: Evaluation context with tool definitions

        Returns:
            EvaluatorResult with tool calling metrics
        """
        # Skip if not applicable
        if not self.applicable_to(ground_truth):
            return None

        # Extract predicted tool and parameters
        predicted_tool = None
        predicted_params = {}
        if prediction.tool_call:
            predicted_tool = prediction.tool_call.get("name")
            predicted_params = prediction.tool_call.get("arguments", {})

        # Compute metrics
        tool_correct = predicted_tool == ground_truth.expected_tool

        # Parameter accuracy requires a tool to have been called
        # If no tool was predicted but one was expected, params cannot be correct
        if predicted_tool is None and ground_truth.expected_tool is not None:
            params_correct = False
        else:
            # Validate parameters against the PREDICTED tool (not expected)
            # This measures parameter extraction capability independently of tool selection
            params_correct = compare_parameters(
                ground_truth.expected_parameters,
                predicted_params,
                tool_name=predicted_tool,  # Use predicted tool for schema validation
                tool_definitions=context.tools,
            )

        # Execution valid requires BOTH correct tool AND correct params
        execution_valid = tool_correct and params_correct

        return EvaluatorResult(
            evaluator_name=self.get_name(),
            metrics={
                "tool_selection_accuracy": 1.0 if tool_correct else 0.0,
                "parameter_accuracy": 1.0 if params_correct else 0.0,
                "execution_valid": 1.0 if execution_valid else 0.0,
            },
            details={
                "expected_tool": ground_truth.expected_tool,
                "predicted_tool": predicted_tool,
                "expected_parameters": ground_truth.expected_parameters,
                "predicted_parameters": predicted_params,
                "tool_match": tool_correct,
                "params_match": params_correct,
            },
        )
