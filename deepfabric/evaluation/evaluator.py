"""Main evaluator for running model evaluation."""

import json

from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from tqdm.auto import tqdm

from ..metrics import trace
from ..schemas import ToolDefinition
from .evaluators import EvaluationContext, EvaluatorRegistry, EvaluatorResult
from .inference import InferenceConfig, ModelResponse, create_inference_backend
from .metrics import (
    EvaluationMetrics,
    SampleEvaluation,
    compute_metrics,
)
from .parser import ExpectedToolCall, GroundTruth, GroundTruthParser
from .reporters import BaseReporter, CloudReporter, FileReporter, MultiReporter

console = Console()

# Mapping for legacy conversation_type values
_CONVERSATION_TYPE_ALIASES = {
    "chain_of_thought": "cot",
}


def _normalize_conversation_type(value: str) -> str:
    """Normalize conversation_type to valid values.

    Handles legacy values like 'chain_of_thought' -> 'cot'.

    Args:
        value: Raw conversation_type value from dataset

    Returns:
        Normalized value ('basic' or 'cot')
    """
    return _CONVERSATION_TYPE_ALIASES.get(value, value)


class EvaluatorConfig(BaseModel):
    """Configuration for evaluation run."""

    dataset_path: str | None = Field(
        default=None,
        description="Path to evaluation dataset (JSONL). Optional if passing dataset to evaluate().",
    )
    output_path: str | None = Field(
        default=None,
        description="Path to save evaluation results",
    )
    model: str | None = Field(
        default=None,
        description="Model to evaluate (overrides inference_config.model)",
    )
    inference_config: InferenceConfig = Field(
        description="Inference backend configuration (includes model)",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for evaluation",
    )
    max_samples: int | None = Field(
        default=None,
        description="Maximum number of samples to evaluate (None for all)",
    )
    save_predictions: bool = Field(
        default=True,
        description="Save individual predictions to output file",
    )
    metric_weights: dict[str, float] | None = Field(
        default=None,
        description="Custom weights for overall score computation",
    )
    evaluators: list[str] | dict[str, dict] = Field(
        default=["tool_calling"],
        description="List of evaluator names or dict of name -> config",
    )
    reporters: list[str] | dict[str, dict] = Field(
        default=["file"],
        description="List of reporter names or dict of name -> config",
    )
    cloud_api_key: str | None = Field(
        default=None,
        description="DeepFabric cloud API key (or use DEEPFABRIC_API_KEY env var)",
    )
    multi_turn: bool = Field(
        default=False,
        description="Enable multi-turn evaluation (loops through conversation using ground truth tool responses)",
    )
    max_turns: int = Field(
        default=10,
        ge=1,
        description="Maximum number of turns in multi-turn evaluation",
    )


class EvaluationResult(BaseModel):
    """Complete evaluation result."""

    metrics: EvaluationMetrics = Field(description="Computed metrics")
    predictions: list[SampleEvaluation] = Field(
        description="Individual sample evaluations",
    )
    config: EvaluatorConfig = Field(description="Evaluation configuration used")


class Evaluator:
    """Orchestrates model evaluation on tool-calling tasks."""

    def __init__(self, config: EvaluatorConfig):
        """Initialize evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.backend = create_inference_backend(config.inference_config)
        # Parser will be configured per-sample based on conversation metadata
        self.parser: GroundTruthParser | None = None

        # Initialize evaluator registry and active evaluators
        self.registry = EvaluatorRegistry()
        self.active_evaluators = self._initialize_evaluators()

        # Initialize reporters
        self.reporter = self._initialize_reporters()

        # Track evaluator creation
        trace(
            "evaluator_created",
            {
                "backend": self.config.inference_config.backend,
                "model": self.config.inference_config.model,
                "has_adapter": self.config.inference_config.adapter_path is not None,
                "evaluators": (
                    list(self.config.evaluators)
                    if isinstance(self.config.evaluators, list)
                    else list(self.config.evaluators.keys())
                ),
                "reporters": (
                    list(self.config.reporters)
                    if isinstance(self.config.reporters, list)
                    else list(self.config.reporters.keys())
                ),
            },
        )

    def _initialize_evaluators(self) -> list:
        """Initialize evaluators based on config.

        Returns:
            List of active evaluator instances
        """
        evaluators = []

        if isinstance(self.config.evaluators, list):
            # Simple list of names
            for name in self.config.evaluators:
                evaluators.append(self.registry.get(name))
        else:
            # Dict with configs
            for name, eval_config in self.config.evaluators.items():
                evaluators.append(self.registry.get(name, config=eval_config))

        return evaluators

    def _initialize_reporters(self) -> BaseReporter:
        """Initialize reporters based on config.

        Returns:
            Reporter instance (may be MultiReporter)
        """
        reporters: list[BaseReporter] = []

        if isinstance(self.config.reporters, list):
            # Simple list of names
            for name in self.config.reporters:
                if name == "file":
                    reporters.append(FileReporter({"path": self.config.output_path}))
                elif name == "cloud":
                    reporters.append(CloudReporter({"api_key": self.config.cloud_api_key}))
        else:
            # Dict with configs
            for name, reporter_config in self.config.reporters.items():
                if name == "file":
                    # Merge output_path if not in config
                    if "path" not in reporter_config and self.config.output_path:
                        reporter_config["path"] = self.config.output_path
                    reporters.append(FileReporter(reporter_config))
                elif name == "cloud":
                    # Merge api_key if not in config
                    if "api_key" not in reporter_config and self.config.cloud_api_key:
                        reporter_config["api_key"] = self.config.cloud_api_key
                    reporters.append(CloudReporter(reporter_config))

        # Return single reporter or MultiReporter
        if len(reporters) == 0:
            # Default to file reporter
            return FileReporter({"path": self.config.output_path})
        if len(reporters) == 1:
            return reporters[0]
        return MultiReporter(reporters)

    def load_dataset(self, dataset: HFDataset | None = None) -> list[dict[str, Any]]:
        """Load evaluation dataset from HFDataset or JSONL file.

        Args:
            dataset: Optional HuggingFace Dataset. If provided, uses this instead
                of loading from config.dataset_path.

        Returns:
            List of dataset samples

        Raises:
            FileNotFoundError: If dataset file doesn't exist (when using file path)
            ValueError: If dataset format is invalid or no dataset source provided
        """
        if dataset is not None:
            # Use provided HuggingFace Dataset
            samples = [dict(sample) for sample in dataset]
        elif self.config.dataset_path is not None:
            # Load from file path
            dataset_path = Path(self.config.dataset_path)
            if not dataset_path.exists():
                msg = f"Dataset file not found: {dataset_path}"
                raise FileNotFoundError(msg)

            samples = []
            with dataset_path.open() as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        msg = f"Invalid JSON on line {line_num}: {e}"
                        raise ValueError(msg) from e
        else:
            msg = "No dataset provided. Either pass a HuggingFace Dataset to evaluate() or set dataset_path in config."
            raise ValueError(msg)

        if self.config.max_samples is not None:
            samples = samples[: self.config.max_samples]

        return samples

    def extract_ground_truth(self, sample: dict[str, Any]) -> GroundTruth:
        """Extract ground truth from sample.

        Args:
            sample: Dataset sample

        Returns:
            Parsed ground truth
        """
        # Create parser for this sample's conversation type
        from ..schemas import Conversation  # noqa: PLC0415

        # Convert sample dict to Conversation object
        conversation = Conversation.model_validate(sample)

        # Determine conversation type from metadata (normalize legacy values)
        metadata = conversation.metadata or {}
        raw_conv_type = metadata.get("conversation_type", "basic")
        conv_type = _normalize_conversation_type(raw_conv_type)
        reasoning_style = metadata.get("reasoning_style")
        agent_mode = metadata.get("agent_mode")

        # Create parser with appropriate config
        parser = GroundTruthParser(
            conversation_type=conv_type,  # type: ignore[arg-type]
            reasoning_style=reasoning_style,  # type: ignore[arg-type]
            agent_mode=agent_mode,  # type: ignore[arg-type]
        )

        return parser.parse(conversation)

    def prepare_messages(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Prepare messages for model inference.

        Extracts conversation up to the assistant's tool call.

        Args:
            sample: Dataset sample

        Returns:
            List of messages for inference
        """
        messages = []
        for msg in sample["messages"]:
            # Stop before first assistant message (where tool call should be generated)
            if msg["role"] == "assistant":
                break
            messages.append({"role": msg["role"], "content": msg["content"]})

        return messages

    def prepare_tools(self, sample: dict[str, Any]) -> list[ToolDefinition]:
        """Prepare tool definitions from sample.

        Args:
            sample: Dataset sample

        Returns:
            List of available tools
        """
        from ..schemas import Conversation  # noqa: PLC0415

        # Convert to Conversation to access tools field
        conversation = Conversation.model_validate(sample)

        if not conversation.tools:
            return []

        # Convert from OpenAI format back to ToolDefinition
        return [ToolDefinition.from_openai(tool) for tool in conversation.tools]

    def build_tool_response_lookup(self, sample: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Build lookup of tool responses by tool name and arguments.

        For multi-turn evaluation, we need to look up tool responses when the
        model makes tool calls. We index by (tool_name, arguments_json) to find
        matching responses from ground truth.

        Args:
            sample: Dataset sample

        Returns:
            Dict mapping "tool_name:args_json" -> {"content": response, "tool_call_id": id}
        """
        lookup: dict[str, dict[str, str]] = {}
        messages = sample.get("messages", [])

        # Track pending tool calls from assistant messages
        pending_tool_calls: dict[str, dict] = {}  # tool_call_id -> {name, arguments}

        for msg in messages:
            role = msg.get("role")

            # Collect tool calls from assistant messages
            if role == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id")
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", "{}")
                    if tc_id:
                        pending_tool_calls[tc_id] = {"name": name, "arguments": args}

            # Match tool responses to their calls
            if role == "tool":
                tc_id = msg.get("tool_call_id")
                content = msg.get("content", "")
                if tc_id and tc_id in pending_tool_calls:
                    call_info = pending_tool_calls[tc_id]
                    # Create lookup key from tool name + normalized arguments
                    try:
                        args_dict = json.loads(call_info["arguments"])
                        normalized_args = json.dumps(args_dict, sort_keys=True)
                        key = f"{call_info['name']}:{normalized_args}"
                    except (json.JSONDecodeError, TypeError):
                        # Fallback if arguments are not a valid JSON string
                        key = f"{call_info['name']}:{call_info['arguments']}"
                    lookup[key] = {"content": content, "tool_call_id": tc_id}

        return lookup

    def find_tool_response(
        self,
        tool_call: dict,
        lookup: dict[str, dict[str, str]],
    ) -> dict[str, str] | None:
        """Find a matching tool response for a predicted tool call.

        Args:
            tool_call: Predicted tool call with 'name' and 'arguments'
            lookup: Tool response lookup from build_tool_response_lookup

        Returns:
            Dict with 'content' and 'tool_call_id' if found, None otherwise
        """
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        # Normalize arguments to JSON string for comparison
        args_json = json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args)

        # Try exact match first
        key = f"{name}:{args_json}"
        if key in lookup:
            return lookup[key]

        # Try matching just by tool name (less strict)
        # This helps when parameter values differ slightly
        for lookup_key, response in lookup.items():
            if lookup_key.startswith(f"{name}:"):
                return response

        return None

    def evaluate_sample(
        self,
        sample: dict[str, Any],
        sample_id: int,
    ) -> SampleEvaluation:
        """Evaluate a single sample using configured evaluators.

        Args:
            sample: Dataset sample
            sample_id: Sample index

        Returns:
            Evaluation result for this sample
        """
        # Use multi-turn evaluation if enabled
        if self.config.multi_turn:
            return self.evaluate_sample_multi_turn(sample, sample_id)

        try:
            # Extract ground truth
            ground_truth = self.extract_ground_truth(sample)

            # Prepare inputs
            messages = self.prepare_messages(sample)
            tools = self.prepare_tools(sample)

            # Run inference
            response: ModelResponse = self.backend.generate(messages, tools)

            # Create evaluation context
            context = EvaluationContext(
                messages=messages,
                tools=tools,
                sample_id=sample_id,
            )

            # Run all active evaluators
            evaluator_results: list[EvaluatorResult] = []
            for evaluator in self.active_evaluators:
                result = evaluator.evaluate(ground_truth, response, context)
                if result is not None:  # Evaluator may skip
                    evaluator_results.append(result)

            # Aggregate results for backwards compatibility
            return self._aggregate_results(
                sample_id=sample_id,
                ground_truth=ground_truth,
                response=response,
                evaluator_results=evaluator_results,
                tools=tools,
            )

        except Exception as e:  # noqa: BLE001
            # Return failed evaluation with safe defaults
            query = ""
            expected_tool = None
            expected_params: dict[str, Any] = {}
            expected_answer = None
            available_tool_names: list[str] = []

            # Try to extract ground truth and tools if available
            try:
                gt = self.extract_ground_truth(sample)
                query = gt.query
                expected_tool = gt.expected_tool
                expected_params = gt.expected_parameters
                expected_answer = gt.expected_answer
            except (KeyError, AttributeError, ValidationError):
                pass

            try:
                tools = self.prepare_tools(sample)
                available_tool_names = [t.name for t in tools]
            except (KeyError, AttributeError, ValidationError):
                pass

            return SampleEvaluation(
                sample_id=sample_id,
                query=query,
                available_tools=available_tool_names,
                expected_tool=expected_tool,
                predicted_tool=None,
                expected_parameters=expected_params,
                predicted_parameters={},
                expected_answer=expected_answer,
                predicted_answer=None,
                tool_selection_correct=False,
                parameters_correct=False,
                execution_valid=False,
                response_score=0.0,
                error=str(e),
            )

    def evaluate_sample_multi_turn(
        self,
        sample: dict[str, Any],
        sample_id: int,
    ) -> SampleEvaluation:
        """Evaluate a single sample using multi-turn conversation.

        Loops through the conversation, feeding tool responses back to the model
        until it generates a final answer (no tool calls) or max turns reached.

        Args:
            sample: Dataset sample
            sample_id: Sample index

        Returns:
            Evaluation result for this sample
        """
        try:
            # Extract ground truth (includes all expected tools)
            ground_truth = self.extract_ground_truth(sample)

            # Prepare initial inputs
            messages = self.prepare_messages(sample)
            tools = self.prepare_tools(sample)

            # Build lookup for tool responses from ground truth
            tool_response_lookup = self.build_tool_response_lookup(sample)

            # Track all predicted tool calls across turns
            all_predicted_tool_calls: list[dict] = []
            final_content = ""

            # Multi-turn loop
            for turn in range(self.config.max_turns):
                # Run inference
                response: ModelResponse = self.backend.generate(messages, tools)
                final_content = response.content

                # Check if model made tool calls
                if not response.tool_calls:
                    # No tool calls - this is the final answer
                    break

                # Process each tool call
                for tool_call in response.tool_calls:
                    all_predicted_tool_calls.append(tool_call)

                    # Find matching tool response from ground truth
                    tool_response = self.find_tool_response(tool_call, tool_response_lookup)

                    if tool_response is None:
                        # Model called a tool we don't have a response for
                        # Continue anyway with an error message
                        tool_response = {
                            "content": json.dumps({"error": "Tool not found in ground truth"}),
                            "tool_call_id": f"generated_{turn}_{len(all_predicted_tool_calls)}",
                        }

                    # Add assistant message with tool call to conversation
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": tool_response["tool_call_id"],
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.get("name", ""),
                                        "arguments": json.dumps(tool_call.get("arguments", {})),
                                    },
                                }
                            ],
                        }
                    )

                    # Add tool response to conversation
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_response["tool_call_id"],
                            "content": tool_response["content"],
                        }
                    )

            # Now compute metrics comparing predicted vs expected tool calls
            return self._compute_multi_turn_metrics(
                sample_id=sample_id,
                ground_truth=ground_truth,
                predicted_tool_calls=all_predicted_tool_calls,
                final_content=final_content,
                tools=tools,
            )

        except Exception as e:  # noqa: BLE001
            # Return failed evaluation with safe defaults
            query = ""
            expected_tool = None
            expected_params: dict[str, Any] = {}
            expected_answer = None
            available_tool_names: list[str] = []

            try:
                gt = self.extract_ground_truth(sample)
                query = gt.query
                expected_tool = gt.expected_tool
                expected_params = gt.expected_parameters
                expected_answer = gt.expected_answer
            except (KeyError, AttributeError, ValidationError):
                pass

            try:
                tools = self.prepare_tools(sample)
                available_tool_names = [t.name for t in tools]
            except (KeyError, AttributeError, ValidationError):
                pass

            return SampleEvaluation(
                sample_id=sample_id,
                query=query,
                available_tools=available_tool_names,
                expected_tool=expected_tool,
                predicted_tool=None,
                expected_parameters=expected_params,
                predicted_parameters={},
                expected_answer=expected_answer,
                predicted_answer=None,
                tool_selection_correct=False,
                parameters_correct=False,
                execution_valid=False,
                response_score=0.0,
                error=str(e),
            )

    def _compute_multi_turn_metrics(
        self,
        sample_id: int,
        ground_truth: GroundTruth,
        predicted_tool_calls: list[dict],
        final_content: str,
        tools: list[ToolDefinition] | None = None,
    ) -> SampleEvaluation:
        """Compute metrics for multi-turn evaluation.

        Compares predicted tool calls against expected tools using set comparison.

        Args:
            sample_id: Sample index
            ground_truth: Expected values including all expected tools
            predicted_tool_calls: All tool calls made by model across turns
            final_content: Final model response content
            tools: List of available tools for this sample

        Returns:
            SampleEvaluation with computed metrics
        """
        # Build sets of tool names for comparison
        expected_tools_list = ground_truth.expected_tools or []
        expected_tool_names = {tc.tool_name for tc in expected_tools_list}
        predicted_tool_names = {tc.get("name", "") for tc in (predicted_tool_calls or [])}

        # Tool set coverage: what fraction of expected tools were called?
        if expected_tool_names:
            matched_tools = expected_tool_names & predicted_tool_names
            tool_coverage = len(matched_tools) / len(expected_tool_names)
        else:
            tool_coverage = 1.0 if not predicted_tool_names else 0.0

        # Tool set precision: what fraction of predicted tools were expected?
        # (Computed but stored in response_score for now, could be expanded later)
        if predicted_tool_names:
            matched_tools = expected_tool_names & predicted_tool_names
            _tool_precision = len(matched_tools) / len(predicted_tool_names)  # noqa: F841
        else:
            _tool_precision = 1.0 if not expected_tool_names else 0.0  # noqa: F841

        # Overall tool selection is correct if coverage is 100%
        tool_selection_correct = tool_coverage == 1.0

        # For backwards compatibility, use first predicted/expected tool
        first_predicted_tool = predicted_tool_calls[0].get("name") if predicted_tool_calls else None
        first_predicted_params = (
            predicted_tool_calls[0].get("arguments", {}) if predicted_tool_calls else {}
        )

        # Parameter accuracy: for matched tools, check if params are structurally correct
        params_correct = self._check_parameter_structure(
            ground_truth.expected_tools,
            predicted_tool_calls,
        )

        # Execution valid if we got through the conversation
        execution_valid = len(predicted_tool_calls) > 0 or final_content != ""

        # Extract tool names for available_tools field
        available_tool_names = [t.name for t in tools] if tools else []

        return SampleEvaluation(
            sample_id=sample_id,
            query=ground_truth.query,
            available_tools=available_tool_names,
            expected_tool=ground_truth.expected_tool,
            predicted_tool=first_predicted_tool,
            expected_parameters=ground_truth.expected_parameters,
            predicted_parameters=first_predicted_params
            if isinstance(first_predicted_params, dict)
            else {},
            expected_answer=ground_truth.expected_answer,
            predicted_answer=final_content,
            tool_selection_correct=tool_selection_correct,
            parameters_correct=params_correct,
            execution_valid=execution_valid,
            response_score=tool_coverage,  # Use coverage as response score
            error=None,
        )

    def _check_parameter_structure(
        self,
        expected_tools: list[ExpectedToolCall],
        predicted_tool_calls: list[dict],
    ) -> bool:
        """Check if predicted tool calls have correct parameter structure.

        For each matched tool, verifies that predicted params have the same keys.
        Does not check parameter values, only structure.

        Args:
            expected_tools: List of ExpectedToolCall from ground truth
            predicted_tool_calls: List of predicted tool call dicts

        Returns:
            True if all matched tools have correct parameter structure
        """
        # Build lookup of expected params by tool name
        expected_params_by_tool: dict[str, set[str]] = {}
        for tc in expected_tools or []:
            if tc.tool_name not in expected_params_by_tool:
                expected_params_by_tool[tc.tool_name] = set(tc.parameters.keys())

        # Check each predicted tool call
        for pred_call in predicted_tool_calls or []:
            tool_name = pred_call.get("name", "")
            pred_args = pred_call.get("arguments", {})

            if tool_name in expected_params_by_tool:
                expected_keys = expected_params_by_tool[tool_name]
                pred_keys = set(pred_args.keys()) if isinstance(pred_args, dict) else set()

                # Check if predicted has all expected keys (may have extra)
                if not expected_keys.issubset(pred_keys):
                    return False

        return True

    def _aggregate_results(
        self,
        sample_id: int,
        ground_truth: GroundTruth,
        response: ModelResponse,
        evaluator_results: list[EvaluatorResult],
        tools: list[ToolDefinition] | None = None,
    ) -> SampleEvaluation:
        """Aggregate evaluator results into SampleEvaluation.

        Args:
            sample_id: Sample index
            ground_truth: Expected values
            response: Model response
            evaluator_results: Results from all evaluators
            tools: List of available tools for this sample

        Returns:
            SampleEvaluation with aggregated metrics
        """
        # Extract tool calling metrics from evaluator results
        tool_correct = False
        params_correct = False
        execution_valid = False
        predicted_tool = None
        predicted_params = {}

        # Extract predictions from response
        if response.tool_call:
            predicted_tool = response.tool_call.get("name")
            predicted_params = response.tool_call.get("arguments", {})

        # Get metrics from tool_calling evaluator
        for result in evaluator_results:
            if result.evaluator_name == "tool_calling":
                metrics = result.metrics
                tool_correct = metrics.get("tool_selection_accuracy", 0.0) == 1.0
                params_correct = metrics.get("parameter_accuracy", 0.0) == 1.0
                execution_valid = metrics.get("execution_valid", 0.0) == 1.0

        # Extract tool names for available_tools field
        available_tool_names = [t.name for t in tools] if tools else []

        # Return backwards-compatible SampleEvaluation
        return SampleEvaluation(
            sample_id=sample_id,
            query=ground_truth.query,
            available_tools=available_tool_names,
            expected_tool=ground_truth.expected_tool,
            predicted_tool=predicted_tool,
            expected_parameters=ground_truth.expected_parameters,
            predicted_parameters=predicted_params,
            expected_answer=ground_truth.expected_answer,
            predicted_answer=response.content,
            tool_selection_correct=tool_correct,
            parameters_correct=params_correct,
            execution_valid=execution_valid,
            response_score=0.0,  # TODO: Could use semantic similarity for response quality evaluation in the future, but disabled for tool-calling mode
            error=None,
        )

    def evaluate(self, dataset: HFDataset | None = None) -> EvaluationResult:
        """Run full evaluation.

        Args:
            dataset: Optional HuggingFace Dataset to evaluate. If not provided,
                loads from config.dataset_path.

        Returns:
            Complete evaluation result with metrics and predictions
        """
        console.print("[bold blue]Loading dataset...[/bold blue]")
        samples = self.load_dataset(dataset)
        console.print(f"Loaded {len(samples)} samples")

        console.print("[bold blue]Running evaluation...[/bold blue]")
        evaluations = []

        pbar = tqdm(enumerate(samples), total=len(samples), desc="Evaluating")
        for idx, sample in pbar:
            eval_result = self.evaluate_sample(sample, idx)
            evaluations.append(eval_result)

            # Stream sample to reporters (for cloud real-time tracking)
            self.reporter.report_sample(eval_result)

            # Force refresh for notebook compatibility
            pbar.refresh()

        console.print("[bold green]Evaluation complete![/bold green]")

        # Compute metrics
        metrics = compute_metrics(evaluations, self.config.metric_weights)

        # Create result
        result = EvaluationResult(
            metrics=metrics,
            predictions=evaluations,
            config=self.config,
        )

        # Track evaluation completion
        trace(
            "evaluation_completed",
            {
                "backend": self.config.inference_config.backend,
                "model": self.config.inference_config.model,
                "has_adapter": self.config.inference_config.adapter_path is not None,
                "samples_evaluated": metrics.samples_evaluated,
                "samples_processed": metrics.samples_processed,
                "processing_errors": metrics.processing_errors,
                "tool_selection_accuracy": round(metrics.tool_selection_accuracy, 4),
                "parameter_accuracy": round(metrics.parameter_accuracy, 4),
                "execution_success_rate": round(metrics.execution_success_rate, 4),
                "overall_score": round(metrics.overall_score, 4),
                "success": metrics.processing_errors == 0,
            },
        )

        # Report results using configured reporters
        if self.config.save_predictions:
            self.reporter.report(result)

        return result

    def cleanup(self) -> None:
        """Clean up resources."""
        self.backend.cleanup()

    def print_summary(self, metrics: EvaluationMetrics) -> None:
        """Print evaluation summary.

        Args:
            metrics: Computed metrics
        """
        console.print("\n[bold]Evaluation Summary[/bold]")
        console.print(f"Samples Evaluated: {metrics.samples_evaluated}")
        console.print(f"Processed Successfully: {metrics.samples_processed}")
        console.print(f"Processing Errors: {metrics.processing_errors}")
        console.print("\n[bold]Metrics[/bold]")
        console.print(f"Tool Selection Accuracy: {metrics.tool_selection_accuracy:.2%}")
        console.print(f"Parameter Accuracy: {metrics.parameter_accuracy:.2%}")
        console.print(f"Execution Success Rate: {metrics.execution_success_rate:.2%}")
        console.print(f"Response Quality: {metrics.response_quality:.2%}")
        console.print(f"\n[bold green]Overall Score: {metrics.overall_score:.2%}[/bold green]")
