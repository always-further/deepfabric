"""Progress reporting system for dataset generation.

This module provides a modular event-based progress reporting system that
allows components to emit progress events (streaming text, step markers, etc.)
without coupling to specific display implementations.

The system uses the Observer pattern to enable multiple observers (TUI, logging,
metrics, etc.) to react to progress events.
"""

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .error_codes import ClassifiedError


class StreamObserver(Protocol):
    """Protocol for observers that react to progress events.

    Implementations can choose which events to handle based on their needs.
    This protocol supports both dataset generation and tree/graph building.
    """

    def on_stream_chunk(self, source: str, chunk: str, metadata: dict[str, Any]) -> None:
        """Called when a chunk of streaming text is received from an LLM.

        Args:
            source: Identifier for the generation source
                - Dataset: "user_question", "agent_reasoning", "tool_sim_weather"
                - Tree/Graph: "topic_generation", "subtopic_expansion"
            chunk: The text chunk received from the LLM
            metadata: Additional context (sample_idx, node_path, depth, etc.)
        """
        ...

    def on_step_start(self, step_name: str, metadata: dict[str, Any]) -> None:
        """Called when a generation step begins.

        Args:
            step_name: Human-readable name of the step
                - Dataset: "Generating user question", "Simulating tool: get_weather"
                - Tree/Graph: "Expanding node: AI/ML", "Generating subtopics (depth 2)"
            metadata: Additional context (sample_idx, turn_idx, depth, node_path, etc.)
        """
        ...

    def on_step_complete(self, step_name: str, metadata: dict[str, Any]) -> None:
        """Called when a generation step completes.

        Args:
            step_name: Human-readable name of the step
            metadata: Additional context including results (tokens_used, duration, success, etc.)
        """
        ...

    def on_error(self, error: "ClassifiedError", metadata: dict[str, Any]) -> None:
        """Called when an error occurs during generation.

        Args:
            error: ClassifiedError with error code and details
            metadata: Additional context (sample_idx, step, etc.)
        """
        ...

    def on_retry(
        self,
        sample_idx: int,
        attempt: int,
        max_attempts: int,
        error_summary: str,
        metadata: dict[str, Any],
    ) -> None:
        """Called when a sample generation will be retried due to validation failure.

        Args:
            sample_idx: 1-based sample index
            attempt: Current attempt number (1-based)
            max_attempts: Total number of attempts allowed
            error_summary: Brief description of the validation error
            metadata: Additional context
        """
        ...


class ProgressReporter:
    """Central progress reporter that notifies observers of generation events.

    This class acts as the subject in the Observer pattern, managing a list of
    observers and broadcasting events to them.

    Example:
        >>> reporter = ProgressReporter()
        >>> reporter.attach(my_tui_observer)
        >>> reporter.emit_step_start("Generating question", sample_idx=1)
        >>> reporter.emit_chunk("user_question", "What is the weather", sample_idx=1)
        >>> reporter.emit_step_complete("Generating question", sample_idx=1)
    """

    def __init__(self):
        """Initialize an empty progress reporter."""
        self._observers: list[StreamObserver] = []

    def attach(self, observer: StreamObserver) -> None:
        """Attach an observer to receive progress events.

        Args:
            observer: Observer implementing StreamObserver protocol
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: StreamObserver) -> None:
        """Detach an observer from receiving progress events.

        Args:
            observer: Observer to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def emit_chunk(self, source: str, chunk: str, **metadata) -> None:
        """Emit a streaming text chunk to all observers.

        Args:
            source: Identifier for the generation source
            chunk: Text chunk from LLM
            **metadata: Additional context as keyword arguments
        """
        for observer in self._observers:
            observer.on_stream_chunk(source, chunk, metadata)

    def emit_step_start(self, step_name: str, **metadata) -> None:
        """Emit a step start event to all observers.

        Args:
            step_name: Human-readable step name
            **metadata: Additional context as keyword arguments
        """
        for observer in self._observers:
            observer.on_step_start(step_name, metadata)

    def emit_step_complete(self, step_name: str, **metadata) -> None:
        """Emit a step complete event to all observers.

        Args:
            step_name: Human-readable step name
            **metadata: Additional context as keyword arguments
        """
        for observer in self._observers:
            observer.on_step_complete(step_name, metadata)

    def emit_error(self, error: "ClassifiedError", **metadata) -> None:
        """Emit an error event to all observers.

        Args:
            error: ClassifiedError with error code and details
            **metadata: Additional context as keyword arguments
        """
        for observer in self._observers:
            if hasattr(observer, "on_error"):
                observer.on_error(error, metadata)

    def emit_retry(
        self,
        sample_idx: int,
        attempt: int,
        max_attempts: int,
        error_summary: str,
        **metadata,
    ) -> None:
        """Emit a retry event to all observers.

        This is used to track validation failures that will be retried,
        allowing the TUI to display them gracefully without cluttering output.

        Args:
            sample_idx: 1-based sample index
            attempt: Current attempt number (1-based)
            max_attempts: Total number of attempts allowed
            error_summary: Brief description of the error
            **metadata: Additional context as keyword arguments
        """
        for observer in self._observers:
            if hasattr(observer, "on_retry"):
                observer.on_retry(sample_idx, attempt, max_attempts, error_summary, metadata)

    def emit_tool_execution(
        self,
        tool_name: str,
        success: bool,
        **metadata,
    ) -> None:
        """Emit a tool execution event to all observers.

        Used to track Spin tool executions in the TUI events panel.

        Args:
            tool_name: Name of the tool being executed
            success: Whether the execution succeeded
            **metadata: Additional context (e.g., error_type, result preview)
        """
        for observer in self._observers:
            if hasattr(observer, "on_tool_execution"):
                observer.on_tool_execution(tool_name, success, metadata)

    def emit_node_retry(
        self,
        node_topic: str,
        attempt: int,
        max_attempts: int,
        error_summary: str,
        **metadata,
    ) -> None:
        """Emit a node expansion retry event to all observers.

        Used to track graph/tree node expansion retries in the TUI events panel.

        Args:
            node_topic: Topic of the node being expanded
            attempt: Current attempt number (1-based)
            max_attempts: Total number of attempts allowed
            error_summary: Brief description of the error
            **metadata: Additional context as keyword arguments
        """
        for observer in self._observers:
            if hasattr(observer, "on_node_retry"):
                observer.on_node_retry(node_topic, attempt, max_attempts, error_summary, metadata)


# Convenience context manager for tracking steps
class ProgressStep:
    """Context manager for automatic step start/complete reporting.

    Example:
        >>> with ProgressStep(reporter, "Generating question", sample_idx=1):
        ...     # Do work
        ...     reporter.emit_chunk("question", "What is...")
    """

    def __init__(self, reporter: ProgressReporter | None, step_name: str, **metadata: Any):
        """Initialize progress step tracker.

        Args:
            reporter: Progress reporter (None = no-op)
            step_name: Human-readable step name
            **metadata: Additional context
        """
        self.reporter = reporter
        self.step_name = step_name
        self.metadata = metadata

    def __enter__(self):
        """Enter context: emit step start."""
        if self.reporter:
            self.reporter.emit_step_start(self.step_name, **self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: emit step complete."""
        if self.reporter:
            self.reporter.emit_step_complete(self.step_name, **self.metadata)
        return False  # Don't suppress exceptions
