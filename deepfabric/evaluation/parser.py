"""Ground truth parsing from DeepFabric dataset samples."""

import json
import re

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..schemas import Conversation, ToolDefinition


class ExpectedToolCall(BaseModel):
    """A single expected tool call with its parameters."""

    tool_name: str = Field(description="Name of the tool")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter names and values",
    )

    def signature(self) -> str:
        """Return a hashable signature for deduplication."""
        params_str = json.dumps(self.parameters, sort_keys=True)
        return f"{self.tool_name}:{params_str}"


class GroundTruth(BaseModel):
    """Parsed ground truth from original dataset sample."""

    query: str = Field(description="The user query")
    expected_tool: str | None = Field(
        default=None,
        description="Expected tool name - first tool (None if no tool use). Kept for backwards compatibility.",
    )
    expected_parameters: dict[str, str | int | float | bool | list | dict] = Field(
        default_factory=dict,
        description="Expected tool parameters - first tool. Kept for backwards compatibility.",
    )
    expected_tools: list[ExpectedToolCall] = Field(
        default_factory=list,
        description="All unique expected tool calls (deduplicated by tool_name + parameters)",
    )
    tool_schema: ToolDefinition | None = Field(
        default=None,
        description="Tool schema from available_tools",
    )
    expected_answer: str | None = Field(
        default=None,
        description="Expected final answer if available",
    )
    conversation_type: Literal["basic", "cot"] = Field(
        description="Type of conversation",
    )
    reasoning_style: Literal["freetext", "agent", "structured", "hybrid"] | None = Field(
        default=None,
        description="Reasoning style if cot",
    )
    agent_mode: Literal["single_turn"] | None = Field(
        default=None,
        description="Agent mode if tools are used (single_turn only)",
    )
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class GroundTruthParser:
    """Parse ground truth from original DeepFabric JSONL format.

    This parser extracts expected tools, parameters, and answers from
    Conversation objects while handling all conversation types and agent modes.
    """

    def __init__(
        self,
        conversation_type: Literal["basic", "cot"],
        reasoning_style: Literal["freetext", "agent", "structured", "hybrid"] | None = None,
        agent_mode: Literal["single_turn"] | None = None,
    ):
        """Initialize parser with conversation configuration.

        Args:
            conversation_type: Type of conversation (basic, cot)
            reasoning_style: Reasoning style for cot
            agent_mode: Agent mode if tools are used (single_turn only)
        """
        self.conversation_type: Literal["basic", "cot"] = conversation_type
        self.reasoning_style: Literal["freetext", "agent", "structured", "hybrid"] | None = (
            reasoning_style
        )
        self.agent_mode: Literal["single_turn"] | None = agent_mode

    def parse(self, conversation: Conversation) -> GroundTruth:
        """Extract ground truth from a conversation sample.

        Args:
            conversation: Conversation object from dataset

        Returns:
            GroundTruth with expected values

        Raises:
            ValueError: If conversation format is invalid
        """
        # Extract query from first user message
        query = self._extract_query(conversation)

        # Extract expected tool and parameters if tool_context present
        expected_tool: str | None = None
        expected_parameters: dict = {}
        expected_tools: list[ExpectedToolCall] = []
        tool_schema: ToolDefinition | None = None

        executions = (
            conversation.tool_context.executions
            if conversation.tool_context is not None and conversation.tool_context.executions
            else []
        )
        if executions:
            # Get first tool execution for backwards compatibility
            first_execution = executions[0]
            expected_tool = first_execution.function_name
            expected_parameters = first_execution.parsed_arguments

            # Extract ALL tool executions and deduplicate
            seen_signatures: set[str] = set()
            for execution in executions:
                tool_call = ExpectedToolCall(
                    tool_name=execution.function_name,
                    parameters=execution.parsed_arguments,
                )
                sig = tool_call.signature()
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    expected_tools.append(tool_call)

            # Get tool schema from tools field (OpenAI format)
            if conversation.tools:
                available_tools = [ToolDefinition.from_openai(tool) for tool in conversation.tools]
                tool_schema = self._get_tool_schema(available_tools, expected_tool)

        # Extract expected answer
        expected_answer = self._extract_expected_answer(conversation)

        # Extract metadata
        metadata_dict: dict[str, str | int | float | bool] = {}
        if conversation.metadata:
            # Filter to only simple types
            for key, value in conversation.metadata.items():
                if isinstance(value, str | int | float | bool):
                    metadata_dict[key] = value

        return GroundTruth(
            query=query,
            expected_tool=expected_tool,
            expected_parameters=expected_parameters,
            expected_tools=expected_tools,
            tool_schema=tool_schema,
            expected_answer=expected_answer,
            conversation_type=self.conversation_type,
            reasoning_style=self.reasoning_style,
            agent_mode=self.agent_mode,
            metadata=metadata_dict,
        )

    def _extract_query(self, conversation: Conversation) -> str:
        """Extract user query from conversation messages.

        Args:
            conversation: Conversation object

        Returns:
            User query string

        Raises:
            ValueError: If no user message found
        """
        # Find first user message
        for message in conversation.messages:
            if message.role == "user":
                return message.content

        # Fallback to question field if present
        if conversation.question:
            return conversation.question

        raise ValueError("No user query found in conversation")

    def _get_tool_schema(
        self,
        available_tools: list[ToolDefinition],
        tool_name: str,
    ) -> ToolDefinition | None:
        """Get tool schema by name from available tools.

        Args:
            available_tools: List of available tool definitions
            tool_name: Name of tool to find

        Returns:
            ToolDefinition if found, None otherwise
        """
        for tool in available_tools:
            if tool.name == tool_name:
                return tool
        return None

    def _extract_expected_answer(self, conversation: Conversation) -> str | None:
        """Extract expected answer from conversation.

        Args:
            conversation: Conversation object

        Returns:
            Expected answer if available, None otherwise
        """
        # Check final_answer field first
        if conversation.final_answer:
            return conversation.final_answer

        # For tool-calling conversations, answer is in last assistant message
        # after tool execution
        if conversation.tool_context:
            # Find last assistant message
            for message in reversed(conversation.messages):
                if message.role == "assistant" and not self._contains_tool_call(message.content):
                    # Skip messages that contain tool calls
                    return message.content

        # For basic conversations, last assistant message is the answer
        for message in reversed(conversation.messages):
            if message.role == "assistant":
                return message.content

        return None

    def _contains_tool_call(self, content: str) -> bool:
        """Check if message content contains a tool call.

        Looks for common tool call patterns:
        - XML: <tool_call>...</tool_call>
        - JSON: {"tool_calls": ...}
        - Function: function_name(...)

        Args:
            content: Message content

        Returns:
            True if tool call detected
        """
        # Check for XML tool call tags
        if "<tool_call>" in content or "</tool_call>" in content:
            return True

        # Check for JSON tool calls
        if "{" in content and "tool_calls" in content:
            try:
                data = json.loads(content)
                if "tool_calls" in data or "function_call" in data:
                    return True
            except json.JSONDecodeError:
                pass

        # Check for function call pattern: func_name(arg1, arg2)
        func_pattern = r"\b[a-z_][a-z0-9_]*\s*\([^)]*\)"
        return bool(re.search(func_pattern, content.lower()))


def parse_batch(
    conversations: list[Conversation],
    conversation_type: Literal["basic", "cot"],
    reasoning_style: Literal["freetext", "agent", "structured", "hybrid"] | None = None,
    agent_mode: Literal["single_turn"] | None = None,
) -> list[GroundTruth]:
    """Parse a batch of conversations to extract ground truth.

    Args:
        conversations: List of Conversation objects
        conversation_type: Type of conversation
        reasoning_style: Reasoning style if cot
        agent_mode: Agent mode if tools are used (single_turn only)

    Returns:
        List of GroundTruth objects
    """
    parser = GroundTruthParser(
        conversation_type=conversation_type,
        reasoning_style=reasoning_style,
        agent_mode=agent_mode,
    )

    ground_truths: list[GroundTruth] = []
    for conversation in conversations:
        try:
            gt = parser.parse(conversation)
            ground_truths.append(gt)
        except ValueError as e:
            # Log error but continue processing
            print(f"Warning: Failed to parse conversation: {e}")
            continue

    return ground_truths
