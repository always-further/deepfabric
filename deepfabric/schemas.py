import ast
import json
import logging
import re
import secrets
import string
import warnings

from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Tool Call ID constants for DeepFabric Format
TOOL_CALL_ID_CHARS = string.ascii_letters + string.digits
TOOL_CALL_ID_LENGTH = 9
TOOL_CALL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9]{9}$")


def generate_tool_call_id() -> str:
    """Generate a 9-character alphanumeric tool call ID.

    Returns:
        A string of exactly 9 alphanumeric characters (A-Z, a-z, 0-9).
    """
    return "".join(secrets.choice(TOOL_CALL_ID_CHARS) for _ in range(TOOL_CALL_ID_LENGTH))


# Type alias for metadata/structured_data fields
# Provider-specific transformations in llm/client.py handle:
# - OpenAI: adds additionalProperties: false
# - Gemini: strips additionalProperties
MetadataDict = dict[str, Any] | None


class ExcludeNoneBaseModel(BaseModel):
    """Base model that excludes None values during serialization."""

    def model_dump(self, **kwargs):
        """Override to always exclude None values for clean serialization."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


# Basic message schema
class ChatMessage(ExcludeNoneBaseModel):
    """A single message in a conversation."""

    model_config = {"extra": "forbid"}

    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The role of the message sender"
    )
    content: str | None = Field(
        default=None, description="The content of the message (optional when tool_calls is present)"
    )
    tool_calls: list["ToolCall"] | None = Field(
        default=None, description="Tool calls made by the assistant (DeepFabric format)"
    )
    tool_call_id: str | None = Field(
        default=None,
        description="ID linking tool result to the original tool call (9-char alphanumeric)",
    )

    @field_validator("tool_call_id")
    @classmethod
    def validate_tool_call_id_format(cls, v: str | None) -> str | None:
        """Validate tool_call_id matches the 9-char alphanumeric format when present."""
        if v is not None and not TOOL_CALL_ID_PATTERN.match(v):
            raise ValueError(f"tool_call_id must be exactly 9 alphanumeric characters, got: '{v}'")
        return v


class ChatTranscript(BaseModel):
    """A complete conversation transcript with messages."""

    messages: list[ChatMessage] = Field(
        description="List of messages in the conversation", min_length=1
    )


class ReasoningStep(BaseModel):
    """A single step in a chain of reasoning."""

    step_number: int = Field(description="The step number in the reasoning chain")
    thought: str = Field(description="The reasoning or thought for this step")
    action: str | None = Field(
        default=None,
        description=(
            "Action taken in this reasoning step. For tool-calling, use one of these formats: "
            "1) Plain function name: 'get_weather' "
            "2) Function call: 'get_weather(city=\"Paris\")' "
            "3) Descriptive text (less reliable): 'I will call the get_weather tool'"
        ),
    )


class StructuredConversation(BaseModel):
    """A conversation with optional structured reasoning and metadata."""

    messages: list[ChatMessage] = Field(
        description="List of messages in the conversation", min_length=1
    )
    reasoning_trace: list[ReasoningStep] | None = Field(
        default=None, description="Optional chain of reasoning steps"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata about the conversation"
    )


# Tool definition schemas for structured tool system
class ToolParameter(BaseModel):
    """A single parameter for a tool/function."""

    name: str = Field(description="Parameter name")
    type: Literal["str", "int", "float", "bool", "list", "dict"] = Field(
        description="Parameter type"
    )
    description: str = Field(description="What this parameter does")
    required: bool = Field(default=True, description="Whether this parameter is required")
    default: str | None = Field(
        default=None,
        description=(
            "Default value if not provided. Stored as string for HuggingFace Datasets compatibility "
            "(Arrow/Parquet requires consistent types). Actual type is preserved in 'type' field."
        ),
    )


# MCP (Model Context Protocol) tool schema models
class MCPInputSchemaProperty(BaseModel):
    """A single property in an MCP input schema."""

    model_config = {"extra": "allow"}

    type: str | list[str] = Field(
        default="string", description="JSON Schema type (string or array for nullable)"
    )
    description: str = Field(default="", description="Property description")
    default: Any | None = Field(default=None, description="Default value")


class MCPInputSchema(BaseModel):
    """MCP tool input schema (JSON Schema format)."""

    model_config = {"extra": "allow"}

    type: str = Field(default="object", description="Schema type")
    properties: dict[str, MCPInputSchemaProperty] = Field(
        default_factory=dict, description="Parameter properties"
    )
    required: list[str] = Field(default_factory=list, description="Required parameter names")


class MCPToolDefinition(BaseModel):
    """MCP (Model Context Protocol) tool definition.

    See: https://modelcontextprotocol.io/specification/2025-06-18/schema#tool
    """

    model_config = {"extra": "allow", "populate_by_name": True}

    name: str = Field(description="Tool name")
    description: str = Field(default="", description="Tool description")
    input_schema: MCPInputSchema | None = Field(
        default=None,
        alias="inputSchema",
        description="JSON Schema for tool parameters (optional, some tools have no params)",
    )

    @property
    def input_schema_safe(self) -> MCPInputSchema:
        """Get input_schema, returning empty schema if None."""
        return self.input_schema or MCPInputSchema()


class ToolDefinition(BaseModel):
    """Complete definition of a tool/function."""

    name: str = Field(description="Tool name (function name)")
    description: str = Field(description="What this tool does")
    parameters: list[ToolParameter] = Field(description="List of parameters this tool accepts")
    returns: str = Field(description="Description of what this tool returns")
    category: str = Field(default="general", description="Tool category for grouping")
    component: str | None = Field(
        default=None,
        description="Spin component name that implements this tool (e.g., 'vfs', 'github-mock')",
    )

    def to_signature(self) -> str:
        """Generate a function signature string."""
        params = []
        for p in self.parameters:
            if p.required:
                params.append(f"{p.name}: {p.type}")
            else:
                params.append(f"{p.name}: {p.type} = {p.default}")
        return f"{self.name}({', '.join(params)}) → {self.returns}"

    def to_openai(self) -> dict[str, Any]:
        """
        Convert tool definition to OpenAI function calling schema format.

        This format is compatible with TRL's SFTTrainer and other HuggingFace
        training frameworks that support tool/function calling.

        Returns:
            Dictionary in OpenAI function calling schema format with:
            - type: Always "function"
            - function: Object containing name, description, and parameters schema
        """
        # Map DeepFabric types to JSON Schema types
        type_mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }

        properties = {}
        required = []

        for param in self.parameters:
            json_type = type_mapping.get(param.type, "string")
            properties[param.name] = {
                "type": json_type,
                "description": param.description,
            }

            # Add default value if present and not required
            # Convert string default back to proper type for JSON Schema
            if not param.required and param.default is not None and param.default != "":
                default_value = param.default
                # Convert string representation back to typed value
                try:
                    if param.type == "int":
                        default_value = int(param.default)
                    elif param.type == "float":
                        default_value = float(param.default)
                    elif param.type == "bool":
                        default_value = param.default.lower() in ("true", "1", "yes")
                    elif param.type in ("list", "dict"):
                        # Handle special cases
                        if param.default.lower() == "none":
                            default_value = None
                        else:
                            # Try JSON first
                            try:
                                default_value = json.loads(param.default)
                            except json.JSONDecodeError:
                                # Fallback: try Python literal (e.g., ['markdown'] -> ["markdown"])
                                try:
                                    default_value = ast.literal_eval(param.default)
                                except (ValueError, SyntaxError):
                                    # Give up and skip this default
                                    continue
                    # str remains as-is
                    properties[param.name]["default"] = default_value
                except (ValueError, AttributeError):
                    # Skip invalid default values (e.g., empty strings for int/float)
                    pass

            # Track required parameters
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @classmethod
    def from_openai(cls, openai_tool: dict[str, Any]) -> "ToolDefinition":
        """Create a ToolDefinition from OpenAI function calling schema format.

        Args:
            openai_tool: Dictionary in OpenAI format with type="function" and function object

        Returns:
            ToolDefinition instance
        """
        # Reverse type mapping
        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        }

        func = openai_tool.get("function", {})
        name = func.get("name", "")
        description = func.get("description", "")
        params_schema = func.get("parameters", {})

        properties = params_schema.get("properties") or {}
        required_params = set(params_schema.get("required") or [])

        parameters = []
        for param_name, param_props in properties.items():
            # Skip None/null property values (can occur in some tool schemas)
            if param_props is None:
                continue
            json_type = param_props.get("type", "string")
            df_type = type_mapping.get(json_type, "str")

            default = param_props.get("default")
            default_str = str(default) if default is not None else ""

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=df_type,  # type: ignore[arg-type]
                    description=param_props.get("description", ""),
                    required=param_name in required_params,
                    default=default_str,
                )
            )

        return cls(
            name=name,
            description=description,
            parameters=parameters,
            returns="",  # OpenAI format doesn't include return description
            category="general",
        )

    @classmethod
    def from_mcp(cls, mcp_tool: MCPToolDefinition | dict[str, Any]) -> "ToolDefinition":
        """Create a ToolDefinition from MCP (Model Context Protocol) tool schema.

        Args:
            mcp_tool: Either an MCPToolDefinition instance or a dict in MCP format

        Returns:
            ToolDefinition instance
        """
        # Type mapping from JSON Schema to DeepFabric types
        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        }

        # Parse dict to MCPToolDefinition if needed
        if isinstance(mcp_tool, dict):
            mcp_tool = MCPToolDefinition.model_validate(mcp_tool)

        # Use safe property to handle None input_schema
        input_schema = mcp_tool.input_schema_safe
        parameters = []
        required_params = set(input_schema.required)

        for param_name, param_props in input_schema.properties.items():
            # Handle type as either string or array (for nullable types like ["string", "null"])
            param_type = param_props.type
            if isinstance(param_type, list):
                # Extract the primary type (non-null type from array)
                primary_type = next((t for t in param_type if t != "null"), "string")
            else:
                primary_type = param_type

            df_type = type_mapping.get(primary_type, "str")
            default_str = str(param_props.default) if param_props.default is not None else ""

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=df_type,  # type: ignore[arg-type]
                    description=param_props.description,
                    required=param_name in required_params,
                    default=default_str,
                )
            )

        return cls(
            name=mcp_tool.name,
            description=mcp_tool.description,
            parameters=parameters,
            returns="",  # MCP format doesn't include return description
            category="general",
        )


class ToolRegistry(BaseModel):
    """Registry of available tools."""

    tools: list[ToolDefinition] = Field(description="List of available tool definitions")

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return next((t for t in self.tools if t.name == name), None)

    def get_tools_by_category(self, category: str) -> list[ToolDefinition]:
        """Get all tools in a category."""
        return [t for t in self.tools if t.category == category]

    def get_tool_names(self) -> list[str]:
        """Get list of all tool names."""
        return [t.name for t in self.tools]

    def to_openai_format(self) -> list[dict[str, Any]]:
        """
        Convert all tools to OpenAI function calling schema format.

        This method is specifically designed for use with HuggingFace TRL's
        SFTTrainer and other training frameworks that require tools to be
        provided in OpenAI function calling format.

        Returns:
            List of tool definitions in OpenAI function calling schema format.
            Each tool includes type="function" and a function object with
            name, description, and parameters.

        Example:
            >>> registry = ToolRegistry(tools=[...])
            >>> trl_tools = registry.to_openai_format()
            >>> # Use in dataset: {"messages": [...], "tools": trl_tools}
        """
        return [tool.to_openai() for tool in self.tools]


# Agent tool-calling schemas
class ToolReasoningStep(BaseModel):
    """A reasoning step that leads to tool selection and parameter construction."""

    step_number: int = Field(description="The step number in the tool planning sequence")
    reasoning: str = Field(description="Why this tool is needed at this point")
    selected_tool_name: str = Field(description="Name of the tool being selected")
    parameter_reasoning: str = Field(description="Reasoning for parameter values")
    expected_result: str = Field(description="What the tool should return and how it helps")


class ToolExecution(BaseModel):
    """Represents actual execution of a tool with reasoning context."""

    function_name: str = Field(min_length=1, description="Name of the function/tool being called")
    arguments: str = Field(
        min_length=2, description="JSON string of arguments passed to the function"
    )
    reasoning: str = Field(min_length=1, description="Brief explanation of why executing now")
    result: str = Field(min_length=1, description="The result returned from the tool execution")

    @field_validator("arguments")
    @classmethod
    def validate_arguments_json(cls, v: str) -> str:
        """Validate that arguments are valid JSON with no null/empty placeholders.

        Empty objects {} are allowed for parameterless tools like list_files().
        """
        stripped = v.strip()

        # Parse and validate JSON
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    if value is None:
                        raise ValueError(f"Argument '{key}' is null - must provide actual value")
                    if isinstance(value, str) and value == "":
                        raise ValueError(
                            f"Argument '{key}' is empty string - must provide actual value"
                        )
        except json.JSONDecodeError as e:
            raise ValueError(f"Arguments must be valid JSON: {e}") from e

        return v

    @property
    def parsed_arguments(self) -> dict[str, Any]:
        """Parse arguments JSON string to dict.

        Uses Any for values as function arguments can be strings, numbers, booleans, lists, nested dicts, etc.
        """
        return json.loads(self.arguments)

    def to_tool_call(self, tool_call_id: str | None = None) -> "ToolCall":
        """Convert ToolExecution to a ToolCall for the final dataset.

        Args:
            tool_call_id: The 9-char alphanumeric ID. If None, generates one.

        Returns:
            ToolCall with JSON string arguments (HuggingFace compatible).
        """
        return ToolCall(
            id=tool_call_id or generate_tool_call_id(),
            type="function",
            function=ToolCallFunction(
                name=self.function_name,
                arguments=self.arguments,  # Pass string directly
            ),
        )

    class Config:
        extra = "forbid"


class PendingToolCall(BaseModel):
    """A tool call request before execution (no result yet).

    Used in AgentStep for the ReAct loop - the LLM generates these,
    then Spin executes them and populates the result separately.
    """

    function_name: str = Field(min_length=1, description="Name of the function/tool to call")
    arguments: str = Field(
        min_length=2, description="JSON string of arguments to pass to the function"
    )
    reasoning: str = Field(min_length=1, description="Brief explanation of why calling this tool")

    @field_validator("arguments")
    @classmethod
    def validate_arguments_json(cls, v: str) -> str:
        """Validate that arguments are valid JSON with no null/empty placeholders.

        Empty objects {} are allowed for parameterless tools like list_files().
        """
        stripped = v.strip()

        # Parse and validate JSON
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    if value is None:
                        raise ValueError(f"Argument '{key}' is null - must provide actual value")
                    if isinstance(value, str) and value == "":
                        raise ValueError(
                            f"Argument '{key}' is empty string - must provide actual value"
                        )
        except json.JSONDecodeError as e:
            raise ValueError(f"Arguments must be valid JSON: {e}") from e

        return v

    @property
    def parsed_arguments(self) -> dict[str, Any]:
        """Parse arguments JSON string to dict."""
        return json.loads(self.arguments)

    def to_tool_execution(self, result: str) -> "ToolExecution":
        """Convert to ToolExecution after getting result from Spin.

        Args:
            result: The result from tool execution

        Returns:
            ToolExecution with the result populated
        """
        return ToolExecution(
            function_name=self.function_name,
            arguments=self.arguments,
            reasoning=self.reasoning,
            result=result,
        )

    class Config:
        extra = "forbid"


class AgentStep(BaseModel):
    """A single step in ReAct-style agent reasoning.

    Each step represents one iteration of the think-act-observe loop:
    1. Agent thinks about what to do next (thought)
    2. Agent decides on tool calls for THIS step only
    3. Tools are executed and results observed
    4. Process repeats until is_final=True

    This ensures tool calls are made based on observed results,
    not hallucinated assumptions about what tools will return.
    """

    thought: str = Field(
        min_length=1,
        description="Agent's reasoning about what to do next based on observations so far",
    )
    tool_calls: list["PendingToolCall"] = Field(
        default_factory=list,
        description="Tool calls for THIS step only. Empty if agent is done.",
    )
    is_final: bool = Field(
        default=False,
        description="True if agent has enough information and is ready to respond to user",
    )

    class Config:
        extra = "forbid"


# Tool calling schemas for conversations that include function calls
class FunctionCall(BaseModel):
    """A function call with arguments."""

    name: str = Field(description="The name of the function to call")
    arguments: dict[str, Any] = Field(description="Arguments to pass to the function")


def _serialize_arguments(v: dict[str, Any] | str) -> str:
    """Serialize arguments to JSON string, stripping None values.

    This ensures consistent schema for HuggingFace datasets (Arrow/Parquet compatibility).
    Accepts both dict and str inputs for backward compatibility.
    """
    match v:
        case dict():
            cleaned = {k: val for k, val in v.items() if val is not None}
            return json.dumps(cleaned, separators=(",", ":"))
        case str():
            # Validate JSON, strip nulls, re-serialize for consistency
            parsed = json.loads(v)
            cleaned = (
                {k: val for k, val in parsed.items() if val is not None}
                if isinstance(parsed, dict)
                else parsed
            )
            return json.dumps(cleaned, separators=(",", ":"))
        case _:
            raise ValueError(f"arguments must be dict or str, got {type(v)}")


# Type alias for JSON-serialized arguments (HuggingFace compatible)
ArgumentsStr = Annotated[str, BeforeValidator(_serialize_arguments)]


class ToolCallFunction(ExcludeNoneBaseModel):
    """Function details within a tool call (DeepFabric Format)."""

    name: str = Field(min_length=1, description="The name of the function to call")
    arguments: ArgumentsStr = Field(description="Arguments as JSON string (HuggingFace compatible)")

    @property
    def parsed_arguments(self) -> dict[str, Any]:
        """Parse arguments JSON string to dict when needed at runtime."""
        return json.loads(self.arguments)

    class Config:
        extra = "forbid"


class ToolCall(ExcludeNoneBaseModel):
    """A tool call in DeepFabric Format.

    Implements the DeepFabric Format specification:
    - ID: Exactly 9 alphanumeric characters (A-Z, a-z, 0-9)
    - Type: Always "function"
    - Arguments: JSON string (for HuggingFace Arrow/Parquet compatibility)
    """

    id: str = Field(
        min_length=9,
        max_length=9,
        description="Unique tool call ID - exactly 9 alphanumeric characters",
    )
    type: Literal["function"] = Field(default="function", description="Tool call type")
    function: ToolCallFunction = Field(description="Function call details")

    @field_validator("id")
    @classmethod
    def validate_tool_call_id(cls, v: str) -> str:
        """Validate that ID is exactly 9 alphanumeric characters."""
        if not TOOL_CALL_ID_PATTERN.match(v):
            raise ValueError(
                f"Tool call ID must be exactly 9 alphanumeric characters (A-Z, a-z, 0-9), got: '{v}'"
            )
        return v

    class Config:
        extra = "forbid"


# Resolve forward reference for ChatMessage.tool_calls
ChatMessage.model_rebuild()


class ToolMessage(BaseModel):
    """A message that includes tool/function calling."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The role of the message sender"
    )
    content: str | None = Field(default=None, description="The text content of the message")
    function_call: FunctionCall | None = Field(
        default=None, description="Function call made by the assistant"
    )
    tool_calls: list[FunctionCall] | None = Field(
        default=None, description="Multiple tool calls made by the assistant"
    )


class ToolConversation(BaseModel):
    """A conversation that may include function/tool calls."""

    messages: list[ToolMessage] = Field(
        description="List of messages that may include tool calls", min_length=1
    )


# Chain of Thought schemas
class FreeTextCoT(BaseModel):
    """Chain of Thought dataset in free-text format (GSM8K style)."""

    question: str = Field(description="The question or problem to solve")
    cot: str = Field(description="Natural language reasoning explanation")
    final_answer: str = Field(description="The definitive answer to the question")


class StructuredCoT(BaseModel):
    """Chain of Thought dataset with structured reasoning trace."""

    messages: list[ChatMessage] = Field(description="Conversation messages", min_length=1)
    reasoning_trace: list[ReasoningStep] = Field(
        description="Structured reasoning steps", min_length=1
    )
    final_answer: str = Field(description="The definitive answer to the question")


class HybridCoT(BaseModel):
    """Chain of Thought dataset with both free-text and structured reasoning."""

    question: str = Field(description="The question or problem to solve")
    cot: str = Field(description="Natural language reasoning explanation")
    reasoning_trace: list[ReasoningStep] = Field(
        description="Structured reasoning steps", min_length=1
    )
    final_answer: str = Field(description="The definitive answer to the question")


# Mathematical variants with numerical-only final answers
class MathematicalAnswerMixin:
    """Mixin class providing mathematical answer formatting and validation."""

    @classmethod
    def _format_mathematical_answer(cls, v: str) -> str:
        """Format mathematical answers with strict consistency rules."""
        v_stripped = v.strip()

        # Handle cases where model returns multiple answers (e.g., "2, 3")
        # Take the first one if comma-separated list detected
        if ", " in v_stripped:
            v_stripped = v_stripped.split(", ")[0].strip()

        # Basic validation pattern
        pattern = r"^-?\d{1,3}(,\d{3})*(\.\d+)?([eE][+-]?\d+)?$|^-?\d+(\.\d+)?([eE][+-]?\d+)?$"
        if not re.match(pattern, v_stripped):
            msg = f"final_answer must be numerical, got: {v}"
            raise ValueError(msg)

        # Remove commas for processing
        v_clean = v_stripped.replace(",", "")

        # Apply formatting rules for consistency
        if cls._is_scientific_notation(v_clean):
            return v_clean  # Preserve scientific notation

        if "." in v_clean:
            decimal_parts = v_clean.split(".")
            if len(decimal_parts) == 2:  # noqa: PLR2004
                decimal_places = len(decimal_parts[1])
                # Round to 2 decimal places for precision artifacts
                if decimal_places >= 3:  # noqa: PLR2004
                    num = Decimal(v_clean)
                    rounded = num.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    v_clean = str(rounded)

        return v_clean

    @staticmethod
    def _is_scientific_notation(value: str) -> bool:
        """Detect scientific notation."""
        return "e" in value.lower()


# Capability Models for Composable Conversation Schema
class ReasoningTrace(BaseModel):
    """Reasoning capability - present when conversation_type='cot'."""

    style: Literal["freetext", "agent"] = Field(
        description="The reasoning style: freetext (natural language) or agent (structured step-by-step for tool-calling)"
    )
    content: str | list[ReasoningStep] = Field(
        description="Reasoning content - string for freetext, list of ReasoningStep for agent"
    )

    class Config:
        extra = "forbid"


class ToolContext(BaseModel):
    """Tool execution history - present when tools are used.

    Note: available_tools has been deprecated in favor of the top-level 'tools'
    field in Conversation. Use 'tools' for the OpenAI-format tool definitions
    needed by chat templates. The field is kept here for backward compatibility
    with existing datasets.
    """

    executions: list[ToolExecution] = Field(
        default_factory=list,
        description="Tool executions performed during the conversation (may be empty if agent answered without tools)",
    )
    # Deprecated: kept for backward compatibility with existing datasets
    available_tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="Deprecated: Use top-level 'tools' field instead. Kept for backward compatibility.",
    )

    @model_validator(mode="after")
    def warn_on_deprecated_available_tools(self) -> "ToolContext":
        """Warn if the deprecated `available_tools` field is used."""
        if self.available_tools is not None:
            warnings.warn(
                (
                    "'tool_context.available_tools' is deprecated and will be removed in a future version. "
                    "Use the top-level 'tools' field in Conversation instead."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        return self

    class Config:
        extra = "forbid"


class AgentContext(BaseModel):
    """Agent capability - present when tools are configured for agent mode."""

    mode: Literal["single_turn"] = Field(
        default="single_turn",
        description="Agent interaction mode (single_turn is the only supported mode)",
    )
    planning_trace: str | None = Field(
        default=None, description="Agent's planning and reasoning about tool usage strategy"
    )
    execution_summary: str | None = Field(
        default=None, description="Summary of agent's execution and results interpretation"
    )

    class Config:
        extra = "forbid"


class Conversation(ExcludeNoneBaseModel):
    """
    Unified conversation schema with optional capability fields.

    This composable schema supports various combinations:
    - Basic conversation: just messages
    - With reasoning: messages + reasoning capability
    - With tools: messages + tool_context capability
    - Agent mode: messages + tool_context + agent_context capabilities
    - Full combination: all capabilities enabled

    The schema validates that capability combinations are consistent
    (e.g., agent_context requires tool_context).
    """

    messages: list[ChatMessage] = Field(description="Core conversation messages", min_length=1)
    metadata: MetadataDict = Field(
        default=None, description="Conversation metadata (topic, domain, etc.)"
    )

    # Optional capability fields - use empty strings/dicts instead of None for OpenAI compatibility
    reasoning: ReasoningTrace | None = Field(
        default=None, description="Reasoning capability - chain of thought traces"
    )
    tool_context: ToolContext | None = Field(
        default=None, description="Tool capability - available tools and executions"
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="OpenAI-compatible tool definitions (populated from tool_context for training)",
    )
    agent_context: AgentContext | None = Field(
        default=None, description="Agent capability - agentic behavior and planning"
    )
    structured_data: MetadataDict = Field(
        default=None, description="Additional structured data for specific formats"
    )

    # Fields for backward compatibility and specific use cases
    question: str = Field(default="", description="Original question (useful for Q&A formats)")
    final_answer: str = Field(default="", description="Final answer (useful for reasoning formats)")

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning_trace(cls, v: ReasoningTrace | None) -> ReasoningTrace | None:
        """Validate reasoning trace content matches style."""
        if v is None:
            return None

        if v.style == "agent" and not isinstance(v.content, list):
            msg = (
                f"Reasoning style '{v.style}' requires list of ReasoningStep, got {type(v.content)}"
            )
            raise ValueError(msg)
        if v.style == "freetext" and not isinstance(v.content, str):
            msg = f"Reasoning style 'freetext' requires string content, got {type(v.content)}"
            raise ValueError(msg)

        return v

    @field_validator("agent_context")
    @classmethod
    def validate_agent_requires_tools(cls, v: AgentContext | None, info) -> AgentContext | None:
        """Validate that agent_context requires tool_context."""
        if v is not None:
            # Access tool_context from the model data
            tool_context = info.data.get("tool_context")
            if tool_context is None:
                msg = "agent_context requires tool_context to be present"
                raise ValueError(msg)
        return v

    class Config:
        extra = "forbid"
        json_schema_extra = {"additionalProperties": False}


class FormattedSample(BaseModel):
    """Output from chat template formatting.

    This model represents a dataset sample after applying a model-specific
    chat template via the Dataset.format() method. The text field contains
    the fully formatted prompt ready for training or inference.

    Attributes:
        text: Formatted text with chat template applied (e.g., ChatML, Llama format)

    Example:
        >>> sample = FormattedSample(text="<|im_start|>user\\nHello<|im_end|>...")
        >>> sample.text[:20]
        '<|im_start|>user\\nHe'
    """

    text: str = Field(..., min_length=1, description="Chat template formatted text")

    class Config:
        extra = "forbid"


# Unified conversation schema mapping
CONVERSATION_SCHEMAS = {
    "basic": Conversation,
    "cot": Conversation,
}


def get_conversation_schema(
    conversation_type: str = "basic",
) -> type[Conversation]:
    """Get the appropriate schema for a conversation configuration.

    With the unified Conversation schema, this now always returns Conversation.
    The schema's capability fields (reasoning, tool_context, agent_context) are
    populated based on the configuration during generation.

    Args:
        conversation_type: Type of conversation (basic, cot)

    Returns:
        Conversation schema (unified for all types)

    Raises:
        ValueError: If conversation_type is not supported
    """
    if conversation_type not in CONVERSATION_SCHEMAS:
        valid_types = ", ".join(CONVERSATION_SCHEMAS.keys())
        msg = f"Unsupported conversation type: {conversation_type}. Valid types: {valid_types}"
        raise ValueError(msg)

    # All types now use the unified Conversation schema
    # Capabilities are populated during generation based on config
    return CONVERSATION_SCHEMAS[conversation_type]


# Topic generation schemas for tree and graph (needed by other modules)
class TopicList(BaseModel):
    """A list of subtopics for tree/graph generation."""

    subtopics: list[str] = Field(
        description="List of subtopic names",
        min_length=1,
    )


class TopicNode(BaseModel):
    """A topic node with subtopics for graph generation."""

    topic: str = Field(description="The topic name")
    subtopics: list[str] = Field(
        description="List of subtopic names",
        default_factory=list,
    )


class GraphSubtopic(BaseModel):
    """A subtopic with connections for graph generation."""

    topic: str = Field(description="The subtopic name")
    connections: list[int] = Field(
        default_factory=list,
        description="List of existing node IDs to connect to, empty list if none",
    )


class GraphSubtopics(BaseModel):
    """List of subtopics with connections for graph generation."""

    subtopics: list[GraphSubtopic] = Field(
        description="List of subtopics with their connections",
        min_length=1,
    )
