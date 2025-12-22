"""Tool call parsers for different model architectures.

Each model family outputs tool calls in a different format. This module provides
a registry of parsers that can extract tool calls from generated text based on
the model architecture.
"""

import json
import logging
import re

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    """Parsed tool call in normalized format."""

    name: str
    arguments: dict[str, Any]

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments_string(cls, v: Any) -> dict[str, Any]:
        """Parse arguments if they're a JSON string."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in arguments field: {e}") from e
        return v


class ToolCallParser(ABC):
    """Abstract base class for tool call parsers."""

    @abstractmethod
    def parse(self, text: str) -> list[dict[str, Any]]:
        """Parse tool calls from generated text.

        Args:
            text: Generated text from model

        Returns:
            List of tool call dicts with 'name' and 'arguments' keys
        """

    @staticmethod
    def _validate_tool_call(data: dict) -> dict | None:
        """Validate and normalize a tool call dict.

        Args:
            data: Raw parsed data

        Returns:
            Normalized tool call dict or None if invalid
        """
        try:
            tool_call = ToolCall.model_validate(data)
            return tool_call.model_dump()
        except ValidationError:
            return None


class QwenToolCallParser(ToolCallParser):
    """Parser for Qwen2.5 and Qwen3 models.

    Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    """

    # Pattern matches <tool_call>...</tool_call> with content inside
    TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def parse(self, text: str) -> list[dict[str, Any]]:
        """Parse Qwen-style tool calls."""
        tool_calls = []

        for match in self.TOOL_CALL_PATTERN.finditer(text):
            content = match.group(1).strip()
            try:
                data = json.loads(content)
                if validated := self._validate_tool_call(data):
                    tool_calls.append(validated)
            except json.JSONDecodeError:
                logger.debug("Failed to parse Qwen tool call JSON: %s", content[:100])

        return tool_calls


class LlamaToolCallParser(ToolCallParser):
    """Parser for Llama 3.1/3.2/3.3 models.

    Llama uses multiple formats:
    - JSON with "name" and "parameters" keys
    - <|python_tag|> for code execution
    - {"type": "function", "function": {"name": ..., "arguments": ...}}
    """

    # Pattern for function call JSON objects
    FUNCTION_PATTERN = re.compile(
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{[^}]*\})\s*\}',
        re.DOTALL,
    )

    # Pattern for OpenAI-style tool calls
    OPENAI_PATTERN = re.compile(
        r'\{\s*"type"\s*:\s*"function"\s*,\s*"function"\s*:\s*(\{[^}]+\})\s*\}',
        re.DOTALL,
    )

    def parse(self, text: str) -> list[dict[str, Any]]:
        """Parse Llama-style tool calls."""
        tool_calls = []

        # Try direct name/parameters format
        for match in self.FUNCTION_PATTERN.finditer(text):
            name = match.group(1)
            try:
                params = json.loads(match.group(2))
                data = {"name": name, "arguments": params}
                if validated := self._validate_tool_call(data):
                    tool_calls.append(validated)
            except json.JSONDecodeError:
                logger.debug("Failed to parse Llama parameters: %s", match.group(2)[:100])

        # Try OpenAI-style format if no matches
        if not tool_calls:
            for match in self.OPENAI_PATTERN.finditer(text):
                try:
                    func_data = json.loads(match.group(1))
                    data = {
                        "name": func_data.get("name", ""),
                        "arguments": func_data.get("arguments", {}),
                    }
                    if validated := self._validate_tool_call(data):
                        tool_calls.append(validated)
                except json.JSONDecodeError:
                    logger.debug("Failed to parse Llama OpenAI-style: %s", match.group(1)[:100])

        # Fallback: try to find any JSON with name/arguments or name/parameters
        if not tool_calls:
            tool_calls = self._fallback_json_parse(text)

        return tool_calls

    def _fallback_json_parse(self, text: str) -> list[dict[str, Any]]:
        """Fallback parser that looks for JSON objects with tool call structure."""
        tool_calls = []
        # Find all JSON-like objects
        for match in re.finditer(r"\{[^{}]*\}", text):
            try:
                data = json.loads(match.group())
                # Check for name + arguments OR name + parameters
                if "name" in data:
                    if "arguments" in data:
                        if validated := self._validate_tool_call(data):
                            tool_calls.append(validated)
                    elif "parameters" in data:
                        normalized = {"name": data["name"], "arguments": data["parameters"]}
                        if validated := self._validate_tool_call(normalized):
                            tool_calls.append(validated)
            except json.JSONDecodeError:
                continue
        return tool_calls


class MistralToolCallParser(ToolCallParser):
    """Parser for Mistral and Mixtral models.

    Format: [TOOL_CALLS] [{"name": "func", "arguments": {...}}]
    """

    # Pattern for [TOOL_CALLS] marker followed by JSON array
    TOOL_CALLS_PATTERN = re.compile(
        r"\[TOOL_CALLS\]\s*(\[.*?\])",
        re.DOTALL,
    )

    # Fallback pattern for JSON array of tool calls
    JSON_ARRAY_PATTERN = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)

    def parse(self, text: str) -> list[dict[str, Any]]:
        """Parse Mistral-style tool calls."""
        tool_calls = []

        # Try [TOOL_CALLS] format first
        match = self.TOOL_CALLS_PATTERN.search(text)
        if match:
            try:
                calls_array = json.loads(match.group(1))
                if isinstance(calls_array, list):
                    for call in calls_array:
                        if validated := self._validate_tool_call(call):
                            tool_calls.append(validated)
            except json.JSONDecodeError:
                logger.debug("Failed to parse Mistral TOOL_CALLS array")

        # Fallback to looking for JSON arrays
        if not tool_calls:
            for match in self.JSON_ARRAY_PATTERN.finditer(text):
                try:
                    calls_array = json.loads(match.group())
                    if isinstance(calls_array, list):
                        for call in calls_array:
                            if isinstance(call, dict):
                                validated = self._validate_tool_call(call)
                                if validated:
                                    tool_calls.append(validated)
                except json.JSONDecodeError:
                    continue

        return tool_calls


class HermesToolCallParser(ToolCallParser):
    """Parser for Hermes/Nous-style models.

    Similar to Qwen but may have variations in formatting.
    Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    """

    TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def parse(self, text: str) -> list[dict[str, Any]]:
        """Parse Hermes-style tool calls."""
        tool_calls = []

        for match in self.TOOL_CALL_PATTERN.finditer(text):
            content = match.group(1).strip()
            try:
                data = json.loads(content)
                if validated := self._validate_tool_call(data):
                    tool_calls.append(validated)
            except json.JSONDecodeError:
                logger.debug("Failed to parse Hermes tool call: %s", content[:100])

        return tool_calls


class GenericToolCallParser(ToolCallParser):
    """Generic fallback parser that tries multiple strategies.

    Used when model architecture is unknown or not specifically supported.
    """

    def __init__(self) -> None:
        """Initialize with sub-parsers to try in order."""
        self._parsers = [
            QwenToolCallParser(),
            MistralToolCallParser(),
            LlamaToolCallParser(),
        ]

    def parse(self, text: str) -> list[dict[str, Any]]:
        """Try each parser until one succeeds."""
        for parser in self._parsers:
            tool_calls = parser.parse(text)
            if tool_calls:
                return tool_calls

        # Final fallback: extract any JSON object with name + arguments
        return self._extract_json_objects(text)

    def _extract_json_objects(self, text: str) -> list[dict[str, Any]]:
        """Extract JSON objects that look like tool calls."""
        tool_calls = []
        depth = 0
        start = -1

        for i, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    json_str = text[start : i + 1]
                    try:
                        data = json.loads(json_str)
                        if "name" in data and ("arguments" in data or "parameters" in data):
                            if "parameters" in data and "arguments" not in data:
                                data["arguments"] = data.pop("parameters")
                            if validated := self._validate_tool_call(data):
                                tool_calls.append(validated)
                    except json.JSONDecodeError:
                        pass
                    start = -1

        return tool_calls


# Architecture to parser mapping
ARCHITECTURE_PARSERS: dict[str, type[ToolCallParser]] = {
    # Qwen family
    "Qwen2ForCausalLM": QwenToolCallParser,
    "Qwen2_5ForCausalLM": QwenToolCallParser,
    "Qwen3ForCausalLM": QwenToolCallParser,
    "Qwen2VLForConditionalGeneration": QwenToolCallParser,
    "Qwen2_5_VLForConditionalGeneration": QwenToolCallParser,
    # Llama family
    "LlamaForCausalLM": LlamaToolCallParser,
    "Llama3ForCausalLM": LlamaToolCallParser,
    # Mistral family
    "MistralForCausalLM": MistralToolCallParser,
    "MixtralForCausalLM": MistralToolCallParser,
    "Mistral3ForConditionalGeneration": MistralToolCallParser,
    # Hermes/Nous (often fine-tuned Llama/Mistral with Hermes format)
    "HermesForCausalLM": HermesToolCallParser,
}


class ToolCallParserRegistry:
    """Registry for tool call parsers by model architecture."""

    def __init__(self) -> None:
        """Initialize the registry with default parsers."""
        self._parsers: dict[str, type[ToolCallParser]] = ARCHITECTURE_PARSERS.copy()
        self._fallback = GenericToolCallParser

    def register(self, architecture: str, parser_class: type[ToolCallParser]) -> None:
        """Register a parser for a model architecture.

        Args:
            architecture: Model architecture name (e.g., "LlamaForCausalLM")
            parser_class: Parser class to use for this architecture
        """
        self._parsers[architecture] = parser_class

    def get_parser(self, architectures: list[str] | None) -> ToolCallParser:
        """Get the appropriate parser for a model's architectures.

        Args:
            architectures: List of architecture names from model config

        Returns:
            Instantiated parser for the model
        """
        if architectures:
            for arch in architectures:
                if arch in self._parsers:
                    logger.debug(
                        "Using %s parser for architecture %s", self._parsers[arch].__name__, arch
                    )
                    return self._parsers[arch]()

        logger.debug("No specific parser found, using generic fallback")
        return self._fallback()

    def get_parser_for_model(self, model: str) -> ToolCallParser:
        """Get parser by loading model config and detecting architecture.

        Args:
            model: Path to model or HuggingFace Hub ID

        Returns:
            Instantiated parser for the model
        """
        from transformers import AutoConfig  # noqa: PLC0415

        try:
            config = AutoConfig.from_pretrained(model)  # nosec
            architectures = getattr(config, "architectures", None)
            return self.get_parser(architectures)
        except Exception as e:
            logger.warning("Failed to load model config for parser detection: %s", e)
            return self._fallback()


# Global registry instance
_registry = ToolCallParserRegistry()


def get_parser(architectures: list[str] | None = None) -> ToolCallParser:
    """Get a parser for the given architectures.

    Args:
        architectures: List of architecture names from model config

    Returns:
        Instantiated parser
    """
    return _registry.get_parser(architectures)


def get_parser_for_model(model: str) -> ToolCallParser:
    """Get a parser for a model.

    Args:
        model: Model path or HuggingFace Hub ID

    Returns:
        Instantiated parser
    """
    return _registry.get_parser_for_model(model)


def register_parser(architecture: str, parser_class: type[ToolCallParser]) -> None:
    """Register a custom parser for an architecture.

    Args:
        architecture: Model architecture name
        parser_class: Parser class to use
    """
    _registry.register(architecture, parser_class)
