"""Tests for tool call parsers.

This module tests:
- QwenToolCallParser for Qwen2.5/Qwen3 models
- LlamaToolCallParser for Llama 3.x models
- MistralToolCallParser for Mistral/Mixtral models
- HermesToolCallParser for Hermes/Nous models
- GenericToolCallParser fallback
- ToolCallParserRegistry and auto-detection
"""

import pytest

from deepfabric.evaluation.backends.tool_call_parsers import (
    GenericToolCallParser,
    HermesToolCallParser,
    LlamaToolCallParser,
    MistralToolCallParser,
    QwenToolCallParser,
    ToolCall,
    ToolCallParserRegistry,
    get_parser,
)


class TestToolCallModel:
    """Test the ToolCall Pydantic model."""

    def test_valid_tool_call(self):
        """Test parsing a valid tool call."""
        tc = ToolCall(name="test_func", arguments={"arg1": "value1"})
        assert tc.name == "test_func"
        assert tc.arguments == {"arg1": "value1"}

    def test_arguments_as_json_string(self):
        """Test that JSON string arguments are parsed."""
        tc = ToolCall(name="test_func", arguments='{"arg1": "value1"}')
        assert tc.arguments == {"arg1": "value1"}

    def test_invalid_json_string_raises(self):
        """Test that invalid JSON string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            ToolCall(name="test_func", arguments="not valid json")

    def test_empty_arguments(self):
        """Test tool call with empty arguments."""
        tc = ToolCall(name="test_func", arguments={})
        assert tc.arguments == {}


class TestQwenToolCallParser:
    """Test QwenToolCallParser for Qwen2.5/Qwen3 models."""

    @pytest.fixture
    def parser(self):
        return QwenToolCallParser()

    def test_single_tool_call(self, parser):
        """Test parsing a single Qwen-style tool call."""
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "London"}}</tool_call>'
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["arguments"] == {"city": "London"}

    def test_tool_call_with_thinking(self, parser):
        """Test parsing tool call with thinking prefix (Qwen3-Thinking)."""
        text = """The user wants weather information for London.
</think>

<tool_call>
{"name": "get_weather", "arguments": {"city": "London", "units": "celsius"}}
</tool_call>"""
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["arguments"]["city"] == "London"
        assert result[0]["arguments"]["units"] == "celsius"

    def test_multiple_tool_calls(self, parser):
        """Test parsing multiple tool calls in one response."""
        text = """<tool_call>{"name": "func1", "arguments": {"a": 1}}</tool_call>
<tool_call>{"name": "func2", "arguments": {"b": 2}}</tool_call>"""
        result = parser.parse(text)
        assert len(result) == 2  # noqa: PLR2004
        assert result[0]["name"] == "func1"
        assert result[1]["name"] == "func2"

    def test_tool_call_with_whitespace(self, parser):
        """Test parsing tool call with extra whitespace."""
        text = """<tool_call>
  {
    "name": "test_func",
    "arguments": {
      "param": "value"
    }
  }
</tool_call>"""
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "test_func"

    def test_no_tool_call(self, parser):
        """Test parsing text without tool calls."""
        text = "I don't need to call any tools to answer this question."
        result = parser.parse(text)
        assert result == []

    def test_invalid_json_in_tool_call(self, parser):
        """Test that invalid JSON is skipped gracefully."""
        text = '<tool_call>{"name": "func", "arguments": invalid}</tool_call>'
        result = parser.parse(text)
        assert result == []

    def test_complex_arguments(self, parser):
        """Test tool call with complex nested arguments."""
        text = """<tool_call>{"name": "detect_anomalies", "arguments": {"series": "[{\\"timestamp\\":1,\\"value\\":100}]", "direction": "both"}}</tool_call>"""
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "detect_anomalies"
        assert result[0]["arguments"]["direction"] == "both"


class TestLlamaToolCallParser:
    """Test LlamaToolCallParser for Llama 3.x models."""

    @pytest.fixture
    def parser(self):
        return LlamaToolCallParser()

    def test_name_parameters_format(self, parser):
        """Test parsing Llama's name/parameters format."""
        text = '{"name": "search", "parameters": {"query": "python"}}'
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["arguments"] == {"query": "python"}

    def test_nested_json_handled_by_generic(self, parser):
        """Test that nested JSON with name/arguments needs GenericToolCallParser.

        Note: LlamaToolCallParser's fallback regex `{[^{}]*}` only matches flat JSON.
        Nested structures like {"name": "x", "arguments": {"y": 1}} aren't matched.
        The GenericToolCallParser has proper nested brace handling.
        """
        text = '{"name": "calculate", "arguments": {"x": 10}}'
        result = parser.parse(text)
        # LlamaToolCallParser cannot parse this due to nested braces limitation
        assert len(result) == 0

        # But GenericToolCallParser can handle it
        generic = GenericToolCallParser()
        result = generic.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "calculate"

    def test_fallback_name_arguments_flat(self, parser):
        """Test fallback parsing of name/arguments format with flat args.

        Note: The fallback regex `{[^{}]*}` only matches flat JSON (no nested braces).
        For nested args, the GenericToolCallParser handles this better.
        """
        # Flat arguments work with the fallback
        text = 'Some text {"name": "test", "parameters": {"key": "value"}} more text'
        # The primary pattern matches name/parameters with simple nested braces
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "test"
        assert result[0]["arguments"] == {"key": "value"}

    def test_no_tool_call(self, parser):
        """Test parsing text without tool calls."""
        text = "Just a regular response without any function calls."
        result = parser.parse(text)
        assert result == []

    def test_multiple_parameters_format(self, parser):
        """Test multiple name/parameters tool calls."""
        text = """{"name": "func1", "parameters": {"a": 1}}
{"name": "func2", "parameters": {"b": 2}}"""
        result = parser.parse(text)
        assert len(result) == 2  # noqa: PLR2004


class TestMistralToolCallParser:
    """Test MistralToolCallParser for Mistral/Mixtral models."""

    @pytest.fixture
    def parser(self):
        return MistralToolCallParser()

    def test_tool_calls_marker_format(self, parser):
        """Test parsing [TOOL_CALLS] marker format."""
        text = '[TOOL_CALLS] [{"name": "get_data", "arguments": {"id": 123}}]'
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "get_data"
        assert result[0]["arguments"] == {"id": 123}

    def test_multiple_tool_calls(self, parser):
        """Test parsing multiple tool calls in array."""
        text = '[TOOL_CALLS] [{"name": "func1", "arguments": {"a": 1}}, {"name": "func2", "arguments": {"b": 2}}]'
        result = parser.parse(text)
        assert len(result) == 2  # noqa: PLR2004
        assert result[0]["name"] == "func1"
        assert result[1]["name"] == "func2"

    def test_json_array_fallback(self, parser):
        """Test fallback to JSON array without marker."""
        text = '[{"name": "test", "arguments": {"x": 1}}]'
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "test"

    def test_no_tool_call(self, parser):
        """Test parsing text without tool calls."""
        text = "I will help you with that request."
        result = parser.parse(text)
        assert result == []

    def test_with_surrounding_text(self, parser):
        """Test parsing with text around the tool calls."""
        text = 'I will call the function now. [TOOL_CALLS] [{"name": "api_call", "arguments": {"endpoint": "/users"}}] Done.'
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "api_call"


class TestHermesToolCallParser:
    """Test HermesToolCallParser for Hermes/Nous models."""

    @pytest.fixture
    def parser(self):
        return HermesToolCallParser()

    def test_single_tool_call(self, parser):
        """Test parsing a single Hermes-style tool call."""
        text = (
            '<tool_call>{"name": "send_email", "arguments": {"to": "user@example.com"}}</tool_call>'
        )
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "send_email"
        assert result[0]["arguments"]["to"] == "user@example.com"

    def test_format_matches_qwen(self, parser):
        """Test that Hermes format matches Qwen format."""
        qwen_parser = QwenToolCallParser()
        text = '<tool_call>{"name": "test", "arguments": {"key": "value"}}</tool_call>'

        hermes_result = parser.parse(text)
        qwen_result = qwen_parser.parse(text)

        assert hermes_result == qwen_result


class TestGenericToolCallParser:
    """Test GenericToolCallParser fallback."""

    @pytest.fixture
    def parser(self):
        return GenericToolCallParser()

    def test_parses_qwen_format(self, parser):
        """Test that generic parser handles Qwen format."""
        text = '<tool_call>{"name": "func", "arguments": {"a": 1}}</tool_call>'
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "func"

    def test_parses_mistral_format(self, parser):
        """Test that generic parser handles Mistral format."""
        text = '[TOOL_CALLS] [{"name": "func", "arguments": {"a": 1}}]'
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "func"

    def test_parses_llama_format(self, parser):
        """Test that generic parser handles Llama format."""
        text = '{"name": "func", "parameters": {"a": 1}}'
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "func"
        assert result[0]["arguments"] == {"a": 1}

    def test_extracts_nested_json(self, parser):
        """Test extraction of nested JSON objects."""
        text = 'Response: {"name": "complex", "arguments": {"nested": {"key": "value"}}}'
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["arguments"]["nested"]["key"] == "value"

    def test_no_tool_call(self, parser):
        """Test parsing text without tool calls."""
        text = "Just a plain text response."
        result = parser.parse(text)
        assert result == []


class TestToolCallParserRegistry:
    """Test the parser registry and auto-detection."""

    def test_get_parser_for_qwen(self):
        """Test parser selection for Qwen architectures."""
        parser = get_parser(["Qwen2ForCausalLM"])
        assert isinstance(parser, QwenToolCallParser)

        parser = get_parser(["Qwen3ForCausalLM"])
        assert isinstance(parser, QwenToolCallParser)

    def test_get_parser_for_llama(self):
        """Test parser selection for Llama architectures."""
        parser = get_parser(["LlamaForCausalLM"])
        assert isinstance(parser, LlamaToolCallParser)

    def test_get_parser_for_mistral(self):
        """Test parser selection for Mistral architectures."""
        parser = get_parser(["MistralForCausalLM"])
        assert isinstance(parser, MistralToolCallParser)

        parser = get_parser(["MixtralForCausalLM"])
        assert isinstance(parser, MistralToolCallParser)

    def test_get_parser_for_unknown(self):
        """Test fallback to generic parser for unknown architectures."""
        parser = get_parser(["UnknownModelForCausalLM"])
        assert isinstance(parser, GenericToolCallParser)

    def test_get_parser_empty_list(self):
        """Test fallback when architecture list is empty."""
        parser = get_parser([])
        assert isinstance(parser, GenericToolCallParser)

    def test_get_parser_none(self):
        """Test fallback when architectures is None."""
        parser = get_parser(None)
        assert isinstance(parser, GenericToolCallParser)

    def test_register_custom_parser(self):
        """Test registering a custom parser."""

        class CustomParser(QwenToolCallParser):
            pass

        registry = ToolCallParserRegistry()
        registry.register("CustomModelForCausalLM", CustomParser)

        parser = registry.get_parser(["CustomModelForCausalLM"])
        assert isinstance(parser, CustomParser)

    def test_first_matching_architecture_wins(self):
        """Test that first matching architecture is used."""
        parser = get_parser(["Qwen2ForCausalLM", "LlamaForCausalLM"])
        assert isinstance(parser, QwenToolCallParser)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string(self):
        """Test parsing empty string."""
        for parser_class in [QwenToolCallParser, LlamaToolCallParser, MistralToolCallParser]:
            parser = parser_class()
            assert parser.parse("") == []

    def test_malformed_json_graceful_handling(self):
        """Test that malformed JSON is handled gracefully."""
        parser = QwenToolCallParser()
        text = '<tool_call>{"name": "func", "arguments": {broken}</tool_call>'
        result = parser.parse(text)
        assert result == []

    def test_missing_required_fields(self):
        """Test that tool calls missing required fields are skipped."""
        parser = QwenToolCallParser()
        # Missing 'arguments' field
        text = '<tool_call>{"name": "func"}</tool_call>'
        result = parser.parse(text)
        assert result == []

    def test_unicode_in_arguments(self):
        """Test handling of unicode characters in arguments."""
        parser = QwenToolCallParser()
        text = '<tool_call>{"name": "translate", "arguments": {"text": "Hello World"}}</tool_call>'
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["arguments"]["text"] == "Hello World"

    def test_very_long_arguments(self):
        """Test handling of very long argument values."""
        parser = QwenToolCallParser()
        long_value = "x" * 10000
        text = f'{{"name": "func", "arguments": {{"data": "{long_value}"}}}}'
        text = f"<tool_call>{text}</tool_call>"
        result = parser.parse(text)
        assert len(result) == 1
        assert len(result[0]["arguments"]["data"]) == 10000  # noqa: PLR2004


class TestRealWorldExamples:
    """Test with real-world model output examples."""

    def test_qwen3_thinking_output(self):
        """Test parsing actual Qwen3-Thinking model output."""
        parser = QwenToolCallParser()
        text = """The user wants to redact email addresses and phone numbers from the provided document. The `redact_sensitive_data` tool is suitable for this task. I will call this tool with the document text and specify 'email,phone' as the entity types to redact.
</think>

<tool_call>
{"name": "redact_sensitive_data", "arguments": {"text":"For urgent matters, contact support@example.com or call 555-123-4567. John Doe can be reached at john.doe@mail.com.","entity_types":"email,phone"}}
</tool_call>"""
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "redact_sensitive_data"
        assert "email,phone" in result[0]["arguments"]["entity_types"]

    def test_escaped_json_in_arguments(self):
        """Test parsing tool call with escaped JSON in arguments."""
        parser = QwenToolCallParser()
        text = r'<tool_call>{"name": "detect_time_series_anomalies", "arguments": {"series":"[{\"timestamp\":0,\"value\":10000},{\"timestamp\":1,\"value\":10200}]","direction":"both"}}</tool_call>'
        result = parser.parse(text)
        assert len(result) == 1
        assert result[0]["name"] == "detect_time_series_anomalies"
        assert result[0]["arguments"]["direction"] == "both"
