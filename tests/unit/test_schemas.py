"""
Tests for DeepFabric schema system.

This module tests:
- Schema framework and registry
- Rich agent CoT schema validation
- Schema mixins and composition
- Mathematical validation
- DeepFabric Format tool call schemas
"""

import pytest

from deepfabric.schemas import (
    CONVERSATION_SCHEMAS,
    TOOL_CALL_ID_PATTERN,
    ChatMessage,
    Conversation,
    ToolCall,
    ToolCallFunction,
    ToolExecution,
    generate_tool_call_id,
    get_conversation_schema,
)


class TestSchemaFramework:
    """Test the schema framework and registry system."""

    def test_conversation_schemas_exist(self):
        """Test that expected schemas are available in new modular system."""
        # Check that new modular schemas are in mapping
        assert "basic" in CONVERSATION_SCHEMAS
        assert "cot" in CONVERSATION_SCHEMAS

    def test_get_conversation_schema(self):
        """Test conversation schema retrieval with new modular types."""

        # All new conversation types return the unified Conversation schema
        basic_schema = get_conversation_schema("basic")
        assert basic_schema is not None
        assert basic_schema == Conversation

        cot_schema = get_conversation_schema("cot")
        assert cot_schema is not None
        assert cot_schema == Conversation


class TestUnifiedConversationSchema:
    """Test the new unified Conversation schema with capability fields."""

    def test_cot_with_reasoning_capability(self):
        """Test Conversation schema with reasoning capability."""

        schema = get_conversation_schema("cot")
        assert schema == Conversation

        # Create instance with reasoning capability (freetext style)
        sample_data = {
            "messages": [
                {"role": "user", "content": "Test question"},
                {"role": "assistant", "content": "Test answer"},
            ],
            "metadata": {},
            "question": "Test question",
            "final_answer": "Test answer",
            "reasoning": {
                "style": "freetext",
                "content": "This is my natural language reasoning...",
            },
        }

        instance = schema(**sample_data)
        assert instance.question == "Test question"
        assert instance.reasoning is not None
        assert instance.reasoning.style == "freetext"
        assert isinstance(instance.reasoning.content, str)

    def test_cot_with_tool_context(self):
        """Test Conversation schema with tool_context capability."""

        schema = get_conversation_schema("cot")
        assert schema == Conversation

        # Create instance with tool_context capability
        # Note: available_tools has been removed from tool_context as it's
        # redundant with the top-level 'tools' field
        sample_data = {
            "messages": [
                {"role": "user", "content": "Test question"},
                {"role": "assistant", "content": "Test answer"},
            ],
            "metadata": {},
            "question": "Test question",
            "final_answer": "Test answer",
            "reasoning": {
                "style": "agent",
                "content": [
                    {"step_number": 1, "thought": "Step 1", "action": None},
                    {"step_number": 2, "thought": "Step 2", "action": None},
                ],
            },
            "tool_context": {
                "executions": [
                    {
                        "function_name": "test_tool",
                        "arguments": '{"param": "value"}',
                        "reasoning": "Testing tool execution",
                        "result": "Test result",
                    }
                ],
            },
            # Tools are now in OpenAI format at top level
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"param": {"type": "string"}},
                            "required": ["param"],
                        },
                    },
                }
            ],
        }

        instance = schema(**sample_data)
        assert instance.question == "Test question"
        assert instance.reasoning is not None
        assert instance.reasoning.style == "agent"
        assert instance.tool_context is not None
        assert len(instance.tool_context.executions) == 1
        assert instance.tools is not None
        assert len(instance.tools) == 1


class TestBasicSchemaFunctionality:
    """Test basic schema functionality with new unified Conversation schema."""

    def test_conversation_with_metadata(self):
        """Test conversation with metadata field."""

        schema = get_conversation_schema("basic")
        assert schema is not None
        assert schema == Conversation

        # Test with conversation data that has metadata
        data_with_metadata = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "I'll check the weather for you."},
            ],
            "metadata": {"topic": "weather"},
        }

        instance = schema(**data_with_metadata)
        assert len(instance.messages) == 2  # noqa: PLR2004

    def test_basic_conversation_schema(self):
        """Test the basic conversation schema."""

        schema = get_conversation_schema("basic")
        assert schema is not None
        assert schema == Conversation

        # Test with basic conversation data
        basic_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "metadata": {},
        }

        instance = schema(**basic_data)
        assert len(instance.messages) == 2  # noqa: PLR2004


class TestSchemaIntegration:
    """Test schema integration with the broader system."""

    def test_conversation_schemas_mapping(self):
        """Test that conversation schemas mapping contains new modular types."""
        # Check that new modular schemas exist
        expected_schemas = ["basic", "cot"]
        for schema_type in expected_schemas:
            assert schema_type in CONVERSATION_SCHEMAS

    def test_schema_retrieval(self):
        """Test schema retrieval for modular types."""

        # Test all modular types return unified Conversation schema
        for schema_type in ["basic", "cot"]:
            schema = get_conversation_schema(schema_type)
            assert schema is not None
            assert schema == Conversation

    def test_unsupported_conversation_type(self):
        """Test error handling for unsupported conversation types."""
        with pytest.raises(ValueError) as exc_info:
            get_conversation_schema("nonexistent_type")

        error_message = str(exc_info.value)
        assert "Unsupported conversation type" in error_message
        assert "nonexistent_type" in error_message


class TestToolCallFormat:
    """Test DeepFabric Format tool call schemas."""

    def test_tool_call_id_generation(self):
        """Test that generated IDs are 9-char alphanumeric."""
        for _ in range(100):
            tool_call_id = generate_tool_call_id()
            assert len(tool_call_id) == 9  # noqa: PLR2004
            assert TOOL_CALL_ID_PATTERN.match(tool_call_id)

    def test_tool_call_valid_id(self):
        """Test ToolCall accepts valid 9-char alphanumeric IDs."""
        tool_call = ToolCall(
            id="callWeath",
            type="function",
            function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"}),
        )
        assert tool_call.id == "callWeath"
        assert tool_call.function.name == "get_weather"
        # Arguments are stored as JSON string for HuggingFace compatibility
        assert tool_call.function.arguments == '{"city":"Paris"}'
        # Use parsed_arguments for dict access
        assert tool_call.function.parsed_arguments == {"city": "Paris"}

    def test_tool_call_invalid_id_too_short(self):
        """Test ToolCall rejects IDs that are too short."""
        with pytest.raises(ValueError) as exc_info:
            ToolCall(
                id="call_0",
                type="function",
                function=ToolCallFunction(name="test", arguments={}),
            )
        assert "9" in str(exc_info.value) or "alphanumeric" in str(exc_info.value)

    def test_tool_call_invalid_id_with_underscore(self):
        """Test ToolCall rejects IDs with non-alphanumeric chars."""
        with pytest.raises(ValueError) as exc_info:
            ToolCall(
                id="call_test",
                type="function",
                function=ToolCallFunction(name="test", arguments={}),
            )
        assert "alphanumeric" in str(exc_info.value)

    def test_tool_call_arguments_json_string(self):
        """Test that arguments are stored as JSON string for HuggingFace compatibility."""
        tool_call = ToolCall(
            id="abc123XYZ",
            type="function",
            function=ToolCallFunction(
                name="search",
                arguments={"query": "test", "limit": 10, "active": True},
            ),
        )
        # Arguments stored as string
        assert isinstance(tool_call.function.arguments, str)
        # Use parsed_arguments for dict access
        assert tool_call.function.parsed_arguments["limit"] == 10  # noqa: PLR2004
        assert tool_call.function.parsed_arguments["active"] is True

    def test_tool_execution_to_tool_call(self):
        """Test ToolExecution conversion to ToolCall."""
        execution = ToolExecution(
            function_name="get_weather",
            arguments='{"city": "Paris", "unit": "C"}',
            reasoning="Getting weather data",
            result='{"temp": 20}',
        )

        tool_call = execution.to_tool_call()

        assert TOOL_CALL_ID_PATTERN.match(tool_call.id)
        # Arguments remain as JSON string
        assert tool_call.function.parsed_arguments == {"city": "Paris", "unit": "C"}

    def test_tool_execution_to_tool_call_with_id(self):
        """Test ToolExecution conversion with provided ID."""
        execution = ToolExecution(
            function_name="get_weather",
            arguments='{"city": "Paris"}',
            reasoning="test",
            result='{"temp": 20}',
        )

        tool_call = execution.to_tool_call("customId1")

        assert tool_call.id == "customId1"
        assert tool_call.function.name == "get_weather"

    def test_chat_message_with_tool_calls(self):
        """Test ChatMessage with typed tool_calls."""
        tool_call = ToolCall(
            id="abc123def",
            type="function",
            function=ToolCallFunction(name="test_func", arguments={"param": "value"}),
        )

        message = ChatMessage(
            role="assistant",
            content="",
            tool_calls=[tool_call],
        )

        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].id == "abc123def"
        # Arguments stored as JSON string, use parsed_arguments for dict access
        assert message.tool_calls[0].function.parsed_arguments == {"param": "value"}

    def test_chat_message_tool_call_id_valid(self):
        """Test valid tool_call_id in ChatMessage."""
        msg = ChatMessage(role="tool", content="result", tool_call_id="callWeath")
        assert msg.tool_call_id == "callWeath"

    def test_chat_message_tool_call_id_invalid(self):
        """Test invalid tool_call_id in ChatMessage."""
        with pytest.raises(ValueError) as exc_info:
            ChatMessage(role="tool", content="result", tool_call_id="call_0")
        assert "9 alphanumeric" in str(exc_info.value)

    def test_chat_message_tool_call_id_none(self):
        """Test that None tool_call_id is allowed."""
        msg = ChatMessage(role="assistant", content="response")
        assert msg.tool_call_id is None

    def test_tool_call_serialization(self):
        """Test ToolCall serializes correctly for dataset export."""
        tool_call = ToolCall(
            id="callWeath",
            type="function",
            function=ToolCallFunction(
                name="get_weather",
                arguments={"city": "Tokyo", "unit": "C"},
            ),
        )

        data = tool_call.model_dump()

        assert data["id"] == "callWeath"
        assert data["type"] == "function"
        assert data["function"]["name"] == "get_weather"
        # Arguments serialized as JSON string for HuggingFace compatibility
        assert data["function"]["arguments"] == '{"city":"Tokyo","unit":"C"}'

    def test_tool_call_function_strips_none_arguments(self):
        """Test that ToolCallFunction strips None values from arguments."""
        func = ToolCallFunction(
            name="search_file",
            arguments={
                "file_path": "script.py",
                "keyword": ".format(",
                "base_branch": None,
                "body": None,
                "repository": None,
            },
        )

        # None values should be stripped - arguments stored as JSON string
        assert func.parsed_arguments == {"file_path": "script.py", "keyword": ".format("}
        assert "base_branch" not in func.arguments
        assert "body" not in func.arguments

    def test_tool_execution_rejects_null_arguments(self):
        """Test ToolExecution rejects null values in arguments."""
        with pytest.raises(ValueError) as exc_info:
            ToolExecution(
                function_name="read_file",
                arguments='{"file_path": "test.py", "unused_param": null}',
                reasoning="test",
                result="file content",
            )
        assert "null" in str(exc_info.value)

    def test_chat_message_serialization_excludes_none(self):
        """Test ChatMessage serialization excludes None fields."""
        # User message without tool_calls or tool_call_id
        msg = ChatMessage(role="user", content="Hello")
        data = msg.model_dump()

        # None fields should be excluded
        assert "tool_calls" not in data
        assert "tool_call_id" not in data
        assert data == {"role": "user", "content": "Hello"}

    def test_chat_message_with_tool_calls_serialization(self):
        """Test ChatMessage with tool_calls serializes cleanly."""
        tool_call = ToolCall(
            id="abc123def",
            type="function",
            function=ToolCallFunction(name="test_func", arguments={"param": "value"}),
        )
        msg = ChatMessage(role="assistant", content="", tool_calls=[tool_call])
        data = msg.model_dump()

        # Should include tool_calls but not tool_call_id (which is None)
        assert "tool_calls" in data
        assert "tool_call_id" not in data
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["id"] == "abc123def"

    def test_tool_message_serialization(self):
        """Test tool message serializes with tool_call_id but not tool_calls."""
        msg = ChatMessage(role="tool", content="result", tool_call_id="abc123def")
        data = msg.model_dump()

        # Should include tool_call_id but not tool_calls (which is None)
        assert "tool_call_id" in data
        assert "tool_calls" not in data
        assert data["tool_call_id"] == "abc123def"

    def test_conversation_serialization_excludes_none(self):
        """Test full Conversation serialization excludes None fields."""
        conv = Conversation(
            messages=[
                ChatMessage(role="user", content="What is 2+2?"),
                ChatMessage(role="assistant", content="4"),
            ],
        )
        data = conv.model_dump()

        # None capability fields should be excluded
        assert "reasoning" not in data
        assert "tool_context" not in data
        assert "tools" not in data
        assert "agent_context" not in data
        assert "structured_data" not in data
        assert "metadata" not in data

        # Messages should also not have None fields
        for msg_data in data["messages"]:
            assert "tool_calls" not in msg_data
            assert "tool_call_id" not in msg_data
