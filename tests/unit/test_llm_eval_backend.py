"""Tests for LLM evaluation backend.

This module tests:
- LLMEvalBackend configuration validation
- Client creation for each provider
- Tool conversion for each provider format
- Tool call response parsing
- Retry behavior on rate limits
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from deepfabric.evaluation.backends.llm_eval_backend import LLMEvalBackend
from deepfabric.evaluation.inference import InferenceConfig, create_inference_backend
from deepfabric.schemas import ToolDefinition, ToolParameter


class TestInferenceConfigValidation:
    """Tests for InferenceConfig validation."""

    def test_requires_provider_for_llm_backend(self):
        """Test that provider is required when backend='llm'."""
        with pytest.raises(ValueError, match="provider must be specified"):
            InferenceConfig(
                model="gpt-4",
                backend="llm",
            )

    def test_valid_openai_config(self):
        """Test valid OpenAI configuration."""
        config = InferenceConfig(
            model="gpt-4o",
            backend="llm",
            provider="openai",
        )
        assert config.backend == "llm"
        assert config.provider == "openai"
        assert config.model == "gpt-4o"

    def test_valid_anthropic_config(self):
        """Test valid Anthropic configuration."""
        config = InferenceConfig(
            model="claude-3-5-sonnet-20241022",
            backend="llm",
            provider="anthropic",
        )
        assert config.provider == "anthropic"

    def test_valid_gemini_config(self):
        """Test valid Gemini configuration."""
        config = InferenceConfig(
            model="gemini-2.0-flash",
            backend="llm",
            provider="gemini",
        )
        assert config.provider == "gemini"

    def test_valid_openrouter_config(self):
        """Test valid OpenRouter configuration."""
        config = InferenceConfig(
            model="openai/gpt-4o",
            backend="llm",
            provider="openrouter",
        )
        assert config.provider == "openrouter"

    def test_transformers_backend_no_provider_required(self):
        """Test that transformers backend doesn't require provider."""
        config = InferenceConfig(
            model="meta-llama/Llama-3-8b",
            backend="transformers",
        )
        assert config.backend == "transformers"
        assert config.provider is None


class TestFactoryFunction:
    """Tests for the create_inference_backend factory."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_creates_llm_backend(self):
        """Test that factory creates LLMEvalBackend for backend='llm'."""
        config = InferenceConfig(
            model="gpt-4o",
            backend="llm",
            provider="openai",
        )
        backend = create_inference_backend(config)
        assert isinstance(backend, LLMEvalBackend)


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("openai.AsyncOpenAI")
    def test_client_creation(self, mock_async_openai):
        """Test OpenAI async client is created correctly."""
        config = InferenceConfig(
            model="gpt-4o",
            backend="llm",
            provider="openai",
        )

        LLMEvalBackend(config)
        mock_async_openai.assert_called_once_with(api_key="test-key")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("openai.AsyncOpenAI")
    def test_client_creation_with_base_url(self, mock_async_openai):
        """Test OpenAI client with custom base URL."""
        config = InferenceConfig(
            model="gpt-4o",
            backend="llm",
            provider="openai",
            base_url="https://custom.api.com/v1",
        )

        LLMEvalBackend(config)
        mock_async_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://custom.api.com/v1",
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        config = InferenceConfig(
            model="gpt-4o",
            backend="llm",
            provider="openai",
        )

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            LLMEvalBackend(config)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("openai.AsyncOpenAI")
    def test_generate_without_tools(self, mock_async_openai):
        """Test generation without tools."""
        # Setup mock response
        mock_message = Mock()
        mock_message.content = "Hello, world!"
        mock_message.tool_calls = None

        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat = Mock()
        mock_client.chat.completions = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client

        config = InferenceConfig(
            model="gpt-4o",
            backend="llm",
            provider="openai",
        )
        backend = LLMEvalBackend(config)

        messages = [{"role": "user", "content": "Hi"}]
        result = backend.generate(messages)

        assert result.content == "Hello, world!"
        assert result.tool_call is None
        assert result.finish_reason == "stop"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("openai.AsyncOpenAI")
    def test_generate_with_tool_call(self, mock_async_openai):
        """Test generation with tool calling."""
        # Setup mock tool call
        mock_function = Mock()
        mock_function.name = "get_weather"
        mock_function.arguments = '{"location": "NYC"}'

        mock_tool_call = Mock()
        mock_tool_call.function = mock_function

        mock_message = Mock()
        mock_message.content = ""
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat = Mock()
        mock_client.chat.completions = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client

        config = InferenceConfig(
            model="gpt-4o",
            backend="llm",
            provider="openai",
        )
        backend = LLMEvalBackend(config)

        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters=[
                ToolParameter(
                    name="location",
                    type="str",
                    description="City name",
                    required=True,
                )
            ],
            returns="Weather information",
        )

        messages = [{"role": "user", "content": "What's the weather in NYC?"}]
        result = backend.generate(messages, tools=[tool])

        assert result.tool_call is not None
        assert result.tool_call["name"] == "get_weather"
        assert result.tool_call["arguments"] == {"location": "NYC"}


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("anthropic.AsyncAnthropic")
    def test_client_creation(self, mock_async_anthropic):
        """Test Anthropic async client is created correctly."""
        config = InferenceConfig(
            model="claude-3-5-sonnet-20241022",
            backend="llm",
            provider="anthropic",
        )

        LLMEvalBackend(config)
        mock_async_anthropic.assert_called_once_with(api_key="test-key")

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        config = InferenceConfig(
            model="claude-3-5-sonnet-20241022",
            backend="llm",
            provider="anthropic",
        )

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            LLMEvalBackend(config)

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("anthropic.AsyncAnthropic")
    def test_tool_conversion(self, _mock_async_anthropic):
        """Test tool conversion to Anthropic format."""
        config = InferenceConfig(
            model="claude-3-5-sonnet-20241022",
            backend="llm",
            provider="anthropic",
        )

        backend = LLMEvalBackend(config)

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter(
                    name="arg1",
                    type="str",
                    description="First argument",
                    required=True,
                )
            ],
            returns="Test result",
        )

        anthropic_tool = backend._convert_tool_to_anthropic(tool)

        assert anthropic_tool["name"] == "test_tool"
        assert anthropic_tool["description"] == "A test tool"
        assert "input_schema" in anthropic_tool
        assert anthropic_tool["input_schema"]["properties"]["arg1"]["type"] == "string"


class TestGeminiProvider:
    """Tests for Gemini provider."""

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
    @patch("google.genai.Client")
    def test_client_creation(self, mock_genai_client):
        """Test Gemini client is created correctly."""
        config = InferenceConfig(
            model="gemini-2.0-flash",
            backend="llm",
            provider="gemini",
        )

        LLMEvalBackend(config)
        mock_genai_client.assert_called_once_with(api_key="test-key")

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("google.genai.Client")
    def test_client_creation_with_gemini_api_key(self, mock_genai_client):
        """Test Gemini client with GEMINI_API_KEY env var."""
        config = InferenceConfig(
            model="gemini-2.0-flash",
            backend="llm",
            provider="gemini",
        )

        LLMEvalBackend(config)
        mock_genai_client.assert_called_once_with(api_key="test-key")

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        config = InferenceConfig(
            model="gemini-2.0-flash",
            backend="llm",
            provider="gemini",
        )

        with pytest.raises(ValueError, match="GOOGLE_API_KEY or GEMINI_API_KEY"):
            LLMEvalBackend(config)

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
    @patch("google.genai.Client")
    def test_schema_conversion_removes_additional_properties(self, _mock_genai_client):
        """Test that additionalProperties is removed for Gemini."""
        config = InferenceConfig(
            model="gemini-2.0-flash",
            backend="llm",
            provider="gemini",
        )
        backend = LLMEvalBackend(config)

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "additionalProperties": False,
        }

        converted = backend._convert_schema_for_gemini(schema)

        assert "additionalProperties" not in converted
        assert converted["properties"]["name"]["type"] == "string"


class TestOpenRouterProvider:
    """Tests for OpenRouter provider."""

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("openai.AsyncOpenAI")
    def test_client_creation(self, mock_async_openai):
        """Test OpenRouter client is created with correct base URL."""
        config = InferenceConfig(
            model="openai/gpt-4o",
            backend="llm",
            provider="openrouter",
        )

        LLMEvalBackend(config)
        mock_async_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
        )

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("openai.AsyncOpenAI")
    def test_client_creation_with_custom_base_url(self, mock_async_openai):
        """Test OpenRouter client with custom base URL."""
        config = InferenceConfig(
            model="openai/gpt-4o",
            backend="llm",
            provider="openrouter",
            base_url="https://custom.openrouter.ai/v1",
        )

        LLMEvalBackend(config)
        mock_async_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://custom.openrouter.ai/v1",
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        config = InferenceConfig(
            model="openai/gpt-4o",
            backend="llm",
            provider="openrouter",
        )

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            LLMEvalBackend(config)


class TestBatchGeneration:
    """Tests for batch generation."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("openai.AsyncOpenAI")
    def test_generate_batch(self, mock_async_openai):
        """Test batch generation processes multiple message lists."""
        # Setup mock response
        mock_message = Mock()
        mock_message.content = "Response"
        mock_message.tool_calls = None

        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat = Mock()
        mock_client.chat.completions = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client

        config = InferenceConfig(
            model="gpt-4o",
            backend="llm",
            provider="openai",
        )
        backend = LLMEvalBackend(config)

        batch_messages = [
            [{"role": "user", "content": "Question 1"}],
            [{"role": "user", "content": "Question 2"}],
            [{"role": "user", "content": "Question 3"}],
        ]

        results = backend.generate_batch(batch_messages)

        expected_count = len(batch_messages)
        assert len(results) == expected_count
        assert all(r.content == "Response" for r in results)
        assert mock_client.chat.completions.create.call_count == expected_count


class TestCleanup:
    """Tests for cleanup method."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("openai.AsyncOpenAI")
    def test_cleanup_does_not_raise(self, _mock_async_openai):
        """Test that cleanup doesn't raise any errors."""
        config = InferenceConfig(
            model="gpt-4o",
            backend="llm",
            provider="openai",
        )
        backend = LLMEvalBackend(config)

        # Should not raise
        backend.cleanup()
