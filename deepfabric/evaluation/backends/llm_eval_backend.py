"""LLM Evaluation Backend for cloud providers.

Supports OpenAI, Anthropic, Gemini, and OpenRouter for tool calling evaluation.
Uses async clients internally with sync wrapper for compatibility with Evaluator.
"""

import asyncio
import json
import logging
import os

from typing import Any

from deepfabric.evaluation.inference import (
    InferenceBackend,
    InferenceConfig,
    ModelResponse,
)
from deepfabric.llm.errors import handle_provider_error
from deepfabric.llm.rate_limit_config import (
    RateLimitConfig,
    create_rate_limit_config,
    get_default_rate_limit_config,
)
from deepfabric.llm.retry_handler import RetryHandler
from deepfabric.schemas import ToolDefinition

logger = logging.getLogger(__name__)


class LLMEvalBackend(InferenceBackend):
    """Inference backend using cloud LLM providers for evaluation.

    Supports:
    - OpenAI (GPT-4, GPT-4o, etc.)
    - Anthropic (Claude models)
    - Gemini (gemini-2.0-flash, etc.)
    - OpenRouter (OpenAI-compatible API)

    Uses async clients internally with sync wrapper for compatibility.
    """

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize LLM evaluation backend.

        Args:
            config: Inference configuration with provider and model details
        """
        super().__init__(config)

        if config.provider is None:
            msg = "provider must be specified for LLM backend"
            raise ValueError(msg)

        self.provider = config.provider
        self.model_name = config.model

        # Initialize rate limiting
        self.rate_limit_config: RateLimitConfig
        if config.rate_limit_config:
            self.rate_limit_config = create_rate_limit_config(
                self.provider, config.rate_limit_config
            )
        else:
            self.rate_limit_config = get_default_rate_limit_config(self.provider)

        self.retry_handler = RetryHandler(self.rate_limit_config, self.provider)

        # Initialize provider-specific async client
        self._client = self._create_client()

    def _create_client(self) -> Any:
        """Create the appropriate async client for the provider."""
        client_creators = {
            "openai": self._create_openai_client,
            "anthropic": self._create_anthropic_client,
            "gemini": self._create_gemini_client,
            "openrouter": self._create_openrouter_client,
        }
        if creator := client_creators.get(self.provider):
            return creator()
        msg = f"Unsupported provider: {self.provider}"
        raise ValueError(msg)

    def _create_openai_client(self) -> Any:
        """Create async OpenAI client."""
        import openai  # noqa: PLC0415

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            msg = "OPENAI_API_KEY environment variable is not set"
            raise ValueError(msg)

        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url

        return openai.AsyncOpenAI(**kwargs)

    def _create_anthropic_client(self) -> Any:
        """Create async Anthropic client."""
        import anthropic  # noqa: PLC0415

        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            msg = "ANTHROPIC_API_KEY environment variable is not set"
            raise ValueError(msg)

        return anthropic.AsyncAnthropic(api_key=api_key)

    def _create_gemini_client(self) -> Any:
        """Create Gemini client (uses aio namespace for async)."""
        from google import genai  # noqa: PLC0415

        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            msg = "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is not set"
            raise ValueError(msg)

        return genai.Client(api_key=api_key)

    def _create_openrouter_client(self) -> Any:
        """Create async OpenRouter client (OpenAI-compatible)."""
        import openai  # noqa: PLC0415

        api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            msg = "OPENROUTER_API_KEY environment variable is not set"
            raise ValueError(msg)

        base_url = self.config.base_url or "https://openrouter.ai/api/v1"
        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Generate response with optional tool calling (sync wrapper).

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of available tools for function calling

        Returns:
            ModelResponse with generated content and parsed tool calls
        """
        return asyncio.run(self.generate_async(messages, tools))

    async def generate_async(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Generate response with optional tool calling (async).

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of available tools for function calling

        Returns:
            ModelResponse with generated content and parsed tool calls
        """
        return await self._generate_with_retry(messages, tools)

    async def _generate_with_retry(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Generate with retry logic."""
        attempt = 0
        max_retries = self.rate_limit_config.max_retries

        while attempt <= max_retries:
            try:
                return await self._do_generate(messages, tools)
            except Exception as e:
                if not self.retry_handler.should_retry(e):
                    raise handle_provider_error(e, self.provider, self.model_name) from e

                if attempt >= max_retries:
                    self.retry_handler.on_giveup_handler({"exception": e, "tries": attempt + 1})
                    raise handle_provider_error(e, self.provider, self.model_name) from e

                delay = self.retry_handler.calculate_delay(attempt, e)
                self.retry_handler.on_backoff_handler(
                    {
                        "exception": e,
                        "wait": delay,
                        "tries": attempt + 1,
                    }
                )
                await asyncio.sleep(delay)
                attempt += 1

        msg = "Unexpected state in retry logic"
        raise RuntimeError(msg)

    async def _do_generate(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Execute generation based on provider."""
        generators = {
            "openai": self._generate_openai_async,
            "openrouter": self._generate_openai_async,
            "anthropic": self._generate_anthropic_async,
            "gemini": self._generate_gemini_async,
        }
        if generator := generators.get(self.provider):
            return await generator(messages, tools)
        msg = f"Unsupported provider: {self.provider}"
        raise ValueError(msg)

    async def _generate_openai_async(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Generate using OpenAI/OpenRouter API."""
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        if tools:
            kwargs["tools"] = [tool.to_openai() for tool in tools]
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        # Parse tool calls
        tool_call = None
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                # Parse arguments from JSON string
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Failed to parse tool call arguments as JSON: %s (%s)",
                        tc.function.arguments,
                        e,
                    )
                    args = {}

                parsed = {
                    "name": tc.function.name,
                    "arguments": args,
                }
                tool_calls.append(parsed)
            tool_call = tool_calls[0] if tool_calls else None

        return ModelResponse(
            content=message.content or "",
            tool_call=tool_call,
            tool_calls=tool_calls,
            raw_output=message.content or "",
            finish_reason=response.choices[0].finish_reason,
        )

    async def _generate_anthropic_async(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Generate using Anthropic API."""
        # Convert messages to Anthropic format (system message separate)
        system_message = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )

        # Anthropic doesn't allow both temperature and top_p together
        # Use temperature only (the more commonly configured parameter)
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if system_message:
            kwargs["system"] = system_message

        if tools:
            kwargs["tools"] = [self._convert_tool_to_anthropic(tool) for tool in tools]

        response = await self._client.messages.create(**kwargs)

        # Parse response - Anthropic uses content blocks
        content = ""
        tool_call = None
        tool_calls: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                parsed = {
                    "name": block.name,
                    "arguments": block.input,
                }
                tool_calls.append(parsed)

        tool_call = tool_calls[0] if tool_calls else None

        return ModelResponse(
            content=content,
            tool_call=tool_call,
            tool_calls=tool_calls if tool_calls else None,
            raw_output=content,
            finish_reason=response.stop_reason,
        )

    def _convert_tool_to_anthropic(self, tool: ToolDefinition) -> dict[str, Any]:
        """Convert ToolDefinition to Anthropic tool format."""
        openai_format = tool.to_openai()
        func = openai_format["function"]

        return {
            "name": func["name"],
            "description": func["description"],
            "input_schema": func["parameters"],
        }

    async def _generate_gemini_async(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Generate using Gemini API."""
        from google.genai import types  # noqa: PLC0415

        # Convert messages to Gemini format
        gemini_contents: list[types.Content] = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            if role == "system":
                # Gemini uses system_instruction parameter
                system_instruction = msg["content"]
            elif role == "assistant":
                gemini_contents.append(
                    types.Content(role="model", parts=[types.Part(text=msg["content"])])
                )
            else:
                gemini_contents.append(
                    types.Content(role="user", parts=[types.Part(text=msg["content"])])
                )

        # Prepare tools for Gemini
        gemini_tools = None
        if tools:
            function_declarations = []
            for tool in tools:
                openai_format = tool.to_openai()
                func = openai_format["function"]

                # Gemini uses slightly different schema format
                params = self._convert_schema_for_gemini(func["parameters"])

                function_declarations.append(
                    types.FunctionDeclaration(
                        name=func["name"],
                        description=func["description"],
                        parameters=params,
                    )
                )

            gemini_tools = [types.Tool(function_declarations=function_declarations)]

        # Configure generation
        generation_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            system_instruction=system_instruction,
            tools=gemini_tools,
        )

        response = await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=gemini_contents,
            config=generation_config,
        )

        # Parse response
        content = ""
        tool_call = None
        tool_calls: list[dict[str, Any]] = []

        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    content += part.text
                elif part.function_call:
                    fc = part.function_call
                    parsed = {
                        "name": fc.name,
                        "arguments": dict(fc.args) if fc.args else {},
                    }
                    tool_calls.append(parsed)

        tool_call = tool_calls[0] if tool_calls else None
        finish_reason = response.candidates[0].finish_reason.name if response.candidates else None

        return ModelResponse(
            content=content,
            tool_call=tool_call,
            tool_calls=tool_calls if tool_calls else None,
            raw_output=content,
            finish_reason=finish_reason,
        )

    def _convert_schema_for_gemini(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Convert JSON Schema to Gemini-compatible format.

        Gemini has specific requirements:
        - Does not support additionalProperties
        - Requires 'items' field for array types
        - Handles nested schemas in anyOf, oneOf, allOf
        """
        if not isinstance(schema, dict):
            return schema

        result = dict(schema)

        # Remove additionalProperties (not supported by Gemini)
        if "additionalProperties" in result:
            del result["additionalProperties"]

        # Ensure array types have items defined (Gemini requires this)
        # Check both explicit type and type within anyOf/oneOf
        is_array = result.get("type") == "array"
        if not is_array and "type" in result and isinstance(result["type"], list):
            is_array = "array" in result["type"]

        if is_array and "items" not in result:
            result["items"] = {"type": "string"}  # Default to string array

        # Recursively process nested schemas in properties
        if "properties" in result and isinstance(result["properties"], dict):
            for prop_name, prop_schema in result["properties"].items():
                if isinstance(prop_schema, dict):
                    result["properties"][prop_name] = self._convert_schema_for_gemini(prop_schema)

        # Process items in arrays
        if "items" in result and isinstance(result["items"], dict):
            result["items"] = self._convert_schema_for_gemini(result["items"])

        # Process anyOf, oneOf, allOf schemas
        for key in ("anyOf", "oneOf", "allOf"):
            if key in result and isinstance(result[key], list):
                result[key] = [
                    self._convert_schema_for_gemini(sub_schema)
                    for sub_schema in result[key]
                    if isinstance(sub_schema, dict)
                ]
                # If any sub-schema is an array type, ensure it has items
                for sub_schema in result[key]:
                    if isinstance(sub_schema, dict):
                        sub_is_array = sub_schema.get("type") == "array"
                        if sub_is_array and "items" not in sub_schema:
                            sub_schema["items"] = {"type": "string"}

        # Process nested definitions/defs
        for key in ("definitions", "$defs"):
            if key in result and isinstance(result[key], dict):
                result[key] = {
                    name: self._convert_schema_for_gemini(def_schema)
                    for name, def_schema in result[key].items()
                    if isinstance(def_schema, dict)
                }

        return result

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        tools: list[ToolDefinition] | None = None,
    ) -> list[ModelResponse]:
        """Generate responses for a batch of message lists.

        Uses asyncio.gather for parallel execution with rate limiting.

        Args:
            batch_messages: List of message lists
            tools: Optional list of available tools

        Returns:
            List of ModelResponse objects
        """
        return asyncio.run(self._generate_batch_async(batch_messages, tools))

    async def _generate_batch_async(
        self,
        batch_messages: list[list[dict[str, str]]],
        tools: list[ToolDefinition] | None = None,
    ) -> list[ModelResponse]:
        """Generate batch responses using asyncio.gather."""
        tasks = [self.generate_async(messages, tools) for messages in batch_messages]
        return list(await asyncio.gather(*tasks))

    def cleanup(self) -> None:
        """Clean up resources.

        Cloud clients don't typically need explicit cleanup.
        """
        pass
