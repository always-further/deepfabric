"""Ollama-based inference backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ollama

from ...schemas import ToolDefinition
from ..inference import InferenceBackend, ModelResponse

if TYPE_CHECKING:
    from ..inference import InferenceConfig


class OllamaBackend(InferenceBackend):
    """Inference backend using Ollama for local model serving.

    Ollama provides optimized local inference for open models with native
    Apple Silicon (M1/M2/M3) support and automatic memory management.
    """

    def __init__(self, config: InferenceConfig):
        """Initialize Ollama backend.

        Args:
            config: Inference configuration

        Note:
            - model should be the Ollama model name (e.g., "mistral", "llama2")
            - Ollama server must be running (ollama serve)
            - Device setting is ignored (Ollama handles device automatically)
        """
        super().__init__(config)

        # Use model directly as Ollama model name
        # Supports: "qwen3:8b", "hf.co/user/model:latest", etc.
        self.model_name = config.model

        # Verify model is available
        try:
            ollama.show(self.model_name)
        except ollama.ResponseError as e:
            msg = (
                f"Model '{self.model_name}' not found in Ollama. "
                f"Pull it first with: ollama pull {self.model_name}"
            )
            raise ValueError(msg) from e

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Generate response from Ollama model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of available tools for function calling

        Returns:
            ModelResponse with generated content and parsed tool calls
        """
        # Convert tools to Ollama format if provided
        ollama_tools = None
        if tools:
            ollama_tools = [self._convert_tool_to_ollama(tool) for tool in tools]

        # Call Ollama API
        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            tools=ollama_tools,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
        )

        # Extract response content (response is a Pydantic object, not dict)
        message = response.message
        content = message.content or ""
        raw_output = content

        # Parse tool calls if present
        tool_call = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            # Ollama returns tool calls in a structured format
            first_tool_call = message.tool_calls[0]
            tool_call = {
                "name": first_tool_call.function.name,
                "arguments": first_tool_call.function.arguments,
            }

        return ModelResponse(
            content=content,
            tool_call=tool_call,
            raw_output=raw_output,
            finish_reason=response.done_reason if hasattr(response, "done_reason") else None,
        )

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        tools: list[ToolDefinition] | None = None,
    ) -> list[ModelResponse]:
        """Generate responses for a batch of message lists.

        Note: Ollama doesn't support true batching, so this processes sequentially.

        Args:
            batch_messages: List of message lists
            tools: Optional list of available tools

        Returns:
            List of ModelResponse objects
        """
        return [self.generate(messages, tools) for messages in batch_messages]

    def _convert_tool_to_ollama(self, tool: ToolDefinition) -> dict:
        """Convert ToolDefinition to Ollama tool format.

        Args:
            tool: DeepFabric ToolDefinition

        Returns:
            Ollama-formatted tool dict
        """
        # Use the built-in OpenAI schema converter (Ollama uses same format)
        return tool.to_openai()

    def cleanup(self) -> None:
        """Clean up resources.

        Note: Ollama manages resources automatically, no cleanup needed.
        """
