"""Model inference interfaces and implementations for evaluation."""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from ..schemas import ToolDefinition


class InferenceConfig(BaseModel):
    """Configuration for model inference."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str | Any = Field(
        description="Model identifier (local path, HuggingFace Hub ID, or model name for cloud providers). "
        "Can also be a pre-loaded model object to avoid reloading.",
    )
    tokenizer: Any | None = Field(
        default=None,
        description="Pre-loaded tokenizer object. Required when model is a pre-loaded model object.",
    )
    adapter_path: str | None = Field(
        default=None,
        description="Path to PEFT/LoRA adapter (if using adapter-based fine-tuning)",
    )
    backend: Literal["transformers", "ollama", "llm"] = Field(
        default="transformers",
        description="Inference backend to use",
    )
    provider: Literal["openai", "anthropic", "gemini", "openrouter"] | None = Field(
        default=None,
        description="Cloud LLM provider (required when backend='llm')",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for the provider (falls back to environment variable if not set)",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom base URL for the API (e.g., for OpenRouter or proxies)",
    )
    rate_limit_config: dict | None = Field(
        default=None,
        description="Rate limiting configuration overrides",
    )
    use_unsloth: bool = Field(
        default=False,
        description="Use Unsloth for loading adapter (for adapters trained with Unsloth)",
    )
    max_seq_length: int = Field(
        default=2048,
        ge=1,
        description="Maximum sequence length for Unsloth models",
    )
    load_in_4bit: bool = Field(
        default=False,
        description="Load model in 4-bit quantization (for Unsloth)",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        description="Maximum tokens to generate",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling top-p",
    )
    device: str | None = Field(
        default=None,
        description="Device to use (cuda, cpu, etc.). None for auto-detection",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for inference",
    )

    @field_serializer("model")
    def serialize_model(self, value: str | Any) -> str:
        """Serialize model field - convert objects to descriptive string."""
        if isinstance(value, str):
            return value
        # For in-memory model objects, return a descriptive string
        model_class = type(value).__name__
        model_name = getattr(getattr(value, "config", None), "name_or_path", "unknown")
        return f"<in-memory:{model_class}:{model_name}>"

    @field_serializer("tokenizer")
    def serialize_tokenizer(self, value: Any | None) -> str | None:
        """Serialize tokenizer field - convert objects to descriptive string."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        # For in-memory tokenizer objects, return a descriptive string
        tokenizer_class = type(value).__name__
        tokenizer_name = getattr(value, "name_or_path", "unknown")
        return f"<in-memory:{tokenizer_class}:{tokenizer_name}>"

    @model_validator(mode="after")
    def validate_config(self) -> "InferenceConfig":
        """Validate configuration consistency."""
        # Ensure provider is set when using LLM backend
        if self.backend == "llm" and self.provider is None:
            msg = "provider must be specified when backend='llm'"
            raise ValueError(msg)

        # Check if model is a pre-loaded object (not a string path)
        is_preloaded_model = not isinstance(self.model, str)

        # If model is pre-loaded, tokenizer must also be provided
        if is_preloaded_model and self.tokenizer is None:
            msg = "tokenizer must be provided when using a pre-loaded model object"
            raise ValueError(msg)

        # Pre-loaded models only work with transformers backend
        if is_preloaded_model and self.backend != "transformers":
            msg = "pre-loaded model objects are only supported with backend='transformers'"
            raise ValueError(msg)

        return self


class ModelResponse(BaseModel):
    """Model inference response."""

    content: str = Field(description="Generated text content")
    tool_call: dict | None = Field(
        default=None,
        description="Parsed tool call if present (first tool call for backwards compatibility)",
    )
    tool_calls: list[dict] | None = Field(
        default=None,
        description="All parsed tool calls if present (for multi-tool responses)",
    )
    raw_output: str = Field(description="Raw model output before parsing")
    finish_reason: str | None = Field(
        default=None,
        description="Reason for completion (stop, length, etc.)",
    )


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    def __init__(self, config: InferenceConfig):
        """Initialize inference backend.

        Args:
            config: Inference configuration
        """
        self.config = config

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Generate response from model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of available tools for function calling

        Returns:
            ModelResponse with generated content and parsed tool calls
        """

    @abstractmethod
    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        tools: list[ToolDefinition] | None = None,
    ) -> list[ModelResponse]:
        """Generate responses for a batch of message sequences.

        Args:
            batch_messages: List of message sequences
            tools: Optional list of available tools for function calling

        Returns:
            List of ModelResponse objects
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources (GPU memory, etc.)."""


def create_inference_backend(config: InferenceConfig) -> InferenceBackend:
    """Factory function to create inference backend.

    Args:
        config: Inference configuration

    Returns:
        Initialized InferenceBackend instance

    Raises:
        ValueError: If backend type is not supported
    """
    if config.backend == "transformers":
        from .backends.transformers_backend import TransformersBackend  # noqa: PLC0415

        return TransformersBackend(config)
    if config.backend == "ollama":
        from .backends.ollama_backend import OllamaBackend  # noqa: PLC0415

        return OllamaBackend(config)
    if config.backend == "llm":
        from .backends.llm_eval_backend import LLMEvalBackend  # noqa: PLC0415

        return LLMEvalBackend(config)

    msg = f"Unsupported backend: {config.backend}"
    raise ValueError(msg)
