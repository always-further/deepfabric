"""Inference backend implementations."""

from .llm_eval_backend import LLMEvalBackend
from .ollama_backend import OllamaBackend
from .tool_call_parsers import (
    GenericToolCallParser,
    HermesToolCallParser,
    LlamaToolCallParser,
    MistralToolCallParser,
    QwenToolCallParser,
    ToolCallParser,
    ToolCallParserRegistry,
    get_parser,
    get_parser_for_model,
    register_parser,
)
from .transformers_backend import TransformersBackend

__all__ = [
    "TransformersBackend",
    "OllamaBackend",
    "LLMEvalBackend",
    # Tool call parsers
    "ToolCallParser",
    "ToolCallParserRegistry",
    "QwenToolCallParser",
    "LlamaToolCallParser",
    "MistralToolCallParser",
    "HermesToolCallParser",
    "GenericToolCallParser",
    "get_parser",
    "get_parser_for_model",
    "register_parser",
]
