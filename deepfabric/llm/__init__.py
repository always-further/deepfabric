"""LLM abstraction layer for DeepFabric."""

from .api_key_verifier import (
    VerificationResult,
    VerificationStatus,
    verify_all_api_keys,
    verify_all_api_keys_async,
    verify_anthropic_api_key,
    verify_azure_openai_api_key,
    verify_gemini_api_key,
    verify_ollama_connection,
    verify_openai_api_key,
    verify_openrouter_api_key,
    verify_provider_api_key,
    verify_provider_api_key_async,
)
from .client import (
    PROVIDER_API_KEY_MAP,
    LLMClient,
    get_required_api_key_env_var,
    make_outlines_model,
    validate_provider_api_key,
)

__all__ = [
    "LLMClient",
    "PROVIDER_API_KEY_MAP",
    "VerificationResult",
    "VerificationStatus",
    "get_required_api_key_env_var",
    "make_outlines_model",
    "validate_provider_api_key",
    "verify_all_api_keys",
    "verify_all_api_keys_async",
    "verify_anthropic_api_key",
    "verify_azure_openai_api_key",
    "verify_gemini_api_key",
    "verify_ollama_connection",
    "verify_openai_api_key",
    "verify_openrouter_api_key",
    "verify_provider_api_key",
    "verify_provider_api_key_async",
]
