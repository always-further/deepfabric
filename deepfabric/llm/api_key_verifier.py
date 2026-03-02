"""API key verification for LLM providers.

This module provides functionality to verify that API keys are valid and working
by making lightweight test API calls to each provider.
"""

import asyncio
import os

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import anthropic
import openai

from google import genai
from google.api_core import exceptions as google_exceptions

from .client import PROVIDER_API_KEY_MAP


class VerificationStatus(Enum):
    """Status of an API key verification check."""

    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"
    CONNECTION_ERROR = "connection_error"
    RATE_LIMITED = "rate_limited"
    UNKNOWN_ERROR = "unknown_error"
    NOT_APPLICABLE = "not_applicable"  # For providers like Ollama that don't need keys


@dataclass
class VerificationResult:
    """Result of an API key verification check."""

    provider: str
    status: VerificationStatus
    message: str
    api_key_env_var: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if the verification passed."""
        return self.status in (VerificationStatus.VALID, VerificationStatus.NOT_APPLICABLE)

    def __str__(self) -> str:
        return f"{self.provider}: {self.status.value} - {self.message}"


def verify_openai_api_key(api_key: str | None = None) -> VerificationResult:
    """Verify an OpenAI API key by making a lightweight API call.

    Args:
        api_key: Optional API key to verify. If not provided, uses OPENAI_API_KEY env var.

    Returns:
        VerificationResult with status and details
    """
    env_var = "OPENAI_API_KEY"

    if api_key is None:
        api_key = os.getenv(env_var)

    if not api_key:
        return VerificationResult(
            provider="openai",
            status=VerificationStatus.MISSING,
            message=f"{env_var} environment variable is not set",
            api_key_env_var=env_var,
        )

    try:
        client = openai.OpenAI(api_key=api_key)
        # Use the models list endpoint - it's lightweight and verifies auth
        models = client.models.list()
        # Just check we can iterate (don't need to consume all)
        model_count = sum(1 for _ in models)

        return VerificationResult(
            provider="openai",
            status=VerificationStatus.VALID,
            message="API key is valid",
            api_key_env_var=env_var,
            details={"models_available": model_count},
        )

    except openai.AuthenticationError as e:
        return VerificationResult(
            provider="openai",
            status=VerificationStatus.INVALID,
            message="Invalid API key",
            api_key_env_var=env_var,
            details={"error": str(e)},
        )

    except openai.RateLimitError as e:
        return VerificationResult(
            provider="openai",
            status=VerificationStatus.RATE_LIMITED,
            message="Rate limit exceeded (key may be valid but quota exhausted)",
            api_key_env_var=env_var,
            details={"error": str(e)},
        )

    except openai.APIConnectionError as e:
        return VerificationResult(
            provider="openai",
            status=VerificationStatus.CONNECTION_ERROR,
            message="Failed to connect to OpenAI API",
            api_key_env_var=env_var,
            details={"error": str(e)},
        )

    except Exception as e:
        return VerificationResult(
            provider="openai",
            status=VerificationStatus.UNKNOWN_ERROR,
            message=f"Unexpected error: {e}",
            api_key_env_var=env_var,
            details={"error": str(e), "error_type": type(e).__name__},
        )


def verify_anthropic_api_key(api_key: str | None = None) -> VerificationResult:
    """Verify an Anthropic API key by making a lightweight API call.

    Args:
        api_key: Optional API key to verify. If not provided, uses ANTHROPIC_API_KEY env var.

    Returns:
        VerificationResult with status and details
    """
    env_var = "ANTHROPIC_API_KEY"

    if api_key is None:
        api_key = os.getenv(env_var)

    if not api_key:
        return VerificationResult(
            provider="anthropic",
            status=VerificationStatus.MISSING,
            message=f"{env_var} environment variable is not set",
            api_key_env_var=env_var,
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        # Use a minimal message call - this is the simplest way to verify auth
        # The API doesn't have a models list endpoint like OpenAI
        # Using count_tokens is a read-only operation that verifies the key
        _result = client.messages.count_tokens(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "test"}],
        )

        return VerificationResult(
            provider="anthropic",
            status=VerificationStatus.VALID,
            message="API key is valid",
            api_key_env_var=env_var,
        )

    except anthropic.AuthenticationError as e:
        return VerificationResult(
            provider="anthropic",
            status=VerificationStatus.INVALID,
            message="Invalid API key",
            api_key_env_var=env_var,
            details={"error": str(e)},
        )

    except anthropic.RateLimitError as e:
        return VerificationResult(
            provider="anthropic",
            status=VerificationStatus.RATE_LIMITED,
            message="Rate limit exceeded (key may be valid but quota exhausted)",
            api_key_env_var=env_var,
            details={"error": str(e)},
        )

    except anthropic.APIConnectionError as e:
        return VerificationResult(
            provider="anthropic",
            status=VerificationStatus.CONNECTION_ERROR,
            message="Failed to connect to Anthropic API",
            api_key_env_var=env_var,
            details={"error": str(e)},
        )

    except Exception as e:
        return VerificationResult(
            provider="anthropic",
            status=VerificationStatus.UNKNOWN_ERROR,
            message=f"Unexpected error: {e}",
            api_key_env_var=env_var,
            details={"error": str(e), "error_type": type(e).__name__},
        )


def verify_gemini_api_key(api_key: str | None = None) -> VerificationResult:
    """Verify a Google Gemini API key by making a lightweight API call.

    Args:
        api_key: Optional API key to verify. If not provided, uses GOOGLE_API_KEY
                 or GEMINI_API_KEY env var.

    Returns:
        VerificationResult with status and details
    """
    env_vars = ["GOOGLE_API_KEY", "GEMINI_API_KEY"]
    env_var_used = None

    if api_key is None:
        for env_var in env_vars:
            api_key = os.getenv(env_var)
            if api_key:
                env_var_used = env_var
                break

    if not api_key:
        return VerificationResult(
            provider="gemini",
            status=VerificationStatus.MISSING,
            message="GOOGLE_API_KEY or GEMINI_API_KEY environment variable is not set",
            api_key_env_var="GOOGLE_API_KEY or GEMINI_API_KEY",
        )

    try:
        client = genai.Client(api_key=api_key)
        # List models - lightweight operation that verifies auth
        models = list(client.models.list())
        model_count = len(models)

        return VerificationResult(
            provider="gemini",
            status=VerificationStatus.VALID,
            message="API key is valid",
            api_key_env_var=env_var_used or "GOOGLE_API_KEY",
            details={"models_available": model_count},
        )

    except google_exceptions.PermissionDenied as e:
        return VerificationResult(
            provider="gemini",
            status=VerificationStatus.INVALID,
            message="Invalid API key",
            api_key_env_var=env_var_used or "GOOGLE_API_KEY",
            details={"error": str(e)},
        )

    except google_exceptions.ResourceExhausted as e:
        return VerificationResult(
            provider="gemini",
            status=VerificationStatus.RATE_LIMITED,
            message="Rate limit exceeded (key may be valid but quota exhausted)",
            api_key_env_var=env_var_used or "GOOGLE_API_KEY",
            details={"error": str(e)},
        )

    except google_exceptions.GoogleAPICallError as e:
        return VerificationResult(
            provider="gemini",
            status=VerificationStatus.CONNECTION_ERROR,
            message="Failed to connect to Gemini API",
            api_key_env_var=env_var_used or "GOOGLE_API_KEY",
            details={"error": str(e)},
        )

    except Exception as e:
        return VerificationResult(
            provider="gemini",
            status=VerificationStatus.UNKNOWN_ERROR,
            message=f"Unexpected error: {e}",
            api_key_env_var=env_var_used or "GOOGLE_API_KEY",
            details={"error": str(e), "error_type": type(e).__name__},
        )


def verify_openrouter_api_key(api_key: str | None = None) -> VerificationResult:
    """Verify an OpenRouter API key by making a lightweight API call.

    Args:
        api_key: Optional API key to verify. If not provided, uses OPENROUTER_API_KEY env var.

    Returns:
        VerificationResult with status and details
    """
    env_var = "OPENROUTER_API_KEY"

    if api_key is None:
        api_key = os.getenv(env_var)

    if not api_key:
        return VerificationResult(
            provider="openrouter",
            status=VerificationStatus.MISSING,
            message=f"{env_var} environment variable is not set",
            api_key_env_var=env_var,
        )

    try:
        # OpenRouter uses OpenAI-compatible API
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        # List models to verify auth
        models = client.models.list()
        model_count = sum(1 for _ in models)

        return VerificationResult(
            provider="openrouter",
            status=VerificationStatus.VALID,
            message="API key is valid",
            api_key_env_var=env_var,
            details={"models_available": model_count},
        )

    except openai.AuthenticationError as e:
        return VerificationResult(
            provider="openrouter",
            status=VerificationStatus.INVALID,
            message="Invalid API key",
            api_key_env_var=env_var,
            details={"error": str(e)},
        )

    except openai.RateLimitError as e:
        return VerificationResult(
            provider="openrouter",
            status=VerificationStatus.RATE_LIMITED,
            message="Rate limit exceeded (key may be valid but quota exhausted)",
            api_key_env_var=env_var,
            details={"error": str(e)},
        )

    except openai.APIConnectionError as e:
        return VerificationResult(
            provider="openrouter",
            status=VerificationStatus.CONNECTION_ERROR,
            message="Failed to connect to OpenRouter API",
            api_key_env_var=env_var,
            details={"error": str(e)},
        )

    except Exception as e:
        return VerificationResult(
            provider="openrouter",
            status=VerificationStatus.UNKNOWN_ERROR,
            message=f"Unexpected error: {e}",
            api_key_env_var=env_var,
            details={"error": str(e), "error_type": type(e).__name__},
        )


def verify_azure_openai_api_key(
    api_key: str | None = None, endpoint: str | None = None
) -> VerificationResult:
    """Verify Azure OpenAI API key and endpoint by making a lightweight API call.

    Args:
        api_key: Optional API key to verify. If not provided, uses AZURE_OPENAI_API_KEY env var.
        endpoint: Optional endpoint URL. If not provided, uses AZURE_OPENAI_ENDPOINT env var.

    Returns:
        VerificationResult with status and details
    """
    api_key_env_var = "AZURE_OPENAI_API_KEY"
    endpoint_env_var = "AZURE_OPENAI_ENDPOINT"

    if api_key is None:
        api_key = os.getenv(api_key_env_var)
    if endpoint is None:
        endpoint = os.getenv(endpoint_env_var)

    if not api_key:
        return VerificationResult(
            provider="azure",
            status=VerificationStatus.MISSING,
            message=f"{api_key_env_var} environment variable is not set",
            api_key_env_var=api_key_env_var,
        )

    if not endpoint:
        return VerificationResult(
            provider="azure",
            status=VerificationStatus.MISSING,
            message=(
                f"{endpoint_env_var} environment variable is not set. "
                f"Please set it to your Azure OpenAI endpoint URL "
                f"(e.g., https://your-resource.openai.azure.com/)"
            ),
            api_key_env_var=endpoint_env_var,
        )

    try:
        # Azure OpenAI uses OpenAI-compatible API
        client = openai.OpenAI(
            api_key=api_key,
            base_url=f"{endpoint.rstrip('/')}/openai",
        )
        # List models to verify auth and connection
        models = client.models.list()
        model_count = sum(1 for _ in models)

        return VerificationResult(
            provider="azure",
            status=VerificationStatus.VALID,
            message="API key and endpoint are valid",
            api_key_env_var=api_key_env_var,
            details={"models_available": model_count, "endpoint": endpoint},
        )

    except openai.AuthenticationError as e:
        return VerificationResult(
            provider="azure",
            status=VerificationStatus.INVALID,
            message="Invalid API key for Azure OpenAI",
            api_key_env_var=api_key_env_var,
            details={"error": str(e)},
        )

    except openai.RateLimitError as e:
        return VerificationResult(
            provider="azure",
            status=VerificationStatus.RATE_LIMITED,
            message="Rate limit exceeded (key may be valid but quota exhausted)",
            api_key_env_var=api_key_env_var,
            details={"error": str(e)},
        )

    except openai.APIConnectionError as e:
        return VerificationResult(
            provider="azure",
            status=VerificationStatus.CONNECTION_ERROR,
            message=f"Failed to connect to Azure OpenAI at {endpoint}",
            api_key_env_var=api_key_env_var,
            details={"error": str(e), "endpoint": endpoint},
        )

    except Exception as e:
        return VerificationResult(
            provider="azure",
            status=VerificationStatus.UNKNOWN_ERROR,
            message=f"Unexpected error: {e}",
            api_key_env_var=api_key_env_var,
            details={"error": str(e), "error_type": type(e).__name__},
        )


def verify_ollama_connection(base_url: str | None = None) -> VerificationResult:
    """Verify Ollama is running and accessible.

    Args:
        base_url: Optional base URL for Ollama server. Defaults to http://localhost:11434.

    Returns:
        VerificationResult with status and details
    """
    if base_url is None:
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    try:
        # Ensure base_url ends with /v1 for OpenAI-compatible API
        api_base = base_url.rstrip("/")
        if not api_base.endswith("/v1"):
            api_base = f"{api_base}/v1"

        client = openai.OpenAI(
            api_key="ollama",  # Dummy key, not needed for Ollama
            base_url=api_base,
        )

        # List models to verify connection
        models = client.models.list()
        model_names = [m.id for m in models]

        return VerificationResult(
            provider="ollama",
            status=VerificationStatus.VALID,
            message="Ollama is running and accessible",
            details={"available_models": model_names, "base_url": base_url},
        )

    except openai.APIConnectionError as e:
        return VerificationResult(
            provider="ollama",
            status=VerificationStatus.CONNECTION_ERROR,
            message=f"Cannot connect to Ollama server at {base_url}. Is Ollama running?",
            details={"error": str(e), "base_url": base_url},
        )

    except Exception as e:
        return VerificationResult(
            provider="ollama",
            status=VerificationStatus.UNKNOWN_ERROR,
            message=f"Unexpected error: {e}",
            details={"error": str(e), "error_type": type(e).__name__, "base_url": base_url},
        )


def verify_provider_api_key(
    provider: str,
    api_key: str | None = None,
    **kwargs,
) -> VerificationResult:
    """Verify an API key for a specific provider.

    Args:
        provider: Provider name (openai, anthropic, gemini, openrouter, ollama, azure)
        api_key: Optional API key to verify. If not provided, uses environment variables.
        **kwargs: Additional provider-specific options (e.g., base_url for ollama, endpoint for azure)

    Returns:
        VerificationResult with status and details
    """
    provider = provider.lower()

    # Dispatch to provider-specific verification functions
    verifiers = {
        "openai": lambda: verify_openai_api_key(api_key),
        "anthropic": lambda: verify_anthropic_api_key(api_key),
        "gemini": lambda: verify_gemini_api_key(api_key),
        "openrouter": lambda: verify_openrouter_api_key(api_key),
        "azure": lambda: verify_azure_openai_api_key(api_key, kwargs.get("endpoint")),
        "ollama": lambda: verify_ollama_connection(kwargs.get("base_url")),
    }

    if provider in verifiers:
        return verifiers[provider]()

    if provider in ("test", "override"):
        return VerificationResult(
            provider=provider,
            status=VerificationStatus.NOT_APPLICABLE,
            message="Test provider, no verification needed",
        )

    return VerificationResult(
        provider=provider,
        status=VerificationStatus.UNKNOWN_ERROR,
        message=f"Unknown provider: {provider}",
    )


def _get_providers_to_check(include_optional: bool = False) -> list[str]:
    """Get list of providers to check based on configured API keys.

    Args:
        include_optional: If True, include providers without keys set.

    Returns:
        List of provider names to verify.
    """
    providers_to_check = []
    for provider, env_vars in PROVIDER_API_KEY_MAP.items():
        if provider in ("test", "override"):
            continue

        if not env_vars:
            # Ollama - always include if checking optional
            if include_optional:
                providers_to_check.append(provider)
            continue

        # Check if any env var is set
        has_key = any(os.getenv(env_var) for env_var in env_vars)
        if has_key or include_optional:
            providers_to_check.append(provider)

    return providers_to_check


def verify_all_api_keys(
    providers: list[str] | None = None,
    include_optional: bool = False,
) -> dict[str, VerificationResult]:
    """Verify API keys for multiple providers.

    Args:
        providers: List of providers to verify. If None, verifies all configured providers
                   (those with API keys set in environment).
        include_optional: If True and providers is None, also check providers without
                          keys set to show which are missing.

    Returns:
        Dictionary mapping provider names to VerificationResult
    """
    if providers is None:
        providers = _get_providers_to_check(include_optional)

    results = {}
    for provider in providers:
        results[provider] = verify_provider_api_key(provider)

    return results


async def verify_provider_api_key_async(
    provider: str,
    api_key: str | None = None,
    **kwargs,
) -> VerificationResult:
    """Async version of verify_provider_api_key.

    Runs the synchronous verification in a thread pool to avoid blocking.
    """
    return await asyncio.to_thread(verify_provider_api_key, provider, api_key, **kwargs)


async def verify_all_api_keys_async(
    providers: list[str] | None = None,
    include_optional: bool = False,
) -> dict[str, VerificationResult]:
    """Async version of verify_all_api_keys.

    Verifies all providers concurrently for faster results.
    """
    if providers is None:
        providers = _get_providers_to_check(include_optional)

    # Run all verifications concurrently
    tasks = [verify_provider_api_key_async(provider) for provider in providers]
    results_list = await asyncio.gather(*tasks)

    return dict(zip(providers, results_list, strict=True))
