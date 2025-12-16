"""LLM client implementation using Outlines for structured generation."""

import asyncio
import os

from typing import Any

import anthropic
import openai
import outlines

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel

from ..exceptions import DataSetGeneratorError
from .errors import handle_provider_error
from .rate_limit_config import RateLimitConfig, create_rate_limit_config
from .retry_handler import RetryHandler, retry_with_backoff, retry_with_backoff_async

# JSON Schema union type keys that need recursive processing
_UNION_KEYS = ("anyOf", "oneOf", "allOf")


def _raise_api_key_error(env_var: str) -> None:
    """Raise an error for missing API key."""
    msg = f"{env_var} environment variable not set"
    raise DataSetGeneratorError(msg)


def _get_gemini_api_key() -> str:
    """Retrieve Gemini API key from environment variables.

    Returns:
        The API key string

    Raises:
        DataSetGeneratorError: If no API key is found
    """
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        if api_key := os.getenv(name):
            return api_key
    _raise_api_key_error("GOOGLE_API_KEY or GEMINI_API_KEY")
    # This return is never reached but satisfies type checker
    raise AssertionError("unreachable")


# Provider to environment variable mapping
PROVIDER_API_KEY_MAP: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "ollama": [],  # No API key required
    # Test providers for unit tests, no API key required
    "test": [],
    "override": [],
}


def validate_provider_api_key(provider: str) -> tuple[bool, str | None]:
    """Validate that the required API key exists for a provider.

    Args:
        provider: Provider name (openai, anthropic, gemini, ollama, openrouter)

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    env_vars = PROVIDER_API_KEY_MAP.get(provider)

    if env_vars is None:
        return False, f"Unknown provider: {provider}"

    # Ollama doesn't need an API key
    if not env_vars:
        return True, None

    # Check if any of the required env vars are set
    for env_var in env_vars:
        if os.getenv(env_var):
            return True, None

    # Build helpful error message
    if len(env_vars) == 1:
        return False, f"Missing API key: {env_vars[0]} environment variable is not set"
    var_list = " or ".join(env_vars)
    return False, f"Missing API key: Set {var_list} environment variable"


def get_required_api_key_env_var(provider: str) -> str | None:
    """Get the environment variable name(s) required for a provider.

    Args:
        provider: Provider name

    Returns:
        Human-readable string of required env var(s), or None if no key required
    """
    env_vars = PROVIDER_API_KEY_MAP.get(provider)
    if not env_vars:
        return None
    if len(env_vars) == 1:
        return env_vars[0]
    return " or ".join(env_vars)


def _raise_unsupported_provider_error(provider: str) -> None:
    """Raise an error for unsupported provider."""
    msg = f"Unsupported provider: {provider}"
    raise DataSetGeneratorError(msg)


def _raise_generation_error(max_retries: int, error: Exception) -> None:
    """Raise an error for generation failure."""
    msg = f"Failed to generate output after {max_retries} attempts: {error!s}"
    raise DataSetGeneratorError(msg) from error


def _create_openai_compatible_client(
    api_key_env_var: str | None,
    default_base_url: str | None,
    dummy_key: str | None = None,
    **kwargs,
) -> openai.OpenAI:
    """Create an OpenAI-compatible client for providers that use OpenAI's API format.

    Args:
        api_key_env_var: Environment variable name for API key (None to skip check)
        default_base_url: Default base URL if not provided in kwargs (None for OpenAI default)
        dummy_key: Dummy API key to use if api_key_env_var is None (e.g., for Ollama)
        **kwargs: Additional client configuration (may include base_url override)

    Returns:
        Configured OpenAI client instance

    Raises:
        DataSetGeneratorError: If required API key is missing
    """
    # Get API key from environment or use dummy key
    if api_key_env_var:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            _raise_api_key_error(api_key_env_var)
    else:
        api_key = dummy_key

    # Set up base_url if provided
    client_kwargs = {k: v for k, v in kwargs.items() if k != "base_url"}
    if default_base_url:
        base_url = kwargs.get("base_url", default_base_url)
        client_kwargs["base_url"] = base_url

    return openai.OpenAI(api_key=api_key, **client_kwargs)


def _create_async_openai_compatible_client(
    api_key_env_var: str | None,
    default_base_url: str | None,
    dummy_key: str | None = None,
    **kwargs,
) -> openai.AsyncOpenAI:
    """Create an async OpenAI-compatible client for providers that use OpenAI's API format.

    Args:
        api_key_env_var: Environment variable name for API key (None to skip check)
        default_base_url: Default base URL if not provided in kwargs (None for OpenAI default)
        dummy_key: Dummy API key to use if api_key_env_var is None (e.g., for Ollama)
        **kwargs: Additional client configuration (may include base_url override)

    Returns:
        Configured AsyncOpenAI client instance

    Raises:
        DataSetGeneratorError: If required API key is missing
    """
    # Get API key from environment or use dummy key
    if api_key_env_var:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            _raise_api_key_error(api_key_env_var)
    else:
        api_key = dummy_key

    # Set up base_url if provided
    client_kwargs = {k: v for k, v in kwargs.items() if k != "base_url"}
    if default_base_url:
        base_url = kwargs.get("base_url", default_base_url)
        client_kwargs["base_url"] = base_url

    return openai.AsyncOpenAI(api_key=api_key, **client_kwargs)


def _is_incompatible_object(schema: dict) -> bool:
    """Check if a schema represents an object incompatible with Gemini.

    Gemini rejects objects with no properties defined (like dict[str, Any]).

    Args:
        schema: JSON schema dictionary

    Returns:
        True if the schema is an incompatible object type
    """
    return schema.get("type") == "object" and "properties" not in schema and "$ref" not in schema


def _is_incompatible_array(schema: dict) -> bool:
    """Check if a schema represents an array with incompatible items.

    Arrays with object items that have no properties (like list[dict[str, Any]])
    are incompatible with Gemini.

    Args:
        schema: JSON schema dictionary

    Returns:
        True if the schema is an incompatible array type
    """
    if schema.get("type") != "array" or "items" not in schema:
        return False

    items = schema["items"]
    return isinstance(items, dict) and _is_incompatible_object(items)


def _inline_refs(schema_dict: dict, defs: dict | None = None) -> dict:
    """
    Recursively inline $ref references in a JSON schema.

    Gemini's structured output can be unreliable with $ref. This function
    resolves all references by inlining the definitions directly.

    Args:
        schema_dict: JSON schema dictionary
        defs: The $defs dictionary from the root schema

    Returns:
        Modified schema dict with all $ref inlined
    """
    if not isinstance(schema_dict, dict):
        return schema_dict

    # Use provided defs or extract from schema (guaranteed to be dict after this)
    resolved_defs: dict = defs if defs is not None else schema_dict.get("$defs", {})

    # Handle $ref - inline the referenced definition
    if "$ref" in schema_dict:
        ref_path = schema_dict["$ref"]
        # Parse reference like "#/$defs/GraphSubtopic"
        if ref_path.startswith("#/$defs/"):
            def_name = ref_path[len("#/$defs/") :]
            if def_name in resolved_defs:
                # Return a copy of the definition with refs inlined
                inlined = _inline_refs(dict(resolved_defs[def_name]), resolved_defs)
                # Preserve any other properties from the original (like description)
                for key, value in schema_dict.items():
                    if key != "$ref":
                        inlined[key] = value
                return inlined
        # If we can't resolve, return as-is
        return schema_dict

    # Create a copy to modify
    result = {}

    for key, value in schema_dict.items():
        if key == "$defs":
            # Skip $defs since we're inlining them
            continue
        if key == "properties" and isinstance(value, dict):
            result[key] = {
                prop_name: _inline_refs(prop_schema, resolved_defs)
                for prop_name, prop_schema in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            result[key] = _inline_refs(value, resolved_defs)
        elif key in _UNION_KEYS and isinstance(value, list):
            result[key] = [_inline_refs(variant, resolved_defs) for variant in value]
        else:
            result[key] = value

    return result


def _strip_additional_properties(schema_dict: dict) -> dict:
    """
    Recursively remove additionalProperties from JSON schema and handle dict[str, Any] fields.

    Gemini doesn't support:
    1. additionalProperties field in JSON schemas
    2. Objects with no properties defined (e.g., dict[str, Any])
    3. Arrays whose items are objects with no properties (e.g., list[dict[str, Any]])

    Fields like dict[str, Any] have additionalProperties: true and no properties defined.
    Gemini requires that object-type fields must have properties, so we exclude these
    fields from the schema entirely.

    Args:
        schema_dict: JSON schema dictionary

    Returns:
        Modified schema dict without additionalProperties and dict[str, Any] fields
    """
    if not isinstance(schema_dict, dict):
        return schema_dict

    # For Gemini, identify and remove incompatible fields
    if "properties" in schema_dict:
        properties_to_remove = []
        for prop_name, prop_schema in schema_dict["properties"].items():
            if not isinstance(prop_schema, dict):
                continue

            # Check for direct incompatibilities
            if prop_schema.get("additionalProperties") is True:
                # Remove fields with additionalProperties: true (e.g., dict[str, Any])
                properties_to_remove.append(prop_name)
            elif _is_incompatible_object(prop_schema):
                # Remove objects with no properties
                properties_to_remove.append(prop_name)
            elif _is_incompatible_array(prop_schema):
                # Remove arrays with incompatible items (e.g., list[dict[str, Any]])
                properties_to_remove.append(prop_name)
            elif "anyOf" in prop_schema and any(
                isinstance(variant, dict)
                and (_is_incompatible_object(variant) or _is_incompatible_array(variant))
                for variant in prop_schema["anyOf"]
            ):
                # Check if anyOf contains incompatible variants
                properties_to_remove.append(prop_name)

        # Remove incompatible properties from the schema
        for prop_name in properties_to_remove:
            del schema_dict["properties"][prop_name]

        # Update required array to exclude removed properties
        if "required" in schema_dict:
            schema_dict["required"] = [
                r for r in schema_dict["required"] if r not in properties_to_remove
            ]

    # Remove additionalProperties from current level
    schema_dict.pop("additionalProperties", None)

    # Recursively process nested structures
    if "$defs" in schema_dict:
        for def_name, def_schema in schema_dict["$defs"].items():
            schema_dict["$defs"][def_name] = _strip_additional_properties(def_schema)

    # Process properties recursively
    if "properties" in schema_dict:
        for prop_name, prop_schema in schema_dict["properties"].items():
            schema_dict["properties"][prop_name] = _strip_additional_properties(prop_schema)

    # Process items (for arrays)
    if "items" in schema_dict:
        schema_dict["items"] = _strip_additional_properties(schema_dict["items"])

    # Process union types (anyOf, oneOf, allOf)
    for union_key in _UNION_KEYS:
        if union_key in schema_dict:
            schema_dict[union_key] = [
                _strip_additional_properties(variant) for variant in schema_dict[union_key]
            ]

    return schema_dict


class DirectGeminiModel:
    """Direct Gemini API client bypassing Outlines for structured output.

    Uses Gemini's native response_json_schema parameter which is more reliable
    than Outlines' wrapper for structured output generation.
    """

    def __init__(self, client: genai.Client, model_name: str):
        self.client = client
        self.model_name = model_name

    def _prepare_json_schema(self, schema: type[BaseModel]) -> dict:
        """Prepare JSON schema for Gemini by inlining refs and stripping incompatible fields."""
        json_schema = schema.model_json_schema()
        return _strip_additional_properties(_inline_refs(json_schema))

    def __call__(
        self,
        prompt: str,
        schema: type[BaseModel],
        **kwargs,
    ) -> str:
        """Generate structured output using Gemini's native JSON schema support.

        Args:
            prompt: The input prompt
            schema: Pydantic model class for structured output
            **kwargs: Additional generation parameters (temperature, max_output_tokens, etc.)

        Returns:
            JSON string matching the schema
        """
        json_schema = self._prepare_json_schema(schema)

        # Build generation config
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=json_schema,
            **kwargs,
        )

        # Call Gemini API directly
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        if response.text is None:
            raise DataSetGeneratorError("Gemini returned empty response")

        return response.text


class AsyncDirectGeminiModel:
    """Async direct Gemini API client bypassing Outlines for structured output.

    Uses Gemini's native response_json_schema parameter which is more reliable
    than Outlines' wrapper for structured output generation.
    """

    def __init__(self, client: genai.Client, model_name: str):
        self.client = client
        self.model_name = model_name

    def _prepare_json_schema(self, schema: type[BaseModel]) -> dict:
        """Prepare JSON schema for Gemini by inlining refs and stripping incompatible fields."""
        json_schema = schema.model_json_schema()
        return _strip_additional_properties(_inline_refs(json_schema))

    async def __call__(
        self,
        prompt: str,
        schema: type[BaseModel],
        **kwargs,
    ) -> str:
        """Asynchronously generate structured output using Gemini's native JSON schema support.

        Args:
            prompt: The input prompt
            schema: Pydantic model class for structured output
            **kwargs: Additional generation parameters (temperature, max_output_tokens, etc.)

        Returns:
            JSON string matching the schema
        """
        json_schema = self._prepare_json_schema(schema)

        # Build generation config
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=json_schema,
            **kwargs,
        )

        # Call Gemini API directly using async method
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        if response.text is None:
            raise DataSetGeneratorError("Gemini returned empty response")

        return response.text


def _strip_ref_sibling_keywords(schema_dict: dict) -> dict:
    """
    Remove sibling keywords from $ref properties.

    OpenAI's strict mode doesn't allow $ref to have additional keywords like 'description'.
    When Pydantic generates schemas, it adds 'description' alongside '$ref' for nested models
    that have Field(description=...). This function strips those extra keywords.

    Args:
        schema_dict: JSON schema dictionary

    Returns:
        Modified schema dict with $ref siblings removed
    """
    if not isinstance(schema_dict, dict):
        return schema_dict

    # If this dict has $ref, remove all sibling keywords except $ref itself
    if "$ref" in schema_dict:
        return {"$ref": schema_dict["$ref"]}

    # Process properties recursively
    if "properties" in schema_dict:
        for prop_name, prop_schema in schema_dict["properties"].items():
            schema_dict["properties"][prop_name] = _strip_ref_sibling_keywords(prop_schema)

    # Process $defs recursively
    if "$defs" in schema_dict:
        for def_name, def_schema in schema_dict["$defs"].items():
            schema_dict["$defs"][def_name] = _strip_ref_sibling_keywords(def_schema)

    # Process items (for arrays)
    if "items" in schema_dict:
        schema_dict["items"] = _strip_ref_sibling_keywords(schema_dict["items"])

    # Process union types (anyOf, oneOf, allOf)
    for union_key in _UNION_KEYS:
        if union_key in schema_dict:
            schema_dict[union_key] = [
                _strip_ref_sibling_keywords(variant) for variant in schema_dict[union_key]
            ]

    return schema_dict


def _ensure_openai_strict_mode_compliance(schema_dict: dict) -> dict:
    """
    Ensure schema complies with OpenAI's strict mode requirements.

    OpenAI's strict mode requires:
    1. For objects, 'additionalProperties' must be explicitly set to false
    2. ALL properties must be in the 'required' array (no optional fields allowed)
    3. No fields with additionalProperties: true (incompatible with strict mode)
    4. $ref cannot have sibling keywords like 'description'

    Fields like dict[str, Any] have additionalProperties: true and cannot be
    represented in strict mode, so they are excluded from the schema entirely.

    Args:
        schema_dict: JSON schema dictionary

    Returns:
        Modified schema dict meeting OpenAI strict mode requirements
    """
    if not isinstance(schema_dict, dict):
        return schema_dict

    # First, strip sibling keywords from $ref properties
    schema_dict = _strip_ref_sibling_keywords(schema_dict)

    # For OpenAI strict mode, identify and remove dict[str, Any] fields
    # These have additionalProperties: true which is incompatible with strict mode
    if "properties" in schema_dict:
        properties_to_remove = []
        for prop_name, prop_schema in schema_dict["properties"].items():
            # Check for direct additionalProperties: true
            if isinstance(prop_schema, dict) and prop_schema.get("additionalProperties") is True:
                # Remove fields with additionalProperties: true (e.g., dict[str, Any])
                properties_to_remove.append(prop_name)
            # Check for anyOf containing object variants with additionalProperties: true
            elif isinstance(prop_schema, dict) and "anyOf" in prop_schema:
                for variant in prop_schema["anyOf"]:
                    if isinstance(variant, dict) and variant.get("additionalProperties") is True:
                        # This anyOf contains an incompatible object variant - remove entire field
                        properties_to_remove.append(prop_name)
                        break

        # Remove incompatible properties from the schema
        for prop_name in properties_to_remove:
            del schema_dict["properties"][prop_name]

        # Update required array to exclude removed properties
        if "required" in schema_dict:
            schema_dict["required"] = [
                r for r in schema_dict["required"] if r not in properties_to_remove
            ]

        # After removing incompatible fields, ensure ALL remaining properties are required
        # OpenAI strict mode doesn't allow optional fields
        property_keys = list(schema_dict["properties"].keys())
        schema_dict["required"] = property_keys
        schema_dict["additionalProperties"] = False

    # For all objects (including those without properties), set additionalProperties to false
    if schema_dict.get("type") == "object":
        schema_dict["additionalProperties"] = False

    # Recursively process nested structures
    if "$defs" in schema_dict:
        for def_name, def_schema in schema_dict["$defs"].items():
            schema_dict["$defs"][def_name] = _ensure_openai_strict_mode_compliance(def_schema)

    # Process properties recursively
    if "properties" in schema_dict:
        for prop_name, prop_schema in schema_dict["properties"].items():
            schema_dict["properties"][prop_name] = _ensure_openai_strict_mode_compliance(
                prop_schema
            )

    # Process items (for arrays)
    if "items" in schema_dict:
        schema_dict["items"] = _ensure_openai_strict_mode_compliance(schema_dict["items"])

    # Process union types (anyOf, oneOf, allOf)
    # This must be done to handle nested structures like list[dict[str, Any]] | None
    # where the dict[str, Any] is inside an array variant
    for union_key in _UNION_KEYS:
        if union_key in schema_dict:
            schema_dict[union_key] = [
                _ensure_openai_strict_mode_compliance(variant) for variant in schema_dict[union_key]
            ]

    return schema_dict


def _create_openai_compatible_schema(schema: type[BaseModel]) -> type[BaseModel]:
    """
    Create an OpenAI-compatible version of a Pydantic schema.

    OpenAI's strict mode requires that all objects have 'additionalProperties: false'.
    This function ensures the schema meets those requirements while preserving
    Pydantic's correct handling of required vs optional fields.

    Args:
        schema: Original Pydantic model

    Returns:
        Wrapper model that generates OpenAI-compatible schemas
    """

    # Create a new model class that overrides model_json_schema
    class OpenAICompatModel(schema):  # type: ignore[misc,valid-type]
        @classmethod
        def model_json_schema(cls, **kwargs):
            # Get the original schema
            original_schema = super().model_json_schema(**kwargs)
            # Ensure OpenAI strict mode compliance
            return _ensure_openai_strict_mode_compliance(original_schema)

    # Set name and docstring
    OpenAICompatModel.__name__ = f"{schema.__name__}OpenAICompat"
    OpenAICompatModel.__doc__ = schema.__doc__

    return OpenAICompatModel


def make_outlines_model(provider: str, model_name: str, **kwargs) -> Any:
    """Create an Outlines model for the specified provider and model.

    Args:
        provider: Provider name (openai, anthropic, gemini, ollama)
        model_name: Model identifier
        **kwargs: Additional parameters passed to the client

    Returns:
        Outlines model instance

    Raises:
        DataSetGeneratorError: If provider is unsupported or configuration fails
    """
    try:
        if provider == "openai":
            client = _create_openai_compatible_client(
                api_key_env_var="OPENAI_API_KEY",
                default_base_url=None,  # Use OpenAI's default
                **kwargs,
            )
            return outlines.from_openai(client, model_name)

        if provider == "ollama":
            client = _create_openai_compatible_client(
                api_key_env_var=None,  # No API key required
                default_base_url="http://localhost:11434/v1",
                dummy_key="ollama",
                **kwargs,
            )
            return outlines.from_openai(client, model_name)

        if provider == "openrouter":
            client = _create_openai_compatible_client(
                api_key_env_var="OPENROUTER_API_KEY",
                default_base_url="https://openrouter.ai/api/v1",
                **kwargs,
            )
            return outlines.from_openai(client, model_name)

        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                _raise_api_key_error("ANTHROPIC_API_KEY")

            client = anthropic.Anthropic(api_key=api_key, **kwargs)
            return outlines.from_anthropic(client, model_name)

        if provider == "gemini":
            api_key = _get_gemini_api_key()
            # Use direct Gemini API instead of Outlines for better structured output reliability
            client = genai.Client(api_key=api_key)
            return DirectGeminiModel(client, model_name)

        _raise_unsupported_provider_error(provider)

    except DataSetGeneratorError:
        # Re-raise our own errors (like missing API keys)
        raise
    except Exception as e:
        # Use the organized error handler
        raise handle_provider_error(e, provider, model_name) from e


def make_async_outlines_model(provider: str, model_name: str, **kwargs) -> Any | None:
    """Create an async Outlines model when the provider supports it.

    Returns ``None`` for providers without async-capable clients.
    """

    try:
        if provider == "openai":
            client = _create_async_openai_compatible_client(
                api_key_env_var="OPENAI_API_KEY",
                default_base_url=None,  # Use OpenAI's default
                **kwargs,
            )
            return outlines.from_openai(client, model_name)

        if provider == "ollama":
            client = _create_async_openai_compatible_client(
                api_key_env_var=None,  # No API key required
                default_base_url="http://localhost:11434/v1",
                dummy_key="ollama",
                **kwargs,
            )
            return outlines.from_openai(client, model_name)

        if provider == "openrouter":
            client = _create_async_openai_compatible_client(
                api_key_env_var="OPENROUTER_API_KEY",
                default_base_url="https://openrouter.ai/api/v1",
                **kwargs,
            )
            return outlines.from_openai(client, model_name)

        if provider == "gemini":
            api_key = _get_gemini_api_key()
            # Use direct async Gemini API for better structured output reliability
            client = genai.Client(api_key=api_key)
            return AsyncDirectGeminiModel(client, model_name)

    except DataSetGeneratorError:
        raise
    except Exception as e:
        raise handle_provider_error(e, provider, model_name) from e

    # Outlines does not currently expose async structured generation wrappers
    # for the remaining providers. Fallback to synchronous execution later.
    return None


class LLMClient:
    """Wrapper for Outlines models with retry logic and error handling."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        rate_limit_config: RateLimitConfig | dict | None = None,
        **kwargs,
    ):
        """Initialize LLM client.

        Args:
            provider: Provider name
            model_name: Model identifier
            rate_limit_config: Rate limiting configuration (None uses provider defaults)
            **kwargs: Additional client configuration
        """
        self.provider = provider
        self.model_name = model_name

        # Initialize rate limiting
        if isinstance(rate_limit_config, dict):
            self.rate_limit_config = create_rate_limit_config(provider, rate_limit_config)
        elif rate_limit_config is None:
            # Use provider-specific defaults
            from .rate_limit_config import (  # noqa: PLC0415
                get_default_rate_limit_config,
            )

            self.rate_limit_config = get_default_rate_limit_config(provider)
        else:
            self.rate_limit_config = rate_limit_config

        self.retry_handler = RetryHandler(self.rate_limit_config, provider)

        self.model: Any = make_outlines_model(provider, model_name, **kwargs)
        self.async_model: Any | None = make_async_outlines_model(provider, model_name, **kwargs)
        if self.model is None:
            msg = f"Failed to create model for {provider}/{model_name}"
            raise DataSetGeneratorError(msg)

    def generate(self, prompt: str, schema: Any, max_retries: int = 3, **kwargs) -> Any:  # noqa: ARG002
        """Generate structured output using the provided schema.

        Args:
            prompt: Input prompt
            schema: Pydantic model or other schema type
            max_retries: Maximum number of retry attempts (deprecated, use rate_limit_config)
            **kwargs: Additional generation parameters

        Returns:
            Generated output matching the schema

        Raises:
            DataSetGeneratorError: If generation fails after all retries

        Note:
            The max_retries parameter is deprecated. Use rate_limit_config in __init__ instead.
            If provided, it will be ignored in favor of the configured retry handler.
        """
        return self._generate_with_retry(prompt, schema, **kwargs)

    @retry_with_backoff
    def _generate_with_retry(self, prompt: str, schema: Any, **kwargs) -> Any:
        """Internal method that performs actual generation with retry wrapper.

        This method is decorated with retry_with_backoff to handle rate limits
        and transient errors automatically.
        """
        # Convert provider-specific parameters
        kwargs = self._convert_generation_params(**kwargs)

        # For Gemini with DirectGeminiModel, pass schema directly (handles transformation internally)
        # For OpenAI and OpenRouter, ensure all properties are in required array
        generation_schema = schema
        if self.provider == "gemini":
            # DirectGeminiModel handles schema transformation internally
            pass
        elif (
            self.provider in ("openai", "openrouter")
            and isinstance(schema, type)
            and issubclass(schema, BaseModel)
        ):
            generation_schema = _create_openai_compatible_schema(schema)

        # Generate JSON string using the model
        json_output = self.model(prompt, generation_schema, **kwargs)

        # Parse and validate the JSON response with the ORIGINAL schema
        # This ensures we still get proper validation
        try:
            return schema.model_validate_json(json_output)
        except Exception as e:
            raise DataSetGeneratorError(
                f"Generation validation failed: {e}",
                context={"raw_content": json_output},
            ) from e

    async def generate_async(self, prompt: str, schema: Any, max_retries: int = 3, **kwargs) -> Any:  # noqa: ARG002
        """Asynchronously generate structured output using provider async clients.

        Args:
            prompt: Input prompt
            schema: Pydantic model or other schema type
            max_retries: Maximum number of retry attempts (deprecated, use rate_limit_config)
            **kwargs: Additional generation parameters

        Returns:
            Generated output matching the schema

        Raises:
            DataSetGeneratorError: If generation fails after all retries

        Note:
            The max_retries parameter is deprecated. Use rate_limit_config in __init__ instead.
            If provided, it will be ignored in favor of the configured retry handler.
        """
        if self.async_model is None:
            # Fallback to running the synchronous path in a worker thread
            return await asyncio.to_thread(self.generate, prompt, schema, **kwargs)

        return await self._generate_async_with_retry(prompt, schema, **kwargs)

    @retry_with_backoff_async
    async def _generate_async_with_retry(self, prompt: str, schema: Any, **kwargs) -> Any:
        """Internal async method that performs actual generation with retry wrapper.

        This method is decorated with retry_with_backoff_async to handle rate limits
        and transient errors automatically.
        """
        kwargs = self._convert_generation_params(**kwargs)

        # For Gemini with DirectGeminiModel, pass schema directly (handles transformation internally)
        # For OpenAI and OpenRouter, ensure all properties are in required array
        generation_schema = schema
        if self.provider == "gemini":
            # DirectGeminiModel handles schema transformation internally
            pass
        elif (
            self.provider in ("openai", "openrouter")
            and isinstance(schema, type)
            and issubclass(schema, BaseModel)
        ):
            generation_schema = _create_openai_compatible_schema(schema)

        # Ensure we have an async model; if not, fall back to running the sync path
        async_model = self.async_model
        if async_model is None:
            return await asyncio.to_thread(self.generate, prompt, schema, **kwargs)

        # Call the async model (guaranteed non-None by check above)
        json_output = await async_model(prompt, generation_schema, **kwargs)
        # Validate with original schema to ensure proper validation
        try:
            return schema.model_validate_json(json_output)
        except Exception as e:
            raise DataSetGeneratorError(
                f"Async generation validation failed: {e}",
                context={"raw_content": json_output},
            ) from e

    async def generate_async_stream(self, prompt: str, schema: Any, max_retries: int = 3, **kwargs):  # noqa: ARG002
        """Asynchronously generate structured output with streaming text chunks.

        This method streams the LLM's output text as it's generated, then returns
        the final parsed Pydantic model. It yields tuples of (chunk, result) where:
        - During streaming: (text_chunk, None)
        - When complete: (None, final_pydantic_model)

        Args:
            prompt: Input prompt
            schema: Pydantic model or other schema type
            max_retries: Maximum number of retry attempts (deprecated, use rate_limit_config)
            **kwargs: Additional generation parameters

        Yields:
            tuple[str | None, Any | None]:
                - (chunk, None) during streaming
                - (None, model) when generation is complete

        Raises:
            DataSetGeneratorError: If generation fails after all retries

        Note:
            The max_retries parameter is deprecated. Use rate_limit_config in __init__ instead.

        Example:
            >>> async for chunk, result in client.generate_async_stream(prompt, MyModel):
            ...     if chunk:
            ...         print(chunk, end='', flush=True)  # Display streaming text
            ...     if result:
            ...         return result  # Final parsed model
        """
        # Call streaming generation directly (retry decorator doesn't work with generators)
        kwargs = self._convert_generation_params(**kwargs)

        # Apply provider-specific schema transformations
        # Note: Gemini uses DirectGeminiModel which handles schema transformation internally
        generation_schema = schema
        if self.provider == "gemini":
            # DirectGeminiModel handles schema transformation internally
            pass
        elif (
            self.provider in ("openai", "openrouter")
            and isinstance(schema, type)
            and issubclass(schema, BaseModel)
        ):
            generation_schema = _create_openai_compatible_schema(schema)

        # Check if model supports streaming
        async_model = self.async_model or self.model
        if not hasattr(async_model, "generate_stream"):
            # Fallback: no streaming support, yield entire result at once
            result = await self.generate_async(prompt, schema, max_retries, **kwargs)
            yield (None, result)
            return

        # Stream generation
        accumulated_text = []
        try:
            # For sync models used in async context
            if self.async_model is None:
                # Run streaming generation in thread pool
                from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

                def _sync_stream():
                    return list(self.model.generate_stream(prompt, generation_schema, **kwargs))

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_sync_stream)
                    # Wait for completion and get all chunks
                    chunks = await asyncio.wrap_future(future)
                    for chunk in chunks:
                        accumulated_text.append(chunk)
                        yield (chunk, None)
            else:
                # True async streaming
                stream = self.async_model.generate_stream(prompt, generation_schema, **kwargs)
                async for chunk in stream:
                    accumulated_text.append(chunk)
                    yield (chunk, None)

            # Parse accumulated JSON with original schema
            full_text = "".join(accumulated_text)
            result = schema.model_validate_json(full_text)
            yield (None, result)

        except Exception as e:
            # Wrap and raise error with raw content for debugging
            raw_content = "".join(accumulated_text) if accumulated_text else None
            raise DataSetGeneratorError(
                f"Streaming generation failed: {e}",
                context={"raw_content": raw_content},
            ) from e

    def _convert_generation_params(self, **kwargs) -> dict:
        """Convert generic parameters to provider-specific ones."""
        # Convert max_tokens to max_output_tokens for Gemini
        if self.provider == "gemini" and "max_tokens" in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        return kwargs

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider}, model={self.model_name})"
