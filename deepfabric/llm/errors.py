"""Error handling for LLM providers."""

import anthropic
import openai

from ..exceptions import DataSetGeneratorError


def handle_openai_error(e: Exception, provider: str, model_name: str) -> DataSetGeneratorError:  # noqa: PLR0911
    """Handle OpenAI-specific errors with helpful messages."""
    if isinstance(e, openai.AuthenticationError):
        return DataSetGeneratorError(
            f"Authentication failed for {provider}. Please check your API key."
        )

    if isinstance(e, openai.NotFoundError):
        if provider == "ollama":
            return DataSetGeneratorError(
                f"Model '{model_name}' not found in Ollama. Please run: ollama pull {model_name}"
            )
        return DataSetGeneratorError(
            f"Model '{model_name}' not found for {provider}. Please check the model name."
        )

    if isinstance(e, openai.APIConnectionError):
        if provider == "ollama":
            return DataSetGeneratorError(
                "Cannot connect to Ollama server. Please ensure Ollama is running (try: ollama serve)"
            )
        return DataSetGeneratorError(
            f"Network error connecting to {provider}. Please check your internet connection."
        )

    if isinstance(e, openai.RateLimitError):
        return DataSetGeneratorError(
            f"Rate limit exceeded for {provider}/{model_name}. Please wait and try again."
        )

    return DataSetGeneratorError(f"OpenAI API error for {provider}/{model_name}: {e}")


def handle_anthropic_error(e: Exception, provider: str, model_name: str) -> DataSetGeneratorError:
    """Handle Anthropic-specific errors with helpful messages."""
    if isinstance(e, anthropic.AuthenticationError):
        return DataSetGeneratorError(
            f"Authentication failed for {provider}. Please check your ANTHROPIC_API_KEY."
        )

    if isinstance(e, anthropic.NotFoundError):
        return DataSetGeneratorError(
            f"Model '{model_name}' not found for {provider}. Please check the model name."
        )

    if isinstance(e, anthropic.APIConnectionError):
        return DataSetGeneratorError(
            f"Network error connecting to {provider}. Please check your internet connection."
        )

    if isinstance(e, anthropic.RateLimitError):
        return DataSetGeneratorError(
            f"Rate limit exceeded for {provider}/{model_name}. Please wait and try again."
        )

    return DataSetGeneratorError(f"Anthropic API error for {provider}/{model_name}: {e}")


def handle_gemini_error(e: Exception, provider: str, model_name: str) -> DataSetGeneratorError:
    """Handle Gemini-specific errors with helpful messages."""
    error_str = str(e).lower()

    if "invalid" in error_str and "model" in error_str:
        return DataSetGeneratorError(
            f"Model '{model_name}' not available for Gemini. Try: gemini-1.5-flash, gemini-1.5-pro"
        )

    if any(
        keyword in error_str
        for keyword in ["permission", "api_key", "authentication", "unauthorized"]
    ):
        return DataSetGeneratorError(
            "Authentication failed for Gemini. Please check your GOOGLE_API_KEY or GEMINI_API_KEY."
        )

    if any(keyword in error_str for keyword in ["quota", "rate limit", "too many requests"]):
        return DataSetGeneratorError(
            f"Rate limit exceeded for Gemini/{model_name}. Please wait and try again."
        )

    if any(keyword in error_str for keyword in ["network", "connection", "timeout"]):
        return DataSetGeneratorError(
            "Network error connecting to Gemini. Please check your internet connection."
        )

    return DataSetGeneratorError(f"Gemini API error for {provider}/{model_name}: {e}")


def handle_provider_error(e: Exception, provider: str, model_name: str) -> DataSetGeneratorError:
    """Handle errors for any provider with appropriate error handler."""
    if provider in ["openai", "ollama", "azure"]:
        return handle_openai_error(e, provider, model_name)
    if provider == "anthropic":
        return handle_anthropic_error(e, provider, model_name)
    if provider == "gemini":
        return handle_gemini_error(e, provider, model_name)
    return DataSetGeneratorError(f"Unknown provider error for {provider}/{model_name}: {e}")
