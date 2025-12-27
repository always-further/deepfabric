"""Shared fixtures and markers for integration tests."""

import os

import pytest

# Skip markers for conditional test execution based on API key availability
requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping OpenAI integration test",
)

requires_gemini = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping Gemini integration test",
)

requires_huggingface = pytest.mark.skipif(
    not os.getenv("HF_TOKEN"),
    reason="HF_TOKEN not set; skipping HuggingFace Hub integration test",
)


# Shared fixtures for provider configurations
@pytest.fixture
def openai_config():
    """Common OpenAI provider configuration for integration tests."""
    return {
        "provider": "openai",
        "model_name": os.getenv("OPENAI_TEST_MODEL", "gpt-4o-mini"),
        "temperature": 0.2,
    }


@pytest.fixture
def gemini_config():
    """Common Gemini provider configuration for integration tests."""
    return {
        "provider": "gemini",
        "model_name": os.getenv("GEMINI_TEST_MODEL", "gemini-2.0-flash"),
        "temperature": 0.2,
    }
