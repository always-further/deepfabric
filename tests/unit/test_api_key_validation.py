"""Tests for API key validation functionality."""

import os

from unittest.mock import patch

import pytest

from click.testing import CliRunner

from deepfabric.cli import cli
from deepfabric.config import DeepFabricConfig
from deepfabric.llm import (
    PROVIDER_API_KEY_MAP,
    get_required_api_key_env_var,
    validate_provider_api_key,
)


class TestProviderApiKeyMap:
    """Tests for the PROVIDER_API_KEY_MAP constant."""

    def test_contains_all_major_providers(self):
        """Verify all major providers are in the map."""
        expected_providers = ["openai", "anthropic", "gemini", "openrouter", "ollama"]
        for provider in expected_providers:
            assert provider in PROVIDER_API_KEY_MAP

    def test_openai_requires_correct_key(self):
        """OpenAI should require OPENAI_API_KEY."""
        assert PROVIDER_API_KEY_MAP["openai"] == ["OPENAI_API_KEY"]

    def test_anthropic_requires_correct_key(self):
        """Anthropic should require ANTHROPIC_API_KEY."""
        assert PROVIDER_API_KEY_MAP["anthropic"] == ["ANTHROPIC_API_KEY"]

    def test_gemini_accepts_either_key(self):
        """Gemini should accept GOOGLE_API_KEY or GEMINI_API_KEY."""
        assert PROVIDER_API_KEY_MAP["gemini"] == ["GOOGLE_API_KEY", "GEMINI_API_KEY"]

    def test_openrouter_requires_correct_key(self):
        """OpenRouter should require OPENROUTER_API_KEY."""
        assert PROVIDER_API_KEY_MAP["openrouter"] == ["OPENROUTER_API_KEY"]

    def test_ollama_requires_no_key(self):
        """Ollama should not require an API key."""
        assert PROVIDER_API_KEY_MAP["ollama"] == []

    def test_test_providers_require_no_key(self):
        """Test providers should not require API keys."""
        assert PROVIDER_API_KEY_MAP["test"] == []
        assert PROVIDER_API_KEY_MAP["override"] == []


class TestValidateProviderApiKey:
    """Tests for the validate_provider_api_key function."""

    def test_unknown_provider_returns_error(self):
        """Unknown provider should return invalid with error message."""
        is_valid, error_msg = validate_provider_api_key("unknown_provider")
        assert is_valid is False
        assert error_msg == "Unknown provider: unknown_provider"

    def test_ollama_always_valid(self):
        """Ollama should always be valid (no key required)."""
        is_valid, error_msg = validate_provider_api_key("ollama")
        assert is_valid is True
        assert error_msg is None

    def test_test_provider_always_valid(self):
        """Test provider should always be valid (no key required)."""
        is_valid, error_msg = validate_provider_api_key("test")
        assert is_valid is True
        assert error_msg is None

    @patch.dict(os.environ, {}, clear=True)
    def test_openai_invalid_without_key(self):
        """OpenAI should be invalid without OPENAI_API_KEY."""
        # Ensure the key is not set
        os.environ.pop("OPENAI_API_KEY", None)

        is_valid, error_msg = validate_provider_api_key("openai")
        assert is_valid is False
        assert "OPENAI_API_KEY" in error_msg
        assert "not set" in error_msg

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    def test_openai_valid_with_key(self):
        """OpenAI should be valid with OPENAI_API_KEY set."""
        is_valid, error_msg = validate_provider_api_key("openai")
        assert is_valid is True
        assert error_msg is None

    @patch.dict(os.environ, {}, clear=True)
    def test_anthropic_invalid_without_key(self):
        """Anthropic should be invalid without ANTHROPIC_API_KEY."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        is_valid, error_msg = validate_provider_api_key("anthropic")
        assert is_valid is False
        assert "ANTHROPIC_API_KEY" in error_msg

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    def test_anthropic_valid_with_key(self):
        """Anthropic should be valid with ANTHROPIC_API_KEY set."""
        is_valid, error_msg = validate_provider_api_key("anthropic")
        assert is_valid is True
        assert error_msg is None

    @patch.dict(os.environ, {}, clear=True)
    def test_gemini_invalid_without_any_key(self):
        """Gemini should be invalid without either key."""
        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)

        is_valid, error_msg = validate_provider_api_key("gemini")
        assert is_valid is False
        assert "GOOGLE_API_KEY" in error_msg
        assert "GEMINI_API_KEY" in error_msg

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_gemini_valid_with_google_key(self):
        """Gemini should be valid with GOOGLE_API_KEY set."""
        is_valid, error_msg = validate_provider_api_key("gemini")
        assert is_valid is True
        assert error_msg is None

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_gemini_valid_with_gemini_key(self):
        """Gemini should be valid with GEMINI_API_KEY set."""
        is_valid, error_msg = validate_provider_api_key("gemini")
        assert is_valid is True
        assert error_msg is None

    @patch.dict(os.environ, {}, clear=True)
    def test_openrouter_invalid_without_key(self):
        """OpenRouter should be invalid without OPENROUTER_API_KEY."""
        os.environ.pop("OPENROUTER_API_KEY", None)

        is_valid, error_msg = validate_provider_api_key("openrouter")
        assert is_valid is False
        assert "OPENROUTER_API_KEY" in error_msg

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"})
    def test_openrouter_valid_with_key(self):
        """OpenRouter should be valid with OPENROUTER_API_KEY set."""
        is_valid, error_msg = validate_provider_api_key("openrouter")
        assert is_valid is True
        assert error_msg is None


class TestGetRequiredApiKeyEnvVar:
    """Tests for the get_required_api_key_env_var function."""

    def test_returns_none_for_unknown_provider(self):
        """Unknown provider should return None."""
        result = get_required_api_key_env_var("unknown")
        assert result is None

    def test_returns_none_for_ollama(self):
        """Ollama should return None (no key required)."""
        result = get_required_api_key_env_var("ollama")
        assert result is None

    def test_returns_single_key_for_openai(self):
        """OpenAI should return the single key name."""
        result = get_required_api_key_env_var("openai")
        assert result == "OPENAI_API_KEY"

    def test_returns_single_key_for_anthropic(self):
        """Anthropic should return the single key name."""
        result = get_required_api_key_env_var("anthropic")
        assert result == "ANTHROPIC_API_KEY"

    def test_returns_joined_keys_for_gemini(self):
        """Gemini should return both keys joined with 'or'."""
        result = get_required_api_key_env_var("gemini")
        assert result == "GOOGLE_API_KEY or GEMINI_API_KEY"

    def test_returns_single_key_for_openrouter(self):
        """OpenRouter should return the single key name."""
        result = get_required_api_key_env_var("openrouter")
        assert result == "OPENROUTER_API_KEY"


class TestCliApiKeyValidation:
    """Tests for CLI-level API key validation."""

    @pytest.fixture
    def sample_config_content(self):
        """Sample YAML content with openai provider (new format)."""
        return """
topics:
  prompt: "Test topic"
  mode: tree
  system_prompt: ""
  depth: 2
  degree: 2
  save_as: "test_tree.jsonl"
  llm:
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.7

generation:
  system_prompt: "Test generation"
  instructions: "Test instructions"
  conversation:
    type: basic
  llm:
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.7

output:
  system_prompt: "Test system prompt"
  include_system_message: true
  num_samples: 1
  batch_size: 1
  save_as: "test_dataset.jsonl"
"""

    @pytest.fixture
    def gemini_config_content(self):
        """Sample YAML content with gemini provider (new format)."""
        return """
topics:
  prompt: "Test topic"
  mode: tree
  system_prompt: ""
  depth: 2
  degree: 2
  save_as: "test_tree.jsonl"
  llm:
    provider: "gemini"
    model: "gemini-2.0-flash"
    temperature: 0.7

generation:
  system_prompt: "Test generation"
  instructions: "Test instructions"
  conversation:
    type: basic
  llm:
    provider: "gemini"
    model: "gemini-2.0-flash"
    temperature: 0.7

output:
  system_prompt: "Test system prompt"
  include_system_message: true
  num_samples: 1
  batch_size: 1
  save_as: "test_dataset.jsonl"
"""

    def test_config_get_configured_providers_tree_mode(self, tmp_path):
        """Test get_configured_providers returns correct providers for tree mode."""
        config_content = """
topics:
  prompt: "Test"
  mode: tree
  depth: 2
  degree: 2
  llm:
    provider: "openai"
    model: "gpt-4o-mini"

generation:
  system_prompt: "Test"
  conversation:
    type: basic
  llm:
    provider: "anthropic"
    model: "claude-3-haiku"

output:
  save_as: "test.jsonl"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        config = DeepFabricConfig.from_yaml(str(config_file))

        providers = config.get_configured_providers()
        assert "openai" in providers
        assert "anthropic" in providers

    def test_config_get_configured_providers_graph_mode(self, tmp_path):
        """Test get_configured_providers returns correct providers for graph mode."""
        config_content = """
topics:
  prompt: "Test"
  mode: graph
  prompt_style: anchored
  depth: 2
  degree: 2
  llm:
    provider: "gemini"
    model: "gemini-2.0-flash"

generation:
  system_prompt: "Test"
  conversation:
    type: basic
  llm:
    provider: "openai"
    model: "gpt-4o-mini"

output:
  save_as: "test.jsonl"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        config = DeepFabricConfig.from_yaml(str(config_file))

        providers = config.get_configured_providers()
        assert "gemini" in providers
        assert "openai" in providers

    @patch.dict(os.environ, {}, clear=True)
    def test_cli_validation_fails_without_openai_key(self, sample_config_content, tmp_path):
        """CLI should fail early when OPENAI_API_KEY is missing."""
        # Remove the key
        os.environ.pop("OPENAI_API_KEY", None)

        config_file = tmp_path / "config.yaml"
        config_file.write_text(sample_config_content)

        runner = CliRunner()
        result = runner.invoke(cli, ["generate", str(config_file)])

        assert result.exit_code == 1
        assert "API key verification failed" in result.output
        assert "OPENAI_API_KEY" in result.output

    @patch.dict(os.environ, {}, clear=True)
    def test_cli_validation_fails_without_gemini_key(self, gemini_config_content, tmp_path):
        """CLI should fail early when GEMINI_API_KEY/GOOGLE_API_KEY is missing."""
        # Remove both keys
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)

        config_file = tmp_path / "config.yaml"
        config_file.write_text(gemini_config_content)

        runner = CliRunner()
        result = runner.invoke(cli, ["generate", str(config_file)])

        assert result.exit_code == 1
        assert "API key verification failed" in result.output
        assert "GOOGLE_API_KEY" in result.output or "GEMINI_API_KEY" in result.output

    @patch.dict(os.environ, {}, clear=True)
    def test_cli_provider_override_validates_override_provider(
        self, sample_config_content, tmp_path
    ):
        """CLI should validate the override provider, not config provider."""
        # Remove both OpenAI and Anthropic keys
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)

        config_file = tmp_path / "config.yaml"
        config_file.write_text(sample_config_content)

        runner = CliRunner()
        # Override provider to anthropic
        result = runner.invoke(
            cli, ["generate", str(config_file), "--provider", "anthropic", "--model", "claude-3"]
        )

        assert result.exit_code == 1
        assert "API key verification failed" in result.output
        # Should check anthropic key, not openai
        assert "ANTHROPIC_API_KEY" in result.output
