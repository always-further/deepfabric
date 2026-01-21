"""
Tests for modular conversation configuration system.

This module tests the new modular configuration architecture where
conversation types, reasoning styles, agent modes, and output formats
are separate, orthogonal concerns that can be combined.
"""

import warnings

import pytest

from deepfabric.config import DataEngineConfig


class TestModularConfigValidation:
    """Test validation rules for modular configuration."""

    def test_cot_requires_reasoning_style(self):
        """Test that cot requires reasoning_style to be set."""
        with pytest.raises(ValueError, match="reasoning_style must be specified"):
            DataEngineConfig(
                generation_system_prompt="Test",
                provider="test",
                model="model",
                conversation_type="cot",
                # Missing reasoning_style
            )

    def test_reasoning_style_only_with_cot(self):
        """Test that reasoning_style can only be set with cot."""
        with pytest.raises(ValueError, match="reasoning_style can only be set"):
            DataEngineConfig(
                generation_system_prompt="Test",
                provider="test",
                model="model",
                conversation_type="basic",
                reasoning_style="freetext",  # Invalid for basic type
            )

    def test_agent_reasoning_requires_cot(self):
        """Test that agent reasoning style requires cot conversation type."""
        with pytest.raises(ValueError, match="reasoning_style can only be set"):
            DataEngineConfig(
                generation_system_prompt="Test",
                provider="test",
                model="model",
                conversation_type="basic",
                reasoning_style="agent",
            )


class TestModularConfigCombinations:
    """Test valid combinations of modular configuration options."""

    def test_basic_conversation(self):
        """Test basic conversation type (no reasoning, no agent)."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="basic",
        )

        assert config.conversation_type == "basic"
        assert config.reasoning_style is None

    def test_cot_freetext(self):
        """Test cot with freetext reasoning."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="cot",
            reasoning_style="freetext",
        )

        assert config.conversation_type == "cot"
        assert config.reasoning_style == "freetext"

    def test_cot_with_agent_reasoning(self):
        """Test cot + agent reasoning style."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="cot",
            reasoning_style="agent",
            available_tools=["get_weather", "calculate"],
        )

        assert config.conversation_type == "cot"
        assert config.reasoning_style == "agent"
        assert "get_weather" in config.available_tools
        assert "calculate" in config.available_tools

    def test_basic_conversation_explicit(self):
        """Test explicitly setting basic conversation type."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="basic",
        )

        assert config.conversation_type == "basic"
        assert config.reasoning_style is None


class TestModularConfigDefaultValues:
    """Test default values for modular configuration fields."""

    def test_default_max_tools_per_query(self):
        """Test that max_tools_per_query has a default value."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="cot",
            reasoning_style="agent",
            available_tools=["tool1"],
        )

        assert config.max_tools_per_query == 3  # noqa: PLR2004

    def test_tools_default_to_empty_lists(self):
        """Test that tool-related fields default to empty lists."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="basic",
        )

        assert config.available_tools == []
        assert config.custom_tools == []


class TestReasoningStyleOptions:
    """Test all reasoning style options."""

    def test_freetext_reasoning(self):
        """Test freetext reasoning style."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="cot",
            reasoning_style="freetext",
        )

        assert config.reasoning_style == "freetext"

    def test_agent_reasoning(self):
        """Test agent reasoning style."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="cot",
            reasoning_style="agent",
        )

        assert config.reasoning_style == "agent"

    def test_deprecated_structured_normalizes_to_agent(self):
        """Test that deprecated 'structured' value normalizes to 'agent'."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = DataEngineConfig(
                generation_system_prompt="Test",
                provider="test",
                model="model",
                conversation_type="cot",
                reasoning_style="structured",
            )

            # Should normalize to 'agent'
            assert config.reasoning_style == "agent"
            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "structured" in str(w[0].message)

    def test_deprecated_hybrid_normalizes_to_agent(self):
        """Test that deprecated 'hybrid' value normalizes to 'agent'."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = DataEngineConfig(
                generation_system_prompt="Test",
                provider="test",
                model="model",
                conversation_type="cot",
                reasoning_style="hybrid",
            )

            # Should normalize to 'agent'
            assert config.reasoning_style == "agent"
            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "hybrid" in str(w[0].message)


class TestAgentReasoningWithTools:
    """Test agent reasoning style with tools."""

    def test_agent_with_available_tools(self):
        """Test agent reasoning with available_tools configured."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="cot",
            reasoning_style="agent",
            available_tools=["tool1", "tool2"],
        )

        assert config.reasoning_style == "agent"
        assert len(config.available_tools) == 2  # noqa: PLR2004

    def test_agent_with_custom_tools(self):
        """Test agent reasoning with custom_tools configured."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="cot",
            reasoning_style="agent",
            custom_tools=[{"name": "my_tool", "description": "A custom tool"}],
        )

        assert config.reasoning_style == "agent"
        assert len(config.custom_tools) == 1
