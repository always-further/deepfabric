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

    def test_chain_of_thought_requires_reasoning_style(self):
        """Test that cot requires reasoning_style to be set."""
        with pytest.raises(ValueError, match="reasoning_style must be specified"):
            DataEngineConfig(
                generation_system_prompt="Test",
                provider="test",
                model="model",
                conversation_type="cot",
                # Missing reasoning_style
            )

    def test_reasoning_style_only_with_chain_of_thought(self):
        """Test that reasoning_style can only be set with cot."""
        with pytest.raises(ValueError, match="reasoning_style can only be set"):
            DataEngineConfig(
                generation_system_prompt="Test",
                provider="test",
                model="model",
                conversation_type="basic",
                reasoning_style="freetext",  # Invalid for basic type
            )

    def test_agent_mode_requires_tools(self):
        """Test that agent_mode requires tools to be configured."""
        with pytest.raises(ValueError, match="agent_mode requires tools"):
            DataEngineConfig(
                generation_system_prompt="Test",
                provider="test",
                model="model",
                conversation_type="cot",
                reasoning_style="agent",
                agent_mode="single_turn",
                # Missing tools configuration
            )

    def test_freetext_not_compatible_with_agent_mode(self):
        """Test that freetext reasoning style cannot be used with agent_mode."""
        with pytest.raises(ValueError, match="freetext.*not compatible with agent_mode"):
            DataEngineConfig(
                generation_system_prompt="Test",
                provider="test",
                model="model",
                conversation_type="cot",
                reasoning_style="freetext",
                agent_mode="single_turn",
                available_tools=["get_weather"],
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
        assert config.agent_mode is None

    def test_chain_of_thought_freetext(self):
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
        assert config.agent_mode is None

    def test_chain_of_thought_with_agent_single_turn(self):
        """Test cot + agent_mode=single_turn."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="cot",
            reasoning_style="agent",
            agent_mode="single_turn",
            available_tools=["get_weather", "calculate"],
        )

        assert config.conversation_type == "cot"
        assert config.reasoning_style == "agent"
        assert config.agent_mode == "single_turn"
        assert "get_weather" in config.available_tools
        assert "calculate" in config.available_tools

    def test_chain_of_thought_agent_multi_turn(self):
        """Test full combination: CoT + agent + multi_turn."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="cot",
            reasoning_style="agent",
            agent_mode="multi_turn",
            available_tools=["tool1", "tool2"],
        )

        assert config.conversation_type == "cot"
        assert config.reasoning_style == "agent"
        assert config.agent_mode == "multi_turn"
        assert len(config.available_tools) == 2  # noqa: PLR2004

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
        assert config.agent_mode is None


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
            agent_mode="single_turn",
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


class TestAgentModeOptions:
    """Test agent mode options with tools."""

    def test_single_turn_agent(self):
        """Test single_turn agent mode."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="cot",
            reasoning_style="agent",
            agent_mode="single_turn",
            available_tools=["tool1"],
        )

        assert config.agent_mode == "single_turn"

    def test_multi_turn_agent(self):
        """Test multi_turn agent mode."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="cot",
            reasoning_style="agent",
            agent_mode="multi_turn",
            available_tools=["tool1", "tool2"],
        )

        assert config.agent_mode == "multi_turn"
