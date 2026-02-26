"""Tests for the configuration module."""

import os
import tempfile

import pytest
import yaml

from deepfabric.config import DeepFabricConfig, ScoringConfig
from deepfabric.exceptions import ConfigurationError


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing (new format)."""
    return {
        "topics": {
            "prompt": "Test root prompt",
            "mode": "tree",
            "system_prompt": "Test topic system prompt",
            "depth": 2,
            "degree": 3,
            "save_as": "test_tree.jsonl",
            "llm": {
                "provider": "test",
                "model": "model",
                "temperature": 0.7,
            },
        },
        "generation": {
            "system_prompt": "Test generation system prompt",
            "instructions": "Test instructions",
            "conversation": {
                "type": "basic",
            },
            "max_retries": 2,
            "llm": {
                "provider": "test",
                "model": "model",
                "temperature": 0.9,
            },
        },
        "output": {
            "system_prompt": "Test output system prompt",
            "include_system_message": True,
            "num_samples": 5,
            "batch_size": 1,
            "save_as": "test_dataset.jsonl",
        },
    }


@pytest.fixture
def sample_config_dict_no_sys_msg():
    """Sample configuration dictionary without include_system_message setting."""
    return {
        "topics": {
            "prompt": "Test root prompt",
            "mode": "tree",
            "system_prompt": "Test topic system prompt",
            "depth": 2,
            "degree": 3,
            "save_as": "test_tree.jsonl",
            "llm": {
                "provider": "test",
                "model": "model",
                "temperature": 0.7,
            },
        },
        "generation": {
            "system_prompt": "Test generation system prompt",
            "instructions": "Test instructions",
            "conversation": {
                "type": "basic",
            },
            "max_retries": 2,
            "llm": {
                "provider": "test",
                "model": "model",
                "temperature": 0.9,
            },
        },
        "output": {
            "num_samples": 5,
            "batch_size": 1,
            "save_as": "test_dataset.jsonl",
        },
    }


@pytest.fixture
def sample_yaml_file(sample_config_dict):
    """Create a temporary YAML file with sample configuration."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config_dict, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_yaml_file_no_sys_msg(sample_config_dict_no_sys_msg):
    """Create a temporary YAML file without include_system_message setting."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config_dict_no_sys_msg, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_load_from_yaml(sample_yaml_file, sample_config_dict):
    """Test loading configuration from YAML file."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file)

    # Verify topics config
    assert config.topics.prompt == sample_config_dict["topics"]["prompt"]
    assert config.topics.mode == sample_config_dict["topics"]["mode"]
    assert config.topics.depth == sample_config_dict["topics"]["depth"]
    assert config.topics.degree == sample_config_dict["topics"]["degree"]

    # Verify generation config
    assert config.generation.system_prompt == sample_config_dict["generation"]["system_prompt"]
    assert config.generation.instructions == sample_config_dict["generation"]["instructions"]

    # Verify output config
    assert config.output.system_prompt == sample_config_dict["output"]["system_prompt"]
    assert config.output.include_system_message is True
    assert config.output.num_samples == sample_config_dict["output"]["num_samples"]


def test_get_topics_params(sample_yaml_file):
    """Test getting topics arguments from config."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file)
    args = config.get_topics_params()

    assert isinstance(args, dict)
    assert args["topic_prompt"] == "Test root prompt"
    assert args["topic_system_prompt"] == "Test topic system prompt"
    assert args["degree"] == 3  # noqa: PLR2004
    assert args["depth"] == 2  # noqa: PLR2004
    assert args["temperature"] == 0.7  # noqa: PLR2004
    assert args["provider"] == "test"
    assert args["model_name"] == "model"


def test_get_generation_params(sample_yaml_file):
    """Test getting generation arguments from config."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file)
    args = config.get_generation_params()

    assert isinstance(args, dict)
    assert args["instructions"] == "Test instructions"
    assert args["generation_system_prompt"] == "Test generation system prompt"
    assert args["model_name"] == "model"
    assert args["provider"] == "test"
    assert args["temperature"] == 0.9  # noqa: PLR2004
    assert args["max_retries"] == 2  # noqa: PLR2004
    assert args["sys_msg"] is True  # From output config


def test_get_generation_params_no_sys_msg(sample_yaml_file_no_sys_msg):
    """Test getting generation arguments without include_system_message setting."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file_no_sys_msg)
    args = config.get_generation_params()

    assert isinstance(args, dict)
    assert args["sys_msg"] is True  # Default value when not specified


def test_get_topics_params_with_overrides(sample_yaml_file):
    """Test getting topics arguments with overrides."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file)
    args = config.get_topics_params(
        provider="override",
        model="override_model",
        temperature=0.5,
    )

    assert args["provider"] == "override"
    assert args["model_name"] == "override_model"
    assert args["temperature"] == 0.5  # noqa: PLR2004


def test_get_generation_params_with_overrides(sample_yaml_file):
    """Test getting generation arguments with overrides."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file)
    args = config.get_generation_params(
        provider="override",
        model="override_model",
        temperature=0.5,
    )

    assert args["provider"] == "override"
    assert args["model_name"] == "override_model"
    assert args["temperature"] == 0.5  # noqa: PLR2004


def test_get_output_config(sample_yaml_file, sample_config_dict):
    """Test getting output configuration."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file)
    output_config = config.get_output_config()

    assert output_config["system_prompt"] == sample_config_dict["output"]["system_prompt"]
    assert output_config["include_system_message"] is True
    assert output_config["num_samples"] == sample_config_dict["output"]["num_samples"]
    assert output_config["batch_size"] == sample_config_dict["output"]["batch_size"]


def test_get_output_config_no_sys_msg(sample_yaml_file_no_sys_msg):
    """Test getting output configuration without include_system_message setting."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file_no_sys_msg)
    output_config = config.get_output_config()

    # Default value is True
    assert output_config["include_system_message"] is True


def test_missing_yaml_file():
    """Test handling of missing YAML file."""
    with pytest.raises(ConfigurationError):
        DeepFabricConfig.from_yaml("nonexistent.yaml")


def test_invalid_yaml_content():
    """Test handling of invalid YAML content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content:")
        temp_path = f.name

    try:
        with pytest.raises(ConfigurationError):
            DeepFabricConfig.from_yaml(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_old_format_rejected():
    """Test that old configuration format is rejected with helpful message."""
    old_format_config = {
        "dataset_system_prompt": "Test",
        "topic_tree": {
            "topic_prompt": "Test",
            "provider": "test",
            "model": "model",
        },
        "data_engine": {
            "generation_system_prompt": "Test",
            "provider": "test",
            "model": "model",
            "conversation_type": "basic",
        },
        "dataset": {
            "creation": {"num_steps": 1, "batch_size": 1},
            "save_as": "test.jsonl",
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(old_format_config, f)
        temp_path = f.name

    try:
        with pytest.raises(ConfigurationError) as exc_info:
            DeepFabricConfig.from_yaml(temp_path)

        # Check that the migration message is included
        assert "Configuration format has changed" in str(exc_info.value)
        assert "output.system_prompt" in str(exc_info.value)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_scoring_config_defaults():
    """ScoringConfig should have sensible defaults."""
    sc = ScoringConfig()
    assert sc.parent_coherence == 0.25  # noqa: PLR2004
    assert sc.sibling_coherence_lower == 0.2  # noqa: PLR2004
    assert sc.sibling_coherence_upper == 0.68  # noqa: PLR2004
    assert sc.prune is True
    assert sc.save_report is False
    assert sc.embedding_model == "all-MiniLM-L6-v2"


def test_scoring_config_from_yaml():
    """Scoring section in YAML should be parsed into TopicsConfig."""
    config_dict = {
        "topics": {
            "prompt": "Test",
            "mode": "graph",
            "scoring": {
                "parent_coherence": 0.3,
                "sibling_coherence_lower": 0.15,
                "sibling_coherence_upper": 0.7,
                "prune": False,
                "save_report": True,
            },
        },
        "generation": {"system_prompt": "Test"},
        "output": {"save_as": "test.jsonl"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name

    try:
        config = DeepFabricConfig.from_yaml(temp_path)
        assert config.topics.scoring is not None
        assert config.topics.scoring.parent_coherence == 0.3  # noqa: PLR2004
        assert config.topics.scoring.sibling_coherence_lower == 0.15  # noqa: PLR2004
        assert config.topics.scoring.sibling_coherence_upper == 0.7  # noqa: PLR2004
        assert config.topics.scoring.prune is False
        assert config.topics.scoring.save_report is True
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_scoring_config_rejects_inverted_sibling_thresholds():
    """sibling_coherence_lower must be less than sibling_coherence_upper."""
    with pytest.raises(ValueError, match="sibling_coherence_lower.*must be less than"):
        ScoringConfig(sibling_coherence_lower=0.7, sibling_coherence_upper=0.3)


def test_scoring_config_rejects_equal_sibling_thresholds():
    """Equal sibling coherence bounds are also rejected."""
    with pytest.raises(ValueError, match="sibling_coherence_lower.*must be less than"):
        ScoringConfig(sibling_coherence_lower=0.5, sibling_coherence_upper=0.5)


def test_scoring_config_absent_is_none():
    """Config without scoring section should have scoring=None."""
    config_dict = {
        "topics": {"prompt": "Test", "mode": "graph"},
        "generation": {"system_prompt": "Test"},
        "output": {"save_as": "test.jsonl"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name

    try:
        config = DeepFabricConfig.from_yaml(temp_path)
        assert config.topics.scoring is None
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
