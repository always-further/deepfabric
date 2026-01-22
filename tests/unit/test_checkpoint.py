"""Unit tests for checkpoint functionality."""

import json
import logging
import os
import tempfile

from pathlib import Path
from unittest.mock import patch

import pytest

from click.testing import CliRunner

from deepfabric.cli import cli
from deepfabric.constants import (
    CHECKPOINT_FAILURES_SUFFIX,
    CHECKPOINT_METADATA_SUFFIX,
    CHECKPOINT_SAMPLES_SUFFIX,
    DEFAULT_CHECKPOINT_DIR,
)
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator, DataSetGeneratorConfig


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoint files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def base_generator_params():
    """Base parameters for creating a DataSetGenerator."""
    return {
        "generation_system_prompt": "You are a helpful assistant.",
        "provider": "openai",
        "model_name": "gpt-4",
        "checkpoint_samples": 5,
        "checkpoint_dir": ".checkpoints",
        "output_save_as": "test_dataset.jsonl",
    }


@pytest.fixture
def generator_with_checkpoint(temp_checkpoint_dir, base_generator_params):
    """Create a generator with checkpoint config and mocked LLM client."""
    params = {**base_generator_params, "checkpoint_dir": temp_checkpoint_dir}
    with patch("deepfabric.generator.LLMClient"):
        return DataSetGenerator(**params)


class TestCheckpointConfig:
    """Tests for checkpoint configuration in DataSetGeneratorConfig."""

    def test_checkpoint_config_defaults(self):
        """Test that checkpoint config has correct defaults."""
        config = DataSetGeneratorConfig(
            generation_system_prompt="Test",
            provider="openai",
            model_name="gpt-4",
        )
        assert config.checkpoint_samples is None
        assert config.checkpoint_dir == DEFAULT_CHECKPOINT_DIR
        assert config.output_save_as is None

    def test_checkpoint_config_with_values(self):
        """Test checkpoint config with custom values."""
        config = DataSetGeneratorConfig(
            generation_system_prompt="Test",
            provider="openai",
            model_name="gpt-4",
            checkpoint_samples=10,
            checkpoint_dir="/custom/dir",
            output_save_as="output.jsonl",
        )
        assert config.checkpoint_samples == 10  # noqa: PLR2004
        assert config.checkpoint_dir == "/custom/dir"
        assert config.output_save_as == "output.jsonl"

    def test_checkpoint_samples_must_be_positive(self):
        """Test that checkpoint_samples must be >= 1."""
        with pytest.raises(ValueError):
            DataSetGeneratorConfig(
                generation_system_prompt="Test",
                provider="openai",
                model_name="gpt-4",
                checkpoint_samples=0,
            )


class TestCheckpointPaths:
    """Tests for checkpoint path generation."""

    def test_get_checkpoint_paths(self, temp_checkpoint_dir, base_generator_params):
        """Test that checkpoint paths are correctly derived from output_save_as."""
        params = {**base_generator_params, "checkpoint_dir": temp_checkpoint_dir}
        with patch("deepfabric.generator.LLMClient"):
            generator = DataSetGenerator(**params)

        metadata_path, samples_path, failures_path = generator._get_checkpoint_paths()

        assert (
            metadata_path == Path(temp_checkpoint_dir) / f"test_dataset{CHECKPOINT_METADATA_SUFFIX}"
        )
        assert (
            samples_path == Path(temp_checkpoint_dir) / f"test_dataset{CHECKPOINT_SAMPLES_SUFFIX}"
        )
        assert (
            failures_path == Path(temp_checkpoint_dir) / f"test_dataset{CHECKPOINT_FAILURES_SUFFIX}"
        )

    def test_get_checkpoint_paths_creates_directory(
        self, temp_checkpoint_dir, base_generator_params
    ):
        """Test that checkpoint directory is created if it doesn't exist."""
        nested_dir = os.path.join(temp_checkpoint_dir, "nested", "checkpoints")
        params = {**base_generator_params, "checkpoint_dir": nested_dir}
        with patch("deepfabric.generator.LLMClient"):
            generator = DataSetGenerator(**params)

        generator._get_checkpoint_paths()

        assert os.path.exists(nested_dir)

    def test_get_checkpoint_paths_requires_output_save_as(self, temp_checkpoint_dir):
        """Test that getting checkpoint paths fails without output_save_as."""
        with patch("deepfabric.generator.LLMClient"):
            generator = DataSetGenerator(
                generation_system_prompt="Test",
                provider="openai",
                model_name="gpt-4",
                checkpoint_samples=5,
                checkpoint_dir=temp_checkpoint_dir,
                output_save_as=None,
            )

        with pytest.raises(DataSetGeneratorError, match="output_save_as not configured"):
            generator._get_checkpoint_paths()


class TestCheckpointSaveLoad:
    """Tests for saving and loading checkpoints."""

    def test_save_checkpoint_creates_files(self, generator_with_checkpoint):
        """Test that saving a checkpoint creates the expected files."""
        generator = generator_with_checkpoint
        generator._initialize_checkpoint_paths()

        # Save some samples
        samples = [{"question": "Q1", "answer": "A1"}]
        failures = [{"error": "Failed sample"}]
        paths = [["Topic1", "Subtopic1"]]

        generator._save_checkpoint(samples, failures, paths)

        # Check files exist
        assert generator._checkpoint_samples_path.exists()
        assert generator._checkpoint_failures_path.exists()
        assert generator._checkpoint_metadata_path.exists()

    def test_save_checkpoint_appends_samples(self, generator_with_checkpoint):
        """Test that checkpoints append samples incrementally."""
        generator = generator_with_checkpoint
        generator._initialize_checkpoint_paths()

        # Save first batch
        samples1 = [{"question": "Q1", "answer": "A1"}]
        generator._save_checkpoint(samples1, [], [["Topic1"]])

        # Save second batch
        samples2 = [{"question": "Q2", "answer": "A2"}]
        generator._save_checkpoint(samples2, [], [["Topic2"]])

        # Read samples file
        with open(generator._checkpoint_samples_path) as f:
            lines = f.readlines()

        assert len(lines) == 2  # noqa: PLR2004
        assert json.loads(lines[0])["question"] == "Q1"
        assert json.loads(lines[1])["question"] == "Q2"

    def test_load_checkpoint_restores_state(self, temp_checkpoint_dir, base_generator_params):
        """Test that loading a checkpoint restores samples and processed paths."""
        params = {**base_generator_params, "checkpoint_dir": temp_checkpoint_dir}

        with patch("deepfabric.generator.LLMClient"):
            # Create and save checkpoint
            generator1 = DataSetGenerator(**params)
            generator1._initialize_checkpoint_paths()
            samples = [{"question": "Q1", "answer": "A1"}]
            failures = [{"error": "Failed"}]
            paths = [["Topic1", "Subtopic1"]]
            generator1._save_checkpoint(samples, failures, paths)

            # Create new generator and load checkpoint
            generator2 = DataSetGenerator(**params)
            loaded = generator2.load_checkpoint()

        assert loaded is True
        assert len(generator2._samples) == 1
        assert generator2._samples[0]["question"] == "Q1"
        assert len(generator2.failed_samples) == 1
        assert "Topic1 -> Subtopic1" in generator2._processed_paths

    def test_load_checkpoint_returns_false_when_no_checkpoint(
        self, temp_checkpoint_dir, base_generator_params
    ):
        """Test that load_checkpoint returns False when no checkpoint exists."""
        params = {**base_generator_params, "checkpoint_dir": temp_checkpoint_dir}
        with patch("deepfabric.generator.LLMClient"):
            generator = DataSetGenerator(**params)

        loaded = generator.load_checkpoint()

        assert loaded is False
        assert len(generator._samples) == 0

    def test_load_checkpoint_with_retry_failed(self, temp_checkpoint_dir, base_generator_params):
        """Test that load_checkpoint with retry_failed=True re-queues failed paths."""
        params = {**base_generator_params, "checkpoint_dir": temp_checkpoint_dir}

        with patch("deepfabric.generator.LLMClient"):
            # Create and save checkpoint with failed samples that have paths
            generator1 = DataSetGenerator(**params)
            generator1._initialize_checkpoint_paths()
            samples = [{"question": "Q1", "answer": "A1"}]
            # Failures include the path for retry functionality
            failures = [{"error": "Rate limit exceeded", "path": "Topic2 -> Subtopic1"}]
            paths = [["Topic1", "Subtopic1"]]
            generator1._save_checkpoint(samples, failures, paths)
            # Also mark the failed path as processed
            generator1._processed_paths.add("Topic2 -> Subtopic1")
            generator1._save_checkpoint_metadata()

            # Load checkpoint without retry_failed - failed path stays processed
            generator2 = DataSetGenerator(**params)
            loaded = generator2.load_checkpoint(retry_failed=False)
            assert loaded is True
            assert "Topic2 -> Subtopic1" in generator2._processed_paths
            assert len(generator2.failed_samples) == 1

            # Load checkpoint with retry_failed=True - failed path removed from processed
            generator3 = DataSetGenerator(**params)
            loaded = generator3.load_checkpoint(retry_failed=True)
            assert loaded is True
            # Failed path should be removed from processed so it can be retried
            assert "Topic2 -> Subtopic1" not in generator3._processed_paths
            # Successfully processed path should still be there
            assert "Topic1 -> Subtopic1" in generator3._processed_paths
            # Failures should be cleared when retrying
            assert len(generator3.failed_samples) == 0

    def test_load_checkpoint_returns_false_when_disabled(self, temp_checkpoint_dir):
        """Test that load_checkpoint returns False when checkpointing is disabled."""
        with patch("deepfabric.generator.LLMClient"):
            generator = DataSetGenerator(
                generation_system_prompt="Test",
                provider="openai",
                model_name="gpt-4",
                checkpoint_samples=None,
                checkpoint_dir=temp_checkpoint_dir,
            )

        loaded = generator.load_checkpoint()

        assert loaded is False

    def test_load_checkpoint_warns_on_config_mismatch(
        self, temp_checkpoint_dir, base_generator_params, caplog
    ):
        """Test that load_checkpoint warns when config differs from checkpoint."""
        params = {**base_generator_params, "checkpoint_dir": temp_checkpoint_dir}

        with patch("deepfabric.generator.LLMClient"):
            # Create checkpoint with gpt-4 model
            generator1 = DataSetGenerator(**params)
            generator1._initialize_checkpoint_paths()
            samples = [{"question": "Q1", "answer": "A1"}]
            generator1._save_checkpoint(samples, [], [["Topic1"]])

            # Create generator with different model and load checkpoint
            different_params = {**params, "model_name": "gpt-3.5-turbo"}
            generator2 = DataSetGenerator(**different_params)

            with caplog.at_level(logging.WARNING):
                loaded = generator2.load_checkpoint()

        assert loaded is True
        assert "Config mismatch" in caplog.text
        assert "model_name" in caplog.text


class TestCheckpointClear:
    """Tests for clearing checkpoint files."""

    def test_clear_checkpoint_removes_files(self, generator_with_checkpoint):
        """Test that clear_checkpoint removes all checkpoint files."""
        generator = generator_with_checkpoint
        generator._initialize_checkpoint_paths()

        # Create checkpoint files
        samples = [{"question": "Q1", "answer": "A1"}]
        generator._save_checkpoint(samples, [], [["Topic1"]])

        # Verify files exist
        assert generator._checkpoint_samples_path.exists()
        assert generator._checkpoint_metadata_path.exists()

        # Clear checkpoint
        generator.clear_checkpoint()

        # Verify files are removed
        assert not generator._checkpoint_samples_path.exists()
        assert not generator._checkpoint_metadata_path.exists()
        assert len(generator._processed_paths) == 0


class TestIsPathProcessed:
    """Tests for checking if a path has been processed."""

    def test_is_path_processed_returns_true_for_processed(self, generator_with_checkpoint):
        """Test that is_path_processed returns True for processed paths."""
        generator = generator_with_checkpoint
        generator._processed_paths.add("Topic1 -> Subtopic1")

        assert generator._is_path_processed(["Topic1", "Subtopic1"]) is True

    def test_is_path_processed_returns_false_for_unprocessed(self, generator_with_checkpoint):
        """Test that is_path_processed returns False for unprocessed paths."""
        generator = generator_with_checkpoint

        assert generator._is_path_processed(["Topic1", "Subtopic1"]) is False

    def test_is_path_processed_handles_none(self, generator_with_checkpoint):
        """Test that is_path_processed returns False for None paths."""
        generator = generator_with_checkpoint

        assert generator._is_path_processed(None) is False


class TestCheckpointMetadata:
    """Tests for checkpoint metadata."""

    def test_metadata_contains_expected_fields(self, generator_with_checkpoint):
        """Test that checkpoint metadata contains expected fields."""
        generator = generator_with_checkpoint
        generator._initialize_checkpoint_paths()

        samples = [{"question": "Q1", "answer": "A1"}]
        generator._samples = samples
        generator._save_checkpoint(samples, [], [["Topic1"]])

        # Read metadata
        with open(generator._checkpoint_metadata_path) as f:
            metadata = json.load(f)

        assert "version" in metadata
        assert "created_at" in metadata
        assert "provider" in metadata
        assert "model_name" in metadata
        assert "total_samples" in metadata
        assert "total_failures" in metadata
        assert "processed_paths" in metadata
        assert "checkpoint_samples" in metadata

        assert metadata["version"] == 1
        assert metadata["provider"] == "openai"
        assert metadata["model_name"] == "gpt-4"


class TestCheckpointStatusCommand:
    """Tests for the checkpoint-status CLI command."""

    def test_checkpoint_status_no_checkpoint(self, temp_checkpoint_dir):
        """Test checkpoint-status when no checkpoint exists."""
        # Create a minimal config file
        config_content = f"""
topics:
  prompt: "Test topic"
  mode: tree
  depth: 2
  degree: 2
generation:
  system_prompt: "Test system prompt"
output:
  save_as: "test_dataset.jsonl"
  checkpoint_dir: "{temp_checkpoint_dir}"
"""
        config_path = os.path.join(temp_checkpoint_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(config_content)

        runner = CliRunner()
        result = runner.invoke(cli, ["checkpoint-status", config_path])

        assert result.exit_code == 0
        assert "No checkpoint found" in result.output

    def test_checkpoint_status_with_checkpoint(self, temp_checkpoint_dir):
        """Test checkpoint-status when checkpoint exists."""
        # Create a minimal config file
        config_content = f"""
topics:
  prompt: "Test topic"
  mode: tree
  depth: 2
  degree: 2
generation:
  system_prompt: "Test system prompt"
output:
  save_as: "test_dataset.jsonl"
  num_samples: 100
  batch_size: 1
  checkpoint_dir: "{temp_checkpoint_dir}"
"""
        config_path = os.path.join(temp_checkpoint_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(config_content)

        # Create checkpoint files
        metadata = {
            "version": 1,
            "created_at": "2024-01-01T00:00:00Z",
            "provider": "openai",
            "model_name": "gpt-4",
            "conversation_type": "basic",
            "reasoning_style": None,
            "total_samples": 25,
            "total_failures": 2,
            "processed_paths": ["Topic1", "Topic2"],
            "checkpoint_samples": 10,
        }

        metadata_path = Path(temp_checkpoint_dir) / f"test_dataset{CHECKPOINT_METADATA_SUFFIX}"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Create samples file
        samples_path = Path(temp_checkpoint_dir) / f"test_dataset{CHECKPOINT_SAMPLES_SUFFIX}"
        with open(samples_path, "w") as f:
            for i in range(25):
                f.write(json.dumps({"question": f"Q{i}", "answer": f"A{i}"}) + "\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["checkpoint-status", config_path])

        assert result.exit_code == 0
        assert "Checkpoint Status" in result.output
        assert "25/100" in result.output  # Progress
        assert "openai" in result.output  # Provider
        assert "gpt-4" in result.output  # Model
        assert "Resume with:" in result.output

    def test_checkpoint_status_with_failures(self, temp_checkpoint_dir):
        """Test checkpoint-status shows failure details."""
        # Create a minimal config file
        config_content = f"""
topics:
  prompt: "Test topic"
  mode: tree
  depth: 2
  degree: 2
generation:
  system_prompt: "Test system prompt"
output:
  save_as: "test_dataset.jsonl"
  num_samples: 100
  batch_size: 1
  checkpoint_dir: "{temp_checkpoint_dir}"
"""
        config_path = os.path.join(temp_checkpoint_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(config_content)

        # Create checkpoint metadata
        metadata = {
            "version": 1,
            "created_at": "2024-01-01T00:00:00Z",
            "provider": "openai",
            "model_name": "gpt-4",
            "conversation_type": "basic",
            "total_samples": 10,
            "total_failures": 3,
            "processed_paths": [],
            "checkpoint_samples": 10,
        }

        metadata_path = Path(temp_checkpoint_dir) / f"test_dataset{CHECKPOINT_METADATA_SUFFIX}"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Create failures file
        failures_path = Path(temp_checkpoint_dir) / f"test_dataset{CHECKPOINT_FAILURES_SUFFIX}"
        with open(failures_path, "w") as f:
            f.write(json.dumps({"error": "Rate limit exceeded"}) + "\n")
            f.write(json.dumps({"error": "JSON parse error"}) + "\n")
            f.write(json.dumps({"error": "Connection timeout"}) + "\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["checkpoint-status", config_path])

        assert result.exit_code == 0
        assert "Failed Topics:" in result.output
        assert "Rate limit exceeded" in result.output
        assert "Retry failed:" in result.output
        assert "--retry-failed" in result.output
