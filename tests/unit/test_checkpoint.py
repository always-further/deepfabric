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
    CHECKPOINT_VERSION,
)
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator, DataSetGeneratorConfig
from deepfabric.topic_model import TopicPath


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
        "checkpoint_interval": 5,
        "checkpoint_path": ".checkpoints",
        "output_save_as": "test_dataset.jsonl",
    }


@pytest.fixture
def generator_with_checkpoint(temp_checkpoint_dir, base_generator_params):
    """Create a generator with checkpoint config and mocked LLM client."""
    params = {**base_generator_params, "checkpoint_path": temp_checkpoint_dir}
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
        assert config.checkpoint_interval is None
        assert config.checkpoint_path is None  # None means auto-resolve to XDG dir
        assert config.checkpoint_retry_failed is False
        assert config.output_save_as is None

    def test_checkpoint_config_with_values(self):
        """Test checkpoint config with custom values."""
        config = DataSetGeneratorConfig(
            generation_system_prompt="Test",
            provider="openai",
            model_name="gpt-4",
            checkpoint_interval=10,
            checkpoint_path="/custom/dir",
            checkpoint_retry_failed=True,
            output_save_as="output.jsonl",
        )
        assert config.checkpoint_interval == 10  # noqa: PLR2004
        assert config.checkpoint_path == "/custom/dir"
        assert config.checkpoint_retry_failed is True
        assert config.output_save_as == "output.jsonl"

    def test_checkpoint_interval_must_be_positive(self):
        """Test that checkpoint_interval must be >= 1."""
        with pytest.raises(ValueError):
            DataSetGeneratorConfig(
                generation_system_prompt="Test",
                provider="openai",
                model_name="gpt-4",
                checkpoint_interval=0,
            )


class TestCheckpointPaths:
    """Tests for checkpoint path generation."""

    def test_get_checkpoint_paths(self, temp_checkpoint_dir, base_generator_params):
        """Test that checkpoint paths are correctly derived from output_save_as."""
        params = {**base_generator_params, "checkpoint_path": temp_checkpoint_dir}
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
        params = {**base_generator_params, "checkpoint_path": nested_dir}
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
                checkpoint_interval=5,
                checkpoint_path=temp_checkpoint_dir,
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

        # Save some samples using TopicPath objects
        samples = [{"question": "Q1", "answer": "A1"}]
        failures = [{"error": "Failed sample"}]
        topic_paths = [TopicPath(path=["Topic1", "Subtopic1"], topic_id="uuid-1")]

        generator._save_checkpoint(samples, failures, topic_paths)

        # Check files exist
        assert generator._checkpoint_samples_path.exists()
        assert generator._checkpoint_failures_path.exists()
        assert generator._checkpoint_metadata_path.exists()

    def test_save_checkpoint_appends_samples(self, generator_with_checkpoint):
        """Test that checkpoints append samples incrementally."""
        generator = generator_with_checkpoint
        generator._initialize_checkpoint_paths()

        # Save first batch with TopicPath
        samples1 = [{"question": "Q1", "answer": "A1"}]
        topic_path1 = [TopicPath(path=["Topic1"], topic_id="uuid-1")]
        generator._save_checkpoint(samples1, [], topic_path1)

        # Save second batch with TopicPath
        samples2 = [{"question": "Q2", "answer": "A2"}]
        topic_path2 = [TopicPath(path=["Topic2"], topic_id="uuid-2")]
        generator._save_checkpoint(samples2, [], topic_path2)

        # Read samples file
        with open(generator._checkpoint_samples_path) as f:
            lines = f.readlines()

        assert len(lines) == 2  # noqa: PLR2004
        assert json.loads(lines[0])["question"] == "Q1"
        assert json.loads(lines[1])["question"] == "Q2"

    def test_load_checkpoint_restores_state(self, temp_checkpoint_dir, base_generator_params):
        """Test that loading a checkpoint restores sample counts and processed IDs.

        Note: With memory optimization, samples are NOT loaded into memory on resume.
        Instead, we track counts via _flushed_samples_count/_flushed_failures_count.
        Samples remain on disk and are loaded only when building the final dataset.
        """
        params = {**base_generator_params, "checkpoint_path": temp_checkpoint_dir}

        with patch("deepfabric.generator.LLMClient"):
            # Create and save checkpoint with TopicPath
            generator1 = DataSetGenerator(**params)
            generator1._initialize_checkpoint_paths()
            samples = [{"question": "Q1", "answer": "A1"}]
            failures = [{"error": "Failed"}]
            topic_paths = [TopicPath(path=["Topic1", "Subtopic1"], topic_id="test-uuid-123")]
            generator1._save_checkpoint(samples, failures, topic_paths)

            # Create new generator and load checkpoint
            generator2 = DataSetGenerator(**params)
            loaded = generator2.load_checkpoint()

        assert loaded is True
        # Memory optimization: samples stay on disk, we track counts
        assert generator2._flushed_samples_count == 1
        assert generator2._flushed_failures_count == 1
        # In-memory lists should be empty (samples are on disk)
        assert len(generator2._samples) == 0
        assert len(generator2.failed_samples) == 0
        # Check processed IDs instead of paths
        assert "test-uuid-123" in generator2._processed_ids

        # Verify we can load samples from disk when needed
        all_samples = generator2._load_all_samples_from_checkpoint()
        assert len(all_samples) == 1
        assert all_samples[0]["question"] == "Q1"

    def test_load_checkpoint_returns_false_when_no_checkpoint(
        self, temp_checkpoint_dir, base_generator_params
    ):
        """Test that load_checkpoint returns False when no checkpoint exists."""
        params = {**base_generator_params, "checkpoint_path": temp_checkpoint_dir}
        with patch("deepfabric.generator.LLMClient"):
            generator = DataSetGenerator(**params)

        loaded = generator.load_checkpoint()

        assert loaded is False
        assert len(generator._samples) == 0

    def test_load_checkpoint_with_retry_failed(self, temp_checkpoint_dir, base_generator_params):
        """Test that load_checkpoint with retry_failed=True re-queues failed IDs.

        Note: With memory optimization, failure counts are tracked via _flushed_failures_count.
        """
        params = {**base_generator_params, "checkpoint_path": temp_checkpoint_dir}

        with patch("deepfabric.generator.LLMClient"):
            # Create and save checkpoint with failed samples that have topic_ids
            generator1 = DataSetGenerator(**params)
            generator1._initialize_checkpoint_paths()
            samples = [{"question": "Q1", "answer": "A1"}]
            # Failures include the topic_id for retry functionality
            failures = [
                {
                    "error": "Rate limit exceeded",
                    "topic_id": "failed-uuid-456",
                    "path": "Topic2 -> Subtopic1",
                }
            ]
            topic_paths = [TopicPath(path=["Topic1", "Subtopic1"], topic_id="success-uuid-123")]
            generator1._save_checkpoint(samples, failures, topic_paths)
            # Also mark the failed topic_id as processed
            generator1._processed_ids.add("failed-uuid-456")
            generator1._save_checkpoint_metadata()

            # Load checkpoint without retry_failed - failed ID stays processed
            generator2 = DataSetGenerator(**params)
            loaded = generator2.load_checkpoint(retry_failed=False)
            assert loaded is True
            assert "failed-uuid-456" in generator2._processed_ids
            # Memory optimization: failures stay on disk, we track count
            assert generator2._flushed_failures_count == 1

            # Load checkpoint with retry_failed=True - failed ID removed from processed
            generator3 = DataSetGenerator(**params)
            loaded = generator3.load_checkpoint(retry_failed=True)
            assert loaded is True
            # Failed ID should be removed from processed so it can be retried
            assert "failed-uuid-456" not in generator3._processed_ids
            # Successfully processed ID should still be there
            assert "success-uuid-123" in generator3._processed_ids
            # Failures count should be cleared when retrying
            assert generator3._flushed_failures_count == 0

    def test_load_checkpoint_returns_false_when_disabled(self, temp_checkpoint_dir):
        """Test that load_checkpoint returns False when checkpointing is disabled."""
        with patch("deepfabric.generator.LLMClient"):
            generator = DataSetGenerator(
                generation_system_prompt="Test",
                provider="openai",
                model_name="gpt-4",
                checkpoint_interval=None,
                checkpoint_path=temp_checkpoint_dir,
            )

        loaded = generator.load_checkpoint()

        assert loaded is False

    def test_load_checkpoint_warns_on_config_mismatch(
        self, temp_checkpoint_dir, base_generator_params, caplog
    ):
        """Test that load_checkpoint warns when config differs from checkpoint."""
        params = {**base_generator_params, "checkpoint_path": temp_checkpoint_dir}

        with patch("deepfabric.generator.LLMClient"):
            # Create checkpoint with gpt-4 model
            generator1 = DataSetGenerator(**params)
            generator1._initialize_checkpoint_paths()
            samples = [{"question": "Q1", "answer": "A1"}]
            topic_paths = [TopicPath(path=["Topic1"], topic_id="uuid-1")]
            generator1._save_checkpoint(samples, [], topic_paths)

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

        # Create checkpoint files with TopicPath
        samples = [{"question": "Q1", "answer": "A1"}]
        topic_paths = [TopicPath(path=["Topic1"], topic_id="uuid-1")]
        generator._save_checkpoint(samples, [], topic_paths)

        # Verify files exist
        assert generator._checkpoint_samples_path.exists()
        assert generator._checkpoint_metadata_path.exists()

        # Clear checkpoint
        generator.clear_checkpoint()

        # Verify files are removed
        assert not generator._checkpoint_samples_path.exists()
        assert not generator._checkpoint_metadata_path.exists()
        assert len(generator._processed_ids) == 0


class TestIsTopicProcessed:
    """Tests for checking if a topic has been processed."""

    def test_is_topic_processed_returns_true_for_processed(self, generator_with_checkpoint):
        """Test that is_topic_processed returns True for processed topics."""
        generator = generator_with_checkpoint
        generator._processed_ids.add("test-uuid-123")

        topic_path = TopicPath(path=["Topic1", "Subtopic1"], topic_id="test-uuid-123")
        assert generator._is_topic_processed(topic_path) is True

    def test_is_topic_processed_returns_false_for_unprocessed(self, generator_with_checkpoint):
        """Test that is_topic_processed returns False for unprocessed topics."""
        generator = generator_with_checkpoint

        topic_path = TopicPath(path=["Topic1", "Subtopic1"], topic_id="new-uuid-456")
        assert generator._is_topic_processed(topic_path) is False

    def test_is_topic_processed_handles_none(self, generator_with_checkpoint):
        """Test that is_topic_processed returns False for None topics."""
        generator = generator_with_checkpoint

        assert generator._is_topic_processed(None) is False


class TestCheckpointMetadata:
    """Tests for checkpoint metadata."""

    def test_metadata_contains_expected_fields(self, generator_with_checkpoint):
        """Test that checkpoint metadata contains expected fields."""
        generator = generator_with_checkpoint
        generator._initialize_checkpoint_paths()

        samples = [{"question": "Q1", "answer": "A1"}]
        generator._samples = samples
        topic_paths = [TopicPath(path=["Topic1"], topic_id="uuid-1")]
        generator._save_checkpoint(samples, [], topic_paths)

        # Read metadata
        with open(generator._checkpoint_metadata_path) as f:
            metadata = json.load(f)

        assert "version" in metadata
        assert "created_at" in metadata
        assert "provider" in metadata
        assert "model_name" in metadata
        assert "total_samples" in metadata
        assert "total_failures" in metadata
        assert "processed_ids" in metadata
        assert "checkpoint_interval" in metadata

        assert metadata["version"] == CHECKPOINT_VERSION
        assert metadata["provider"] == "openai"
        assert metadata["model_name"] == "gpt-4"
        assert "uuid-1" in metadata["processed_ids"]


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
  checkpoint:
    interval: 10
    path: "{temp_checkpoint_dir}"
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
  mode: graph
  depth: 2
  degree: 2
generation:
  system_prompt: "Test system prompt"
output:
  save_as: "test_dataset.jsonl"
  num_samples: 100
  batch_size: 1
  checkpoint:
    interval: 10
    path: "{temp_checkpoint_dir}"
"""
        config_path = os.path.join(temp_checkpoint_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(config_content)

        # Create checkpoint files with version 2 format
        metadata = {
            "version": 2,
            "created_at": "2024-01-01T00:00:00Z",
            "provider": "openai",
            "model_name": "gpt-4",
            "conversation_type": "basic",
            "reasoning_style": None,
            "total_samples": 25,
            "total_failures": 2,
            "processed_ids": ["uuid-1", "uuid-2"],
            "checkpoint_interval": 10,
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
  mode: graph
  depth: 2
  degree: 2
generation:
  system_prompt: "Test system prompt"
output:
  save_as: "test_dataset.jsonl"
  num_samples: 100
  batch_size: 1
  checkpoint:
    interval: 10
    path: "{temp_checkpoint_dir}"
"""
        config_path = os.path.join(temp_checkpoint_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(config_content)

        # Create checkpoint metadata with version 2 format
        metadata = {
            "version": 2,
            "created_at": "2024-01-01T00:00:00Z",
            "provider": "openai",
            "model_name": "gpt-4",
            "conversation_type": "basic",
            "total_samples": 10,
            "total_failures": 3,
            "processed_ids": [],
            "checkpoint_interval": 10,
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
