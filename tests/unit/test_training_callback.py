"""Tests for DeepFabric training callback and metrics sender."""

from __future__ import annotations

import os
import time

from unittest.mock import MagicMock, patch

from deepfabric.training import api_key_prompt
from deepfabric.training.api_key_prompt import (
    _is_interactive_terminal,
    _is_notebook,
    clear_api_key_cache,
    get_api_key,
)
from deepfabric.training.callback import DeepFabricCallback
from deepfabric.training.metrics_sender import MetricsSender


class TestMetricsSender:
    """Tests for MetricsSender class."""

    def test_sender_disabled_without_api_key(self):
        """Sender should be disabled when no API key provided."""
        sender = MetricsSender(endpoint="https://api.test.com", api_key=None)

        assert not sender.enabled
        assert sender.send_metrics({"loss": 1.0}) is False

    def test_sender_enabled_with_api_key(self):
        """Sender should be enabled when API key provided."""
        sender = MetricsSender(endpoint="https://api.test.com", api_key="test-key")

        assert sender.enabled
        sender.shutdown()

    def test_send_metrics_queues_data(self):
        """send_metrics should queue data for async sending."""
        sender = MetricsSender(
            endpoint="https://api.test.com",
            api_key="test-key",
            batch_size=100,  # Large batch to prevent immediate flush
        )

        result = sender.send_metrics({"loss": 2.5, "step": 100})

        assert result is True
        assert sender._queue.qsize() == 1

        sender.shutdown()

    def test_queue_overflow_drops_metrics(self):
        """Metrics should be dropped when queue is full."""
        sender = MetricsSender(
            endpoint="https://api.test.com",
            api_key="test-key",
            max_queue_size=2,
            batch_size=100,
        )

        # Fill the queue
        sender.send_metrics({"step": 1})
        sender.send_metrics({"step": 2})

        # This should be dropped
        result = sender.send_metrics({"step": 3})

        assert result is False
        assert sender._metrics_dropped == 1

        sender.shutdown()

    def test_stats_tracking(self):
        """Sender should track statistics."""
        sender = MetricsSender(endpoint="https://api.test.com", api_key="test-key")

        stats = sender.stats

        assert "metrics_sent" in stats
        assert "metrics_dropped" in stats
        assert "send_errors" in stats
        assert "queue_size" in stats

        sender.shutdown()

    def test_send_run_start_queues_event(self):
        """send_run_start should queue a run_start event."""
        sender = MetricsSender(
            endpoint="https://api.test.com",
            api_key="test-key",
            batch_size=100,
        )

        result = sender.send_run_start({"run_id": "test-123", "model_name": "test-model"})

        assert result is True
        item = sender._queue.get_nowait()
        assert item["type"] == "run_start"
        assert item["data"]["run_id"] == "test-123"

        sender.shutdown()

    def test_send_run_end_queues_event(self):
        """send_run_end should queue a run_end event."""
        sender = MetricsSender(
            endpoint="https://api.test.com",
            api_key="test-key",
            batch_size=100,
        )

        result = sender.send_run_end({"run_id": "test-123", "final_step": 1000})

        assert result is True
        item = sender._queue.get_nowait()
        assert item["type"] == "run_end"

        sender.shutdown()

    @patch("deepfabric.training.metrics_sender.requests.post")
    def test_batch_sending(self, mock_post):
        """Metrics should be sent in batches."""
        mock_post.return_value = MagicMock(ok=True)

        sender = MetricsSender(
            endpoint="https://api.test.com",
            api_key="test-key",
            pipeline_id="test-pipeline",
            batch_size=2,
            flush_interval=0.1,
        )

        # Send 2 metrics to trigger batch
        sender.send_metrics({"loss": 1.0})
        sender.send_metrics({"loss": 2.0})

        # Wait for batch to be sent
        time.sleep(0.5)

        # Verify API was called with metrics batch
        sender.shutdown()

        # Should have been called (may be multiple times depending on timing)
        assert mock_post.called


class TestDeepFabricCallback:
    """Tests for DeepFabricCallback class."""

    def test_callback_disabled_without_api_key(self):
        """Callback should be disabled when no API key provided."""
        # Clear environment
        env_key = os.environ.pop("DEEPFABRIC_API_KEY", None)
        clear_api_key_cache()

        try:
            callback = DeepFabricCallback(api_key=None)

            assert not callback.enabled
            assert not callback.sender.enabled
        finally:
            if env_key:
                os.environ["DEEPFABRIC_API_KEY"] = env_key
            clear_api_key_cache()

    def test_callback_enabled_with_api_key(self):
        """Callback should be enabled when API key provided."""
        callback = DeepFabricCallback(api_key="test-key")

        assert callback.enabled
        assert callback.sender.enabled
        assert callback.run_id is not None

        callback.sender.shutdown()

    def test_callback_uses_env_api_key(self):
        """Callback should use DEEPFABRIC_API_KEY env var."""
        os.environ["DEEPFABRIC_API_KEY"] = "env-test-key"
        clear_api_key_cache()

        try:
            callback = DeepFabricCallback()

            assert callback.enabled
            assert callback.api_key == "env-test-key"

            callback.sender.shutdown()
        finally:
            del os.environ["DEEPFABRIC_API_KEY"]
            clear_api_key_cache()

    def test_callback_uses_custom_endpoint(self):
        """Callback should use custom endpoint if provided."""
        callback = DeepFabricCallback(
            api_key="test-key",
            endpoint="https://custom.api.com",
        )

        assert callback.endpoint == "https://custom.api.com"

        callback.sender.shutdown()

    def test_callback_accepts_trainer(self):
        """Callback should accept trainer instance."""
        mock_trainer = MagicMock()
        mock_trainer.model = MagicMock()
        mock_trainer.model.config.name_or_path = "test-model"

        callback = DeepFabricCallback(trainer=mock_trainer, api_key="test-key")

        assert callback._trainer is mock_trainer

        callback.sender.shutdown()

    def test_on_log_filters_none_values(self):
        """on_log should filter out None values from logs."""
        callback = DeepFabricCallback(api_key="test-key")

        # Mock state and args
        mock_state = MagicMock()
        mock_state.global_step = 100
        mock_state.epoch = 0.5
        mock_args = MagicMock()
        mock_control = MagicMock()

        logs = {"loss": 2.5, "learning_rate": 0.001, "grad_norm": None}

        callback.on_log(mock_args, mock_state, mock_control, logs=logs)

        # Check that metrics were queued
        assert callback.sender._queue.qsize() == 1

        callback.sender.shutdown()

    def test_on_log_skips_empty_logs(self):
        """on_log should skip if logs is empty or None."""
        callback = DeepFabricCallback(api_key="test-key")

        mock_state = MagicMock()
        mock_state.global_step = 100
        mock_args = MagicMock()
        mock_control = MagicMock()

        # Test with None
        callback.on_log(mock_args, mock_state, mock_control, logs=None)
        assert callback.sender._queue.qsize() == 0

        callback.sender.shutdown()

    def test_on_train_begin_sends_run_start(self):
        """on_train_begin should send run start event."""
        callback = DeepFabricCallback(api_key="test-key")

        mock_state = MagicMock()
        mock_state.max_steps = 1000
        mock_state.num_train_epochs = 3
        mock_state.is_world_process_zero = True

        mock_args = MagicMock()
        mock_args.num_train_epochs = 3
        mock_args.per_device_train_batch_size = 8
        mock_args.learning_rate = 2e-5

        mock_control = MagicMock()

        callback.on_train_begin(mock_args, mock_state, mock_control)

        # Check that run_start was queued
        assert callback._run_started
        assert callback.sender._queue.qsize() == 1

        callback.sender.shutdown()

    def test_on_train_end_sends_run_end(self):
        """on_train_end should send run end event."""
        callback = DeepFabricCallback(api_key="test-key")
        callback._run_started = True

        mock_state = MagicMock()
        mock_state.global_step = 1000
        mock_state.epoch = 3.0
        mock_state.total_flos = 1e15
        mock_state.best_metric = 0.95

        mock_args = MagicMock()
        mock_control = MagicMock()

        with patch.object(callback.sender, "flush"):
            callback.on_train_end(mock_args, mock_state, mock_control)

        # Check that run_end was queued
        assert callback.sender._queue.qsize() == 1

        callback.sender.shutdown()

    def test_callback_has_all_required_methods(self):
        """Callback should have all TrainerCallback methods."""
        callback = DeepFabricCallback(api_key="test-key")

        # Required methods
        required_methods = [
            "on_train_begin",
            "on_train_end",
            "on_log",
            "on_evaluate",
            "on_save",
            "on_step_begin",
            "on_step_end",
            "on_epoch_begin",
            "on_epoch_end",
            "on_pre_optimizer_step",
        ]

        for method in required_methods:
            assert hasattr(callback, method), f"Missing method: {method}"
            assert callable(getattr(callback, method))

        callback.sender.shutdown()


class TestAPIKeyPrompt:
    """Tests for API key prompt functions."""

    def test_is_notebook_returns_false_in_tests(self):
        """_is_notebook should return False in test environment."""
        assert _is_notebook() is False

    def test_is_interactive_terminal(self):
        """_is_interactive_terminal should detect terminal."""
        # In CI/tests, stdin is typically not a tty
        result = _is_interactive_terminal()
        assert isinstance(result, bool)

    def test_get_api_key_from_env(self):
        """get_api_key should return env var if set."""
        clear_api_key_cache()
        os.environ["DEEPFABRIC_API_KEY"] = "env-key-123"

        try:
            result = get_api_key()
            assert result == "env-key-123"
        finally:
            del os.environ["DEEPFABRIC_API_KEY"]
            clear_api_key_cache()

    def test_get_api_key_caches_result(self):
        """get_api_key should cache result to avoid repeated prompts."""
        clear_api_key_cache()

        # First call in non-interactive env returns None
        result1 = get_api_key()

        assert api_key_prompt._api_key_checked is True

        # Second call should use cache
        result2 = get_api_key()
        assert result1 == result2

        clear_api_key_cache()

    def test_clear_api_key_cache(self):
        """clear_api_key_cache should reset state."""
        # Set some state
        api_key_prompt._api_key_cache = "cached-key"
        api_key_prompt._api_key_checked = True

        clear_api_key_cache()

        assert api_key_prompt._api_key_cache is None
        assert api_key_prompt._api_key_checked is False


class TestIntegrationWithMockTrainer:
    """Integration tests with mock Trainer."""

    def test_callback_integration_with_mock_trainer(self):
        """Test callback works with a mock Trainer-like object."""
        # Create mock trainer
        mock_trainer = MagicMock()
        mock_trainer.model = MagicMock()
        mock_trainer.model.config.name_or_path = "test-model"

        # Create callback with trainer
        callback = DeepFabricCallback(trainer=mock_trainer, api_key="test-key")

        # Mock trainer components
        mock_args = MagicMock()
        mock_args.num_train_epochs = 3
        mock_args.per_device_train_batch_size = 8
        mock_args.learning_rate = 2e-5
        mock_args.warmup_steps = 100
        mock_args.output_dir = "./output"

        mock_state = MagicMock()
        mock_state.max_steps = 1000
        mock_state.num_train_epochs = 3
        mock_state.global_step = 0
        mock_state.epoch = 0.0
        mock_state.is_world_process_zero = True

        mock_control = MagicMock()

        # Simulate training lifecycle
        callback.on_train_begin(mock_args, mock_state, mock_control)
        assert callback._run_started

        # Simulate logging
        for step in range(10):
            mock_state.global_step = step
            mock_state.epoch = step / 100
            logs = {"loss": 2.5 - step * 0.1, "learning_rate": 2e-5}
            callback.on_log(mock_args, mock_state, mock_control, logs=logs)

        # Simulate evaluation
        eval_metrics = {"eval_loss": 2.0, "eval_accuracy": 0.85}
        callback.on_evaluate(mock_args, mock_state, mock_control, metrics=eval_metrics)

        # Simulate training end
        mock_state.global_step = 1000
        mock_state.epoch = 3.0

        with patch.object(callback.sender, "flush"):
            callback.on_train_end(mock_args, mock_state, mock_control)

        callback.sender.shutdown()
