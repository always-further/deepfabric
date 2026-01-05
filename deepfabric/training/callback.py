"""DeepFabric TrainerCallback for automatic metrics logging."""

from __future__ import annotations

import logging
import os
import uuid

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .api_key_prompt import get_api_key
from .metrics_sender import MetricsSender

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState
    from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class DeepFabricCallback:
    """Callback that sends training metrics to DeepFabric SaaS.

    This callback integrates with HuggingFace Trainer and TRL trainers to
    automatically log training metrics (loss, learning rate, epoch, global step,
    throughput, TRL-specific metrics, and custom metrics) to the DeepFabric
    backend.

    The callback is designed to be non-blocking and gracefully handles failures
    without impacting training.

    Example:
        from deepfabric.training import DeepFabricCallback

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.add_callback(DeepFabricCallback(trainer))
        trainer.train()

    Environment Variables:
        DEEPFABRIC_API_KEY: API key (alternative to constructor arg)
        DEEPFABRIC_API_URL: Backend URL (default: https://api.deepfabric.cloud)
    """

    def __init__(
        self,
        trainer: Any | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        pipeline_id: str | None = None,
        enabled: bool = True,
    ):
        """Initialize the DeepFabric callback.

        Args:
            trainer: Optional Trainer instance to extract model info from
            api_key: DeepFabric API key (falls back to DEEPFABRIC_API_KEY env var,
                     then prompts in interactive environments)
            endpoint: API endpoint URL (falls back to DEEPFABRIC_API_URL env var)
            pipeline_id: Pipeline ID to associate training with (falls back to
                         DEEPFABRIC_PIPELINE_ID env var or pipeline_id.txt file)
            enabled: Whether logging is enabled (default: True)
        """
        # Get API key from arg, env, or prompt
        self.api_key = api_key or get_api_key()
        self.endpoint = endpoint or os.getenv("DEEPFABRIC_API_URL", "https://api.deepfabric.cloud")
        self.pipeline_id = pipeline_id or self._get_pipeline_id()
        self.run_id = str(uuid.uuid4())
        self.enabled = enabled and self.api_key is not None

        # Store trainer reference for model extraction
        self._trainer = trainer

        # Initialize sender (handles None api_key gracefully)
        self.sender = MetricsSender(
            endpoint=self.endpoint,
            api_key=self.api_key if self.enabled else None,
            pipeline_id=self.pipeline_id,
        )

        self._run_started = False
        self._model_name: str | None = None
        self._training_args_logged = False
        self._start_time: datetime | None = None

        if self.enabled:
            if self.pipeline_id:
                logger.debug(
                    f"DeepFabric callback initialized (run_id={self.run_id}, "
                    f"pipeline_id={self.pipeline_id})"
                )
            else:
                logger.warning(
                    "DeepFabric callback initialized but no pipeline_id set. "
                    "Metrics will not be sent. Set DEEPFABRIC_PIPELINE_ID env var "
                    "or create pipeline_id.txt file."
                )
        else:
            logger.debug("DeepFabric callback disabled (no API key)")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        """Called at the beginning of training.

        Sends run start event with training configuration.
        """
        if not self.enabled or self._run_started:
            return

        self._run_started = True
        self._start_time = datetime.now(timezone.utc)

        # Extract model name from various sources
        model = kwargs.get("model")
        if model is None and self._trainer is not None:
            model = getattr(self._trainer, "model", None)
        self._model_name = self._extract_model_name(args, model)

        # Build training args dict (safe extraction)
        training_config = self._extract_training_args(args)

        self.sender.send_run_start(
            {
                "run_id": self.run_id,
                "model_name": self._model_name,
                "training_config": training_config,
                "state": {
                    "max_steps": state.max_steps,
                    "num_train_epochs": state.num_train_epochs,
                    "is_world_process_zero": getattr(state, "is_world_process_zero", True),
                },
                "started_at": self._start_time.isoformat(),
            }
        )

    def on_log(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        logs: dict[str, float] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Called when metrics are logged.

        Sends all logged metrics to DeepFabric (loss, learning_rate, epoch,
        global_step, throughput, TRL metrics, custom metrics, etc.).
        """
        if not self.enabled or logs is None:
            return

        # Filter out None values and non-numeric values
        filtered_logs = {}
        for key, value in logs.items():
            if value is not None:
                if isinstance(value, int | float):
                    filtered_logs[key] = value
                elif isinstance(value, str):
                    # Keep string values for metadata
                    filtered_logs[key] = value

        if not filtered_logs:
            return

        payload = {
            "run_id": self.run_id,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "log",
            "metrics": filtered_logs,
        }

        self.sender.send_metrics(payload)

    def on_evaluate(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        metrics: dict[str, float] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Called after evaluation.

        Sends evaluation metrics to DeepFabric.
        """
        if not self.enabled or metrics is None:
            return

        payload = {
            "run_id": self.run_id,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "eval",
            "metrics": metrics,
        }

        self.sender.send_metrics(payload)

    def on_train_end(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Called at the end of training.

        Sends run end event and flushes pending metrics.
        """
        if not self.enabled or not self._run_started:
            return

        completed_at = datetime.now(timezone.utc)

        self.sender.send_run_end(
            {
                "run_id": self.run_id,
                "final_step": state.global_step,
                "final_epoch": state.epoch,
                "total_flos": getattr(state, "total_flos", None),
                "best_metric": getattr(state, "best_metric", None),
                "best_model_checkpoint": getattr(state, "best_model_checkpoint", None),
                "completed_at": completed_at.isoformat(),
            }
        )

        # Flush remaining metrics
        self.sender.flush(timeout=30.0)

        logger.debug(f"DeepFabric run completed: {self.sender.stats}")

    def on_save(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Called when a checkpoint is saved.

        Optionally logs checkpoint events.
        """
        if not self.enabled:
            return

        # Log checkpoint event
        self.sender.send_metrics(
            {
                "run_id": self.run_id,
                "global_step": state.global_step,
                "epoch": state.epoch,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "checkpoint",
                "metrics": {"checkpoint_step": state.global_step},
            }
        )

    def _get_pipeline_id(self) -> str | None:
        """Get pipeline ID from environment or file.

        Returns:
            Pipeline ID or None
        """
        # Try environment variable first
        pipeline_id = os.getenv("DEEPFABRIC_PIPELINE_ID", "")
        if pipeline_id:
            return pipeline_id

        # Try pipeline_id.txt file
        pipeline_file = "pipeline_id.txt"
        if os.path.exists(pipeline_file):
            with open(pipeline_file) as f:
                pipeline_id = f.read().strip()
                if pipeline_id:
                    return pipeline_id

        return None

    def _extract_model_name(self, args: TrainingArguments, model: Any | None) -> str | None:
        """Extract model name from various sources.

        Args:
            args: Training arguments
            model: Model instance (may be None)

        Returns:
            Model name or None
        """
        # Try args first (model_name_or_path exists on SFTConfig and similar subclasses)
        model_name_or_path = getattr(args, "model_name_or_path", None)
        if model_name_or_path is not None:
            return model_name_or_path

        # Try model config
        if model is not None:
            if hasattr(model, "config") and hasattr(model.config, "name_or_path"):
                return model.config.name_or_path
            if hasattr(model, "name_or_path"):
                return model.name_or_path

        # Try output_dir as fallback
        output_dir = getattr(args, "output_dir", None)
        if output_dir is not None:
            return os.path.basename(output_dir)

        return None

    def _extract_training_args(self, args: TrainingArguments) -> dict[str, Any]:
        """Extract training arguments for logging.

        Args:
            args: Training arguments

        Returns:
            Dictionary of training configuration
        """
        config = {}

        # Core training args
        safe_attrs = [
            "num_train_epochs",
            "max_steps",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "max_grad_norm",
            "warmup_steps",
            "warmup_ratio",
            "lr_scheduler_type",
            "logging_steps",
            "eval_steps",
            "save_steps",
            "seed",
            "fp16",
            "bf16",
            "gradient_checkpointing",
            "deepspeed",
            "local_rank",
            "dataloader_num_workers",
        ]

        for attr in safe_attrs:
            if hasattr(args, attr):
                value = getattr(args, attr)
                # Convert enums to strings
                if hasattr(value, "value"):
                    value = value.value
                config[attr] = value

        return config


# Make it compatible with transformers TrainerCallback protocol
# by ensuring it has all required methods (even as no-ops)
def _ensure_trainer_callback_compatibility():
    """Ensure DeepFabricCallback has all TrainerCallback methods."""
    # These methods are optional but good to have for completeness
    # Include all methods that transformers might call on callbacks
    optional_methods = [
        "on_step_begin",
        "on_step_end",
        "on_substep_end",
        "on_epoch_begin",
        "on_epoch_end",
        "on_prediction_step",
        "on_init_end",
        # Newer transformers versions
        "on_pre_optimizer_step",
        "on_optimizer_step",
        "on_post_optimizer_step",
        "on_pre_scheduler_step",
        "on_scheduler_step",
        "on_post_scheduler_step",
    ]

    def _make_noop(name):
        """Create a no-op method that returns control unchanged."""

        def noop(self, args, state, control, **kwargs):  # noqa: ARG001
            return control

        noop.__name__ = name
        return noop

    for method in optional_methods:
        if not hasattr(DeepFabricCallback, method):
            setattr(DeepFabricCallback, method, _make_noop(method))


_ensure_trainer_callback_compatibility()
