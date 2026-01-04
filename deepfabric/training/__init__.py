"""DeepFabric training utilities.

This module provides:
- Integration with HuggingFace Trainer and TRL trainers for metrics logging
- Dataset preparation utilities for optimizing training data

Features:
- Non-blocking async metrics sending
- Notebook-friendly API key prompts (like wandb)
- Graceful handling of failures without impacting training
- Tool filtering to reduce sequence lengths and memory usage

Usage:
    from deepfabric.training import DeepFabricCallback, prepare_dataset_for_training

    # Prepare dataset (reduces tool overhead)
    dataset = load_dataset("your/dataset", split="train")
    prepared = prepare_dataset_for_training(dataset, tool_strategy="used_only")

    # Train with metrics logging
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepared,
    )
    trainer.add_callback(DeepFabricCallback(trainer))
    trainer.train()

Environment Variables:
    DEEPFABRIC_API_KEY: API key for authentication
    DEEPFABRIC_API_URL: SaaS backend URL (default: https://api.deepfabric.cloud)
"""

from __future__ import annotations

from .callback import DeepFabricCallback
from .dataset_utils import (
    ToolInclusionStrategy,
    clean_tool_schema,
    filter_tools_for_sample,
    get_used_tool_names,
    prepare_dataset_for_training,
)
from .metrics_sender import MetricsSender

__all__ = [
    "DeepFabricCallback",
    "MetricsSender",
    "ToolInclusionStrategy",
    "clean_tool_schema",
    "filter_tools_for_sample",
    "get_used_tool_names",
    "prepare_dataset_for_training",
]
