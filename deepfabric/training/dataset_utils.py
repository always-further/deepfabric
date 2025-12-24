"""Dataset preparation utilities for training.

This module provides utilities for preparing DeepFabric datasets for training,
including tool filtering to reduce sequence lengths and memory usage.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

ToolInclusionStrategy = Literal["all", "used_only", "used_plus_related"]


def get_used_tool_names(messages: list[dict[str, Any]]) -> set[str]:
    """Extract tool names that are actually called in a conversation.

    Args:
        messages: List of message dicts from the conversation

    Returns:
        Set of tool names that were called
    """
    used_tools: set[str] = set()

    for msg in messages:
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        # OpenAI format: {"function": {"name": "..."}}
                        func = tc.get("function", {})
                        if isinstance(func, dict) and func.get("name"):
                            used_tools.add(func["name"])
                        # Alternative format: {"name": "..."}
                        elif tc.get("name"):
                            used_tools.add(tc["name"])

    return used_tools


def clean_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Remove null/None values from tool schema to reduce size.

    Args:
        tool: Tool definition in OpenAI format

    Returns:
        Cleaned tool definition with nulls removed
    """
    if not isinstance(tool, dict):
        return tool

    cleaned: dict[str, Any] = {}

    for key, value in tool.items():
        if value is None:
            continue
        if isinstance(value, dict):
            cleaned_value = clean_tool_schema(value)
            # Only include if dict is not empty after cleaning
            if cleaned_value:
                cleaned[key] = cleaned_value
        elif isinstance(value, list):
            cleaned_list = []
            for item in value:
                if isinstance(item, dict):
                    cleaned_item = clean_tool_schema(item)
                    if cleaned_item:
                        cleaned_list.append(cleaned_item)
                elif item is not None:
                    cleaned_list.append(item)
            if cleaned_list:
                cleaned[key] = cleaned_list
        else:
            cleaned[key] = value

    return cleaned


def filter_tools_for_sample(
    sample: dict[str, Any],
    strategy: ToolInclusionStrategy = "used_only",
    min_tools: int = 1,
    clean_schemas: bool = True,
) -> dict[str, Any]:
    """Filter tools in a sample to only include relevant ones.

    Args:
        sample: Dataset sample with 'messages' and 'tools' fields
        strategy: Tool inclusion strategy:
            - "all": Keep all tools (no filtering)
            - "used_only": Only include tools that are called in the conversation
            - "used_plus_related": Include used tools plus related ones (not implemented)
        min_tools: Minimum number of tools to include (fallback if filtering
            removes all tools)
        clean_schemas: Whether to remove null values from tool schemas

    Returns:
        Modified sample with filtered tools
    """
    if strategy == "all" and not clean_schemas:
        return sample

    messages = sample.get("messages", [])
    all_tools = sample.get("tools", [])

    if not all_tools:
        return sample

    # Clean schemas if requested
    if clean_schemas:
        all_tools = [clean_tool_schema(tool) for tool in all_tools]

    if strategy == "all":
        sample["tools"] = all_tools
        return sample

    # Get tools actually used
    used_names = get_used_tool_names(messages)

    if not used_names:
        # No tools used - keep minimum number of tools
        sample["tools"] = all_tools[:min_tools] if min_tools > 0 else []
        return sample

    # Filter to used tools
    filtered_tools = []
    for tool in all_tools:
        func = tool.get("function", {})
        if isinstance(func, dict) and func.get("name") in used_names:
            filtered_tools.append(tool)

    # Ensure minimum tools
    if len(filtered_tools) < min_tools:
        # Add more tools from the original list
        for tool in all_tools:
            if tool not in filtered_tools:
                filtered_tools.append(tool)
                if len(filtered_tools) >= min_tools:
                    break

    sample["tools"] = filtered_tools
    return sample


def prepare_dataset_for_training(
    dataset: Dataset,
    tool_strategy: ToolInclusionStrategy = "used_only",
    clean_tool_schemas: bool = True,
    min_tools: int = 1,
    num_proc: int | None = None,
) -> Dataset:
    """Prepare a DeepFabric dataset for training with optimizations.

    This function applies various optimizations to reduce dataset size and
    memory usage during training:
    - Filters tools to only include those actually used in each conversation
    - Removes null values from tool schemas
    - Can be extended with additional preprocessing steps

    Args:
        dataset: HuggingFace Dataset with DeepFabric conversation format
        tool_strategy: How to filter tools (see filter_tools_for_sample)
        clean_tool_schemas: Whether to remove null values from tool schemas
        min_tools: Minimum tools to keep per sample
        num_proc: Number of processes for parallel processing

    Returns:
        Processed dataset ready for training

    Example:
        >>> from datasets import load_dataset
        >>> from deepfabric.training import prepare_dataset_for_training
        >>>
        >>> dataset = load_dataset("your/dataset", split="train")
        >>> prepared = prepare_dataset_for_training(
        ...     dataset,
        ...     tool_strategy="used_only",
        ...     clean_tool_schemas=True,
        ... )
        >>> # Now use prepared dataset for training
    """
    logger.info(
        "Preparing dataset for training: tool_strategy=%s, clean_schemas=%s",
        tool_strategy,
        clean_tool_schemas,
    )

    # Get initial stats
    if "tools" in dataset.column_names:
        initial_tool_counts = [len(sample.get("tools", []) or []) for sample in dataset]
        avg_initial = (
            sum(initial_tool_counts) / len(initial_tool_counts) if initial_tool_counts else 0
        )
        logger.info("Initial average tools per sample: %.1f", avg_initial)

    # Apply tool filtering
    processed = dataset.map(
        lambda x: filter_tools_for_sample(
            x,
            strategy=tool_strategy,
            min_tools=min_tools,
            clean_schemas=clean_tool_schemas,
        ),
        num_proc=num_proc,
        desc="Filtering tools",
    )

    # Log final stats
    if "tools" in processed.column_names:
        final_tool_counts = [len(sample.get("tools", []) or []) for sample in processed]
        avg_final = sum(final_tool_counts) / len(final_tool_counts) if final_tool_counts else 0
        logger.info("Final average tools per sample: %.1f", avg_final)

    return processed
