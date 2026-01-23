import asyncio
import contextlib
import json
import os
import traceback

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Dataset as HFDataset
from rich.layout import Layout
from rich.live import Live

from .config import DeepFabricConfig
from .config_manager import DEFAULT_MODEL
from .exceptions import ConfigurationError
from .generator import DataSetGenerator
from .progress import ProgressReporter
from .tui import STREAM_PANEL_WIDTH, get_dataset_tui, get_tui
from .utils import ensure_not_running_loop


# Lazy/defensive access to TUI settings to avoid early import issues
def _get_tui_settings():
    try:
        from .tui import get_tui_settings as _gts  # noqa: PLC0415

        return _gts()
    except Exception:

        class _S:
            mode = "rich"

        return _S()


def _get_preview_lines() -> int:
    try:
        from .tui import get_preview_lines as _gpl  # noqa: PLC0415

        return _gpl()
    except Exception:
        return 16


if TYPE_CHECKING:
    from .topic_model import TopicModel

# Constants for debug output
DEBUG_MAX_FAILURES_TO_SHOW = 10


def resolve_num_samples(num_samples: int | str, topic_count: int) -> int:
    """Resolve num_samples to an integer based on topic count.

    Args:
        num_samples: Integer, "auto", or percentage string like "50%"
        topic_count: Number of available topic paths

    Returns:
        Resolved integer sample count

    Raises:
        ConfigurationError: If topic_count is 0 and dynamic sampling is requested
    """
    if isinstance(num_samples, int):
        return num_samples

    if topic_count == 0:
        raise ConfigurationError(
            "Cannot use 'auto' or percentage num_samples with empty topic model. "
            "Ensure topic generation produced paths."
        )

    if num_samples == "auto":
        return topic_count

    if isinstance(num_samples, str) and num_samples.endswith("%"):
        percentage = float(num_samples[:-1]) / 100.0
        return max(1, int(topic_count * percentage))

    # Fallback - try to parse as int (shouldn't reach here if validated properly)
    return int(num_samples)


async def handle_dataset_events_async(
    generator: AsyncIterator[dict | HFDataset], engine=None, debug: bool = False
) -> HFDataset | None:
    """Handle dataset generation with TUI progress and streaming feedback."""
    tui = get_dataset_tui()
    footer_prog = None
    task = None
    live = None
    simple_task = None

    final_result: HFDataset | None = None
    try:
        async for event in generator:
            if isinstance(event, dict) and "event" in event:
                if event["event"] == "generation_start":
                    settings = _get_tui_settings()
                    # Build header and params panels for layout
                    header_panel, params_panel = tui.build_generation_panels(
                        event["model_name"], event["num_steps"], event["batch_size"]
                    )
                    # Capture context for the run
                    tui.root_topic_prompt = event.get("root_topic_prompt")
                    tui.topic_model_type = event.get("topic_model_type")

                    if settings.mode == "rich":
                        # Initialize status tracking
                        tui.init_status(
                            total_steps=event["num_steps"],
                            total_samples=event["total_samples"],
                            checkpoint_enabled=event.get("checkpoint_enabled", False),
                        )

                        # Build layout with footer card
                        layout = Layout(name="root")
                        layout.split(Layout(name="main"), Layout(name="footer", size=3))
                        left = Layout(name="left", ratio=3)
                        right = Layout(name="right", ratio=2)
                        right.minimum_size = STREAM_PANEL_WIDTH
                        # Right column: status on top, streaming preview fills remaining space
                        right.split(
                            Layout(name="status", size=8),
                            Layout(name="preview"),
                        )
                        left.split(
                            Layout(name="header", size=4),
                            Layout(name="params", size=6),
                            Layout(name="context", size=5),
                            Layout(name="events"),
                        )
                        left["header"].update(header_panel)
                        left["params"].update(params_panel)
                        left["context"].update(tui._context_panel())
                        left["events"].update(tui.tui.build_events_panel([], title="Events"))
                        right["status"].update(tui._status_panel())
                        right["preview"].update(
                            tui.tui.build_stream_panel("Waiting for LLM output...")
                        )
                        layout["main"].split_row(left, right)

                        # Footer run status
                        footer_prog = tui.tui.create_footer(layout, title="Run Status")
                        task = footer_prog.add_task(
                            "Generating dataset samples",
                            total=event["total_samples"],
                            completed=event.get("resumed_samples", 0),
                        )

                        # Use alternate screen to avoid scroll trails; leave a clean terminal
                        live = Live(
                            layout,
                            console=tui.console,
                            refresh_per_second=15,
                            screen=True,
                        )
                        tui.live_display = live  # Give TUI reference to update it
                        tui.live_layout = layout  # Allow TUI to update panes
                        live.start()
                    else:
                        # Simple/headless mode: print and proceed without Live
                        tui.show_generation_header(
                            event["model_name"], event["num_steps"], event["batch_size"]
                        )
                        simple_task = {
                            "count": event.get("resumed_samples", 0),
                            "total": event["total_samples"],
                        }
                elif event["event"] == "step_complete":
                    samples_generated = event.get("samples_generated", 0)
                    if footer_prog and task is not None:
                        if samples_generated > 0:
                            with contextlib.suppress(Exception):
                                footer_prog.update(task, advance=samples_generated)
                            tui.log_event(f"✓ Generated +{samples_generated} samples")
                            # Update status totals
                            tui.status_step_complete(
                                samples_generated, int(event.get("failed_in_step", 0))
                            )
                    elif isinstance(simple_task, dict):
                        simple_task["count"] += samples_generated
                        failed_in_step = int(event.get("failed_in_step", 0))
                        retry_summary = tui.get_step_retry_summary()

                        # Build step summary message
                        step_msg = f"Step {event.get('step')}: +{samples_generated}"
                        if failed_in_step > 0:
                            step_msg += f" (-{failed_in_step} failed)"
                        step_msg += f" (total {simple_task['count']}/{simple_task['total']})"

                        # Display with appropriate style based on failures
                        if failed_in_step > 0:
                            tui.warning(step_msg)
                        else:
                            tui.info(step_msg)

                        # Show retry summary if there were retries
                        if retry_summary:
                            tui.console.print(f"   [dim]{retry_summary}[/dim]")

                        # Clear retries for next step
                        tui.clear_step_retries()
                elif event["event"] == "step_start":
                    # Keep status panel in sync
                    step = int(event.get("step", 0))
                    total = int(event.get("total_steps", 0))
                    tui.status_step_start(step, total)

                elif event["event"] == "checkpoint_saved":
                    # Display checkpoint save notification
                    total_samples = event.get("total_samples", 0)
                    total_failures = event.get("total_failures", 0)
                    is_final = event.get("final", False)

                    if footer_prog and task is not None:
                        # Rich mode: log to events panel and update status
                        if is_final:
                            tui.log_event(f"💾 Final checkpoint: {total_samples} samples")
                        else:
                            tui.log_event(f"💾 Checkpoint: {total_samples} samples")
                        tui.status_checkpoint_saved(total_samples)
                    elif isinstance(simple_task, dict):
                        # Simple mode: print checkpoint notification
                        checkpoint_msg = f"Checkpoint saved: {total_samples} samples"
                        if total_failures > 0:
                            checkpoint_msg += f" ({total_failures} failures)"
                        if is_final:
                            checkpoint_msg = "Final " + checkpoint_msg.lower()
                        tui.info(checkpoint_msg)

                elif event["event"] == "generation_stopped":
                    # Graceful stop at checkpoint
                    if live:
                        live.stop()
                    tui.console.print()
                    tui.success(
                        f"Gracefully stopped: {event['total_samples']} samples saved to checkpoint"
                    )
                    if event.get("total_failures", 0) > 0:
                        tui.info(f"({event['total_failures']} failures recorded)")
                    tui.info("Resume with: --resume flag")

                elif event["event"] == "generation_complete":
                    if live:
                        live.stop()
                    tui.console.print()  # Add blank line after live display
                    tui.success(f"Successfully generated {event['total_samples']} samples")
                    tui.log_event(
                        f"Done • total={event['total_samples']} failed={event['failed_samples']}"
                    )
                    if event["failed_samples"] > 0:
                        tui.warning(f"Failed to generate {event['failed_samples']} samples")

                        # Show detailed failure information in debug mode
                        if debug and engine and hasattr(engine, "failed_samples"):
                            get_tui().error("\n🔍 Debug: Dataset generation failures:")
                            for idx, failure in enumerate(
                                engine.failed_samples[:DEBUG_MAX_FAILURES_TO_SHOW], 1
                            ):
                                get_tui().error(f"  [{idx}] {failure}")
                            if len(engine.failed_samples) > DEBUG_MAX_FAILURES_TO_SHOW:
                                remaining = len(engine.failed_samples) - DEBUG_MAX_FAILURES_TO_SHOW
                                get_tui().error(f"  ... and {remaining} more failures")

            elif isinstance(event, HFDataset):
                final_result = event
            else:
                # Handle unexpected non-dict, non-Dataset events
                get_tui().warning(f"Unexpected event type: {type(event)}")
    except Exception as e:
        if live:
            live.stop()
        if debug:
            get_tui().error(f"🔍 Debug: Full traceback:\n{traceback.format_exc()}")
        get_tui().error(f"Dataset generation failed: {str(e)}")
        raise

    return final_result


def handle_dataset_events(generator, engine=None, debug: bool = False) -> HFDataset | None:
    """Synchronous wrapper for async dataset event handling."""
    ensure_not_running_loop("handle_dataset_events")
    return asyncio.run(handle_dataset_events_async(generator, engine=engine, debug=debug))


def create_dataset(
    engine: DataSetGenerator,
    topic_model: "TopicModel",
    config: DeepFabricConfig,
    num_samples: int | str | None = None,
    batch_size: int | None = None,
    include_system_message: bool | None = None,
    provider: str | None = None,  # noqa: ARG001
    model: str | None = None,
    generation_overrides: dict | None = None,
    debug: bool = False,
) -> HFDataset:
    """
    Create dataset using the data engine and topic model.

    Args:
        engine: DataSetGenerator instance
        topic_model: TopicModel (Tree or Graph) to use for generation
        config: DeepFabricConfig object
        num_samples: Override for number of samples (int, "auto", or percentage like "50%")
        batch_size: Override for batch size
        include_system_message: Override for including system message
        provider: Override for LLM provider
        model: Override for model name
        generation_overrides: Additional generation parameter overrides

    Returns:
        Generated HuggingFace Dataset object

    Raises:
        ConfigurationError: If dataset generation fails
    """
    ensure_not_running_loop("create_dataset")
    return asyncio.run(
        create_dataset_async(
            engine=engine,
            topic_model=topic_model,
            config=config,
            num_samples=num_samples,
            batch_size=batch_size,
            include_system_message=include_system_message,
            provider=provider,
            model=model,
            generation_overrides=generation_overrides,
            debug=debug,
        )
    )


async def create_dataset_async(
    engine: DataSetGenerator,
    topic_model: "TopicModel",
    config: DeepFabricConfig,
    num_samples: int | str | None = None,
    batch_size: int | None = None,
    include_system_message: bool | None = None,
    provider: str | None = None,  # noqa: ARG001
    model: str | None = None,
    generation_overrides: dict | None = None,
    debug: bool = False,
) -> HFDataset:
    output_config = config.get_output_config()

    raw_num_samples = num_samples if num_samples is not None else output_config["num_samples"]
    final_batch_size = batch_size or output_config["batch_size"]

    # Resolve "auto" or percentage to actual count based on topic paths
    topic_count = len(topic_model.get_all_paths())
    final_num_samples = resolve_num_samples(raw_num_samples, topic_count)

    # Log resolution for dynamic values
    tui = get_dataset_tui()
    if isinstance(raw_num_samples, str):
        tui.info(f"Resolved num_samples: {raw_num_samples} → {final_num_samples} samples")

    generation_params = config.get_generation_params(**(generation_overrides or {}))
    final_model = model or generation_params.get("model_name", DEFAULT_MODEL)

    # Convert total samples to number of steps (batches)
    # The generator expects num_steps where total_samples = num_steps * batch_size
    import math  # noqa: PLC0415

    final_num_steps = math.ceil(final_num_samples / final_batch_size)

    tui.info(
        f"Dataset generation: {final_num_samples} samples in {final_num_steps} steps "
        f"(batch_size={final_batch_size})"
    )

    # Create progress reporter and attach TUI as observer for streaming feedback
    progress_reporter = ProgressReporter()
    progress_reporter.attach(tui)

    # Attach progress reporter to engine
    engine.progress_reporter = progress_reporter

    try:
        generator = engine.create_data_with_events_async(
            num_steps=final_num_steps,
            batch_size=final_batch_size,
            topic_model=topic_model,
            model_name=final_model,
            sys_msg=include_system_message,
            num_example_demonstrations=output_config.get("num_example_demonstrations") or 3,
        )
        dataset = await handle_dataset_events_async(generator, engine=engine, debug=debug)
    except Exception as e:  # noqa: BLE001
        raise ConfigurationError(f"Error creating dataset: {str(e)}") from e

    if dataset is None:
        raise ConfigurationError("Dataset generation failed - no dataset returned")

    return dataset


def _upload_to_service(
    service_name: str,
    dataset_path: str,
    config: dict,
    credential_check_func,
    uploader_import_func,
    uploader_args_func,
    push_args_func,
    tui,
) -> None:
    """Generic function to upload dataset to any configured service."""
    try:
        tui.info(f"Uploading dataset to {service_name}...")

        # Check credentials
        credentials = credential_check_func()
        if not credentials:
            return

        # Import uploader class
        uploader_class = uploader_import_func()

        # Create uploader instance
        uploader_args = uploader_args_func(credentials)
        uploader = (
            uploader_class(*uploader_args)
            if isinstance(uploader_args, tuple)
            else uploader_class(**uploader_args)
        )

        # Prepare push arguments
        push_args = push_args_func(config, dataset_path)

        # Upload dataset
        result = uploader.push_to_hub(**push_args)

        if result["status"] == "success":
            tui.success(result["message"])
        else:
            tui.warning(f"{service_name} upload failed: {result['message']}")

    except Exception as e:
        tui.warning(f"Error uploading to {service_name}: {str(e)}")


def _upload_to_huggingface(dataset_path: str, hf_config: dict, tui) -> None:
    """Upload dataset to HuggingFace Hub if configured."""

    def check_credentials():
        token = os.getenv("HF_TOKEN")
        if not token:
            tui.warning("HF_TOKEN not set. Skipping HuggingFace upload.")
            return None
        return token

    def import_uploader():
        from .hf_hub import HFUploader  # noqa: PLC0415

        return HFUploader

    def get_uploader_args(credentials):
        return (credentials,)  # HFUploader takes token as single argument

    def get_push_args(config, dataset_path):
        return {
            "hf_dataset_repo": config["repository"],
            "jsonl_file_path": dataset_path,
            "tags": config.get("tags", []),
        }

    _upload_to_service(
        "HuggingFace Hub",
        dataset_path,
        hf_config,
        check_credentials,
        import_uploader,
        get_uploader_args,
        get_push_args,
        tui,
    )


def _upload_to_kaggle(dataset_path: str, kaggle_config: dict, tui) -> None:
    """Upload dataset to Kaggle if configured."""

    def check_credentials():
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")
        if not username or not key:
            tui.warning("KAGGLE_USERNAME or KAGGLE_KEY not set. Skipping Kaggle upload.")
            return None
        return (username, key)

    def import_uploader():
        from .kaggle_hub import KaggleUploader  # noqa: PLC0415

        return KaggleUploader

    def get_uploader_args(credentials):
        return credentials  # KaggleUploader takes username, key as tuple

    def get_push_args(config, dataset_path):
        return {
            "dataset_handle": config["handle"],
            "jsonl_file_path": dataset_path,
            "tags": config.get("tags", []),
            "version_notes": config.get("version_notes"),
            "description": config.get("description"),
        }

    _upload_to_service(
        "Kaggle",
        dataset_path,
        kaggle_config,
        check_credentials,
        import_uploader,
        get_uploader_args,
        get_push_args,
        tui,
    )


def _strip_nulls(obj: Any) -> Any:
    """Recursively strip null values from nested dicts and lists.

    HuggingFace Dataset's Arrow schema injects null for missing fields across rows.
    This function removes those nulls for clean JSON output.
    """
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nulls(item) for item in obj]
    return obj


def _save_jsonl_without_nulls(dataset: HFDataset, save_path: str) -> None:
    """Save HF Dataset to JSONL, stripping null values injected by Arrow schema."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        for row in dataset:
            cleaned = _strip_nulls(dict(row))
            f.write(json.dumps(cleaned, separators=(",", ":")) + "\n")


def _save_failed_samples(save_path: str, failed_samples: list, tui) -> None:
    """Save failed samples to a timestamped file alongside the main dataset.

    Args:
        save_path: Path to the main dataset file (e.g., "my-dataset.jsonl")
        failed_samples: List of failed samples - can be dicts with 'error' and 'raw_content' keys,
                       or plain strings/other types for legacy compatibility
        tui: TUI instance for output
    """
    # Generate timestamped filename: my-dataset.jsonl -> my-dataset_failures_20231130_143022.jsonl
    base_path = save_path.rsplit(".", 1)[0] if "." in save_path else save_path
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    failures_path = f"{base_path}_failures_{timestamp}.jsonl"

    try:
        with open(failures_path, "w") as f:
            for idx, failure in enumerate(failed_samples):
                # Structure each failure as a JSON object with metadata
                failure_record = {
                    "index": idx,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
                if isinstance(failure, dict):
                    # New format: dict with 'error' and optionally 'raw_content'
                    failure_record["error"] = failure.get("error", str(failure))
                    if "raw_content" in failure:
                        failure_record["raw_content"] = failure["raw_content"]
                else:
                    # Legacy format: plain string or other type
                    failure_record["error"] = str(failure)
                f.write(json.dumps(failure_record) + "\n")
        tui.warning(f"Failed samples saved to: {failures_path} ({len(failed_samples)} failures)")
    except Exception as e:
        tui.error(f"Could not save failed samples: {str(e)}")


def save_dataset(
    dataset: HFDataset,
    save_path: str,
    config: DeepFabricConfig | None = None,
    engine: DataSetGenerator | None = None,
) -> None:
    """
    Save dataset to file.

    Args:
        dataset: HuggingFace Dataset object to save
        save_path: Path where to save the dataset
        config: Optional configuration for upload settings
        engine: Optional DataSetGenerator to save failed samples from

    Raises:
        ConfigurationError: If saving fails
    """
    tui = get_tui()
    try:
        # Save the dataset as JSONL, stripping null values injected by HF Dataset
        # HuggingFace Dataset's Arrow schema adds null for missing fields across rows,
        # but we want clean output without null values for optional fields
        _save_jsonl_without_nulls(dataset, save_path)
        tui.success(f"Dataset saved to: {save_path}")

        # Save failed samples if engine has any (including flushed to checkpoint)
        if engine:
            all_failures = engine.get_all_failures()
            if all_failures:
                _save_failed_samples(save_path, all_failures, tui)

        # Handle automatic uploads if configured
        if config:
            # HuggingFace upload
            if config.huggingface:
                _upload_to_huggingface(save_path, config.get_huggingface_config(), tui)

            # Kaggle upload
            if config.kaggle:
                _upload_to_kaggle(save_path, config.get_kaggle_config(), tui)

    except Exception as e:
        raise ConfigurationError(f"Error saving dataset: {str(e)}") from e
