import contextlib
import json
import math
import os
import signal
import sys

from pathlib import Path
from typing import Literal, NoReturn, cast

import click
import yaml

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from .auth import auth as auth_group
from .config import DeepFabricConfig
from .config_manager import apply_cli_overrides, get_final_parameters, load_config
from .constants import (
    CHECKPOINT_FAILURES_SUFFIX,
    CHECKPOINT_METADATA_SUFFIX,
    CHECKPOINT_SAMPLES_SUFFIX,
)
from .dataset_manager import create_dataset, save_dataset
from .exceptions import ConfigurationError
from .generator import DataSetGenerator
from .graph import Graph
from .llm import VerificationStatus, verify_provider_api_key
from .metrics import set_trace_debug, trace
from .topic_manager import load_or_build_topic_model, save_topic_model
from .topic_model import TopicModel
from .tui import configure_tui, get_dataset_tui, get_tui
from .update_checker import check_for_updates
from .utils import (
    check_dir_writable,
    check_path_writable,
    get_bool_env,
    get_checkpoint_dir,
    parse_num_samples,
)
from .validation import show_validation_success, validate_path_requirements

OverrideValue = str | int | float | bool | None
OverrideMap = dict[str, OverrideValue]


def handle_error(ctx: click.Context, error: Exception) -> NoReturn:
    """Handle errors in CLI commands."""
    _ = ctx  # Unused but required for click context
    tui = get_tui()

    # Check if this is formatted error from our event handlers
    error_msg = str(error)
    if not error_msg.startswith("Error: "):
        tui.error(f"Error: {error_msg}")
    else:
        tui.error(error_msg)

    sys.exit(1)


def _get_checkpoint_topics_path(
    checkpoint_dir: str,
    output_save_as: str,
) -> str | None:
    """
    Read checkpoint metadata to get the topics path used in the original run.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        output_save_as: Output file path (used to derive checkpoint file names)

    Returns:
        Topics file path from checkpoint metadata, or None if not available
    """
    checkpoint_path = Path(checkpoint_dir)
    output_stem = Path(output_save_as).stem
    metadata_path = checkpoint_path / f"{output_stem}{CHECKPOINT_METADATA_SUFFIX}"

    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata.get("topics_file") or metadata.get("topics_save_as")
    except (OSError, json.JSONDecodeError):
        return None


@click.group()
@click.version_option()
@click.option(
    "--debug",
    is_flag=True,
    envvar="DEEPFABRIC_DEBUG",
    help="Enable debug mode for detailed output",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool):
    """DeepFabric CLI - Generate synthetic training data for language models."""
    # Store debug flag in context for subcommands to access
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    # Check for updates on CLI startup (silently fail if any issues occur)
    with contextlib.suppress(Exception):
        check_for_updates()


class GenerateOptions(BaseModel):
    """
    Validated command options for dataset generation.

    These options can be provided via CLI arguments or a configuration file.
    so they are marked as optional here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config_file: str | None = None
    # New naming convention
    output_system_prompt: str | None = None
    topic_prompt: str | None = None
    topics_system_prompt: str | None = None
    generation_system_prompt: str | None = None
    topics_save_as: str | None = None
    topics_load: str | None = None
    output_save_as: str | None = None
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    degree: int | None = None
    depth: int | None = None
    num_samples: int | str | None = None
    batch_size: int | None = None
    base_url: str | None = None
    include_system_message: bool | None = None
    mode: Literal["tree", "graph"] = Field(default="tree")
    debug: bool = False
    topic_only: bool = False
    tui: Literal["rich", "simple"] = Field(default="rich")

    # Modular conversation configuration
    conversation_type: Literal["basic", "cot"] | None = None
    reasoning_style: Literal["freetext", "agent", "structured", "hybrid"] | None = None
    agent_mode: Literal["single_turn", "multi_turn"] | None = (
        None  # Deprecated, kept for backward compat
    )

    # Cloud upload (experimental)
    cloud_upload: Literal["all", "dataset", "graph", "none"] | None = None

    # Checkpointing options
    checkpoint_interval: int | None = None
    checkpoint_path: str | None = None
    resume: bool = False
    retry_failed: bool = False

    @model_validator(mode="after")
    def validate_mode_constraints(self) -> "GenerateOptions":
        if self.topic_only and self.topics_load:
            raise ValueError("--topic-only cannot be used with --topics-load")
        return self


class GenerationPreparation(BaseModel):
    """Validated state required to run dataset generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: DeepFabricConfig
    topics_overrides: OverrideMap = Field(default_factory=dict)
    generation_overrides: OverrideMap = Field(default_factory=dict)
    num_samples: int | str  # Can be int, "auto", or percentage like "50%"
    batch_size: int
    depth: int
    degree: int
    loading_existing: bool

    @model_validator(mode="after")
    def validate_positive_dimensions(self) -> "GenerationPreparation":
        # Skip num_samples validation for dynamic values (auto or percentage)
        if isinstance(self.num_samples, int) and self.num_samples <= 0:
            raise ValueError("num_samples must be greater than zero")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.depth <= 0:
            raise ValueError("depth must be greater than zero")
        if self.degree <= 0:
            raise ValueError("degree must be greater than zero")
        return self


def _validate_api_keys(
    config: DeepFabricConfig,
    provider_override: str | None = None,
) -> None:
    """Validate that required API keys are present and working for configured providers.

    Args:
        config: The loaded configuration
        provider_override: Optional CLI provider override that takes precedence

    Raises:
        ConfigurationError: If any required API key is missing or invalid
    """
    tui = get_tui()

    # Get providers from config
    providers = config.get_configured_providers()

    # If there's a provider override from CLI, that takes precedence for all components
    if provider_override:
        providers = {provider_override}

    # Display what we're checking
    provider_list = ", ".join(sorted(providers))
    tui.info(f"Validating API keys for: {provider_list}")

    # Verify each provider's API key
    errors = []
    validated_providers = []
    for provider in providers:
        result = verify_provider_api_key(provider)

        # Determine the primary env var to show in error messages
        env_var = result.api_key_env_var or f"{provider.upper()}_API_KEY"
        primary_var = env_var.split(" or ")[0] if " or " in env_var else env_var

        if result.status == VerificationStatus.MISSING:
            errors.append(
                f"  {provider}: API key not found.\n"
                f"    Export it with: export {primary_var}=your-api-key"
            )
        elif result.status == VerificationStatus.INVALID:
            errors.append(
                f"  {provider}: API key is invalid.\n"
                f"    Check your key and re-export: export {primary_var}=your-api-key"
            )
        elif result.status == VerificationStatus.CONNECTION_ERROR:
            if provider == "ollama":
                errors.append(
                    f"  {provider}: Cannot connect to Ollama server.\n"
                    f"    Make sure Ollama is running: ollama serve"
                )
            else:
                errors.append(
                    f"  {provider}: Connection failed.\n"
                    f"    Check your internet connection and try again."
                )
        elif result.status == VerificationStatus.RATE_LIMITED:
            errors.append(
                f"  {provider}: Rate limit exceeded.\n"
                f"    Wait a moment and try again, or check your API quota."
            )
        else:
            # VALID or NOT_APPLICABLE (e.g., ollama)
            validated_providers.append(provider)

    if errors:
        error_list = "\n".join(errors)
        raise ConfigurationError(f"API key verification failed:\n\n{error_list}")

    # Show success message
    tui.success(f"API keys validated for: {', '.join(sorted(validated_providers))}")
    print()  # Visual separator before next section


def _load_and_prepare_generation_context(
    options: GenerateOptions,
    *,
    skip_path_validation: bool = False,
) -> GenerationPreparation:
    """Load configuration, compute overrides, and validate derived parameters."""
    tui = get_tui()

    # Step 1: Load configuration
    tui.info("Loading configuration...")
    config = load_config(
        config_file=options.config_file,
        topic_prompt=options.topic_prompt,
        topics_system_prompt=options.topics_system_prompt,
        generation_system_prompt=options.generation_system_prompt,
        output_system_prompt=options.output_system_prompt,
        provider=options.provider,
        model=options.model,
        temperature=options.temperature,
        degree=options.degree,
        depth=options.depth,
        num_samples=options.num_samples,
        batch_size=options.batch_size,
        topics_save_as=options.topics_save_as,
        output_save_as=options.output_save_as,
        include_system_message=options.include_system_message,
        mode=options.mode,
        conversation_type=options.conversation_type,
        reasoning_style=options.reasoning_style,
        agent_mode=options.agent_mode,
    )

    # Step 2: Validate API keys EARLY - this is critical for user feedback
    # This must happen before any LLM operations and should be clearly visible
    _validate_api_keys(config, options.provider)

    topics_overrides_raw, generation_overrides_raw = apply_cli_overrides(
        output_system_prompt=options.output_system_prompt,
        topic_prompt=options.topic_prompt,
        topics_system_prompt=options.topics_system_prompt,
        generation_system_prompt=options.generation_system_prompt,
        provider=options.provider,
        model=options.model,
        temperature=options.temperature,
        degree=options.degree,
        depth=options.depth,
        base_url=options.base_url,
    )

    final_num_samples, final_batch_size, final_depth, final_degree = get_final_parameters(
        config=config,
        num_samples=options.num_samples,
        batch_size=options.batch_size,
        depth=options.depth,
        degree=options.degree,
    )

    loading_existing = bool(options.topics_load)

    # Skip path validation for topic-only mode since we're not generating dataset samples
    if not skip_path_validation:
        validate_path_requirements(
            mode=options.mode,
            depth=final_depth,
            degree=final_degree,
            num_samples=final_num_samples,
            batch_size=final_batch_size,
            loading_existing=loading_existing,
        )

        show_validation_success(
            mode=options.mode,
            depth=final_depth,
            degree=final_degree,
            num_samples=final_num_samples,
            batch_size=final_batch_size,
            loading_existing=loading_existing,
        )

    try:
        return GenerationPreparation(
            config=config,
            topics_overrides=cast(OverrideMap, topics_overrides_raw),
            generation_overrides=cast(OverrideMap, generation_overrides_raw),
            num_samples=final_num_samples,
            batch_size=final_batch_size,
            depth=final_depth,
            degree=final_degree,
            loading_existing=loading_existing,
        )
    except (ValueError, PydanticValidationError) as error:
        raise ConfigurationError(str(error)) from error


def _initialize_topic_model(
    *,
    preparation: GenerationPreparation,
    options: GenerateOptions,
) -> TopicModel:
    """Load existing topic structures or build new ones and persist when needed."""

    topic_model = load_or_build_topic_model(
        config=preparation.config,
        topics_load=options.topics_load,
        topics_overrides=preparation.topics_overrides,
        provider=options.provider,
        model=options.model,
        base_url=options.base_url,
        debug=options.debug,
    )

    if not options.topics_load:
        save_topic_model(
            topic_model=topic_model,
            config=preparation.config,
            topics_save_as=options.topics_save_as,
        )

    return topic_model


def _trigger_cloud_upload(
    *,
    preparation: GenerationPreparation,
    options: GenerateOptions,
    dataset_path: str | None = None,
) -> None:
    """Trigger cloud upload if EXPERIMENTAL_DF is enabled and mode is 'graph'.

    Args:
        preparation: Generation preparation context
        options: CLI options including cloud_upload flag
        dataset_path: Path to dataset file (None for topic-only mode)
    """
    # Cloud upload only supports graph mode, not tree mode
    # Use config.topics.mode since options.mode may have CLI default value
    actual_mode = preparation.config.topics.mode
    if not (get_bool_env("EXPERIMENTAL_DF") and actual_mode == "graph"):
        return

    from .cloud_upload import handle_cloud_upload  # noqa: PLC0415

    graph_path = options.topics_save_as or preparation.config.topics.save_as or "topic_graph.json"

    handle_cloud_upload(
        dataset_path=dataset_path,
        graph_path=graph_path,
        cloud_upload_flag=options.cloud_upload,
    )


def _run_generation(
    *,
    preparation: GenerationPreparation,
    topic_model: TopicModel,
    options: GenerateOptions,
    checkpoint_dir: str,
) -> None:
    """Create the dataset using the prepared configuration and topic model."""
    tui = get_tui()

    # Apply CLI checkpoint overrides
    checkpoint_overrides = {}
    if options.checkpoint_interval is not None:
        checkpoint_overrides["checkpoint_interval"] = options.checkpoint_interval
    if options.checkpoint_path is not None:
        checkpoint_overrides["checkpoint_path"] = options.checkpoint_path
    if options.retry_failed:
        checkpoint_overrides["checkpoint_retry_failed"] = options.retry_failed

    generation_params = preparation.config.get_generation_params(
        **preparation.generation_overrides, **checkpoint_overrides
    )

    # Use provided checkpoint_dir if not explicitly set via CLI
    if generation_params.get("checkpoint_path") is None:
        generation_params["checkpoint_path"] = checkpoint_dir

    # Resolve and pass topics_file for checkpoint metadata
    # Prioritize: loaded file > save path > config > default
    # Store absolute path for reliable resume from any working directory
    topics_mode = preparation.config.topics.mode
    default_topics_path = "topic_graph.json" if topics_mode == "graph" else "topic_tree.jsonl"
    resolved_topics_path = (
        options.topics_load
        or options.topics_save_as
        or preparation.config.topics.save_as
        or default_topics_path
    )
    generation_params["topics_file"] = str(Path(resolved_topics_path).resolve())

    engine = DataSetGenerator(**generation_params)

    # Check for existing checkpoint when not resuming
    if not options.resume and engine.has_checkpoint():
        tui.warning("Existing checkpoint found for this configuration")
        tui.console.print()
        tui.console.print("  [cyan]1)[/cyan] Resume from checkpoint")
        tui.console.print("  [cyan]2)[/cyan] Clear checkpoint and start fresh")
        tui.console.print("  [cyan]3)[/cyan] Abort")
        tui.console.print()

        choice = click.prompt(
            "Choose an option",
            type=click.Choice(["1", "2", "3"]),
            default="1",
        )

        if choice == "1":
            # User wants to resume
            options.resume = True
        elif choice == "2":
            # Clear and start fresh
            engine.clear_checkpoint()
            tui.info("Checkpoint cleared, starting fresh generation")
        else:
            # Abort
            tui.info("Aborted")
            sys.exit(0)

    # Handle resume from checkpoint
    if options.resume:
        if engine.load_checkpoint(retry_failed=options.retry_failed):
            samples_done = engine._flushed_samples_count
            failures_done = engine._flushed_failures_count
            ids_processed = len(engine._processed_ids)
            retry_msg = " (retrying failed samples)" if options.retry_failed else ""

            # Update TUI status panel with checkpoint progress
            get_dataset_tui().set_checkpoint_resume_status(samples_done, failures_done)

            # Log resume info including failures
            if failures_done > 0:
                tui.info(
                    f"Resuming from checkpoint: {samples_done} samples, "
                    f"{failures_done} failed, {ids_processed} IDs processed{retry_msg}"
                )
            else:
                tui.info(
                    f"Resuming from checkpoint: {samples_done} samples, "
                    f"{ids_processed} IDs processed{retry_msg}"
                )
        else:
            tui.info("No checkpoint found, starting fresh generation")

    # Set up graceful Ctrl+C handling for checkpoint-based stop
    interrupt_count = 0

    def handle_sigint(_signum, _frame):
        nonlocal interrupt_count
        interrupt_count += 1

        if interrupt_count == 1:
            engine.stop_requested = True
            tui.warning("Stopping after current checkpoint... (Ctrl+C again to force quit)")
            dataset_tui = get_dataset_tui()
            dataset_tui.log_event("⚠ Graceful stop requested")
            dataset_tui.status_stop_requested()
        else:
            tui.error("Force quit!")
            sys.exit(1)

    original_handler = signal.signal(signal.SIGINT, handle_sigint)
    try:
        dataset = create_dataset(
            engine=engine,
            topic_model=topic_model,
            config=preparation.config,
            num_samples=preparation.num_samples,
            batch_size=preparation.batch_size,
            include_system_message=options.include_system_message,
            provider=options.provider,
            model=options.model,
            generation_overrides=preparation.generation_overrides,
            debug=options.debug,
        )
    finally:
        signal.signal(signal.SIGINT, original_handler)

    # If gracefully stopped, don't save partial dataset or clean up checkpoints
    if engine.stop_requested:
        return

    output_config = preparation.config.get_output_config()
    output_save_path = options.output_save_as or output_config["save_as"]
    save_dataset(dataset, output_save_path, preparation.config, engine=engine)

    # Clean up checkpoint files after successful completion
    if generation_params.get("checkpoint_interval") is not None:
        try:
            engine.clear_checkpoint()
            tui.info("Checkpoint files cleaned up after successful generation")
        except OSError as e:
            tui.warning(f"Failed to clean up checkpoint files: {e}")

    trace(
        "dataset_generated",
        {"samples": len(dataset)},
    )

    # Cloud upload (experimental feature)
    _trigger_cloud_upload(
        preparation=preparation,
        options=options,
        dataset_path=output_save_path,
    )


@cli.command()
@click.argument("config_file", type=click.Path(exists=True), required=False)
@click.option(
    "--output-system-prompt",
    help="System prompt for final dataset output (if include_system_message is true)",
)
@click.option("--topic-prompt", help="Starting topic/seed for topic generation")
@click.option("--topics-system-prompt", help="System prompt for topic generation")
@click.option("--generation-system-prompt", help="System prompt for dataset content generation")
@click.option("--topics-save-as", help="Save path for the generated topics")
@click.option(
    "--topics-load",
    type=click.Path(exists=True),
    help="Path to existing topics file (JSONL for tree, JSON for graph)",
)
@click.option("--output-save-as", help="Save path for the dataset")
@click.option("--provider", help="LLM provider (e.g., ollama, openai)")
@click.option("--model", help="Model name (e.g., qwen3:8b, gpt-4o)")
@click.option("--temperature", type=float, help="Temperature setting")
@click.option("--degree", type=int, help="Degree (branching factor)")
@click.option("--depth", type=int, help="Depth setting")
@click.option(
    "--num-samples",
    type=str,
    help="Number of samples: integer, 'auto' (all topics), or percentage like '50%'",
)
@click.option("--batch-size", type=int, help="Batch size")
@click.option("--base-url", help="Base URL for LLM provider API endpoint")
@click.option(
    "--include-system-message/--no-system-message",
    default=None,
    help="Include system message in dataset output (default: true)",
)
@click.option(
    "--mode",
    type=click.Choice(["tree", "graph"]),
    default="tree",
    help="Topic generation mode (default: tree)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode for detailed error output",
)
@click.option(
    "--tui",
    type=click.Choice(["rich", "simple"]),
    default="rich",
    show_default=True,
    help="TUI mode: rich (two-pane with preview) or simple (headless-friendly)",
)
@click.option(
    "--topic-only",
    is_flag=True,
    help="Only create topic assets, no dataset",
)
@click.option(
    "--conversation-type",
    type=click.Choice(["basic", "cot"]),
    help="Base conversation type: basic (simple chat), cot (with reasoning)",
)
@click.option(
    "--reasoning-style",
    type=click.Choice(["freetext", "agent"]),
    help="Reasoning style for cot: freetext (natural language) or agent (structured for tool-calling)",
)
@click.option(
    "--agent-mode",
    type=click.Choice(["single_turn", "multi_turn"]),
    help="[Deprecated] Agent mode is now implicit when tools are configured. 'multi_turn' is no longer supported.",
)
@click.option(
    "--cloud-upload",
    type=click.Choice(["all", "dataset", "graph", "none"], case_sensitive=False),
    default=None,
    help="Upload to DeepFabric Cloud (experimental): all, dataset, graph, or none. "
    "Enables headless mode for CI. Requires DEEPFABRIC_API_KEY or prior auth.",
)
@click.option(
    "--checkpoint-interval",
    type=int,
    help="Save checkpoint every N samples. Enables resumable generation.",
)
@click.option(
    "--checkpoint-path",
    type=click.Path(),
    help="Override checkpoint directory (default: XDG data dir)",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from existing checkpoint if available",
)
@click.option(
    "--retry-failed",
    is_flag=True,
    help="When resuming, retry previously failed samples",
)
def generate(  # noqa: PLR0913
    config_file: str | None,
    output_system_prompt: str | None = None,
    topic_prompt: str | None = None,
    topics_system_prompt: str | None = None,
    generation_system_prompt: str | None = None,
    topics_save_as: str | None = None,
    topics_load: str | None = None,
    output_save_as: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    degree: int | None = None,
    depth: int | None = None,
    num_samples: str | None = None,
    batch_size: int | None = None,
    base_url: str | None = None,
    include_system_message: bool | None = None,
    mode: Literal["tree", "graph"] = "tree",
    debug: bool = False,
    topic_only: bool = False,
    conversation_type: Literal["basic", "cot"] | None = None,
    reasoning_style: Literal["freetext", "agent"] | None = None,
    agent_mode: Literal["single_turn", "multi_turn"] | None = None,
    cloud_upload: Literal["all", "dataset", "graph", "none"] | None = None,
    tui: Literal["rich", "simple"] = "rich",
    checkpoint_interval: int | None = None,
    checkpoint_path: str | None = None,
    resume: bool = False,
    retry_failed: bool = False,
) -> None:
    """Generate training data from a YAML configuration file or CLI parameters."""
    # Handle deprecated --agent-mode flag
    if agent_mode == "multi_turn":
        click.echo(
            "Error: --agent-mode multi_turn is deprecated and no longer supported. "
            "Omit --agent-mode and the default supported agent mode will be used.",
            err=True,
        )
        sys.exit(1)
    elif agent_mode == "single_turn":
        click.echo(
            "Note: --agent-mode single_turn is deprecated. "
            "Single-turn agent mode is now implicit when tools are configured."
        )

    set_trace_debug(debug)
    trace(
        "cli_generate",
        {
            "mode": mode,
            "has_config": config_file is not None,
            "provider": provider,
            "model": model,
        },
    )

    try:
        # Parse num_samples from CLI string (could be int, "auto", or "50%")
        parsed_num_samples = parse_num_samples(num_samples)

        options = GenerateOptions(
            config_file=config_file,
            output_system_prompt=output_system_prompt,
            topic_prompt=topic_prompt,
            topics_system_prompt=topics_system_prompt,
            generation_system_prompt=generation_system_prompt,
            topics_save_as=topics_save_as,
            topics_load=topics_load,
            output_save_as=output_save_as,
            provider=provider,
            model=model,
            temperature=temperature,
            degree=degree,
            depth=depth,
            num_samples=parsed_num_samples,
            batch_size=batch_size,
            base_url=base_url,
            include_system_message=include_system_message,
            mode=mode,
            debug=debug,
            topic_only=topic_only,
            conversation_type=conversation_type,
            reasoning_style=reasoning_style,
            agent_mode=agent_mode,
            cloud_upload=cloud_upload,
            tui=tui,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=checkpoint_path,
            resume=resume,
            retry_failed=retry_failed,
        )
    except (PydanticValidationError, ValueError) as error:
        handle_error(click.get_current_context(), ConfigurationError(str(error)))
        return

    try:
        # Configure TUI before any output
        configure_tui(options.tui)
        tui = get_tui()  # type: ignore

        # Show initialization header
        tui.info("Initializing DeepFabric...")  # type: ignore
        print()

        preparation = _load_and_prepare_generation_context(options, skip_path_validation=topic_only)

        # Compute checkpoint directory once for consistent use throughout generation
        # Use config file for hash, fallback to output path for config-less runs
        path_source = options.config_file or options.output_save_as or preparation.config.output.save_as
        checkpoint_dir = options.checkpoint_path or get_checkpoint_dir(path_source)

        # Auto-infer topics-load when resuming from checkpoint
        if options.resume and not options.topics_load:
            output_path = options.output_save_as or preparation.config.output.save_as

            inferred_topics_path = _get_checkpoint_topics_path(checkpoint_dir, output_path)
            if inferred_topics_path:
                if Path(inferred_topics_path).exists():
                    tui.info(f"Resume: auto-loading topics from {inferred_topics_path}")
                    options.topics_load = inferred_topics_path
                else:
                    tui.warning(
                        f"Checkpoint references topics at {inferred_topics_path} but file not found. "
                        "Topic graph will be regenerated."
                    )

        topic_model = _initialize_topic_model(
            preparation=preparation,
            options=options,
        )

        if topic_only:
            # Cloud upload for topic-only mode (graph only, no dataset)
            _trigger_cloud_upload(
                preparation=preparation,
                options=options,
                dataset_path=None,
            )
            return

        _run_generation(
            preparation=preparation,
            topic_model=topic_model,
            options=options,
            checkpoint_dir=checkpoint_dir,
        )

    except ConfigurationError as e:
        handle_error(click.get_current_context(), e)
    except Exception as e:
        tui = get_tui()  # type: ignore
        tui.error(f"Unexpected error: {str(e)}")  # type: ignore
        sys.exit(1)


@cli.command("upload-hf")
@click.argument("dataset_file", type=click.Path(exists=True))
@click.option(
    "--repo",
    required=True,
    help="Hugging Face repository (e.g., username/dataset-name)",
)
@click.option(
    "--token",
    help="Hugging Face API token (can also be set via HF_TOKEN env var)",
)
@click.option(
    "--tags",
    multiple=True,
    help="Tags for the dataset (can be specified multiple times)",
)
def upload_hf(
    dataset_file: str,
    repo: str,
    token: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Upload a dataset to Hugging Face Hub."""
    trace("cli_upload_hf", {"has_tags": len(tags) > 0 if tags else False})

    try:
        # Get token from CLI arg or env var
        token = token or os.getenv("HF_TOKEN")
        if not token:
            handle_error(
                click.get_current_context(),
                ValueError("Hugging Face token not provided. Set via --token or HF_TOKEN env var."),
            )

        # Lazy import to avoid slow startup when not using HF features
        from .hf_hub import HFUploader  # noqa: PLC0415

        uploader = HFUploader(token)
        result = uploader.push_to_hub(str(repo), dataset_file, tags=list(tags) if tags else [])

        tui = get_tui()
        if result["status"] == "success":
            tui.success(result["message"])
        else:
            tui.error(result["message"])
            sys.exit(1)

    except Exception as e:
        tui = get_tui()
        tui.error(f"Error uploading to Hugging Face Hub: {str(e)}")
        sys.exit(1)


@cli.command("upload-kaggle")
@click.argument("dataset_file", type=click.Path(exists=True))
@click.option(
    "--handle",
    required=True,
    help="Kaggle dataset handle (e.g., username/dataset-name)",
)
@click.option(
    "--username",
    help="Kaggle username (can also be set via KAGGLE_USERNAME env var)",
)
@click.option(
    "--key",
    help="Kaggle API key (can also be set via KAGGLE_KEY env var)",
)
@click.option(
    "--tags",
    multiple=True,
    help="Tags for the dataset (can be specified multiple times)",
)
@click.option(
    "--version-notes",
    help="Version notes for the dataset update",
)
@click.option(
    "--description",
    help="Description for the dataset",
)
def upload_kaggle(
    dataset_file: str,
    handle: str,
    username: str | None = None,
    key: str | None = None,
    tags: list[str] | None = None,
    version_notes: str | None = None,
    description: str | None = None,
) -> None:
    """Upload a dataset to Kaggle."""
    trace("cli_upload_kaggle", {"has_tags": len(tags) > 0 if tags else False})

    try:
        # Get credentials from CLI args or env vars
        username = username or os.getenv("KAGGLE_USERNAME")
        key = key or os.getenv("KAGGLE_KEY")

        if not username or not key:
            handle_error(
                click.get_current_context(),
                ValueError(
                    "Kaggle credentials not provided. "
                    "Set via --username/--key or KAGGLE_USERNAME/KAGGLE_KEY env vars."
                ),
            )

        # Lazy import to avoid slow startup when not using Kaggle features
        from .kaggle_hub import KaggleUploader  # noqa: PLC0415

        uploader = KaggleUploader(username, key)
        result = uploader.push_to_hub(
            str(handle),
            dataset_file,
            tags=list(tags) if tags else [],
            version_notes=version_notes,
            description=description,
        )

        tui = get_tui()
        if result["status"] == "success":
            tui.success(result["message"])
        else:
            tui.error(result["message"])
            sys.exit(1)

    except Exception as e:
        tui = get_tui()
        tui.error(f"Error uploading to Kaggle: {str(e)}")
        sys.exit(1)


# DeepFabric Cloud upload command group
@click.group()
def upload() -> None:
    """Upload datasets and graphs to DeepFabric Cloud."""
    pass


def _upload_to_cloud(
    file: str,
    resource_type: Literal["dataset", "graph"],
    handle: str | None,
    name: str | None,
    description: str | None,
    tags: list[str] | None,
    config_file: str | None,
) -> None:
    """Shared helper for uploading datasets and graphs to DeepFabric Cloud.

    Args:
        file: Path to the file to upload
        resource_type: Either "dataset" or "graph"
        handle: Resource handle (e.g., username/resource-name)
        name: Display name for the resource
        description: Description for the resource
        tags: Tags for the resource (only used for datasets)
        config_file: Path to config file with upload settings
    """
    # Lazy imports to avoid slow startup
    import httpx  # noqa: PLC0415

    from .auth import DEFAULT_API_URL  # noqa: PLC0415
    from .cloud_upload import (  # noqa: PLC0415
        _get_user_friendly_error,
        build_urls,
        derive_frontend_url,
        derive_name_and_slug,
        ensure_authenticated,
        get_current_user,
        upload_dataset,
        upload_topic_graph,
    )

    tui = get_tui()
    config_key = resource_type  # "dataset" or "graph"
    url_resource_type = "datasets" if resource_type == "dataset" else "graphs"

    # Load handle from config if not provided via CLI
    final_handle = handle
    final_description = description or ""
    final_tags = list(tags) if tags else []

    if config_file:
        config = DeepFabricConfig.from_yaml(config_file)
        cloud_config = config.get_deepfabric_cloud_config()
        if not final_handle:
            final_handle = cloud_config.get(config_key)
        if not description and cloud_config.get("description"):
            final_description = cloud_config.get("description", "")
        if resource_type == "dataset" and not tags and cloud_config.get("tags"):
            final_tags = cloud_config.get("tags", [])

    # Ensure authenticated
    if not ensure_authenticated(DEFAULT_API_URL, headless=False):
        tui.error("Authentication required. Run 'deepfabric auth login' first.")
        sys.exit(1)

    # Derive name and slug from filename if not provided
    default_name, default_slug = derive_name_and_slug(file)
    final_name = name or default_name

    # Use slug from handle if provided, otherwise use derived slug
    if final_handle and "/" in final_handle:
        final_slug = final_handle.split("/")[-1]
    else:
        final_slug = final_handle or default_slug

    tui.info(f"Uploading {resource_type} '{final_name}'...")

    try:
        # Call the appropriate upload function
        if resource_type == "dataset":
            result = upload_dataset(
                dataset_path=file,
                name=final_name,
                slug=final_slug,
                description=final_description,
                tags=final_tags,
                api_url=DEFAULT_API_URL,
            )
            resource_id = result.get("dataset_id") or result.get("id")
        else:
            result = upload_topic_graph(
                graph_path=file,
                name=final_name,
                description=final_description,
                slug=final_slug,
                api_url=DEFAULT_API_URL,
            )
            resource_id = result.get("id")

        # Display success message
        tui.success(f"{resource_type.capitalize()} '{final_name}' uploaded successfully!")

        # Display URL if available
        if resource_id:
            user_info = get_current_user(DEFAULT_API_URL)
            username = user_info.get("username") if user_info else None
            frontend_url = derive_frontend_url(DEFAULT_API_URL)
            public_url, internal_url = build_urls(
                url_resource_type, resource_id, final_slug, username, frontend_url
            )
            tui.info(f"View at: {public_url or internal_url}")

    except httpx.HTTPStatusError as e:
        error_msg = _get_user_friendly_error(e)
        if "already exists" in error_msg.lower():
            tui.error(
                f"A {resource_type} with slug '{final_slug}' already exists. "
                "Use a different --handle value."
            )
        else:
            tui.error(f"Error uploading {resource_type}: {error_msg}")
        sys.exit(1)
    except Exception as e:
        tui.error(f"Error uploading {resource_type}: {str(e)}")
        sys.exit(1)


@upload.command("dataset")
@click.argument("file", type=click.Path(exists=True))
@click.option("--handle", help="Dataset handle (e.g., username/dataset-name)")
@click.option("--name", help="Display name for the dataset")
@click.option("--description", help="Description for the dataset")
@click.option(
    "--tags", multiple=True, help="Tags for the dataset (can be specified multiple times)"
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    help="Config file with upload settings",
)
def upload_dataset_cmd(
    file: str,
    handle: str | None,
    name: str | None,
    description: str | None,
    tags: tuple[str, ...],
    config_file: str | None,
) -> None:
    """Upload a dataset to DeepFabric Cloud.

    FILE is the path to the JSONL dataset file.

    Examples:

        deepfabric upload dataset my-dataset.jsonl --handle myuser/my-dataset

        deepfabric upload dataset output.jsonl --config config.yaml
    """
    trace(
        "cli_upload_dataset",
        {"has_config": config_file is not None, "has_handle": handle is not None},
    )
    _upload_to_cloud(
        file=file,
        resource_type="dataset",
        handle=handle,
        name=name,
        description=description,
        tags=list(tags) if tags else None,
        config_file=config_file,
    )


@upload.command("graph")
@click.argument("file", type=click.Path(exists=True))
@click.option("--handle", help="Graph handle (e.g., username/graph-name)")
@click.option("--name", help="Display name for the graph")
@click.option("--description", help="Description for the graph")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    help="Config file with upload settings",
)
def upload_graph_cmd(
    file: str,
    handle: str | None,
    name: str | None,
    description: str | None,
    config_file: str | None,
) -> None:
    """Upload a topic graph to DeepFabric Cloud.

    FILE is the path to the JSON graph file.

    Examples:

        deepfabric upload graph topic_graph.json --handle myuser/my-graph

        deepfabric upload graph graph.json --config config.yaml
    """
    trace(
        "cli_upload_graph",
        {"has_config": config_file is not None, "has_handle": handle is not None},
    )
    _upload_to_cloud(
        file=file,
        resource_type="graph",
        handle=handle,
        name=name,
        description=description,
        tags=None,
        config_file=config_file,
    )


@cli.command()
@click.argument("graph_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output SVG file path",
)
def visualize(graph_file: str, output: str) -> None:
    """Visualize a topic graph as an SVG file."""
    try:
        # Load the graph
        with open(graph_file) as f:
            import json  # noqa: PLC0415

            graph_data = json.load(f)

        # Create a minimal Graph object for visualization
        # We need to get the args from somewhere - for now, use defaults
        from .constants import (  # noqa: PLC0415
            TOPIC_GRAPH_DEFAULT_DEGREE,
            TOPIC_GRAPH_DEFAULT_DEPTH,
        )

        # Create parameters for Graph instantiation
        graph_params = {
            "topic_prompt": "placeholder",  # Not needed for visualization
            "model_name": "placeholder/model",  # Not needed for visualization
            "degree": graph_data.get("degree", TOPIC_GRAPH_DEFAULT_DEGREE),
            "depth": graph_data.get("depth", TOPIC_GRAPH_DEFAULT_DEPTH),
            "temperature": 0.7,  # Default, not used for visualization
        }

        # Use the Graph.from_json method to properly load the graph structure
        import tempfile  # noqa: PLC0415

        # Create a temporary file with the graph data and use from_json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            json.dump(graph_data, tmp_file)
            temp_path = tmp_file.name

        try:
            graph = Graph.from_json(temp_path, graph_params)
        finally:
            import os  # noqa: PLC0415

            os.unlink(temp_path)

        # Visualize the graph
        graph.visualize(output)
        tui = get_tui()
        tui.success(f"Graph visualization saved to: {output}.svg")

    except Exception as e:
        tui = get_tui()
        tui.error(f"Error visualizing graph: {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--check-api/--no-check-api",
    default=True,
    help="Validate API keys by making test calls (default: enabled)",
)
def validate(config_file: str, check_api: bool) -> None:  # noqa: PLR0912
    """Validate a DeepFabric configuration file."""
    try:
        # Try to load the configuration
        config = DeepFabricConfig.from_yaml(config_file)

        # Check required sections
        errors = []
        warnings = []

        # Check for system prompt
        if not config.generation.system_prompt:
            warnings.append("No generation.system_prompt defined")

        # Check topics configuration
        if not config.topics.prompt:
            errors.append("topics.prompt is required")

        # Check output configuration
        if not config.output.save_as:
            warnings.append("No output.save_as path defined for dataset")

        # Report results
        tui = get_tui()
        if errors:
            tui.error("Configuration validation failed:")
            for error in errors:
                tui.console.print(f"  - {error}", style="red")
            sys.exit(1)

        if warnings:
            tui.console.print("Warnings:", style="yellow bold")
            for warning in warnings:
                tui.warning(warning)
            tui.console.print()

        # Print configuration summary
        tui.console.print("Configuration Summary:", style="cyan bold")

        # Topics summary with estimated paths
        depth = config.topics.depth
        degree = config.topics.degree
        # Estimated paths = degree^depth (each level branches by degree)
        estimated_paths = degree**depth
        tui.info(
            f"Topics: mode={config.topics.mode}, depth={depth}, degree={degree}, "
            f"estimated_paths={estimated_paths} ({degree}^{depth})"
        )

        # Output summary with step size and checkpoint info
        num_samples = config.output.num_samples
        batch_size = config.output.batch_size
        # Calculate num_steps - handle 'auto' and percentage strings
        if isinstance(num_samples, int):
            num_steps = math.ceil(num_samples / batch_size)
            output_info = f"Output: num_samples={num_samples}, batch_size={batch_size}, num_steps={num_steps}"
        else:
            # For 'auto' or percentage, we can't compute steps without topic count
            output_info = f"Output: num_samples={num_samples}, batch_size={batch_size}"

        # Add checkpoint info if enabled
        if config.output.checkpoint:
            checkpoint = config.output.checkpoint
            output_info += f", checkpoint_interval={checkpoint.interval}"
        tui.info(output_info)

        if config.huggingface:
            hf_config = config.get_huggingface_config()
            tui.info(f"Hugging Face: repo={hf_config.get('repository', 'not set')}")

        if config.kaggle:
            kaggle_config = config.get_kaggle_config()
            tui.info(f"Kaggle: handle={kaggle_config.get('handle', 'not set')}")

        # Check path writability
        tui.console.print("\nPath Writability:", style="cyan bold")
        path_errors = []

        # Check topics.save_as if configured
        if config.topics.save_as:
            is_writable, error_msg = check_path_writable(config.topics.save_as, "topics.save_as")
            if is_writable:
                tui.success(f"topics.save_as: {config.topics.save_as}")
            else:
                path_errors.append(error_msg)
                tui.error(f"topics.save_as: {error_msg}")

        # Check output.save_as
        if config.output.save_as:
            is_writable, error_msg = check_path_writable(config.output.save_as, "output.save_as")
            if is_writable:
                tui.success(f"output.save_as: {config.output.save_as}")
            else:
                path_errors.append(error_msg)
                tui.error(f"output.save_as: {error_msg}")

        # Check checkpoint directory if enabled
        if config.output.checkpoint:
            checkpoint_path = config.output.checkpoint.path or get_checkpoint_dir(config_file)
            is_writable, error_msg = check_dir_writable(checkpoint_path, "checkpoint directory")
            if is_writable:
                tui.success(f"checkpoint directory: {checkpoint_path}")
            else:
                path_errors.append(error_msg)
                tui.error(f"checkpoint directory: {error_msg}")

        if path_errors:
            tui.console.print()
            tui.error("Some paths are not writable. Fix permissions or choose different paths.")
            sys.exit(1)

        # Validate API keys if requested
        if check_api:
            tui.console.print("\nAPI Keys:", style="cyan bold")
            try:
                _validate_api_keys(config)
            except ConfigurationError as e:
                tui.error(str(e))
                sys.exit(1)
        else:
            tui.console.print("\nSkipping API key validation (use --check-api to enable)")

        # Final success message
        tui.console.print()
        tui.success("Configuration is valid")

    except FileNotFoundError:
        handle_error(
            click.get_current_context(),
            ValueError(f"Config file not found: {config_file}"),
        )
    except yaml.YAMLError as e:
        handle_error(
            click.get_current_context(),
            ValueError(f"Invalid YAML in config file: {str(e)}"),
        )
    except Exception as e:
        handle_error(
            click.get_current_context(),
            ValueError(f"Error validating config file: {str(e)}"),
        )


@cli.command()
def info() -> None:
    """Show DeepFabric version and configuration information."""
    try:
        import importlib.metadata  # noqa: PLC0415

        # Get version
        try:
            version = importlib.metadata.version("deepfabric")
        except importlib.metadata.PackageNotFoundError:
            version = "development"

        tui = get_tui()
        header = tui.create_header(
            f"DeepFabric v{version}",
            "Large Scale Topic based Synthetic Data Generation",
        )
        tui.console.print(header)

        tui.console.print("\nAvailable Commands:", style="cyan bold")
        commands = [
            ("generate", "Generate training data from configuration"),
            ("validate", "Validate a configuration file"),
            ("visualize", "Create SVG visualization of a topic graph"),
            ("upload-hf", "Upload dataset to Hugging Face Hub"),
            ("upload-kaggle", "Upload dataset to Kaggle"),
            ("evaluate", "Evaluate a fine-tuned model on tool-calling tasks"),
            ("import-tools", "Import tool definitions from external sources"),
            ("info", "Show this information"),
        ]
        for cmd, desc in commands:
            tui.console.print(f"  [cyan]{cmd}[/cyan] - {desc}")

        tui.console.print("\nEnvironment Variables:", style="cyan bold")
        env_vars = [
            ("OPENAI_API_KEY", "OpenAI API key"),
            ("ANTHROPIC_API_KEY", "Anthropic API key"),
            ("HF_TOKEN", "Hugging Face API token"),
        ]
        for var, desc in env_vars:
            tui.console.print(f"  [yellow]{var}[/yellow] - {desc}")

        tui.console.print(
            "\nFor more information, visit: [link]https://github.com/always-further/deepfabric[/link]"
        )

    except Exception as e:
        tui = get_tui()
        tui.error(f"Error getting info: {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=click.Path())
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Path to save evaluation results (JSON)",
)
@click.option(
    "--adapter-path",
    type=click.Path(),
    help="Path to PEFT/LoRA adapter (for adapter-based fine-tuning)",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help="Batch size for evaluation",
)
@click.option(
    "--max-samples",
    type=int,
    help="Maximum number of samples to evaluate (default: all)",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature",
)
@click.option(
    "--max-tokens",
    type=int,
    default=2048,
    help="Maximum tokens to generate",
)
@click.option(
    "--top-p",
    type=float,
    default=0.9,
    help="Nucleus sampling top-p",
)
@click.option(
    "--backend",
    type=click.Choice(["transformers", "ollama"]),
    default="transformers",
    help="Inference backend to use",
)
@click.option(
    "--device",
    type=str,
    help="Device to use (cuda, cpu, mps, etc.) - only for transformers backend",
)
@click.option(
    "--no-save-predictions",
    is_flag=True,
    help="Don't save individual predictions to output file",
)
def evaluate(
    model_path: str,
    dataset_path: str,
    output: str | None,
    adapter_path: str | None,
    batch_size: int,
    max_samples: int | None,
    temperature: float,
    max_tokens: int,
    top_p: float,
    backend: str,
    device: str | None,
    no_save_predictions: bool,
):
    """Evaluate a fine-tuned model on tool-calling tasks.

    MODEL_PATH: Path to base model or fine-tuned model (local directory or HuggingFace Hub ID)

    DATASET_PATH: Path to evaluation dataset (JSONL format)

    Typical workflow:

        # Full fine-tuning: evaluate checkpoint
        deepfabric evaluate ./checkpoints/final ./eval.jsonl --output results.json

        # LoRA/PEFT: evaluate adapter on base model
        deepfabric evaluate unsloth/Qwen3-4B-Instruct ./eval.jsonl \\
            --adapter-path ./lora_model \\
            --output results.json

        # Quick evaluation during development
        deepfabric evaluate ./my-model ./eval.jsonl --max-samples 50

        # Evaluate HuggingFace model
        deepfabric evaluate username/model-name ./eval.jsonl \\
            --temperature 0.5 \\
            --device cuda
    """
    tui = get_tui()

    try:
        from typing import Literal, cast  # noqa: PLC0415

        from .evaluation import EvaluatorConfig, InferenceConfig  # noqa: PLC0415
        from .evaluation.evaluator import Evaluator  # noqa: PLC0415

        # Create inference configuration
        inference_config = InferenceConfig(
            model=model_path,
            adapter_path=adapter_path,
            backend=cast(Literal["transformers", "ollama"], backend),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            device=device,
            batch_size=batch_size,
        )

        # Create evaluator configuration
        evaluator_config = EvaluatorConfig(
            dataset_path=dataset_path,
            output_path=output,
            inference_config=inference_config,
            batch_size=batch_size,
            max_samples=max_samples,
            save_predictions=not no_save_predictions,
            metric_weights={
                "accuracy": 1.0,
                "exact_match": 1.0,
                "f1_score": 0.5,
            },
            evaluators=["tool_calling"],
            reporters=["console", "json"] if output else ["console"],
            cloud_api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Display configuration
        tui.console.print("\n[bold]Evaluation Configuration:[/bold]")
        tui.console.print(f"  Model: {model_path}")
        tui.console.print(f"  Backend: {backend}")
        if adapter_path:
            tui.console.print(f"  Adapter: {adapter_path}")
        tui.console.print(f"  Dataset: {dataset_path}")
        if output:
            tui.console.print(f"  Output: {output}")
        tui.console.print(f"  Batch size: {batch_size}")
        if max_samples:
            tui.console.print(f"  Max samples: {max_samples}")
        tui.console.print(f"  Temperature: {temperature}")
        tui.console.print(f"  Max tokens: {max_tokens}")
        if device and backend == "transformers":
            tui.console.print(f"  Device: {device}")
        tui.console.print()

        # Create evaluator and run evaluation
        tui.console.print("[bold blue]Loading model...[/bold blue]")
        tui.console.print(
            "[dim]This may take several minutes for large models (downloading + loading into memory)[/dim]"
        )
        evaluator = Evaluator(evaluator_config)
        tui.console.print("[green]Model loaded successfully![/green]\n")

        # Track evaluation start
        trace(
            "evaluation_started",
            {
                "model_path": model_path,
                "backend": backend,
                "has_adapter": adapter_path is not None,
                "dataset_path": dataset_path,
                "batch_size": batch_size,
                "max_samples": max_samples,
                "temperature": temperature,
            },
        )

        try:
            result = evaluator.evaluate()

            # Print summary
            evaluator.print_summary(result.metrics)

            if output:
                tui.console.print(f"\n[green]Full results saved to {output}[/green]")

        finally:
            evaluator.cleanup()

    except FileNotFoundError as e:
        trace(
            "evaluation_failed",
            {
                "model_path": model_path,
                "backend": backend,
                "dataset_path": dataset_path,
                "error_type": "FileNotFoundError",
            },
        )
        handle_error(click.get_current_context(), e)
    except ValueError as e:
        trace(
            "evaluation_failed",
            {
                "model_path": model_path,
                "backend": backend,
                "dataset_path": dataset_path,
                "error_type": "ValueError",
            },
        )
        handle_error(click.get_current_context(), e)
    except Exception as e:
        trace(
            "evaluation_failed",
            {
                "model_path": model_path,
                "backend": backend,
                "dataset_path": dataset_path,
                "error_type": type(e).__name__,
            },
        )
        handle_error(click.get_current_context(), e)


# Register the auth and upload command groups
# EXPERIMENTAL: Only enable cloud features if explicitly opted in
if get_bool_env("EXPERIMENTAL_DF"):
    cli.add_command(auth_group)
    cli.add_command(upload)


@cli.command("import-tools")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"]),
    required=True,
    help="MCP transport type: stdio (subprocess) or http (Streamable HTTP)",
)
@click.option(
    "--command",
    "-c",
    help="Shell command to launch MCP server (required for stdio transport)",
)
@click.option(
    "--endpoint",
    "-e",
    help="HTTP endpoint URL for MCP server (required for http transport)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (.json or .yaml). Optional if --spin is used.",
)
@click.option(
    "--spin",
    "-s",
    "spin_endpoint",
    help="Spin server URL to push tools to (e.g., http://localhost:3000)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["deepfabric", "openai"]),
    default="deepfabric",
    help="Output format: deepfabric (native) or openai (TRL compatible)",
)
@click.option(
    "--env",
    multiple=True,
    help="Environment variables for stdio (format: KEY=VALUE, can be repeated)",
)
@click.option(
    "--header",
    multiple=True,
    help="HTTP headers for authentication (format: KEY=VALUE, can be repeated)",
)
@click.option(
    "--timeout",
    type=float,
    default=30.0,
    help="Request timeout in seconds (default: 30)",
)
def import_tools(
    transport: str,
    command: str | None,
    endpoint: str | None,
    output: str | None,
    spin_endpoint: str | None,
    output_format: str,
    env: tuple[str, ...],
    header: tuple[str, ...],
    timeout: float,
) -> None:
    """Import tool definitions from an MCP (Model Context Protocol) server.

    This command connects to an MCP server, fetches available tools via the
    tools/list method, and either saves them to a file or pushes them to a
    Spin server (or both).

    Supports both MCP transport types:

    \b
    STDIO: Launches the MCP server as a subprocess
      deepfabric import-tools --transport stdio \\
        --command "npx -y @modelcontextprotocol/server-filesystem /tmp" \\
        --output tools.json

    \b
    HTTP: Connects to a running MCP server via HTTP
      deepfabric import-tools --transport http \\
        --endpoint "http://localhost:3000/mcp" \\
        --output tools.json

    \b
    Push directly to Spin server:
      deepfabric import-tools --transport stdio \\
        --command "npx -y figma-developer-mcp --stdio" \\
        --env "FIGMA_API_KEY=your-key" \\
        --spin http://localhost:3000

    \b
    Both save to file and push to Spin:
      deepfabric import-tools --transport stdio \\
        --command "your-mcp-server" \\
        --output tools.json \\
        --spin http://localhost:3000
    """
    tui = get_tui()

    # Validate that at least one output is specified
    if not output and not spin_endpoint:
        tui.error("At least one of --output or --spin is required")
        sys.exit(1)

    def parse_key_value_pairs(pairs: tuple[str, ...], pair_type: str) -> dict[str, str]:
        """Parse a tuple of 'KEY=VALUE' strings into a dictionary."""
        result: dict[str, str] = {}
        for p in pairs:
            if "=" not in p:
                tui.error(f"Invalid {pair_type} format: {p} (expected KEY=VALUE)")
                sys.exit(1)
            key, value = p.split("=", 1)
            result[key] = value
        return result

    env_dict = parse_key_value_pairs(env, "env")
    header_dict = parse_key_value_pairs(header, "header")

    # Validate transport-specific options
    if transport == "stdio" and not command:
        tui.error("--command is required for stdio transport")
        sys.exit(1)
    if transport == "http" and not endpoint:
        tui.error("--endpoint is required for http transport")
        sys.exit(1)

    try:
        # Lazy import to avoid slow startup
        from typing import Literal, cast  # noqa: PLC0415

        from .tools.mcp_client import (  # noqa: PLC0415
            fetch_and_push_to_spin,
            fetch_tools_from_mcp,
            save_tools_to_file,
        )

        tui.info(f"Connecting to MCP server via {transport}...")

        # If pushing to Spin, use the combined function
        if spin_endpoint:
            registry, spin_result = fetch_and_push_to_spin(
                transport=cast(Literal["stdio", "http"], transport),
                spin_endpoint=spin_endpoint,
                command=command,
                endpoint=endpoint,
                env=env_dict if env_dict else None,
                headers=header_dict if header_dict else None,
                timeout=timeout,
            )
            tui.success(f"Pushed {spin_result.loaded} tools to Spin server at {spin_endpoint}")
        else:
            # Just fetch without pushing to Spin
            registry = fetch_tools_from_mcp(
                transport=cast(Literal["stdio", "http"], transport),
                command=command,
                endpoint=endpoint,
                env=env_dict if env_dict else None,
                headers=header_dict if header_dict else None,
                timeout=timeout,
            )

        if not registry.tools:
            tui.warning("No tools found from MCP server")
            sys.exit(0)

        # Save to file if output path specified
        if output:
            save_tools_to_file(
                registry,
                output,
                output_format=cast(Literal["deepfabric", "openai"], output_format),
            )
            tui.success(f"Saved {len(registry.tools)} tools to {output}")

        # List the tools
        tui.console.print("\nImported tools:", style="cyan bold")
        for tool in registry.tools:
            param_count = len(tool.parameters)
            desc = tool.description[:60] if tool.description else "(no description)"
            tui.console.print(f"  - {tool.name} ({param_count} params): {desc}...")

    except Exception as e:
        tui.error(f"Failed to import tools: {str(e)}")
        sys.exit(1)


@cli.command("checkpoint-status")
@click.argument("config_file", type=click.Path(exists=True))
def checkpoint_status(config_file: str) -> None:
    """Show checkpoint status for a generation config.

    Displays the current state of any checkpoint files associated with
    the given configuration file, including progress, failures, and
    resume instructions.
    """
    tui = get_tui()

    try:
        config = DeepFabricConfig.from_yaml(config_file)
    except Exception as e:
        tui.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Get checkpoint configuration
    checkpoint_config = config.get_checkpoint_config()
    output_config = config.get_output_config()
    checkpoint_dir = checkpoint_config.get("path") or get_checkpoint_dir(config_file)
    save_as = output_config.get("save_as")

    if not save_as:
        tui.error("Config does not specify output.save_as - cannot determine checkpoint paths")
        sys.exit(1)

    # Derive checkpoint paths
    output_stem = Path(save_as).stem
    checkpoint_dir_path = Path(checkpoint_dir)
    metadata_path = checkpoint_dir_path / f"{output_stem}{CHECKPOINT_METADATA_SUFFIX}"
    samples_path = checkpoint_dir_path / f"{output_stem}{CHECKPOINT_SAMPLES_SUFFIX}"
    failures_path = checkpoint_dir_path / f"{output_stem}{CHECKPOINT_FAILURES_SUFFIX}"

    # Check if checkpoint exists
    if not metadata_path.exists():
        tui.info(f"No checkpoint found at: {metadata_path}")
        tui.info("\nTo enable checkpointing, run:")
        tui.info(f"  deepfabric generate {config_file} --checkpoint-interval 10")
        return

    # Load and display checkpoint metadata
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        tui.error(f"Failed to read checkpoint metadata: {e}")
        sys.exit(1)

    # Count samples in checkpoint file
    checkpoint_sample_count = 0
    if samples_path.exists():
        with open(samples_path) as f:
            checkpoint_sample_count = sum(1 for line in f if line.strip())

    # Count failures
    checkpoint_failures = 0
    failure_details = []
    if failures_path.exists():
        with open(failures_path) as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped:
                    checkpoint_failures += 1
                    try:
                        failure = json.loads(stripped)
                        failure_details.append(failure)
                    except json.JSONDecodeError:
                        pass

    # Get target samples from config
    # num_samples is the total target, not per-batch. It can be int, "auto", or percentage like "50%"
    target_samples = output_config.get("num_samples", 0)
    # "auto" or percentage strings can't be resolved without topic model
    total_target = 0 if isinstance(target_samples, str) else (target_samples or 0)

    # Display status
    tui.console.print()
    tui.console.print(f"[bold]Checkpoint Status:[/bold] {metadata_path}")
    tui.console.print()

    # Progress
    progress_pct = (checkpoint_sample_count / total_target * 100) if total_target > 0 else 0
    tui.console.print(
        f"  [cyan]Progress:[/cyan]     {checkpoint_sample_count}/{total_target} samples ({progress_pct:.1f}%)"
    )
    tui.console.print(f"  [cyan]Failed:[/cyan]       {checkpoint_failures} samples")

    # Paths processed
    processed_ids = metadata.get("processed_ids", [])
    tui.console.print(f"  [cyan]Paths done:[/cyan]   {len(processed_ids)}")

    # Config info
    tui.console.print()
    tui.console.print(f"  [dim]Provider:[/dim]      {metadata.get('provider', 'unknown')}")
    tui.console.print(f"  [dim]Model:[/dim]         {metadata.get('model_name', 'unknown')}")
    tui.console.print(f"  [dim]Conv type:[/dim]     {metadata.get('conversation_type', 'unknown')}")
    if metadata.get("reasoning_style"):
        tui.console.print(f"  [dim]Reasoning:[/dim]     {metadata.get('reasoning_style')}")
    tui.console.print(f"  [dim]Last saved:[/dim]    {metadata.get('created_at', 'unknown')}")

    # Show topics file path if available
    topics_path = metadata.get("topics_file") or metadata.get("topics_save_as")
    if topics_path:
        topics_exists = Path(topics_path).exists()
        status = "[green]exists[/green]" if topics_exists else "[red]missing[/red]"
        tui.console.print(f"  [dim]Topics file:[/dim]  {topics_path} ({status})")

    # Show failed topics if any
    max_failures_to_show = 5
    max_error_length = 60
    if failure_details:
        tui.console.print()
        tui.console.print("[yellow]Failed Topics:[/yellow]")
        for failure in failure_details[:max_failures_to_show]:
            error_msg = failure.get("error", "Unknown error")
            # Truncate long error messages
            if len(error_msg) > max_error_length:
                error_msg = error_msg[: max_error_length - 3] + "..."
            tui.console.print(f"  - {error_msg}")
        if len(failure_details) > max_failures_to_show:
            remaining = len(failure_details) - max_failures_to_show
            tui.console.print(f"  ... and {remaining} more failures")

    # Resume instructions
    tui.console.print()
    checkpoint_interval_arg = metadata.get("checkpoint_interval", 10)
    tui.console.print("[green]Resume with:[/green]")
    tui.console.print(
        f"  deepfabric generate {config_file} --checkpoint-interval {checkpoint_interval_arg} --resume"
    )
    if metadata.get("total_failures", 0) > 0:
        tui.console.print("[green]Retry failed:[/green]")
        tui.console.print(
            f"  deepfabric generate {config_file} --checkpoint-interval {checkpoint_interval_arg} --resume --retry-failed"
        )


if __name__ == "__main__":
    cli()
