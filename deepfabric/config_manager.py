import yaml

from pydantic import ValidationError

from .config import DeepFabricConfig
from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    ENGINE_DEFAULT_BATCH_SIZE,
    ENGINE_DEFAULT_NUM_EXAMPLES,
    ENGINE_DEFAULT_TEMPERATURE,
    TOPIC_GRAPH_DEFAULT_DEGREE,
    TOPIC_GRAPH_DEFAULT_DEPTH,
    TOPIC_GRAPH_DEFAULT_TEMPERATURE,
    TOPIC_TREE_DEFAULT_DEGREE,
    TOPIC_TREE_DEFAULT_DEPTH,
    TOPIC_TREE_DEFAULT_TEMPERATURE,
)
from .exceptions import ConfigurationError
from .tui import get_tui


def load_config(  # noqa: PLR0913
    config_file: str | None,
    topic_prompt: str | None = None,
    topics_system_prompt: str | None = None,
    generation_system_prompt: str | None = None,
    output_system_prompt: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    degree: int | None = None,
    depth: int | None = None,
    num_samples: int | None = None,
    batch_size: int | None = None,
    topics_save_as: str | None = None,
    output_save_as: str | None = None,
    include_system_message: bool | None = None,
    mode: str = "tree",
    # Modular conversation configuration
    conversation_type: str | None = None,
    reasoning_style: str | None = None,
    agent_mode: str | None = None,
) -> DeepFabricConfig:
    """
    Load configuration from YAML file or create minimal config from CLI arguments.

    Args:
        config_file: Path to YAML configuration file
        topic_prompt: Starting topic/seed for topic generation
        topics_system_prompt: System prompt for topic generation
        generation_system_prompt: System prompt for dataset content generation
        output_system_prompt: System prompt for final dataset output
        provider: LLM provider
        model: Model name
        temperature: Temperature setting
        degree: Branching factor
        depth: Depth of tree/graph
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        topics_save_as: Path to save topics
        output_save_as: Path to save dataset
        include_system_message: Include system message in dataset
        mode: Topic generation mode (tree or graph)
        conversation_type: Base conversation type (basic, cot)
        reasoning_style: Reasoning style for cot (freetext, agent)
        agent_mode: [Deprecated] Agent mode (single_turn only, multi_turn no longer supported)

    Returns:
        DeepFabricConfig object

    Raises:
        ConfigurationError: If config file is invalid or required parameters are missing
    """
    if config_file:
        try:
            config = DeepFabricConfig.from_yaml(config_file)
        except FileNotFoundError as e:
            raise ConfigurationError(f"Config file not found: {config_file}") from e
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {str(e)}") from e
        except Exception as e:
            raise ConfigurationError(f"Error loading config file: {str(e)}") from e
        else:
            return config

    # No config file provided - create minimal configuration from CLI args
    if not topic_prompt:
        raise ConfigurationError("--topic-prompt is required when no config file is provided")

    tui = get_tui()
    tui.info("No config file provided - using CLI parameters")

    # Create minimal config dict with new structure
    default_prompt = generation_system_prompt or "You are a helpful AI assistant."

    # Build conversation config
    conversation_config = {"type": conversation_type or "basic"}
    if reasoning_style:
        conversation_config["reasoning_style"] = reasoning_style
    if agent_mode:
        conversation_config["agent_mode"] = agent_mode

    minimal_config = {
        "topics": {
            "prompt": topic_prompt,
            "mode": mode,
            "system_prompt": topics_system_prompt or "",
            "depth": depth
            or (TOPIC_GRAPH_DEFAULT_DEPTH if mode == "graph" else TOPIC_TREE_DEFAULT_DEPTH),
            "degree": degree
            or (TOPIC_GRAPH_DEFAULT_DEGREE if mode == "graph" else TOPIC_TREE_DEFAULT_DEGREE),
            "save_as": topics_save_as
            or ("topic_graph.json" if mode == "graph" else "topic_tree.jsonl"),
            "llm": {
                "provider": provider or DEFAULT_PROVIDER,
                "model": model or DEFAULT_MODEL,
                "temperature": temperature
                or (
                    TOPIC_GRAPH_DEFAULT_TEMPERATURE
                    if mode == "graph"
                    else TOPIC_TREE_DEFAULT_TEMPERATURE
                ),
            },
        },
        "generation": {
            "system_prompt": default_prompt,
            "instructions": "Generate diverse and educational examples",
            "conversation": conversation_config,
            "max_retries": DEFAULT_MAX_RETRIES,
            "llm": {
                "provider": provider or DEFAULT_PROVIDER,
                "model": model or DEFAULT_MODEL,
                "temperature": temperature or ENGINE_DEFAULT_TEMPERATURE,
            },
        },
        "output": {
            "system_prompt": output_system_prompt,
            "include_system_message": include_system_message
            if include_system_message is not None
            else True,
            "num_samples": num_samples if num_samples is not None else ENGINE_DEFAULT_NUM_EXAMPLES,
            "batch_size": batch_size or ENGINE_DEFAULT_BATCH_SIZE,
            "save_as": output_save_as or "dataset.jsonl",
        },
    }

    try:
        return DeepFabricConfig.model_validate(minimal_config)
    except ValidationError as e:
        raise ConfigurationError(f"Invalid configuration: {str(e)}") from e


def apply_cli_overrides(
    output_system_prompt: str | None = None,
    topic_prompt: str | None = None,
    topics_system_prompt: str | None = None,
    generation_system_prompt: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    degree: int | None = None,
    depth: int | None = None,
    base_url: str | None = None,
) -> tuple[dict, dict]:
    """
    Build override dictionaries from CLI parameters.

    Args:
        output_system_prompt: Override for output system prompt
        topic_prompt: Override for topic prompt
        topics_system_prompt: Override for topics system prompt
        generation_system_prompt: Override for generation system prompt
        provider: Override for LLM provider
        model: Override for model name
        temperature: Override for temperature
        degree: Override for branching factor
        depth: Override for depth
        base_url: Override for base URL

    Returns:
        Tuple of (topics_overrides, generation_overrides) dictionaries
    """
    # Prepare topics overrides
    topics_overrides = {}
    if topic_prompt:
        topics_overrides["topic_prompt"] = topic_prompt
    if topics_system_prompt:
        topics_overrides["topic_system_prompt"] = topics_system_prompt
    if provider:
        topics_overrides["provider"] = provider
    if model:
        topics_overrides["model"] = model
    if temperature:
        topics_overrides["temperature"] = temperature
    if degree:
        topics_overrides["degree"] = degree
    if depth:
        topics_overrides["depth"] = depth
    if base_url:
        topics_overrides["base_url"] = base_url

    # Prepare generation overrides
    generation_overrides = {}
    if generation_system_prompt:
        generation_overrides["generation_system_prompt"] = generation_system_prompt
    if output_system_prompt:
        generation_overrides["dataset_system_prompt"] = output_system_prompt
    if provider:
        generation_overrides["provider"] = provider
    if model:
        generation_overrides["model"] = model
    if temperature:
        generation_overrides["temperature"] = temperature
    if base_url:
        generation_overrides["base_url"] = base_url

    return topics_overrides, generation_overrides


def get_final_parameters(
    config: DeepFabricConfig,
    num_samples: int | str | None = None,
    batch_size: int | None = None,
    depth: int | None = None,
    degree: int | None = None,
) -> tuple[int | str, int, int, int]:
    """
    Get final parameters from config and CLI overrides.

    Args:
        config: DeepFabricConfig object
        num_samples: CLI override for num_samples (int, "auto", or percentage like "50%")
        batch_size: CLI override for batch_size
        depth: CLI override for depth
        degree: CLI override for degree

    Returns:
        Tuple of (num_samples, batch_size, depth, degree)
        Note: num_samples may be int, "auto", or percentage string
    """
    output_config = config.get_output_config()

    # Use 'is not None' to allow passing through "auto" or percentage strings
    final_num_samples = num_samples if num_samples is not None else output_config["num_samples"]
    final_batch_size = batch_size or output_config["batch_size"]

    # Get depth and degree from topics config
    final_depth = depth or config.topics.depth
    final_degree = degree or config.topics.degree

    return final_num_samples, final_batch_size, final_depth, final_degree
