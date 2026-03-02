"""DeepFabric Python SDK for programmatic dataset generation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DeepFabricConfig
from .dataset import Dataset
from .exceptions import ConfigurationError, DeepFabricError
from .generator import DataSetGenerator
from .graph import Graph, GraphConfig
from .llm import validate_provider_api_key
from .tree import Tree, TreeConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a dataset generation operation."""

    success: bool
    dataset_path: str | None = None
    topic_path: str | None = None
    samples_generated: int = 0
    samples_failed: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        """Total samples attempted (generated + failed). """
        return self.samples_generated + self.samples_failed


class DeepFabricSDK:
    """Python SDK for DeepFabric dataset generation.

    Example usage:
        from deepfabric import DeepFabricSDK

        sdk = DeepFabricSDK()
        result = sdk.generate("config.yaml")

        if result.success:
            print(f"Generated {result.samples_generated} samples")
            print(f"Dataset saved to: {result.dataset_path}")
        else:
            print(f"Generation failed: {result.error}")
    """

    def __init__(self, *, verbose: bool = False):
        """Initialize the SDK.

        Args:
            verbose: Enable verbose logging output
        """
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)

    def generate(
        self,
        config_path: str | Path,
        *,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        num_samples: int | str | None = None,
        batch_size: int | None = None,
        topic_only: bool = False,
    ) -> GenerationResult:
        """Generate a dataset from a config file.

        Args:
            config_path: Path to the YAML configuration file
            provider: Override provider from config
            model: Override model from config
            temperature: Override temperature from config
            num_samples: Number of samples to generate (int or "auto")
            batch_size: Batch size for generation
            topic_only: Only generate topic structure, not dataset

        Returns:
            GenerationResult with generation status and metadata

        Raises:
            FileNotFoundError: If config_path doesn't exist
            ConfigurationError: If config is invalid
            DeepFabricError: If generation fails
        """
        start_time = time.time()

        try:
            # Load configuration
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            logger.info("Loading configuration from %s", config_path)
            config = DeepFabricConfig.from_yaml(str(config_path))

            # Determine effective provider
            effective_provider = provider
            if not effective_provider and config.llm:
                effective_provider = config.llm.provider
            if effective_provider:
                logger.info("Using provider: %s", effective_provider)

            # Validate API keys
            if effective_provider:
                result = validate_provider_api_key(effective_provider)
                if not result.is_valid:
                    raise ConfigurationError(
                        f"API key validation failed for {effective_provider}: {result.message}"
                    )

            # Build topic tree or graph
            topic_model = self._build_topic_model(config)

            if topic_only:
                topic_path = config.topics.save_as if config.topics else None
                return GenerationResult(
                    success=True,
                    topic_path=topic_path,
                    duration_seconds=time.time() - start_time,
                    metadata={"mode": config.topics.mode if config.topics else "tree"},
                )

            # Generate dataset
            dataset, stats = self._generate_dataset(
                config=config,
                topic_model=topic_model,
                provider=provider,
                model=model,
                temperature=temperature,
                num_samples=num_samples,
                batch_size=batch_size,
            )

            output_path = config.output.save_as if config.output else None
            return GenerationResult(
                success=True,
                dataset_path=output_path,
                topic_path=config.topics.save_as if config.topics else None,
                samples_generated=stats.get("generated", 0),
                samples_failed=stats.get("failed", 0),
                duration_seconds=time.time() - start_time,
                metadata={"provider": effective_provider, "mode": config.topics.mode if config.topics else "tree"},
            )

        except FileNotFoundError as e:
            return GenerationResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )
        except ConfigurationError as e:
            return GenerationResult(
                success=False,
                error=f"Configuration error: {e}",
                duration_seconds=time.time() - start_time,
            )
        except DeepFabricError as e:
            return GenerationResult(
                success=False,
                error=f"Generation error: {e}",
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.exception("Unexpected error during generation")
            return GenerationResult(
                success=False,
                error=f"Unexpected error: {e}",
                duration_seconds=time.time() - start_time,
            )

    def _build_topic_model(self, config: DeepFabricConfig) -> Tree | Graph:
        """Build topic tree or graph from configuration."""
        # Get LLM params from config
        llm = config.get_llm_config(config.topics)

        if config.topics.mode == "graph":
            logger.info("Building topic graph...")
            graph_config = GraphConfig(
                root_prompt=config.topics.prompt,
                system_prompt=config.topics.system_prompt,
                model_name=llm.model,
                provider=llm.provider,
                temperature=llm.temperature,
            )
            topic_model = Graph(graph_config)
        else:
            logger.info("Building topic tree...")
            tree_config = TreeConfig(
                root_prompt=config.topics.prompt,
                system_prompt=config.topics.system_prompt,
                model_name=llm.model,
                provider=llm.provider,
                temperature=llm.temperature,
            )
            topic_model = Tree(tree_config)

        # Build the topic structure
        topic_model.build()

        # Save if path specified
        if config.topics.save_as:
            topic_model.save(config.topics.save_as)
            logger.info("Topic structure saved to %s", config.topics.save_as)

        return topic_model

    def _generate_dataset(
        self,
        config: DeepFabricConfig,
        topic_model: Tree | Graph,
        **overrides,
    ) -> tuple[Dataset, dict[str, int]]:
        """Generate dataset from topic model."""
        # Get effective LLM config
        llm = config.get_llm_config(config.generation)

        # Apply overrides
        provider = overrides.get("provider") or llm.provider
        model = overrides.get("model") or llm.model
        temperature = overrides.get("temperature") or llm.temperature

        logger.info("Generating dataset with %s/%s", provider, model)

        # Create generator
        generator_config = DataSetGenerator.create_config(
            system_prompt=config.generation.system_prompt,
            model_name=model,
            provider=provider,
            temperature=temperature,
        )

        generator = DataSetGenerator(generator_config)

        # Get topic paths
        if hasattr(topic_model, "get_all_paths"):
            topic_paths = topic_model.get_all_paths()
        else:
            topic_paths = list(topic_model.traverse())

        # Generate samples
        num_samples = overrides.get("num_samples") or config.output.num_samples
        batch_size = overrides.get("batch_size") or config.output.batch_size

        dataset = Dataset()
        generated = 0
        failed = 0

        for i, topic_path in enumerate(topic_paths[:num_samples]):
            try:
                sample = generator.generate_sample(topic_path)
                dataset.add_sample(sample)
                generated += 1

                if batch_size and (i + 1) % batch_size == 0:
                    logger.info("Generated %d samples...", i + 1)
            except Exception as e:
                logger.warning("Failed to generate sample %d: %s", i, e)
                failed += 1

        # Save dataset
        if config.output.save_as:
            dataset.save(config.output.save_as)
            logger.info("Dataset saved to %s", config.output.save_as)

        stats = {"generated": generated, "failed": failed}
        return dataset, stats

    def __repr__(self) -> str:
        return "DeepFabricSDK()"
