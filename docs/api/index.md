# API Reference

The DeepFabric Python API provides programmatic access to all synthetic data generation capabilities, enabling integration into existing workflows, custom automation, and advanced use cases that require fine-grained control over the generation process.

The API design mirrors the CLI structure while providing additional flexibility through direct object manipulation and advanced configuration options. This approach supports both simple scripting scenarios and complex applications that require sophisticated dataset generation workflows.

## Core Architecture

DeepFabric's API centers around four primary classes that correspond to the main components of the generation pipeline:

**Tree and Graph Classes** handle topic modeling through hierarchical or graph-based approaches, transforming root prompts into comprehensive domain structures.

**DataSetGenerator** converts topic structures into training examples using configurable templates and generation parameters.

**Dataset** manages collections of training examples with validation, export, and analysis capabilities.

**DeepFabricConfig** provides programmatic access to YAML configuration loading and parameter management.

## Import Patterns

Standard imports provide access to the core functionality:

```python
from deepfabric import (
    Tree, Graph, DataSetGenerator,
    DeepFabricConfig
)
```

These imports give you direct access to all essential classes while maintaining clean namespace organization.

## Configuration Management

The API supports both programmatic configuration and YAML file loading:

```python
# Direct configuration
config = DeepFabricConfig.from_yaml("config.yaml")

# Programmatic configuration
tree = Tree(
    topic_prompt="Machine Learning Concepts",
    model_name="openai/gpt-4",
    degree=4,
    depth=3,
    temperature=0.7
)
```

This flexibility enables workflows that combine configuration files with runtime parameter adjustments.

## Basic Generation Pattern

The standard generation workflow follows a consistent pattern across all API usage:

```python
# 1. Create topic model
tree = Tree(
    topic_prompt="Machine Learning Concepts",
    model_name="openai/gpt-4",
    degree=4,
    depth=3,
    temperature=0.7
)

import asyncio

async def build_tree() -> None:
    async for _ in tree.build_async():
        pass

asyncio.run(build_tree())

# 2. Create dataset generator
generator = DataSetGenerator(
    instructions="Create detailed explanations with practical examples.",
    model_name="openai/gpt-4",
    temperature=0.8
)

# 3. Generate dataset
dataset = asyncio.run(generator.create_data_async(
    num_steps=100,
    batch_size=5,
    topic_model=tree
))

# 4. Save results
dataset.save("output.jsonl")
```

This pattern provides clear separation of concerns while enabling customization at each stage.

## Advanced Usage Patterns

The API supports sophisticated workflows including iterative refinement, multi-stage processing, and custom quality control:

**Iterative Development** allows you to build topic structures incrementally and test dataset generation with subsets before scaling to full production.

**Multi-Provider Workflows** enable different components to use optimized model providers, balancing cost, speed, and quality requirements.

**Custom Validation** supports application-specific quality control through custom dataset filtering and analysis.

## Error Handling

The API provides comprehensive exception handling through a hierarchy of custom exceptions:

```python
from deepfabric import (
    DeepFabricError, ConfigurationError, 
    ModelError, ValidationError
)

try:
    dataset = asyncio.run(generator.create_data_async(topic_model=tree))
except ModelError as e:
    # Handle API or model-specific issues
    print(f"Model error: {e}")
except ValidationError as e:
    # Handle configuration or data validation issues
    print(f"Validation error: {e}")
```

This structured approach enables robust error handling and graceful degradation in production environments.

## API Sections

Detailed documentation for each major component:

[**Tree**](tree.md) - Hierarchical topic modeling API
[**Graph**](graph.md) - Graph-based topic modeling API  
[**DataSetGenerator**](generator.md) - Dataset generation and management
[**Configuration**](config.md) - Configuration loading and parameter management

Each section provides comprehensive coverage of class methods, parameters, return values, and usage examples that demonstrate both basic and advanced usage patterns.