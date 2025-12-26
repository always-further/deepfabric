# validate

The `validate` command performs comprehensive analysis of DeepFabric configuration files, identifying potential issues before expensive generation processes begin. This proactive approach saves significant time and resources by catching configuration problems, authentication issues, and parameter incompatibilities early in the development cycle.

Validation encompasses structural analysis, parameter compatibility checking, provider authentication verification, and logical consistency assessment across all configuration sections. The command provides detailed feedback with actionable guidance for resolving identified issues.

## Basic Usage

Validate a configuration file for common issues:

```bash
deepfabric validate config.yaml
```

The command analyzes your configuration structure, checks parameter values, and reports any problems with clear descriptions and suggested fixes. Successful validation confirms that your configuration is ready for the generation process.

## Validation Categories

The validation process examines multiple aspects of your configuration:

**Structural Validation** ensures all required sections are present and properly formatted. Missing critical sections like `topics`, `generation`, or `output`, or malformed YAML syntax are identified immediately.

**Parameter Compatibility** checks that parameter values are within acceptable ranges and compatible with each other. For example, extremely high temperature values or inconsistent model provider specifications are flagged.

**Provider Authentication** attempts to verify that required environment variables are set for the specified model providers, helping identify authentication issues before they cause generation failures.

**Logical Consistency** examines relationships between different configuration sections, ensuring that file paths, placeholder references, and parameter dependencies are coherent.

## Validation Output

Successful validation produces a summary of your configuration:

```bash
 Configuration is valid

Configuration Summary:
  Topic Tree: depth=3, degree=4
  Dataset: steps=100, batch_size=5
  Hugging Face: repo=username/dataset-name

Warnings:
  ⚠️  High temperature value (0.95) may produce inconsistent results
  ⚠️  No save_as path defined for topic tree
```

The summary provides an overview of key parameters, while warnings highlight potential issues that don't prevent execution but may affect results.

## Error Reporting

Configuration problems are reported with clear categorization and guidance:

```bash
Configuration validation failed:
  - topics section is required
  - generation section is required
  - output section is required
```

Each error includes sufficient detail to identify the problem location and suggested corrections. Error messages reference specific configuration sections and parameter names for efficient problem resolution.

## Configuration Analysis

Beyond basic validation, the command provides insights into your configuration choices:

```bash
deepfabric validate config.yaml
```

```bash
Configuration Analysis:
  Estimated generation time: 15-25 minutes
  Estimated API costs: $2.50-4.00 (OpenAI GPT-4)
  Output size: ~500 training examples
  Topic coverage: Comprehensive (degree=4, depth=3)
```

This analysis helps you understand the implications of your configuration choices in terms of time, cost, and output characteristics.

## Provider-Specific Validation

The validation process includes provider-specific checks based on your configuration:

For OpenAI configurations, it verifies model name formats and availability. For Anthropic configurations, it checks Claude model specifications. For Ollama configurations, it attempts to verify local model availability.

```bash
# Example validation with provider details
deepfabric validate config.yaml
```

```bash
Provider Validation:
   OpenAI API key detected (OPENAI_API_KEY)
   Model gpt-4 is available
  ⚠️  Model gpt-4 has higher costs than gpt-4-turbo
```

## Development Workflow Integration

Integrate validation into your development workflow to catch issues early:

```bash
# Validate before generation
deepfabric validate config.yaml && deepfabric generate config.yaml
```

This pattern ensures that configuration problems are identified before expensive generation processes begin, reducing development cycle time and preventing resource waste.

## Batch Validation

Validate multiple configurations simultaneously:

```bash
for config in configs/*.yaml; do
  echo "Validating $config"
  deepfabric validate "$config"
done
```

This approach is useful when maintaining multiple configuration variants for different experiments or use cases.

## Common Issues

The validation process identifies several categories of common configuration problems:

**Missing Required Sections**: Configurations lacking essential components like `topics`, `generation`, or `output` sections are flagged immediately.

**Parameter Range Issues**: Values outside reasonable ranges, such as negative depths or extremely high temperatures, are identified with suggested corrections.

**Provider Mismatches**: Inconsistencies between specified providers and model names are detected and reported with compatible alternatives.

**File Path Problems**: Invalid or potentially conflicting output paths are identified to prevent generation failures or accidental overwrites.

## Validation Exit Codes

The validate command uses standard exit codes for scripting integration:

- **0**: Configuration is valid and ready for generation
- **1**: Configuration has errors that prevent generation
- **2**: Configuration file not found or unreadable

These exit codes enable reliable automation and continuous integration workflows where configuration validation is part of the build process.

??? tip "Continuous Validation Strategy"
    Consider adding configuration validation to your version control hooks or continuous integration pipeline. This practice catches configuration regressions and ensures that all committed configurations are functional and ready for use.
