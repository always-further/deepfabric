# validate

The `validate` command performs comprehensive analysis of DeepFabric configuration files, identifying potential issues before expensive generation processes begin.

!!! tip "Save Time and Resources"
    Catch configuration problems, authentication issues, and parameter incompatibilities early in the development cycle.

## Basic Usage

Validate a configuration file for common issues:

```bash title="Basic validation"
deepfabric validate config.yaml
```

The command analyzes your configuration structure, checks parameter values, and reports any problems with clear descriptions and suggested fixes.

## Validation Categories

The validation process examines multiple aspects of your configuration:

:material-file-check: **Structural Validation**
:   Ensures all required sections (`topics`, `generation`, `output`) are present and properly formatted.

:material-check-all: **Parameter Compatibility**
:   Checks that parameter values are within acceptable ranges and compatible with each other.

:material-key: **Provider Authentication**
:   Verifies that required environment variables are set for the specified model providers.

:material-link-variant: **Logical Consistency**
:   Examines relationships between configuration sections, ensuring file paths and dependencies are coherent.

## Validation Output

??? example "Successful validation output"

    ```
    Configuration is valid

    Configuration Summary:
      Topic Tree: depth=3, degree=4
      Dataset: steps=100, batch_size=5
      Hugging Face: repo=username/dataset-name

    Warnings:
      High temperature value (0.95) may produce inconsistent results
      No save_as path defined for topic tree
    ```

The summary provides an overview of key parameters, while warnings highlight potential issues that don't prevent execution but may affect results.

## Error Reporting

??? example "Validation error output"

    ```
    Configuration validation failed:
      - topics section is required
      - generation section is required
      - output section is required
    ```

Each error includes sufficient detail to identify the problem location and suggested corrections.

## Configuration Analysis

Beyond basic validation, the command provides insights into your configuration choices:

??? example "Configuration analysis output"

    ```
    Configuration Analysis:
      Estimated generation time: 15-25 minutes
      Estimated API costs: $2.50-4.00 (OpenAI GPT-4)
      Output size: ~500 training examples
      Topic coverage: Comprehensive (degree=4, depth=3)
    ```

This analysis helps you understand the implications of your configuration choices in terms of time, cost, and output characteristics.

## Provider-Specific Validation

The validation process includes provider-specific checks based on your configuration:

=== "OpenAI"

    Verifies model name formats and availability.

=== "Anthropic"

    Checks Claude model specifications.

=== "Ollama"

    Attempts to verify local model availability.

??? example "Provider validation output"

    ```
    Provider Validation:
       OpenAI API key detected (OPENAI_API_KEY)
       Model gpt-4 is available
       Model gpt-4 has higher costs than gpt-4-turbo
    ```

## Development Workflow Integration

Integrate validation into your development workflow to catch issues early:

```bash title="Validate before generation"
deepfabric validate config.yaml && deepfabric generate config.yaml
```

!!! tip "Best Practice"
    This pattern ensures configuration problems are identified before expensive generation processes begin.

## Batch Validation

Validate multiple configurations simultaneously:

```bash title="Batch validation"
for config in configs/*.yaml; do
  echo "Validating $config"
  deepfabric validate "$config"
done
```

## Common Issues

!!! warning "Missing Required Sections"
    Configurations lacking essential components like `topics`, `generation`, or `output` sections are flagged immediately.

!!! warning "Parameter Range Issues"
    Values outside reasonable ranges, such as negative depths or extremely high temperatures, are identified with suggested corrections.

!!! warning "Provider Mismatches"
    Inconsistencies between specified providers and model names are detected and reported with compatible alternatives.

!!! warning "File Path Problems"
    Invalid or potentially conflicting output paths are identified to prevent generation failures or accidental overwrites.

## Validation Exit Codes

The validate command uses standard exit codes for scripting integration:

| Exit Code | Meaning |
|-----------|---------|
| **0** | Configuration is valid and ready for generation |
| **1** | Configuration has errors that prevent generation |
| **2** | Configuration file not found or unreadable |

??? tip "Continuous Validation Strategy"
    Consider adding configuration validation to your version control hooks or CI pipeline. This practice catches configuration regressions and ensures all committed configurations are functional.
