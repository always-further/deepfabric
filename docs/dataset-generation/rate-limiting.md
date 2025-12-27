# Rate Limiting

DeepFabric includes intelligent retry handling for API rate limits across all LLM providers.

## Overview

The system provides:

- **Provider-aware defaults** for OpenAI, Anthropic, Gemini, Ollama, OpenRouter
- **Exponential backoff with jitter** to prevent thundering herd
- **Retry-after header support** when providers specify wait times
- **Fail-fast detection** for non-retryable errors (e.g., daily quota exhaustion)

## Provider Defaults

Each provider has optimized defaults:

| Provider | Max Retries | Base Delay | Max Delay |
|----------|-------------|------------|-----------|
| OpenAI | 5 | 1.0s | 60s |
| Anthropic | 5 | 1.0s | 60s |
| Gemini | 5 | 2.0s | 120s |
| Ollama | 2 | 0.5s | 5s |
| OpenRouter | 5 | 1.0s | 60s |

## Configuration

### Using Defaults

Omit rate limiting config to use provider defaults:

```yaml title="config.yaml"
generation:
  llm:
    provider: "gemini"
    model: "gemini-2.0-flash-exp"
  # Rate limiting uses defaults automatically
```

### Custom Configuration

```yaml title="config.yaml"
generation:
  llm:
    provider: "gemini"
    model: "gemini-2.0-flash-exp"

  rate_limit:
    max_retries: 7
    base_delay: 3.0
    max_delay: 180.0
    backoff_strategy: "exponential_jitter"
    exponential_base: 2.0
    jitter: true
    respect_retry_after: true
```

## Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_retries` | int | 5 | Maximum retry attempts (0-20) |
| `base_delay` | float | 1.0 | Initial delay in seconds (0.1-60) |
| `max_delay` | float | 60.0 | Maximum delay cap in seconds (1-300) |
| `backoff_strategy` | string | exponential_jitter | See strategies below |
| `exponential_base` | float | 2.0 | Multiplier for exponential backoff (1.1-10) |
| `jitter` | bool | true | Add randomization to prevent synchronized retries |
| `respect_retry_after` | bool | true | Honor server-specified wait times |

### Backoff Strategies

| Strategy | Formula | Use Case |
|----------|---------|----------|
| `exponential_jitter` | `delay = base * (exp_base ^ attempt) +/- 25%` | Recommended default |
| `exponential` | `delay = base * (exp_base ^ attempt)` | Predictable timing |
| `linear` | `delay = base * attempt` | Gentle increase |
| `constant` | Always use `base_delay` | Fixed intervals |

## Provider-Specific Behavior

=== "OpenAI"

    Monitors `x-ratelimit-*` headers and respects `retry-after`:

    ```yaml title="config.yaml"
    rate_limit:
      max_retries: 5
      respect_retry_after: true
    ```

=== "Anthropic"

    Uses token bucket algorithm with RPM/ITPM/OTPM limits:

    ```yaml title="config.yaml"
    rate_limit:
      max_retries: 5
      base_delay: 1.0
    ```

=== "Gemini"

    No retry-after header. Detects daily quota exhaustion and fails fast:

    ```yaml title="config.yaml"
    rate_limit:
      max_retries: 5
      base_delay: 2.0      # Higher default
      max_delay: 120.0     # Longer for daily quota
    ```

    !!! warning "Daily Quota"
        When Gemini's RPD (requests per day) limit is hit, the system fails fast rather than retrying since quota resets at midnight Pacific time.

=== "Ollama"

    Minimal retries for local deployment:

    ```yaml title="config.yaml"
    rate_limit:
      max_retries: 2
      base_delay: 0.5
      max_delay: 5.0
    ```

## Python API

```python title="rate_limit_example.py"
from deepfabric import DataSetGenerator

generator = DataSetGenerator(
    generation_system_prompt="You are a helpful assistant.",
    provider="gemini",
    model_name="gemini-2.0-flash-exp",
    rate_limit={
        "max_retries": 7,
        "base_delay": 3.0,
        "max_delay": 180.0,
        "backoff_strategy": "exponential_jitter",
    }
)
```

## Retry Behavior

!!! success "Retries On"
    - `429` (rate limit)
    - `500`, `502`, `503`, `504` (server errors)
    - Timeout, connection, network errors

!!! failure "Does NOT Retry"
    - `4xx` errors (except 429)
    - Authentication failures
    - Daily quota exhaustion (Gemini)

## Best Practices

=== "High Volume (Paid Tier)"

    ```yaml title="Paid tier config"
    rate_limit:
      max_retries: 3
      base_delay: 0.5
      max_delay: 10.0
    ```

=== "Free Tier (Aggressive Limits)"

    ```yaml title="Free tier config"
    rate_limit:
      max_retries: 10
      base_delay: 5.0
      max_delay: 300.0
    ```

=== "Combined with Batch Size"

    ```yaml title="Batch size optimization"
    output:
      batch_size: 2    # Smaller batches reduce rate limit pressure
      num_samples: 20

    generation:
      rate_limit:
        max_retries: 5
        base_delay: 2.0
    ```

## Troubleshooting

### Still Hitting Rate Limits

!!! tip "Solutions"
    1. Reduce `batch_size` in dataset creation
    2. Increase `base_delay`
    3. Check your provider tier/quota

### Daily Quota Exhausted (Gemini)

The system detects this and fails immediately:

```
ERROR - Failing fast for gemini: 429 RESOURCE_EXHAUSTED (daily_quota_exhausted=True)
```

!!! info "Options"
    - Wait until midnight Pacific time
    - Upgrade Gemini tier
    - Switch providers temporarily

### Too Many Retries

Reduce `max_retries` to fail faster:

```yaml
rate_limit:
  max_retries: 3
```
