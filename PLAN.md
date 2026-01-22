# Implementation Plan: Checkpoint-Based Resume (#575)

## Overview

Implement checkpoint-based resume capability for dataset generation, allowing users to pause and resume long-running generation processes without losing progress.

## Requirements (from Issue #575)

1. During dataset generation, at specified intervals, save samples to file
2. Record the node UUID as a checkpoint alongside saved samples
3. Make checkpoints configurable by users

## Dependencies

- **PR #585** (auto/percentage num_samples) - This PR fixes the `num_samples`/`num_steps` relationship so that `num_samples` means actual samples and `num_steps = ceil(num_samples / batch_size)`. Checkpoint logic depends on this corrected behavior.

## Architecture

### Checkpoint Strategy: Samples-Based Checkpoints

Checkpoint after a threshold number of samples have been generated (more intuitive for users than step-based):

```python
checkpoint_samples: 500  # Save every ~500 samples
```

**Write Strategy: Incremental Append** (minimal overhead)
- Append only NEW samples since last checkpoint
- Write amplification: 1× (same as single final write)
- Overhead: <0.1% of generation time

### Memory Optimization

**Current Problem:** All samples accumulate in `self._samples` list, causing memory issues for large datasets (e.g., 50,000+ samples).

**Solution:** When checkpointing is enabled, flush samples to disk and clear from memory:

```python
# After checkpoint save
if checkpoint_samples:
    append_to_checkpoint(samples_since_last_checkpoint)
    save_checkpoint_metadata()

    # Memory optimization: clear flushed samples from memory
    self._samples.clear()  # Or keep only last batch for immediate use
    samples_since_checkpoint = 0
```

**Memory profile comparison:**

| Scenario | Peak Memory (50K samples, ~1KB each) |
|----------|--------------------------------------|
| No checkpointing | ~50MB (all samples in memory) |
| Checkpoint every 500 | ~0.5MB (only current batch + pending) |

**Trade-off:** Final dataset assembly reads from checkpoint file instead of memory.

**Final Output Flow:**
1. Generation completes → all samples already in checkpoint file
2. Copy/rename checkpoint file to final output path
3. Or stream-read checkpoint and apply any post-processing

### What Gets Saved

1. **Samples file** (`.checkpoint.jsonl`) - Incremental append of successful samples
2. **Failures file** (`.checkpoint.failures.jsonl`) - Incremental append of failed samples
3. **Metadata file** (`.checkpoint.json`) - Current state:
   - Samples generated count
   - Failed samples count
   - Steps completed
   - Topic paths processed (completed successfully)
   - Topic paths failed (generation errors)
   - Topic paths pending (in-flight, for future parallelism - currently always empty)
   - Node UUIDs (for validation)
   - Generation config snapshot

### Resume Strategy

When resuming:
1. Load existing samples from checkpoint file
2. Load failed samples from failures file
3. Load metadata to determine starting point
4. Re-queue any pending topic paths (future parallelism support)
5. Skip already-processed topic paths
6. If `retry_failed_samples=True`, add failed topic paths back to queue
6. Continue generation from next step

### Failed Samples Retry

**Option:** `retry_failed_samples` (default: `False`)

- When `False`: Failed samples are saved to checkpoint but not retried on resume
- When `True`: Failed samples from checkpoint are added back to the generation queue

This allows users to:
- Review failures before deciding to retry
- Resume without wasting time on persistently failing topics
- Explicitly opt-in to retry behavior when desired

## Implementation Phases

### Phase 1: Core Checkpoint Logic (MVP)

#### 1.1 Config Changes (`config.py`)

Add new `CheckpointConfig` class:
```python
class CheckpointConfig(BaseModel):
    interval: int = Field(
        ...,
        ge=1,
        description="Save checkpoint every N samples"
    )
    path: str = Field(
        default=".checkpoints",
        description="Directory to store checkpoint files"
    )
    retry_failed: bool = Field(
        default=False,
        description="When resuming, retry previously failed samples"
    )
```

Add to `DeepFabricConfig`:
```python
checkpoint: CheckpointConfig | None = Field(
    None, description="Checkpoint configuration for resumable generation"
)
```

#### 1.2 Constants (`constants.py`)

Add checkpoint filename patterns:
```python
CHECKPOINT_METADATA_SUFFIX = ".checkpoint.json"
CHECKPOINT_SAMPLES_SUFFIX = ".checkpoint.jsonl"
CHECKPOINT_FAILURES_SUFFIX = ".checkpoint.failures.jsonl"
```

#### 1.3 Generator Changes (`generator.py`)

**New Methods:**
- `_save_checkpoint(samples_since_last: list)` - Append new samples to checkpoint
- `_save_checkpoint_metadata()` - Update metadata file
- `_load_checkpoint(checkpoint_path: str)` - Load checkpoint for resume
- `_get_checkpoint_paths()` - Get checkpoint file paths

**Modified:** `_run_generation_loop_async()`
- Track `samples_since_checkpoint` counter
- After each step, check if threshold reached
- Use incremental append for samples

#### 1.4 Checkpoint Logic

```python
samples_since_checkpoint = 0
checkpoint_file = None  # Opened once, appended to

for step in range(num_steps):
    # Generate batch
    new_samples = generate_batch()  # batch_size samples
    self._samples.extend(new_samples)
    samples_since_checkpoint += len(new_samples)

    # Check if checkpoint threshold reached
    if checkpoint_samples and samples_since_checkpoint >= checkpoint_samples:
        # Append only new samples (incremental)
        append_to_checkpoint(new_samples_since_last_checkpoint)
        save_checkpoint_metadata()
        samples_since_checkpoint = 0
```

#### 1.5 Checkpoint File Format

**Metadata file** (`dataset.checkpoint.json`):
```json
{
  "version": 1,
  "samples_generated": 500,
  "samples_failed": 12,
  "steps_completed": 100,
  "total_steps": 1000,
  "batch_size": 5,
  "checkpoint_samples": 500,
  "timestamp": "2026-01-22T14:30:45Z",
  "topic_paths_processed": [
    ["Root", "Topic1", "Subtopic1"],
    ["Root", "Topic1", "Subtopic2"]
  ],
  "topic_paths_failed": [
    ["Root", "Topic2", "Subtopic3"]
  ],
  "topic_paths_pending": [],
  "generation_config": {
    "model_name": "gpt-4o",
    "temperature": 0.7,
    "num_samples": 5000
  }
}
```

**Note on `topic_paths_pending`:** Currently always empty (sequential step execution). Included in schema now for forward-compatibility with future step-level parallelism. On resume, any pending paths are re-queued.

**Samples file** (`dataset.checkpoint.jsonl`):
- Standard JSONL format, incrementally appended

### Phase 2: CLI & Resume Support

#### 2.1 CLI Arguments (`cli.py`)

```python
@click.option("--checkpoint-interval", type=int, help="Save checkpoint every N samples")
@click.option("--checkpoint-path", type=click.Path(), help="Checkpoint directory")
@click.option("--resume", is_flag=True, help="Resume from existing checkpoint")
@click.option("--retry-failed", is_flag=True, default=False, help="Retry failed samples when resuming")
```

#### 2.2 Resume Logic (`generator.py`)

- Load samples from checkpoint file
- Load failed samples from failures file
- Skip steps already completed
- Filter out already-processed topic paths
- If `retry_failed_samples=True`:
  - Extract topic paths from failed samples
  - Add them back to the generation queue
  - Clear the failures file (will be repopulated if they fail again)
- Validate config compatibility (warn if different)

### Phase 3: Polish & UX

- TUI status for checkpoint saves
- Checkpoint cleanup (keep last N)
- Checkpoint integrity validation
- Better error messages for resume failures
- `deepfabric checkpoint status config.yaml` command to inspect checkpoint state

#### 3.1 Checkpoint Status Command (`cli.py`)

```python
@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def checkpoint_status(config_path: str):
    """Show checkpoint status for a generation config."""
```

**Behavior:**
1. Load config to get `checkpoint_dir` and `save_as`
2. Derive checkpoint paths from `save_as` (e.g., `dataset.jsonl` → `dataset.checkpoint.json`)
3. Display status summary

**Example output:**
```
Checkpoint Status: .checkpoints/dataset.checkpoint.json

Progress:    500/5000 samples (10.0%)
Failed:      12 samples
Steps:       100/1000 completed

Failed Topics:
  - Root > Topic2 > Subtopic3 (Rate limit exceeded)
  - Root > Topic4 > Subtopic1 (JSON parse error)
  ...

Resume with: deepfabric generate config.yaml --resume
Retry failed: deepfabric generate config.yaml --resume --retry-failed
```

## Files to Modify

| File | Changes |
|------|---------|
| `deepfabric/config.py` | Add `CheckpointConfig` class with `interval`, `path`, `retry_failed` fields |
| `deepfabric/constants.py` | Add checkpoint filename constants |
| `deepfabric/generator.py` | Checkpoint save/load logic, modify generation loop |
| `deepfabric/cli.py` | Add CLI arguments for checkpointing (`--checkpoint-interval`, `--checkpoint-path`, `--resume`, `--retry-failed`) |
| `deepfabric/dataset_manager.py` | Handle checkpoint paths in output flow |
| `tests/unit/test_checkpoint.py` | Add checkpoint tests |
| `docs/dataset-generation/configuration.md` | Document checkpoint config options |
| `docs/cli.md` | Document `--checkpoint-samples`, `--resume`, `--retry-failed` flags |
| `mkdocs.yml` | Add new docs pages if needed |

## Example Calculations

### Scenario: 5,000 samples, batch_size=5, checkpoint_samples=500

```
num_samples: 5000
batch_size: 5
num_steps: ceil(5000 / 5) = 1000 steps
checkpoint_samples: 500

Checkpoints at samples: 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000
Checkpoints at steps:   100,  200,  300,  400,  500,  600,  700,  800,  900, 1000
Total checkpoints: 10
Samples per checkpoint: 500
```

### Write Overhead (Incremental Append)

| Checkpoint | New Samples Written | Cumulative in File |
|------------|--------------------|--------------------|
| 1 | 500 | 500 |
| 2 | 500 | 1,000 |
| ... | 500 | ... |
| 10 | 500 | 5,000 |

**Total writes: 5,000 samples** (same as final dataset - no amplification)

## Testing Strategy

### Unit Tests
- Checkpoint save/load round-trip
- Resume skips completed steps correctly
- Config validation with checkpoint options
- Topic path deduplication on resume
- Incremental append works correctly
- Failed samples saved to checkpoint
- Resume with `retry_failed_samples=False` skips failures
- Resume with `retry_failed_samples=True` retries failures

### Integration Tests
- Full generation with checkpoints
- Interrupt simulation and resume
- Verify final dataset matches non-interrupted run

## Edge Cases

1. **Interrupted mid-step**: Checkpoint saves after step completes (may lose up to batch_size samples)
2. **Config changes between runs**: Warn user, validate compatibility
3. **Topic model changes**: Detect and fail gracefully
4. **Disk full**: Handle checkpoint write failures
5. **Corrupt checkpoint**: Validate on load, clear error message
6. **batch_size doesn't divide evenly into checkpoint interval**: Checkpoint after threshold exceeded

## Usage Examples

### YAML Config
```yaml
output:
  num_samples: 5000
  batch_size: 5
  save_as: "final_dataset.jsonl"
  checkpoint:
    interval: 500          # Checkpoint every 500 samples
    path: "./my-checkpoints"
    retry_failed: false    # Default: don't retry failures on resume
```

### CLI
```bash
# Generate with checkpoints every 500 samples
deepfabric generate config.yaml --checkpoint-interval 500

# Resume from checkpoint (skips failed samples by default)
deepfabric generate config.yaml --resume

# Resume and retry previously failed samples
deepfabric generate config.yaml --resume --retry-failed
```

## Success Criteria

- [ ] Checkpoints saved at configured sample intervals
- [ ] Incremental append mode (no write amplification)
- [ ] Resume loads samples and continues from correct point
- [ ] No duplicate samples in final dataset
- [ ] Failed samples saved to checkpoint failures file
- [ ] Resume with `retry_failed=False` (default) skips failures
- [ ] Resume with `retry_failed=True` retries failed samples
- [ ] Memory optimization: constant memory usage regardless of dataset size
- [ ] Performance: <0.1% overhead for checkpoint saves
- [ ] Existing tests pass
- [ ] New tests for checkpoint functionality
- [ ] Documentation updated (config options, CLI flags, usage examples)

## Future Enhancements

### Step-Level Parallelism (Future)

Current architecture runs steps sequentially with batch-level parallelism. For true step-level parallelism:

1. ~~Add `topic_paths_pending` to checkpoint metadata~~ ✅ Already included in schema
2. Use thread-safe queue for sample collection
3. Single writer thread for checkpoint appends
4. Resume logic re-queues pending paths (already implemented)

The checkpoint schema is forward-compatible - `topic_paths_pending` is included but currently always empty. When step-level parallelism is added, it will populate this field with in-flight work.
