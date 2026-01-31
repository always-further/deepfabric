# Generation Model Refactor Plan

## Status: IMPLEMENTED

This refactor has been implemented on branch `fix/generation-byuuid`. See commits:
- `43ebd47` - feat: add UUID-based topic identification for generation
- `f44ad09` - wip: checkpoint format v4 with (uuid, cycle) tuples
- `ebd7739` - feat: implement cycle-based generation with UUID tracking
- `f144ca4` - feat: add TUI handlers for cycle-based generation events

---

## Overview

Simplify dataset generation from step-based batching to UUID-based iteration with concurrency control.

## Previous Model

```
num_steps = total_samples / batch_size
for step in range(num_steps):
    start_idx = step * batch_size
    prompts = topic_paths[start_idx:start_idx+batch_size]
    process_batch(prompts)
```

- Progress: "Step 3/50"
- Checkpoint tracks `_processed_ids` (topic_ids, may have duplicates in graphs)
- Cycles topics by multiplying path list
- Filtering by `topic_id` causes issues with duplicate UUIDs

## New Model (Implemented)

```python
unique_topics = topic_model.get_unique_topics()  # deduplicated by UUID
cycles_needed = ceil(total_samples / len(unique_topics))

for cycle in range(cycles_needed):
    for topic in unique_topics:
        if not _is_completed(topic.uuid, cycle):
            await generate_sample(topic)  # with semaphore concurrency
            mark_completed(topic.uuid, cycle)
```

- Progress: "Cycle 2/3: +1875 samples (total 3750/5000)"
- Checkpoint tracks `completed: list[[uuid, cycle]]` tuples
- No path-based iteration for cycle calculation
- `batch_size` controls `concurrency` (parallel LLM calls via asyncio.Semaphore)

---

## Implementation Summary

### Phase 1: Topic Model Updates (COMPLETE)

| File | Changes |
|------|---------|
| `topic_model.py` | Added `Topic(uuid, topic)` namedtuple and `get_unique_topics()` abstract method |
| `graph.py` | Implemented `get_unique_topics()` - deduplicates by node UUID from metadata |
| `tree.py` | Added `_leaf_uuids` list, `leaf_uuid` field in JSONL, implemented `get_unique_topics()` |

### Phase 2: Generator Core (COMPLETE)

| File | Changes |
|------|---------|
| `constants.py` | Bumped `CHECKPOINT_VERSION` to 4 |
| `generator.py` | Changed `_processed_ids: set[str]` to `_completed: set[tuple[str, int]]` |
| `generator.py` | Added `_prepare_unique_topics()` for cycle calculation |
| `generator.py` | Added `_run_cycle_based_generation_async()` with asyncio.Semaphore |
| `generator.py` | Added `_is_completed(uuid, cycle)` and `_is_uuid_completed_any_cycle(uuid)` |
| `generator.py` | Updated `create_data_async()` and `create_data_with_events_async()` to use cycle-based generation |

### Phase 3: TUI & Event Handlers (COMPLETE)

| File | Changes |
|------|---------|
| `dataset_manager.py` | Added handlers for `cycle_start` and `cycle_complete` events |
| `dataset_manager.py` | Updated `generation_start` handler to support both cycle-based and step-based formats |

### Phase 4: Testing (COMPLETE)

| File | Changes |
|------|---------|
| `test_checkpoint.py` | Updated all tests for v4 checkpoint format with `completed` tuples |
| `test_checkpoint.py` | Renamed `TestIsTopicProcessed` to `TestIsCompleted` with new method tests |
| `test_generator.py` | Added `get_unique_topics` mock to topic_tree fixtures |

---

## Checkpoint Format

### Version 4 (Current)
```json
{
  "version": 4,
  "completed": [
    ["uuid1", 0],
    ["uuid1", 1],
    ["uuid2", 0]
  ],
  "total_samples": 500,
  "total_failures": 10,
  "checkpoint_interval": 100
}
```

Where `completed` is a list of `[uuid, cycle]` tuples (JSON arrays).
In Python: `self._completed: set[tuple[str, int]]`

### Migration
- Checkpoints with version < 4 are rejected with clear error message
- Users must delete old checkpoints and restart

---

## Tree JSONL Format

```json
{"path": ["Root", "Branch", "Leaf"], "leaf_uuid": "550e8400-e29b-41d4-a716-446655440000"}
```

- UUID generated at build time using `uuid.uuid4()`
- Persisted with tree, stable across loads
- Existing trees without `leaf_uuid` field will error on load

---

## Event System

### Cycle-Based Events (New)
- `cycle_start`: `{cycle, total_cycles, topics_in_cycle}`
- `cycle_complete`: `{cycle, samples_in_cycle, failures_in_cycle}`

### Updated Events
- `generation_start`: Now includes `unique_topics`, `cycles_needed`, `concurrency` when cycle-based
- `generation_complete`: Includes `cycles_completed`, `unique_topics`

### Backward Compatibility
- Step-based events (`step_start`, `step_complete`) still emitted when no topic_model provided
- TUI handles both event types

---

## Remaining Work (Future)

### CLI Updates (Not Yet Implemented)
- Add `--concurrency` option as alias for `--batch-size`
- Update `validate` command to show unique topics and cycles

### Config Updates (Not Yet Implemented)
- Accept `concurrency` as alias for `batch_size` in YAML config

---

## Key Decisions Made

1. **Tree UUIDs**: Generate on build and persist in JSONL
   - New format: `{"path": [...], "leaf_uuid": "..."}`
   - Existing trees require regeneration

2. **Checkpoint migration**: Require fresh start
   - Detect old checkpoint, warn user, refuse to resume
   - User must delete checkpoint or start fresh

3. **Partial cycles**: Track as `(uuid, cycle)` tuples
   - Example: 5000 samples from 1875 topics = 3 cycles
   - Cycle 1: 1875 topics, Cycle 2: 1875 topics, Cycle 3: 1250 topics (partial)
   - Checkpoint stores: `{("uuid-abc", 0), ("uuid-abc", 1), ("uuid-def", 0), ...}`

4. **Concurrency model**: asyncio.Semaphore
   - `batch_size` parameter controls maximum parallel LLM calls
   - Each topic processed independently with semaphore-controlled concurrency
