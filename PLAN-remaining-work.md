# Remaining Work: CLI & Documentation Updates

## Overview

Update the `validate` command output and documentation to reflect the new cycle-based generation model.

---

## 1. CLI `validate` Command Updates

### Current Output (lines 1294-1319 in cli.py)

```
Topics: mode=tree, depth=2, degree=3, estimated_paths=9 (3^2)
Output: num_samples=5000, batch_size=10, num_steps=500
```

### New Output

```
Topics: mode=tree, depth=2, degree=3, estimated_paths=9 (3^2)
Output: num_samples=5000, concurrency=10
  → Cycles needed: 556 (5000 samples ÷ 9 unique topics)
  → Final cycle: 5 topics (partial)
```

### Changes Required

| Location | Change |
|----------|--------|
| cli.py:1306-1313 | Replace `num_steps` calculation with `cycles_needed` calculation |
| cli.py:1310 | Change `num_steps` display to show `concurrency` and `cycles_needed` |
| cli.py:1310 | Add partial cycle info if `num_samples % estimated_paths != 0` |

### Implementation

```python
# Calculate cycles for the new model
if isinstance(num_samples, int):
    cycles_needed = math.ceil(num_samples / estimated_paths)
    final_cycle_size = num_samples - (cycles_needed - 1) * estimated_paths
    is_partial = final_cycle_size < estimated_paths

    output_info = f"Output: num_samples={num_samples}, concurrency={batch_size}"
    tui.info(output_info)
    tui.info(f"  → Cycles needed: {cycles_needed} ({num_samples} samples ÷ {estimated_paths} unique topics)")
    if is_partial:
        tui.info(f"  → Final cycle: {final_cycle_size} topics (partial)")
```

---

## 2. Documentation Updates

### Files to Update

| File | Changes |
|------|---------|
| `docs/cli/validate.md` | Update example outputs to show cycles instead of steps |
| `docs/dataset-generation/configuration.md` | Update `batch_size` description, add info about cycles |
| `docs/dataset-generation/basic.md` | Update "Sample Size" and "Topic Cycling" sections |

---

### 2.1 docs/cli/validate.md

**Section: Validation Output (lines 36-49)**

Current:
```
Configuration Summary:
  Topic Tree: depth=3, degree=4
  Dataset: steps=100, batch_size=5
```

New:
```
Configuration Summary:
  Topics: mode=tree, depth=3, degree=4, unique_topics=64 (4^3)
  Output: num_samples=500, concurrency=5
    → Cycles needed: 8 (500 ÷ 64)
    → Final cycle: 52 topics (partial)
```

---

### 2.2 docs/dataset-generation/configuration.md

**Section: output (lines 178-189)**

Update table:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | int | 1 | Parallel generation concurrency (number of simultaneous LLM calls) |

**Section: output.checkpoint tip (lines 226-234)**

Current text mentions "steps" - update to explain cycle-based model:

```
!!! tip "Understanding Generation Model"
    DeepFabric uses a cycle-based generation model:

    - **Unique topics**: Deduplicated topics from your tree/graph (by UUID)
    - **Cycles**: Number of times to iterate through all topics
    - **Concurrency**: Maximum parallel LLM calls (`batch_size`)

    For example, with 100 unique topics and `num_samples: 250`:
    - Cycles needed: 3 (ceil(250/100))
    - Full cycles: 2 (100 topics each)
    - Final cycle: 50 topics (partial)
```

---

### 2.3 docs/dataset-generation/basic.md

**Section: Tips - Sample Size (lines 98-105)**

Current:
```
- Steps = ceil(`num_samples` / `batch_size`)
For example, `num_samples: 10` with `batch_size: 2` runs 5 steps, generating 2 samples each.
```

New:
```
- **Unique topics**: Deduplicated count from topic tree/graph
- **Cycles**: ceil(`num_samples` / unique_topics)
- **Concurrency**: `batch_size` controls parallel LLM calls

For example, with 4 unique topics and `num_samples: 10`:
- Cycles needed: 3 (ceil(10/4))
- Cycle 1: 4 samples, Cycle 2: 4 samples, Cycle 3: 2 samples
```

**Section: Graph to Sample Ratio (lines 113-131)**

Update "Topic Cycling" info box:
```
!!! info "Cycle-Based Generation"
    When `num_samples` exceeds unique topics, DeepFabric iterates through multiple cycles:

    - Each unique topic is processed once per cycle
    - Cycles continue until `num_samples` is reached
    - The final cycle may be partial

    For example, with 4 unique topics and `num_samples: 10`:
    - Cycle 1: Topics 1-4 (4 samples)
    - Cycle 2: Topics 1-4 (4 samples)
    - Cycle 3: Topics 1-2 (2 samples, partial)

    Checkpoint tracks progress as `(topic_uuid, cycle_number)` tuples,
    allowing precise resume from any point.
```

---

## 3. Implementation Order

1. **Update cli.py validate command** (30 min)
   - Replace step calculation with cycle calculation
   - Update output formatting
   - Add tests for new output format

2. **Update docs/cli/validate.md** (15 min)
   - Update example outputs
   - Add cycle-related information

3. **Update docs/dataset-generation/configuration.md** (20 min)
   - Update batch_size description
   - Add cycle model explanation
   - Update checkpoint tip

4. **Update docs/dataset-generation/basic.md** (15 min)
   - Update Sample Size section
   - Update Topic Cycling info box

5. **Run tests and verify** (10 min)
   - Ensure all tests pass
   - Verify docs build correctly

---

## 4. Optional: Add `--concurrency` CLI Option

Currently `--batch-size` is used. Could add `--concurrency` as an alias:

```python
@click.option("--concurrency", "--batch-size", type=int, help="Parallel generation concurrency")
```

This maintains backward compatibility while introducing the new terminology.
