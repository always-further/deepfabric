# topic score

The `topic score` command evaluates graph quality using embedding-based coherence metrics:

- **Global Coherence**: cosine similarity to the root topic (how on-topic a node is)
- **Parent Coherence**: max cosine similarity to parent topic(s) (local relevance)
- **Sibling Coherence**: mean cosine similarity to sibling nodes (peer consistency)

It writes a JSON report with per-node metrics, flagged nodes, and estimated removal impact using a 4-step cascading pruning pipeline.

## Usage

```bash
deepfabric topic score topic_graph.json
```

By default, this writes `<input>_score_report.json`.

## Options

| Option | Description |
|--------|-------------|
| `--output-report, -o` | Output path for report JSON |
| `--parent-coherence` | Flag nodes with parent coherence below this threshold (default: 0.25) |
| `--sibling-coherence-lower` | Flag nodes with sibling coherence below this threshold — outliers (default: 0.2) |
| `--sibling-coherence-upper` | Flag nodes with sibling coherence above this threshold — repetitive (default: 0.68) |
| `--embedding-key` | Metadata key used for node embeddings |
| `--embedding-model` | SentenceTransformer model used if embeddings are missing |

## Example

```bash
deepfabric topic score topic_graph.json \
  --parent-coherence 0.25 \
  --sibling-coherence-lower 0.2 \
  --sibling-coherence-upper 0.68 \
  --output-report report.json
```

## 4-Step Pruning Pipeline

Nodes that fail a threshold are flagged **along with all their descendants**. Each step operates on the nodes surviving from previous steps:

| Step | Condition | Purpose |
|------|-----------|---------|
| 1 | `global_coherence < 0` (hardcoded) | Remove nodes pointing opposite to root |
| 2 | `parent_coherence < threshold` | Remove off-topic children |
| 3 | `sibling_coherence < lower threshold` | Remove outlier siblings |
| 4 | `sibling_coherence > upper threshold` | Remove repetitive siblings |

## Report Contents

The report includes:

- `summary`
    - node counts (original, flagged, estimated removed, estimated remaining)
    - global coherence / parent coherence / sibling coherence distribution stats
    - depth distribution
    - threshold config used
    - per-step removal counts
- `metrics_per_node` with global_coherence, parent_coherence, and sibling_coherence values
- `flagged_nodes` with reasons
- `removed_node_ids` estimated by subtree propagation from flagged nodes

!!! note
    If embeddings are not present in node metadata, DeepFabric attempts to generate them with `sentence-transformers`. Install with: `pip install deepfabric[scoring]`

## Pipeline Integration

Scoring can also run automatically during `deepfabric generate` by adding a `scoring` section to your YAML configuration. When `prune: true`, the pipeline removes flagged subtrees before dataset generation. See [Configuration Reference](../dataset-generation/configuration.md#topicsscoring-graph-mode-only-optional) for details.
