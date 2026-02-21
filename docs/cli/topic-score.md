# topic score

The `topic score` command evaluates graph quality using embedding-based drift metrics:

- **GTD (Global Topic Drift)**: cosine similarity to the root topic
- **LTD (Local Topic Drift)**: max cosine similarity to parent topic(s)

It writes a JSON report with per-node metrics, flagged nodes, and estimated removal impact.

## Usage

```bash
deepfabric topic score topic_graph.json
```

By default, this writes `<input>_score_report.json`.

## Options

| Option | Description |
|--------|-------------|
| `--output-report, -o` | Output path for report JSON |
| `--depth1-gtd` | Flag depth-1 nodes with GTD below this threshold |
| `--gtd-neg` | Flag nodes with GTD below this threshold |
| `--ltd` | Flag nodes with LTD below this threshold |
| `--embedding-key` | Metadata key used for node embeddings |
| `--embedding-model` | SentenceTransformer model used if embeddings are missing |

## Example

```bash
deepfabric topic score topic_graph.json \
  --depth1-gtd 0.25 \
  --gtd-neg 0.0 \
  --ltd 0.25 \
  --output-report report.json
```

## Report Contents

The report includes:

- `summary`
  - node counts (original, flagged, estimated removed, estimated remaining)
  - GTD/LTD distribution stats
  - depth distribution
  - threshold config used
- `metrics_per_node` with GTD/LTD values
- `flagged_nodes` with reasons
- `removed_node_ids` estimated by subtree propagation from flagged nodes

!!! note
    If embeddings are not present in node metadata, DeepFabric attempts to generate them with `sentence-transformers`.
