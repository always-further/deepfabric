# topic optimize-thresholds

The `topic optimize-thresholds` command searches for better coherence cutoff values on an existing graph.

It does **not** regenerate the graph. It re-scores the same graph across multiple threshold combinations.

The search space covers three tunable thresholds: `parent_coherence`, `sibling_coherence_lower`, and `sibling_coherence_upper`. The `global_coherence < 0` check is hardcoded and not part of the optimization.

## Usage

```bash
deepfabric topic optimize-thresholds topic_graph.json
```

By default this runs random search with 40 trials and writes:
`<input>_threshold_optimization.json`

## Example

```bash
deepfabric topic optimize-thresholds topic_graph.json \
  --search random \
  --trials 40 \
  --parent-coherence-min 0.10 --parent-coherence-max 0.50 \
  --sibling-coherence-lower-min 0.05 --sibling-coherence-lower-max 0.40 \
  --sibling-coherence-upper-min 0.50 --sibling-coherence-upper-max 0.85 \
  --output-report best_thresholds.json
```

## Constraints

You can constrain acceptable solutions:

```bash
deepfabric topic optimize-thresholds topic_graph.json \
  --max-removed-ratio 0.35 \
  --max-internal-removed 120
```

If no trial satisfies constraints, the command falls back to the best unconstrained result and records this in the output report.
