# topic optimize-thresholds

The `topic optimize-thresholds` command searches for better GTD/LTD cutoff values on an existing graph.

It does **not** regenerate the graph. It re-scores the same graph across multiple threshold combinations.

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
  --depth1-gtd-min 0.10 --depth1-gtd-max 0.50 \
  --gtd-neg-min -0.10 --gtd-neg-max 0.10 \
  --ltd-min 0.10 --ltd-max 0.50 \
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
