# topic

The `topic` command group provides tools for inspecting and managing topic structures after generation. These commands work with both tree (JSONL) and graph (JSON) formats.

## Subcommands

<div class="grid cards" markdown>

-   :material-file-search: **topic inspect**

    ---

    Explore topic structure, browse levels, and discover node UUIDs

    [:octicons-arrow-right-24: Reference](topic-inspect.md)

-   :material-content-cut: **topic prune**

    ---

    Remove branches or depth levels from topic graphs

    [:octicons-arrow-right-24: Reference](topic-prune.md)

-   :material-chart-line: **topic score**

    ---

    Score topic graphs with GTD/LTD quality metrics

    [:octicons-arrow-right-24: Reference](topic-score.md)

-   :material-tune-variant: **topic optimize-thresholds**

    ---

    Search for better GTD/LTD cutoff values on an existing graph

    [:octicons-arrow-right-24: Reference](topic-optimize-thresholds.md)

</div>

## Workflow

Topic management fits naturally into the generation workflow:

```bash title="Generate, inspect, and refine"
# 1. Generate topic structure
deepfabric generate config.yaml --topic-only

# 2. Inspect the result
deepfabric topic inspect topics.json --level 1 --expand

# 3. Score graph quality
deepfabric topic score topics.json

# 4. Prune if needed (preview first)
deepfabric topic prune topics.json --level 2 --dry-run
deepfabric topic prune topics.json --level 2 -o refined_topics.json

# 5. Verify the pruned structure
deepfabric topic inspect refined_topics.json --all

# 6. Generate dataset from refined topics
deepfabric generate config.yaml --topics-load refined_topics.json
```

!!! tip "Iterative Refinement"
    Use `inspect` and `prune` iteratively to shape your topic structure before committing to dataset generation. This avoids wasting API calls on unwanted branches.
