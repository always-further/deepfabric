# topic inspect

The `topic inspect` command explores the structure of generated topic files without regenerating them. It auto-detects the file format and provides multiple views for navigating the hierarchy.

## Basic Usage

```bash title="Inspect a topic file"
deepfabric topic inspect topics.json
```

The command auto-detects whether the file is a tree (JSONL) or graph (JSON) and displays a summary with format, total paths, max depth, and metadata.

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--level N` | `-l` | Show topics at a specific depth (0=root, 1=children, etc.) |
| `--expand [N]` | `-e` | Show subtree from level. Use alone for all sublevels, or specify depth |
| `--all` | `-a` | Show the complete tree structure |
| `--format FMT` | `-f` | Output format: `tree` (default), `table`, or `json` |
| `--uuid` | `-u` | Show UUID/topic_id for each node |
| `--score-report PATH` |  | Score report JSON from `topic score` (graph files only) |
| `--show-pruned MODE` |  | Overlay flagged/pruned nodes from `--score-report`: `all` or `flagged-only` |

## Exploring by Level

Browse the hierarchy one level at a time:

```bash title="Level exploration"
# Show root topic
deepfabric topic inspect topics.json --level 0

# Show first-level children
deepfabric topic inspect topics.json --level 1

# Show second-level topics
deepfabric topic inspect topics.json --level 2
```

Each level shows a simple bullet list of unique topics at that depth.

## Expanding Subtrees

Combine `--level` with `--expand` to see the subtree from a given level:

```bash title="Subtree expansion"
# Show all sublevels from level 1
deepfabric topic inspect topics.json --level 1 --expand

# Show only 2 sublevels from level 1
deepfabric topic inspect topics.json --level 1 --expand 2
```

!!! info "Expand Behavior"
    Without a depth argument, `--expand` shows all sublevels. With a number, it limits the subtree depth. This is useful for large topic structures where showing everything would be overwhelming.

## Viewing the Full Structure

Display the complete topic hierarchy:

```bash title="Full tree view"
deepfabric topic inspect topics.json --all
```

This renders the entire structure as an indented tree with all paths and depth levels.

## Output Formats

=== "Tree (Default)"

    ```bash
    deepfabric topic inspect topics.json --all
    ```

    Rich tree rendering with indentation and branch lines. Best for visual exploration in the terminal.

=== "Table"

    ```bash
    deepfabric topic inspect topics.json --all --format table
    ```

    Tabular output with columns for path and depth. Useful for structured review of all paths.

=== "JSON"

    ```bash
    deepfabric topic inspect topics.json --format json
    ```

    Machine-readable output with metadata, paths, and statistics. Useful for scripting and automation.

    ```json title="Example JSON output"
    {
      "format": "graph",
      "total_paths": 12,
      "max_depth": 3,
      "metadata": {
        "total_nodes": 15,
        "has_cycles": false,
        "root_topic": "Python programming"
      }
    }
    ```

## UUID Discovery

The `--uuid` flag shows node identifiers alongside topic names:

```bash title="Show UUIDs"
# UUIDs for all nodes
deepfabric topic inspect topics.json --all --uuid

# UUIDs at a specific level
deepfabric topic inspect topics.json --level 2 --uuid
```

!!! info "UUID Types"
    - **Graph format**: Shows the actual UUID stored in each node's metadata
    - **Tree format**: Shows a 16-character hex hash derived from the full topic path

!!! tip "Use with Prune"
    UUIDs discovered here can be passed to `deepfabric topic prune --uuid` to remove specific branches. See [topic prune](topic-prune.md).

## Prune Overlay

For graph JSON files, you can overlay score output to preview which nodes would be pruned:

```bash title="Inspect with prune overlay"
deepfabric topic inspect topics.json --all \
  --score-report topics_score_report.json \
  --show-pruned all
```

Show only affected branches:

```bash
deepfabric topic inspect topics.json --all \
  --score-report topics_score_report.json \
  --show-pruned flagged-only
```

Overlay legend:

- yellow: directly flagged by thresholds
- red: would be pruned (including descendants of flagged nodes)

## Format Auto-Detection

The inspect command determines the file format automatically:

| File Content | Detected Format | Key Indicators |
|-------------|----------------|----------------|
| JSONL with `path` arrays | Tree | Line-delimited JSON objects with `"path"` field |
| JSON with `nodes` and `root_id` | Graph | Single JSON object with `"nodes"` and `"root_id"` |

Graph files display additional metadata including total node count and cycle detection status.

## Use Cases

<div class="grid cards" markdown>

-   :material-magnify: **Structure Inspection**

    ---

    Review topic hierarchy after generation to verify coverage and depth

-   :material-layers: **Level Exploration**

    ---

    Browse specific depth levels to understand topic distribution

-   :material-code-json: **Scripting**

    ---

    Use `--format json` to extract metadata for automated workflows

-   :material-identifier: **UUID Discovery**

    ---

    Find node UUIDs for targeted pruning with `topic prune`

</div>
