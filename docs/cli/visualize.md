# visualize

The `visualize` command creates SVG diagrams of topic graphs, enabling visual exploration of domain structures and relationship patterns.

!!! info "Graph Visualization"
    This capability is particularly valuable for topic graphs where cross-connections between concepts create complex structures that benefit from visual representation.

## Basic Usage

Create an SVG visualization from a topic graph file:

```bash title="Basic visualization"
deepfabric visualize topic_graph.json --output domain_structure
```

This command reads the graph structure and generates `domain_structure.svg` showing nodes as topics and edges as relationships between concepts.

!!! note "Required Option"
    The `--output` (`-o`) option is required.

## Graph File Requirements

The visualize command works with JSON files containing topic graph structures:

```json title="topic_graph.json"
{
  "nodes": {
    "root": {"prompt": "Root concept", "children": ["child1", "child2"]},
    "child1": {"prompt": "First subtopic", "connections": ["child2"]}
  },
  "edges": [
    {"from": "root", "to": "child1", "type": "hierarchical"},
    {"from": "child1", "to": "child2", "type": "cross_connection"}
  ]
}
```

The visualization interprets both hierarchical parent-child relationships and cross-connections between topics.

## Output Options

Control the output file location and naming:

```bash title="Custom output paths"
# Specify custom output location
deepfabric visualize research_graph.json --output analysis/research_structure

# Use different naming pattern
deepfabric visualize ml_topics.json -o visualizations/ml_domain_map
```

!!! info "Automatic Extension"
    The command automatically appends `.svg` to the output filename, creating scalable vector graphics suitable for both screen viewing and print output.

## Visual Elements

The generated diagrams use distinct visual elements:

:material-checkbox-blank: **Node Representation**
:   Topics displayed as labeled boxes with the seed prompt text, sized according to hierarchy position.

:material-arrow-right: **Hierarchical Edges**
:   Solid lines connecting parent topics to direct children, showing primary organizational structure.

:material-arrow-decision: **Cross-Connections**
:   Dashed lines linking related concepts across different hierarchical branches.

:material-star: **Root Highlighting**
:   Distinctive formatting for the root node to identify the domain's central concept.

## Use Cases

<div class="grid cards" markdown>

-   :material-magnify: **Domain Analysis**

    ---

    Reveals structure and balance of topic coverage, identifying dense interconnections versus isolated branches.

-   :material-check-circle: **Quality Assessment**

    ---

    Visual inspection of relationships to identify illogical connections or missing relationships.

-   :material-presentation: **Stakeholder Communication**

    ---

    Intuitive visual representations of complex domain structures for presentations.

-   :material-sync: **Iterative Development**

    ---

    Visualize effects of different configuration choices during refinement.

</div>

## Large Graph Considerations

Complex topic graphs with many nodes benefit from several visualization strategies:

:material-layers: **Hierarchical Layout**
:   Organizes nodes by distance from root, creating clear visual layers.

:material-filter: **Connection Filtering**
:   Emphasizes important relationships while de-emphasizing less significant connections.

:material-focus-field: **Subgraph Focus**
:   Visualize specific portions of large graphs for detailed analysis.

## Integration with Generation Workflow

Visualization fits naturally into the iterative development process:

```bash title="Complete workflow"
# Generate topic graph
deepfabric generate research-config.yaml

# Visualize the structure
deepfabric visualize research_graph.json --output research_analysis

# Review visualization and adjust parameters
# Regenerate with modified configuration
```

!!! tip "Rapid Iteration"
    This workflow enables rapid iteration on topic generation parameters with immediate visual feedback on structural changes.

## File Format Compatibility

!!! warning "JSON Only"
    The visualization command specifically requires JSON files in the DeepFabric topic graph format. JSONL topic tree files are not directly compatible.

## Technical Specifications

Generated SVG files are self-contained vector graphics that:

- :material-resize: Scale cleanly to any resolution without quality loss
- :material-text-search: Include embedded text that remains searchable and selectable
- :material-web: Work in modern web browsers, vector graphics applications, and print workflows
- :material-palette: Use standard web colors and fonts for maximum compatibility

??? tip "Visualization Best Practices"
    Generate visualizations after topic graph creation but before dataset generation to validate topic coverage and relationships. Use visualizations to identify overly dense or sparse areas in your domain coverage.
