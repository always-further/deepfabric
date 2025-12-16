# Graph API

The Graph class provides programmatic access to graph-based topic modeling, enabling complex domain representation through networks of interconnected concepts. This experimental API supports both hierarchical relationships and cross-connections between topics in different branches.

```python
import asyncio

def consume_graph(graph):
    async def _run():
        async for _ in graph.build_async():
            pass
    asyncio.run(_run())
```

## Graph Configuration

Graph configuration is passed directly to the Graph constructor with parameters similar to trees but extended for graph-specific features:

```python
from deepfabric import Graph

graph = Graph(
    topic_prompt="Artificial intelligence research areas",
    model_name="anthropic/claude-sonnet-4-5",
    topic_system_prompt="You are mapping interconnected research concepts.",
    degree=4,           # Connections per node
    depth=3,            # Maximum distance from root
    temperature=0.8,    # Higher creativity for connections
    max_concurrent=4    # Limit concurrent LLM calls to avoid rate limits
)
```

### Parameters

**topic_prompt** (str): Central concept from which the graph expands. Should support rich interconnections.

**model** (str): model specification in `provider/model` format.

**provider** (str): provider name , e.g `openai`, `anthropic`.

**topic_system_prompt** (str): System prompt guiding both hierarchical and lateral relationship generation.

**degree** (int): Maximum connections per node, including both children and cross-connections.

**depth** (int): Maximum shortest-path distance from root to any node.

**temperature** (float): Controls creativity in connection generation. Higher values encourage more diverse relationships.

**max_concurrent** (int): Maximum number of concurrent LLM calls during graph expansion. Lower values help avoid rate limits but increase build time. Default is 4. Range: 1-20.

## Graph Class

The Graph class manages construction and manipulation of topic graph structures:

```python
import asyncio
from deepfabric import Graph

# Create and build a graph
graph = Graph(
    topic_prompt="Artificial intelligence research areas",
    model_name="anthropic/claude-sonnet-4-5",
    degree=4,
    depth=3,
    temperature=0.8
)

async def build_graph() -> None:
    async for event in graph.build_async():
        if event["event"] == "build_complete":
            print(f"Graph built with {event['nodes_count']} nodes")

asyncio.run(build_graph())

# Access graph structure
print(f"Generated {len(graph.nodes)} nodes")

# Save and visualize
graph.save("research_graph.json")
graph.visualize("research_structure")
```

### Core Methods

#### build_async()

Constructs the complete graph structure through multi-phase generation using a generator pattern:

```python
import asyncio

async def consume_events() -> None:
    async for event in graph.build_async():
        if event['event'] == 'depth_start':
            print(f"Starting depth {event['depth']} with {event['leaf_count']} nodes")
        elif event['event'] == 'node_expanded':
            print(
                f"Expanded '{event['node_topic']}' -> {event['subtopics_added']} subtopics,"
                f" {event['connections_added']} connections"
            )
        elif event['event'] == 'build_complete':
            print(
                f"Graph complete! {event['nodes_count']} nodes,"
                f" {event.get('failed_generations', 0)} failures"
            )

asyncio.run(consume_events())
```

To run the build without handling progress events:

```python
async def build_silently() -> None:
    async for _ in graph.build_async():
        pass

asyncio.run(build_silently())
```

**Yields**: Progress events with the following types:
- `depth_start`: Beginning depth level processing
- `node_expanded`: Node expansion completed
- `depth_complete`: Depth level finished
- `build_complete`: Graph construction finished
- `error`: Build error occurred

The build process includes hierarchical construction followed by cross-connection analysis. The async generator pattern provides real-time progress monitoring or silent consumption.

#### save(filepath: str)

Persists graph structure in JSON format preserving nodes, edges, and metadata:

```python
graph.save("domain_graph.json")
```

Output format includes complete structural information:

```json
{
  "nodes": {
    "node_id": {
      "prompt": "Node topic",
      "children": ["child1", "child2"],
      "connections": ["related_node"],
      "depth": 2
    }
  },
  "edges": [
    {"from": "parent", "to": "child", "type": "hierarchical"},
    {"from": "node1", "to": "node2", "type": "cross_connection"}
  ]
}
```

#### load(filepath: str)

Reconstructs graph from previously saved JSON files:

```python
graph = Graph(
    topic_prompt="Default prompt",
    model_name="anthropic/claude-sonnet-4-5"
)
graph.load("existing_graph.json")
```

#### from_json(filepath: str, **kwargs)

Class method for loading graphs with specific configuration:

```python
graph = Graph.from_json(
    "saved_graph.json",
    topic_prompt="Research areas",
    model_name="anthropic/claude-sonnet-4-5"
)
```

#### visualize(output_path: str)

Generates SVG visualization of the graph structure:

```python
graph.visualize("analysis/domain_map")
```

Creates `domain_map.svg` showing nodes, hierarchical relationships, and cross-connections with distinct visual styling.

### Graph Analysis

Access structural information through analysis methods:

```python
# Basic statistics
node_count = len(graph.nodes)
edge_count = len(graph.edges)

# Connection analysis
hierarchical_edges = [e for e in graph.edges if e["type"] == "hierarchical"]
cross_connections = [e for e in graph.edges if e["type"] == "cross_connection"]

# Path analysis
shortest_paths = graph.find_shortest_paths()
centrality_scores = graph.calculate_centrality()
```

### Advanced Construction

#### Phase-by-Phase Building

Control graph construction through individual phases:

```python
graph = Graph(
    topic_prompt="Complex domain",
    model_name="anthropic/claude-sonnet-4-5",
    degree=4,
    depth=3
)
graph.build_hierarchical_structure()  # Create tree backbone
graph.analyze_connections()           # Find potential cross-connections
graph.create_cross_connections()      # Add lateral relationships
graph.validate_structure()            # Ensure acyclic property
```

#### Custom Connection Logic

Implement domain-specific connection strategies:

```python
def connection_filter(node1, node2, relationship_strength):
    # Custom logic for determining valid connections
    return relationship_strength > 0.7 and not creates_cycle(node1, node2)

graph.set_connection_filter(connection_filter)
consume_graph(graph)
```

#### Connection Strength Tuning

Adjust parameters controlling cross-connection generation:

```python
graph.set_connection_parameters(
    min_similarity=0.6,        # Minimum semantic similarity
    max_connections_per_node=3, # Limit connections per node
    prefer_distant_connections=True  # Favor connections across distant branches
)
```

## Graph Navigation

Navigate complex graph structures through specialized methods:

```python
# Find all paths between nodes
paths = graph.find_all_paths("node1", "node2")

# Get connected components
components = graph.get_connected_components()

# Analyze node relationships
neighbors = graph.get_neighbors("node_id")
related_concepts = graph.get_cross_connected_nodes("node_id")

# Depth-based queries
nodes_at_depth = graph.get_nodes_at_depth(2)
max_depth = graph.get_maximum_depth()
```

## Integration with Dataset Generation

Graphs integrate seamlessly with dataset generation:

```python
# Generate dataset from graph
generator = DataSetGenerator(
    instructions="Create interconnected explanations",
    model_name="anthropic/claude-sonnet-4-5",
    temperature=0.7
)
dataset = generator.create_data(
    topic_model=graph,
    num_steps=150,
    batch_size=5
)

# Graph-aware topic sampling
sampler = graph.create_balanced_sampler()  # Ensures cross-connection coverage
dataset = generator.create_data(
    topic_model=graph,
    topic_sampler=sampler,
    num_steps=100
)
```

## Error Handling

Graph-specific error handling addresses connectivity and structure issues:

```python
from deepfabric import GraphError, CyclicGraphError

try:
    consume_graph(graph)
except CyclicGraphError as e:
    print(f"Cycle detected: {e.cycle_path}")
except GraphError as e:
    print(f"Graph construction failed: {e}")
```

## Performance Considerations

Graph construction is more computationally intensive than tree generation:

```python
# Monitor construction progress
graph.enable_progress_monitoring(verbose=True)
consume_graph(graph)

# Optimize for large graphs
graph.set_batch_size(smaller_batch)  # Reduce memory usage
graph.enable_incremental_saves(checkpoint_frequency=50)  # Regular checkpointing
```

Graph complexity scales quadratically with node count during connection analysis, making parameter selection important for large-scale generation.