import asyncio
import json
import textwrap

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .constants import (
    DEFAULT_MAX_TOKENS,
    MAX_RETRY_ATTEMPTS,
    TOPIC_GRAPH_DEFAULT_MODEL,
    TOPIC_GRAPH_DEFAULT_TEMPERATURE,
    TOPIC_GRAPH_SUMMARY,
)
from .llm import LLMClient
from .metrics import trace
from .prompts import GRAPH_EXPANSION_PROMPT
from .schemas import GraphSubtopics
from .stream_simulator import simulate_stream
from .topic_model import TopicModel

if TYPE_CHECKING:  # only for type hints to avoid runtime cycles
    from .progress import ProgressReporter


def validate_graph_response(response_text: str) -> dict[str, Any] | None:
    """Clean and validate the JSON response for the graph from the LLM."""
    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse the input string as JSON.\n{e}")
        return None


class GraphConfig(BaseModel):
    """Configuration for constructing a topic graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    topic_prompt: str = Field(
        ..., min_length=1, description="The initial prompt to start the topic graph"
    )
    topic_system_prompt: str = Field(
        default="", description="System prompt for topic exploration and generation"
    )
    provider: str = Field(
        default="ollama",
        min_length=1,
        description="LLM provider (openai, anthropic, gemini, ollama)",
    )
    model_name: str = Field(
        default=TOPIC_GRAPH_DEFAULT_MODEL,
        min_length=1,
        description="The name of the model to be used",
    )
    temperature: float = Field(
        default=TOPIC_GRAPH_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    degree: int = Field(default=3, ge=1, le=10, description="The branching factor of the graph")
    depth: int = Field(default=2, ge=1, le=5, description="The depth of the graph")
    max_concurrent: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum concurrent LLM calls during graph expansion (helps avoid rate limits)",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for API endpoint (e.g., custom OpenAI-compatible servers)",
    )


class NodeModel(BaseModel):
    """Pydantic model for a node in the graph."""

    id: int
    topic: str
    children: list[int] = Field(default_factory=list)
    parents: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphModel(BaseModel):
    """Pydantic model for the entire topic graph."""

    nodes: dict[int, NodeModel]
    root_id: int


class Node:
    """Represents a node in the Graph for runtime manipulation."""

    def __init__(self, topic: str, node_id: int, metadata: dict[str, Any] | None = None):
        self.topic: str = topic
        self.id: int = node_id
        self.children: list[Node] = []
        self.parents: list[Node] = []
        self.metadata: dict[str, Any] = metadata.copy() if metadata is not None else {}

    def to_pydantic(self) -> NodeModel:
        """Converts the runtime Node to its Pydantic model representation."""
        return NodeModel(
            id=self.id,
            topic=self.topic,
            children=[child.id for child in self.children],
            parents=[parent.id for parent in self.parents],
            metadata=self.metadata,
        )


class Graph(TopicModel):
    """Represents the topic graph and manages its structure."""

    def __init__(self, **kwargs):
        try:
            self.config = GraphConfig.model_validate(kwargs)
        except Exception as e:
            raise ValueError(f"Invalid graph configuration: {str(e)}") from e  # noqa: TRY003

        # Initialize from config
        self.topic_prompt = self.config.topic_prompt
        self.model_system_prompt = self.config.topic_system_prompt
        self.provider = self.config.provider
        self.model_name = self.config.model_name
        self.temperature = self.config.temperature
        self.degree = self.config.degree
        self.depth = self.config.depth
        self.max_concurrent = self.config.max_concurrent

        # Initialize LLM client
        llm_kwargs = {}
        if self.config.base_url:
            llm_kwargs["base_url"] = self.config.base_url

        self.llm_client = LLMClient(
            provider=self.provider,
            model_name=self.model_name,
            **llm_kwargs,
        )

        # Progress reporter for streaming feedback (set by topic_manager)
        self.progress_reporter: ProgressReporter | None = None

        trace(
            "graph_created",
            {
                "provider": self.provider,
                "model_name": self.model_name,
                "degree": self.degree,
                "depth": self.depth,
            },
        )

        self.root: Node = Node(self.config.topic_prompt, 0)
        self.nodes: dict[int, Node] = {0: self.root}
        self._next_node_id: int = 1
        self.failed_generations: list[dict[str, Any]] = []

    def _wrap_text(self, text: str, width: int = 30) -> str:
        """Wrap text to a specified width."""
        return "\n".join(textwrap.wrap(text, width=width))

    def add_node(self, topic: str, metadata: dict[str, Any] | None = None) -> Node:
        """Adds a new node to the graph."""
        node = Node(topic, self._next_node_id, metadata)
        self.nodes[node.id] = node
        self._next_node_id += 1
        return node

    def add_edge(self, parent_id: int, child_id: int) -> None:
        """Adds a directed edge from a parent to a child node, avoiding duplicates."""
        parent_node = self.nodes.get(parent_id)
        child_node = self.nodes.get(child_id)
        if parent_node and child_node:
            if child_node not in parent_node.children:
                parent_node.children.append(child_node)
            if parent_node not in child_node.parents:
                child_node.parents.append(parent_node)

    def to_pydantic(self) -> GraphModel:
        """Converts the runtime graph to its Pydantic model representation."""
        return GraphModel(
            nodes={node_id: node.to_pydantic() for node_id, node in self.nodes.items()},
            root_id=self.root.id,
        )

    def to_json(self) -> str:
        """Returns a JSON representation of the graph."""
        pydantic_model = self.to_pydantic()
        return pydantic_model.model_dump_json(indent=2)

    def save(self, save_path: str) -> None:
        """Save the topic graph to a file."""
        with open(save_path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, json_path: str, params: dict) -> "Graph":
        """Load a topic graph from a JSON file."""
        with open(json_path) as f:
            data = json.load(f)

        graph_model = GraphModel(**data)
        graph = cls(**params)
        graph.nodes = {}

        # Create nodes
        for node_model in graph_model.nodes.values():
            node = Node(node_model.topic, node_model.id, node_model.metadata)
            graph.nodes[node.id] = node
            if node.id == graph_model.root_id:
                graph.root = node

        # Create edges
        for node_model in graph_model.nodes.values():
            for child_id in node_model.children:
                graph.add_edge(node_model.id, child_id)

        graph._next_node_id = max(graph.nodes.keys()) + 1
        return graph

    def visualize(self, save_path: str) -> None:
        """Visualize the graph and save it to a file."""
        try:
            from mermaid import Mermaid  # noqa: PLC0415
        except ImportError:
            print("Please install mermaid-py to visualize the graph: uv add mermaid-py")
            return

        graph_definition = "graph TD\n"
        for node in self.nodes.values():
            graph_definition += f'    {node.id}["{self._wrap_text(node.topic)}"]\n'

        for node in self.nodes.values():
            for child in node.children:
                graph_definition += f"    {node.id} --> {child.id}\n"

        mermaid = Mermaid(graph_definition)
        mermaid.to_svg(f"{save_path}.svg")

    async def build_async(self):
        """Builds the graph by iteratively calling the LLM to get subtopics and connections.

        Yields:
            dict: Progress events with event type and associated data
        """

        def _raise_if_build_failed():
            """Check if build failed completely and raise appropriate error."""
            if len(self.nodes) == 1 and self.failed_generations:
                # Surface the actual first error instead of a generic message
                first_error = self.failed_generations[0]["last_error"]
                raise RuntimeError(first_error)

        try:
            for depth in range(self.depth):
                leaf_nodes = [node for node in self.nodes.values() if not node.children]
                yield {"event": "depth_start", "depth": depth + 1, "leaf_count": len(leaf_nodes)}

                if leaf_nodes:
                    # Use semaphore to limit concurrent LLM calls and avoid rate limits
                    semaphore = asyncio.Semaphore(self.max_concurrent)

                    async def bounded_expand(
                        node: Node, sem: asyncio.Semaphore = semaphore
                    ) -> tuple[int, int]:
                        async with sem:
                            return await self.get_subtopics_and_connections(node, self.degree)

                    tasks = [bounded_expand(node) for node in leaf_nodes]
                    results = await asyncio.gather(*tasks)

                    for node, (subtopics_added, connections_added) in zip(
                        leaf_nodes, results, strict=True
                    ):
                        yield {
                            "event": "node_expanded",
                            "node_topic": node.topic,
                            "subtopics_added": subtopics_added,
                            "connections_added": connections_added,
                        }

                yield {"event": "depth_complete", "depth": depth + 1}

            # Check if build was completely unsuccessful (only root node exists)
            _raise_if_build_failed()

            trace(
                "graph_built",
                {
                    "provider": self.provider,
                    "model_name": self.model_name,
                    "nodes_count": len(self.nodes),
                    "failed_generations": len(self.failed_generations),
                    "success": len(self.nodes) > 1,
                },
            )

            yield {
                "event": "build_complete",
                "nodes_count": len(self.nodes),
                "failed_generations": len(self.failed_generations),
            }

        except Exception as e:
            yield {"event": "error", "error": str(e)}
            raise

    async def get_subtopics_and_connections(  # noqa: PLR0912
        self, parent_node: Node, num_subtopics: int
    ) -> tuple[int, int]:
        """Generate subtopics and connections for a given node. Returns (subtopics_added, connections_added)."""
        subtopics_added = 0
        connections_added = 0
        graph_summary = (
            self.to_json()
            if len(self.nodes) <= TOPIC_GRAPH_SUMMARY
            else "Graph too large to display"
        )

        graph_prompt = GRAPH_EXPANSION_PROMPT.replace("{{current_graph_summary}}", graph_summary)
        graph_prompt = graph_prompt.replace("{{current_topic}}", parent_node.topic)
        graph_prompt = graph_prompt.replace("{{num_subtopics}}", str(num_subtopics))

        last_error = None
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # Always use non-streaming for reliable structured output
                response = await self.llm_client.generate_async(
                    prompt=graph_prompt,
                    schema=GraphSubtopics,
                    max_retries=1,  # Don't retry inside - we handle it here
                    max_tokens=DEFAULT_MAX_TOKENS,
                    temperature=self.temperature,
                )

                # Fire-and-forget: simulate streaming for TUI preview (non-blocking)
                simulate_stream(
                    self.progress_reporter,
                    response.model_dump_json(indent=2),
                    source="graph_generation",
                )

                # Process structured response
                for subtopic_data in response.subtopics:
                    new_node = self.add_node(subtopic_data.topic)
                    self.add_edge(parent_node.id, new_node.id)
                    subtopics_added += 1
                    for connection_id in subtopic_data.connections:
                        if connection_id in self.nodes:
                            self.add_edge(connection_id, new_node.id)
                            connections_added += 1

            except Exception as e:
                last_error = str(e)
                # Check if it's an API key related error - don't retry these
                error_str = str(e).lower()
                if any(
                    keyword in error_str
                    for keyword in ["api_key", "api key", "authentication", "unauthorized"]
                ):
                    error_msg = f"Authentication failed for provider '{self.provider}'. Please set the required API key environment variable."
                    raise RuntimeError(error_msg) from e

                # Log retry attempt if not the last one
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    # Emit retry event to TUI if progress reporter available
                    if self.progress_reporter:
                        self.progress_reporter.emit_node_retry(
                            node_topic=parent_node.topic,
                            attempt=attempt + 1,
                            max_attempts=MAX_RETRY_ATTEMPTS,
                            error_summary=last_error or "Unknown error",
                        )
                    # Brief delay before retry with exponential backoff
                    delay = (2**attempt) * 0.5  # 0.5s, 1s, 2s
                    await asyncio.sleep(delay)
                    continue

            else:
                # Success - return the results
                return subtopics_added, connections_added

        # All retries exhausted - record failure
        self.failed_generations.append(
            {"node_id": parent_node.id, "attempts": MAX_RETRY_ATTEMPTS, "last_error": last_error}
        )
        return 0, 0  # No subtopics or connections added on failure

    def get_all_paths(self) -> list[list[str]]:
        """Returns all paths from the root to leaf nodes."""
        paths = []
        visited: set[int] = set()
        self._dfs_paths(self.root, [self.root.topic], paths, visited)
        return paths

    def _dfs_paths(
        self, node: Node, current_path: list[str], paths: list[list[str]], visited: set[int]
    ) -> None:
        """Helper function for DFS traversal to find all paths.

        Args:
            node: Current node being visited
            current_path: Path from root to current node
            paths: Accumulated list of complete paths
            visited: Set of node IDs already visited in current path to prevent cycles
        """
        # Prevent cycles by tracking visited nodes in the current path
        if node.id in visited:
            return

        visited.add(node.id)

        if not node.children:
            paths.append(current_path)

        for child in node.children:
            self._dfs_paths(child, current_path + [child.topic], paths, visited)

        # Remove node from visited when backtracking to allow it in other paths
        visited.remove(node.id)

    def _has_cycle_util(self, node: Node, visited: set[int], recursion_stack: set[int]) -> bool:
        """Utility function for cycle detection."""
        visited.add(node.id)
        recursion_stack.add(node.id)

        for child in node.children:
            if child.id not in visited:
                if self._has_cycle_util(child, visited, recursion_stack):
                    return True
            elif child.id in recursion_stack:
                return True

        recursion_stack.remove(node.id)
        return False

    def has_cycle(self) -> bool:
        """Checks if the graph contains a cycle."""
        visited: set[int] = set()
        recursion_stack: set[int] = set()
        for node_id in self.nodes:
            if node_id not in visited and self._has_cycle_util(
                self.nodes[node_id], visited, recursion_stack
            ):
                return True
        return False
