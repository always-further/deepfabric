import asyncio
import json
import tempfile

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest  # type: ignore

from deepfabric.graph import (
    Graph,
    GraphConfig,
    GraphModel,
    Node,
    NodeModel,
)


@pytest.fixture
def topic_graph_params():
    """Fixture for Graph parameters."""
    return {
        "topic_prompt": "Test root topic",
        "provider": "openai",
        "model_name": "test-model",
        "temperature": 0.7,
        "degree": 3,
        "depth": 2,
    }


@pytest.fixture
def topic_graph(topic_graph_params):
    """Fixture for Graph instance."""
    with patch("deepfabric.graph.LLMClient"):
        return Graph(**topic_graph_params)


async def _consume_async_iter(async_iterable):
    async for _ in async_iterable:
        pass


class TestNode:
    """Tests for Node class."""

    def test_node_initialization(self):
        """Test Node initialization."""
        node = Node("Test topic", 42)
        assert node.topic == "Test topic"
        assert node.id == 42  # noqa: PLR2004
        assert node.children == []
        assert node.parents == []

    def test_node_to_pydantic(self):
        """Test Node to Pydantic conversion."""
        parent = Node("Parent", 1)
        child = Node("Child", 2)
        parent.children.append(child)
        child.parents.append(parent)

        parent_model = parent.to_pydantic()
        assert isinstance(parent_model, NodeModel)
        assert parent_model.id == 1
        assert parent_model.topic == "Parent"
        assert parent_model.children == [2]
        assert parent_model.parents == []

        child_model = child.to_pydantic()
        assert child_model.id == 2  # noqa: PLR2004
        assert child_model.topic == "Child"
        assert child_model.children == []
        assert child_model.parents == [1]


class TestGraphConfig:
    """Tests for GraphConfig model."""

    def test_valid_arguments(self):
        """Test creation with valid arguments."""
        config = GraphConfig(
            topic_prompt="Test", model_name="gpt-4", temperature=0.5, degree=2, depth=3
        )
        assert config.topic_prompt == "Test"
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.5  # noqa: PLR2004
        assert config.depth == 3  # noqa: PLR2004

    def test_default_arguments(self):
        """Test default values."""
        config = GraphConfig(
            topic_prompt="Test",
            model_name="openai/gpt-4o",
            temperature=0.2,
            degree=3,
            depth=2,
        )
        assert config.model_name == "openai/gpt-4o"  # From constants
        assert config.temperature == 0.2  # From constants  # noqa: PLR2004
        assert config.degree == 3  # noqa: PLR2004
        assert config.depth == 2  # noqa: PLR2004

    def test_invalid_arguments(self):
        """Test validation errors."""
        with pytest.raises(ValueError):
            GraphConfig(
                topic_prompt="",  # Empty topic_prompt
                model_name="openai/gpt-4o",
                temperature=0.2,
                degree=3,
                depth=2,
            )

        with pytest.raises(ValueError):
            GraphConfig(
                topic_prompt="Test",
                model_name="openai/gpt-4o",
                temperature=-0.1,
                degree=3,
                depth=2,
            )  # Invalid temperature

        with pytest.raises(ValueError):
            GraphConfig(
                topic_prompt="Test",
                model_name="openai/gpt-4o",
                temperature=-0.1,
                degree=0,
                depth=2,
            )  # Invalid degree


class TestGraph:
    """Tests for Graph class."""

    def test_initialization(self, topic_graph_params):
        """Test Graph initialization."""
        with patch("deepfabric.graph.LLMClient"):
            graph = Graph(**topic_graph_params)
        assert graph.config.topic_prompt == topic_graph_params["topic_prompt"]
        assert graph.root.topic == "Test root topic"
        assert graph.root.id == 0
        assert len(graph.nodes) == 1
        assert graph._next_node_id == 1
        assert graph.failed_generations == []

    def test_add_node(self, topic_graph):
        """Test adding nodes to the graph."""
        node1 = topic_graph.add_node("First node")
        assert node1.id == 1
        assert node1.topic == "First node"
        assert len(topic_graph.nodes) == 2  # noqa: PLR2004

        node2 = topic_graph.add_node("Second node")
        assert node2.id == 2  # noqa: PLR2004
        assert topic_graph._next_node_id == 3  # noqa: PLR2004

    def test_add_edge(self, topic_graph):
        """Test adding edges between nodes."""
        node1 = topic_graph.add_node("Child 1")
        node2 = topic_graph.add_node("Child 2")

        # Add edge from root to node1
        topic_graph.add_edge(0, node1.id)
        assert node1 in topic_graph.root.children
        assert topic_graph.root in node1.parents

        # Add edge from node1 to node2
        topic_graph.add_edge(node1.id, node2.id)
        assert node2 in node1.children
        assert node1 in node2.parents

        # Test duplicate edge prevention
        topic_graph.add_edge(0, node1.id)
        assert topic_graph.root.children.count(node1) == 1

    def test_add_edge_invalid_nodes(self, topic_graph):
        """Test adding edges with invalid node IDs."""
        topic_graph.add_edge(0, 999)  # Non-existent child
        assert len(topic_graph.root.children) == 0

        topic_graph.add_edge(999, 0)  # Non-existent parent
        assert len(topic_graph.root.parents) == 0

    def test_to_pydantic(self, topic_graph):
        """Test conversion to Pydantic model."""
        node1 = topic_graph.add_node("Child")
        topic_graph.add_edge(0, node1.id)

        pydantic_model = topic_graph.to_pydantic()
        assert isinstance(pydantic_model, GraphModel)
        assert pydantic_model.root_id == 0
        assert len(pydantic_model.nodes) == 2  # noqa: PLR2004
        assert pydantic_model.nodes[0].topic == "Test root topic"
        assert pydantic_model.nodes[1].topic == "Child"

    def test_to_json(self, topic_graph):
        """Test JSON serialization."""
        node1 = topic_graph.add_node("Child")
        topic_graph.add_edge(0, node1.id)

        json_str = topic_graph.to_json()
        data = json.loads(json_str)
        assert data["root_id"] == 0
        assert len(data["nodes"]) == 2  # noqa: PLR2004

    def test_save_and_load(self, topic_graph_params):
        """Test saving and loading a graph."""
        # Create and populate a graph
        with patch("deepfabric.graph.LLMClient"):
            graph = Graph(**topic_graph_params)
        node1 = graph.add_node("Child 1")
        node2 = graph.add_node("Child 2")
        graph.add_edge(0, node1.id)
        graph.add_edge(node1.id, node2.id)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            graph.save(temp_path)

            # Load the graph
            with patch("deepfabric.graph.LLMClient"):
                loaded_graph = Graph.from_json(temp_path, topic_graph_params)

            # Verify structure
            assert len(loaded_graph.nodes) == 3  # noqa: PLR2004
            assert loaded_graph.root.id == 0
            assert loaded_graph.root.topic == "Test root topic"
            assert len(loaded_graph.root.children) == 1
            assert loaded_graph.nodes[2].topic == "Child 2"
            assert loaded_graph._next_node_id == 3  # noqa: PLR2004
        finally:
            Path(temp_path).unlink()

    def test_wrap_text(self, topic_graph):
        """Test text wrapping utility."""
        long_text = "This is a very long text that should be wrapped at the specified width"
        wrapped = topic_graph._wrap_text(long_text, width=20)
        lines = wrapped.split("\n")
        assert all(len(line) <= 20 for line in lines)  # noqa: PLR2004

    def test_get_all_paths_simple(self, topic_graph):
        """Test getting all paths in a simple tree."""
        node1 = topic_graph.add_node("Child 1")
        node2 = topic_graph.add_node("Child 2")
        node3 = topic_graph.add_node("Grandchild")

        topic_graph.add_edge(0, node1.id)
        topic_graph.add_edge(0, node2.id)
        topic_graph.add_edge(node1.id, node3.id)

        paths = topic_graph.get_all_paths()
        expected_paths = [
            ["Test root topic", "Child 1", "Grandchild"],
            ["Test root topic", "Child 2"],
        ]
        assert sorted(paths) == sorted(expected_paths)

    def test_has_cycle_no_cycle(self, topic_graph):
        """Test cycle detection with no cycles."""
        node1 = topic_graph.add_node("Child 1")
        node2 = topic_graph.add_node("Child 2")
        node3 = topic_graph.add_node("Grandchild")

        topic_graph.add_edge(0, node1.id)
        topic_graph.add_edge(0, node2.id)
        topic_graph.add_edge(node1.id, node3.id)

        assert topic_graph.has_cycle() is False

    def test_has_cycle_with_cycle(self, topic_graph):
        """Test cycle detection with a cycle."""
        node1 = topic_graph.add_node("Child 1")
        node2 = topic_graph.add_node("Child 2")

        topic_graph.add_edge(0, node1.id)
        topic_graph.add_edge(node1.id, node2.id)
        topic_graph.add_edge(node2.id, node1.id)  # Create cycle

        assert topic_graph.has_cycle() is True

    def test_get_subtopics_and_connections_success(self, topic_graph):
        """Test successful subtopic generation."""
        # Mock the generate method on the topic_graph's llm_client instance
        from deepfabric.schemas import GraphSubtopic, GraphSubtopics  # noqa: PLC0415

        topic_graph.llm_client.generate_async = AsyncMock(
            return_value=GraphSubtopics(
                subtopics=[
                    GraphSubtopic(topic="Subtopic 1", connections=[]),
                    GraphSubtopic(topic="Subtopic 2", connections=[0]),
                ]
            )
        )

        # Generate subtopics
        asyncio.run(topic_graph.get_subtopics_and_connections(topic_graph.root, 2))

        # Verify nodes were added
        assert len(topic_graph.nodes) == 3  # noqa: PLR2004
        assert any(node.topic == "Subtopic 1" for node in topic_graph.nodes.values())
        assert any(node.topic == "Subtopic 2" for node in topic_graph.nodes.values())

        # Verify connections
        subtopic2 = next(node for node in topic_graph.nodes.values() if node.topic == "Subtopic 2")
        assert topic_graph.root in subtopic2.parents  # Connection to root

    def test_get_subtopics_and_connections_retry(self, topic_graph):
        """Test subtopic generation with retries."""
        # Mock success - LLMClient handles retries internally, so Graph only sees the final result
        from deepfabric.schemas import GraphSubtopic, GraphSubtopics  # noqa: PLC0415

        topic_graph.llm_client.generate_async = AsyncMock(
            return_value=GraphSubtopics(subtopics=[GraphSubtopic(topic="Subtopic", connections=[])])
        )

        asyncio.run(topic_graph.get_subtopics_and_connections(topic_graph.root, 1))

        # Verify node was added
        assert len(topic_graph.nodes) == 2  # noqa: PLR2004

    def test_get_subtopics_and_connections_max_retries(
        self,
        topic_graph,
        capsys,  # noqa: ARG002
    ):  # noqa: ARG002
        """Test subtopic generation hitting max retries."""
        # All calls fail
        topic_graph.llm_client.generate_async = AsyncMock(side_effect=Exception("API Error"))

        asyncio.run(topic_graph.get_subtopics_and_connections(topic_graph.root, 1))

        # Verify failure was recorded
        assert len(topic_graph.failed_generations) == 1
        assert topic_graph.failed_generations[0]["node_id"] == 0
        assert "API Error" in topic_graph.failed_generations[0]["last_error"]

    def test_build(self, topic_graph):
        """Test building the entire graph."""
        # Mock responses for each call
        from deepfabric.schemas import GraphSubtopic, GraphSubtopics  # noqa: PLC0415

        topic_graph.llm_client.generate_async = AsyncMock(
            return_value=GraphSubtopics(
                subtopics=[
                    GraphSubtopic(topic=f"Topic {i}", connections=[])
                    for i in range(topic_graph.degree)
                ]
            )
        )

        asyncio.run(_consume_async_iter(topic_graph.build_async()))

        # With depth=2 and degree=3, we should have:
        # 1 root + 3 children + (3 * 3) grandchildren = 13 nodes
        assert len(topic_graph.nodes) == 13  # noqa: PLR2004


class TestIntegration:
    """Integration tests for the Graph system."""

    def test_complex_graph_creation(self):
        """Test creating a complex graph with cross-connections."""
        graph_params = {
            "topic_prompt": "Machine Learning",
            "provider": "openai",
            "model_name": "test-model",
            "temperature": 0.7,
            "degree": 2,
            "depth": 2,
        }
        with patch("deepfabric.graph.LLMClient"):
            graph = Graph(**graph_params)

        # First level response
        _first_response = {
            "subtopics": [
                {"topic": "Supervised Learning", "connections": []},
                {"topic": "Unsupervised Learning", "connections": []},
            ]
        }

        # Second level responses with cross-connections
        _supervised_response = {
            "subtopics": [
                {"topic": "Classification", "connections": []},
                {"topic": "Regression", "connections": [2]},  # Connect to Unsupervised
            ]
        }

        _unsupervised_response = {
            "subtopics": [
                {"topic": "Clustering", "connections": [1]},  # Connect to Supervised
                {"topic": "Dimensionality Reduction", "connections": []},
            ]
        }

        # Set up mock to return different responses
        from deepfabric.schemas import GraphSubtopic, GraphSubtopics  # noqa: PLC0415

        graph.llm_client.generate_async = AsyncMock(
            side_effect=[
                GraphSubtopics(
                    subtopics=[
                        GraphSubtopic(topic="Supervised Learning", connections=[]),
                        GraphSubtopic(topic="Unsupervised Learning", connections=[]),
                    ]
                ),
                GraphSubtopics(
                    subtopics=[
                        GraphSubtopic(topic="Classification", connections=[]),
                        GraphSubtopic(
                            topic="Regression", connections=[2]
                        ),  # Connect to Unsupervised
                    ]
                ),
                GraphSubtopics(
                    subtopics=[
                        GraphSubtopic(topic="Clustering", connections=[1]),  # Connect to Supervised
                        GraphSubtopic(topic="Dimensionality Reduction", connections=[]),
                    ]
                ),
            ]
        )

        asyncio.run(_consume_async_iter(graph.build_async()))

        # Verify structure
        assert len(graph.nodes) == 7  # 1 root + 2 level1 + 4 level2  # noqa: PLR2004

        # Verify cross-connections exist
        regression_node = next(node for node in graph.nodes.values() if node.topic == "Regression")
        unsupervised_node = next(
            node for node in graph.nodes.values() if node.topic == "Unsupervised Learning"
        )
        assert unsupervised_node in regression_node.parents

        clustering_node = next(node for node in graph.nodes.values() if node.topic == "Clustering")
        supervised_node = next(
            node for node in graph.nodes.values() if node.topic == "Supervised Learning"
        )
        assert supervised_node in clustering_node.parents

    def test_graph_persistence_roundtrip(self):
        """Test complete save/load roundtrip with complex graph."""
        graph_params = {
            "topic_prompt": "Science",
            "model_name": "test-model",
            "temperature": 0.5,
            "degree": 2,
            "depth": 2,
        }
        graph = Graph(**graph_params)

        # Build a complex graph manually
        physics = graph.add_node("Physics")
        chemistry = graph.add_node("Chemistry")
        biology = graph.add_node("Biology")
        quantum = graph.add_node("Quantum Mechanics")
        organic = graph.add_node("Organic Chemistry")

        graph.add_edge(0, physics.id)
        graph.add_edge(0, chemistry.id)
        graph.add_edge(0, biology.id)
        graph.add_edge(physics.id, quantum.id)
        graph.add_edge(chemistry.id, organic.id)
        graph.add_edge(chemistry.id, biology.id)  # Cross-connection

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save and load
            graph.save(temp_path)
            loaded = Graph.from_json(temp_path, graph_params)

            # Verify complete structure
            assert len(loaded.nodes) == len(graph.nodes)
            assert loaded.has_cycle() == graph.has_cycle()
            assert sorted(loaded.get_all_paths()) == sorted(graph.get_all_paths())

            # Verify specific relationships
            loaded_chemistry = loaded.nodes[chemistry.id]
            assert len(loaded_chemistry.children) == 2  # noqa: PLR2004
            assert any(child.topic == "Biology" for child in loaded_chemistry.children)
        finally:
            Path(temp_path).unlink()
