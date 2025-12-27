"""Integration tests for Graph with real API calls."""

import asyncio
import json

import pytest  # pyright: ignore[reportMissingImports]

from deepfabric import Graph

from .conftest import requires_gemini, requires_openai


class TestGraphOpenAI:
    """Integration tests for Graph with OpenAI provider."""

    @requires_openai
    @pytest.mark.openai
    def test_graph_builds_basic(self, openai_config):
        """Test basic graph building with OpenAI."""
        degree = 2
        depth = 1
        topic = "Machine Learning"

        graph = Graph(
            topic_prompt=topic,
            provider=openai_config["provider"],
            model_name=openai_config["model_name"],
            temperature=openai_config["temperature"],
            degree=degree,
            depth=depth,
        )

        async def run_build():
            return [event async for event in graph.build_async()]

        events = asyncio.run(run_build())

        # Verify build completion
        completes = [e for e in events if e.get("event") == "build_complete"]
        assert len(completes) == 1

        # Verify paths were built
        paths = graph.get_all_paths()
        assert len(paths) >= 1
        assert all(p[0] == topic for p in paths)

    @requires_openai
    @pytest.mark.openai
    def test_graph_save_and_load_roundtrip(self, tmp_path, openai_config):
        """Test saving and loading a graph with OpenAI."""
        degree = 2
        depth = 1
        topic = "Data Science"

        graph = Graph(
            topic_prompt=topic,
            provider=openai_config["provider"],
            model_name=openai_config["model_name"],
            temperature=openai_config["temperature"],
            degree=degree,
            depth=depth,
        )

        async def build_graph():
            async for _ in graph.build_async():
                pass

        asyncio.run(build_graph())

        # Save to file
        out_path = tmp_path / "graph.json"
        graph.save(str(out_path))

        # Verify file structure
        data = json.loads(out_path.read_text())
        assert "nodes" in data
        assert "root_id" in data
        assert len(data["nodes"]) >= 1

        # Load into new graph and verify
        graph_params = {
            "topic_prompt": topic,
            "provider": openai_config["provider"],
            "model_name": openai_config["model_name"],
            "degree": degree,
            "depth": depth,
        }
        new_graph = Graph.from_json(str(out_path), graph_params)

        assert new_graph.get_all_paths() == graph.get_all_paths()


class TestGraphGemini:
    """Integration tests for Graph with Gemini provider."""

    @requires_gemini
    @pytest.mark.gemini
    def test_graph_builds_basic(self, gemini_config):
        """Test basic graph building with Gemini."""
        degree = 2
        depth = 1
        topic = "Cloud Computing"

        graph = Graph(
            topic_prompt=topic,
            provider=gemini_config["provider"],
            model_name=gemini_config["model_name"],
            temperature=gemini_config["temperature"],
            degree=degree,
            depth=depth,
        )

        async def run_build():
            return [event async for event in graph.build_async()]

        events = asyncio.run(run_build())

        # Verify build completion
        completes = [e for e in events if e.get("event") == "build_complete"]
        assert len(completes) == 1

        # Verify paths were built
        paths = graph.get_all_paths()
        assert len(paths) >= 1
        assert all(p[0] == topic for p in paths)

    @requires_gemini
    @pytest.mark.gemini
    def test_graph_no_cycles(self, gemini_config):
        """Test that generated graphs have no cycles."""
        graph = Graph(
            topic_prompt="Software Engineering",
            provider=gemini_config["provider"],
            model_name=gemini_config["model_name"],
            temperature=gemini_config["temperature"],
            degree=2,
            depth=2,
        )

        async def build_graph():
            async for _ in graph.build_async():
                pass

        asyncio.run(build_graph())

        # Verify no cycles
        assert not graph.has_cycle()
        # Verify we got paths
        paths = graph.get_all_paths()
        assert len(paths) >= 1
