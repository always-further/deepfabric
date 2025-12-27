import asyncio
import json

import pytest  # pyright: ignore[reportMissingImports]

from deepfabric import Tree, topic_manager
from deepfabric.utils import read_topic_tree_from_jsonl

from .conftest import requires_openai


class TestTreeIntegration:
    """Integration tests for the Tree class, requiring real LLM calls."""

    # Centralize test configuration
    TEST_PROVIDER = "openai"
    TEST_MODEL = "gpt-4o-mini"
    TEST_TEMPERATURE = 0.2

    @pytest.fixture
    def tree_builder(self):
        """Fixture to provide a factory for creating Tree instances."""

        def _builder(
            topic_prompt: str,
            degree: int,
            depth: int,
            system_prompt: str = "",
        ) -> Tree:
            """Helper to build a Tree with common test parameters."""
            return Tree(
                topic_prompt=topic_prompt,
                topic_system_prompt=system_prompt,
                degree=degree,
                depth=depth,
                provider=self.TEST_PROVIDER,
                model_name=self.TEST_MODEL,
                temperature=self.TEST_TEMPERATURE,
            )

        return _builder

    async def _run_build(self, tree: Tree):
        """Helper to run the async build and collect all events."""
        return [event async for event in tree.build_async()]

    @requires_openai
    @pytest.mark.openai
    def test_tree_builds_basic(self, tree_builder):
        degree = 2
        depth = 1
        topic = "AI Ethics"
        tree = tree_builder(topic_prompt=topic, degree=degree, depth=depth)

        events = asyncio.run(self._run_build(tree))

        # Verify build completion event
        completes = [e for e in events if e.get("event") == "build_complete"]
        assert len(completes) == 1
        assert completes[0]["total_paths"] == degree**depth

        # Verify paths were built as expected
        paths = tree.get_all_paths()
        assert len(paths) == degree**depth
        assert all(len(p) == depth + 1 for p in paths)
        assert all(p[0] == topic for p in paths)

    @requires_openai
    @pytest.mark.openai
    def test_tree_builds_with_deeper_recursion(self, tree_builder):
        # Single branch but deeper depth to exercise recursion with real LLM
        degree = 1
        depth = 2  # two LLM calls
        topic = "Quantum Computing"
        tree = tree_builder(topic_prompt=topic, degree=degree, depth=depth)

        events = asyncio.run(self._run_build(tree))
        last_event = events[-1] if events else None

        assert last_event is not None
        assert last_event.get("event") == "build_complete"
        assert last_event["total_paths"] == degree**depth
        paths = tree.get_all_paths()
        assert len(paths) == degree**depth
        assert all(len(p) == depth + 1 for p in paths)

    @requires_openai
    @pytest.mark.openai
    def test_tree_generates_and_saves_jsonl_structure(self, tmp_path, tree_builder):
        # Generate a small tree, save to JSONL, and validate file structure
        degree = 2
        depth = 1
        root_topic = "Software Engineering"
        tree = tree_builder(topic_prompt=root_topic, degree=degree, depth=depth)

        asyncio.run(self._run_build(tree))

        # Save to a temp file
        out_path = tmp_path / "topics.jsonl"
        tree.save(str(out_path))

        # Read and validate structure from file
        lines = out_path.read_text().splitlines()
        assert len(lines) == degree**depth

        items = [json.loads(line) for line in lines]
        assert all(isinstance(obj, dict) for obj in items)
        assert all("path" in obj for obj in items)
        assert all(isinstance(obj["path"], list) for obj in items)
        assert all(len(obj["path"]) == depth + 1 for obj in items)
        assert all(obj["path"][0] == root_topic for obj in items)
        assert all(list(obj.keys()) == ["path"] for obj in items)

    @requires_openai
    @pytest.mark.openai
    def test_tree_save_and_load_round_trip(self, tmp_path, tree_builder):
        degree = 2
        depth = 1
        root = "RoundTrip Root"

        # Build original tree
        tree = tree_builder(topic_prompt=root, degree=degree, depth=depth)
        asyncio.run(self._run_build(tree))

        # Save to JSONL
        out = tmp_path / "topics.jsonl"
        tree.save(str(out))

        # Load dict list and construct a new tree from it
        dict_list = read_topic_tree_from_jsonl(str(out))
        rebuilt = tree_builder(topic_prompt="placeholder", degree=degree, depth=depth)
        rebuilt.from_dict_list(dict_list)

        # Assert round-trip equality
        assert rebuilt.get_all_paths() == tree.get_all_paths()
        # And both reflect expected shape
        assert len(rebuilt.get_all_paths()) == degree**depth
        assert all(len(p) == depth + 1 for p in rebuilt.get_all_paths())
        assert all(p[0] == root for p in rebuilt.get_all_paths())

    @requires_openai
    @pytest.mark.openai
    def test_tree_tui_streaming_and_events(self, monkeypatch, tree_builder):
        # Use topic_manager + TUI to drive build and capture streaming chunks

        class DummyTreeTUI:
            def __init__(self):
                self.started = []
                self.finished = []
                self.failures = 0
                self.chunks = []

            # Methods used by topic_manager
            def start_building(
                self, model_name: str, depth: int, degree: int, topic_prompt: str
            ) -> None:
                self.started.append((model_name, depth, degree, topic_prompt))

            def add_failure(self) -> None:
                self.failures += 1

            def finish_building(self, total_paths: int, failed_generations: int) -> None:
                self.finished.append((total_paths, failed_generations))

            # StreamObserver callback used by ProgressReporter
            def on_stream_chunk(self, source: str, chunk: str, metadata: dict) -> None:  # noqa: D401, ARG002
                self.chunks.append(chunk)

        fake_tui = DummyTreeTUI()
        monkeypatch.setattr(topic_manager, "get_tree_tui", lambda: fake_tui, raising=True)

        degree = 1
        depth = 1
        tree = tree_builder(topic_prompt="Networking", degree=degree, depth=depth)

        final_event = topic_manager.handle_tree_events(tree, debug=False)

        # Validate TUI callbacks and final event
        assert final_event is not None and final_event.get("event") == "build_complete"
        assert final_event["total_paths"] == degree**depth
        assert len(fake_tui.started) == 1
        assert fake_tui.started[0][1] == depth and fake_tui.started[0][2] == degree
        assert len(fake_tui.finished) == 1
        assert fake_tui.finished[0][0] == degree**depth
        assert fake_tui.failures == 0
        # Streaming should have produced at least one chunk
        assert isinstance(fake_tui.chunks, list)
        assert len(fake_tui.chunks) >= 1
