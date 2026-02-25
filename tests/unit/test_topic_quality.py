"""Tests for topic quality scoring."""

import json
import tempfile

from pathlib import Path

import pytest

from click.testing import CliRunner

from deepfabric.cli import cli
from deepfabric.topic_quality import (
    derive_topic_score_report_path,
    derive_topic_threshold_optimization_report_path,
    optimize_topic_thresholds,
    score_topic_graph,
)


@pytest.fixture
def graph_with_embeddings_file():
    """Create a temporary graph JSON file with precomputed embeddings.

    Graph structure:
        0 (root) -> 1, 2, 5
        1 -> 4
        2 -> 3

    Node 2 has negative global coherence (embedding [-0.3, 0.9] vs root [1.0, 0.0]).
    Nodes 1 and 5 are siblings with high mutual coherence.
    """
    content = {
        "nodes": {
            "0": {
                "id": 0,
                "topic": "SEO Root",
                "children": [1, 2, 5],
                "parents": [],
                "metadata": {"uuid": "uuid-0", "embedding": [1.0, 0.0]},
            },
            "1": {
                "id": 1,
                "topic": "On-topic branch",
                "children": [4],
                "parents": [0],
                "metadata": {"uuid": "uuid-1", "embedding": [0.9, 0.1]},
            },
            "2": {
                "id": 2,
                "topic": "Off-topic branch",
                "children": [3],
                "parents": [0],
                "metadata": {"uuid": "uuid-2", "embedding": [-0.3, 0.9]},
            },
            "3": {
                "id": 3,
                "topic": "Off-topic child",
                "children": [],
                "parents": [2],
                "metadata": {"uuid": "uuid-3", "embedding": [-0.7, 0.1]},
            },
            "4": {
                "id": 4,
                "topic": "On-topic child",
                "children": [],
                "parents": [1],
                "metadata": {"uuid": "uuid-4", "embedding": [0.85, 0.15]},
            },
            "5": {
                "id": 5,
                "topic": "Another on-topic branch",
                "children": [],
                "parents": [0],
                "metadata": {"uuid": "uuid-5", "embedding": [0.8, 0.2]},
            },
        },
        "root_id": 0,
        "metadata": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.7,
            "created_at": "2024-01-01T00:00:00+00:00",
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(content, f)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


def test_score_topic_graph_flags_and_removals(graph_with_embeddings_file):
    """Scoring should flag negative-coherence nodes and include descendant removal estimate."""
    report = score_topic_graph(graph_with_embeddings_file)
    summary = report["summary"]

    assert summary["original_node_count"] == 6  # noqa: PLR2004
    assert summary["flagged_node_count"] >= 1
    # Node 2 (negative global coherence) and its descendant node 3 should be removed
    assert "2" in report["removed_node_ids"]
    assert "3" in report["removed_node_ids"]

    flagged = {item["node_id"]: item for item in report["flagged_nodes"]}
    assert "2" in flagged
    assert "NEGATIVE_GLOBAL_COHERENCE" in flagged["2"]["reasons"]

    # Verify new metric keys are present in flagged nodes
    assert "global_coherence" in flagged["2"]
    assert "parent_coherence" in flagged["2"]
    assert "sibling_coherence" in flagged["2"]

    # Verify metrics_per_node uses new key names
    assert "global_coherence" in report["metrics_per_node"]["0"]
    assert "parent_coherence" in report["metrics_per_node"]["0"]
    assert "sibling_coherence" in report["metrics_per_node"]["0"]

    # Verify summary has new stats keys
    assert "global_coherence_stats" in summary
    assert "parent_coherence_stats" in summary
    assert "sibling_coherence_stats" in summary
    assert "step_removals" in summary


def test_derive_topic_score_report_path(graph_with_embeddings_file):
    output_path = derive_topic_score_report_path(graph_with_embeddings_file)
    assert output_path.endswith("_score_report.json")


def test_topic_score_cli_writes_report(graph_with_embeddings_file, tmp_path):
    runner = CliRunner()
    report_path = tmp_path / "quality_report.json"

    result = runner.invoke(
        cli,
        [
            "topic",
            "score",
            graph_with_embeddings_file,
            "--output-report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert "Topic graph scored successfully" in result.output
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["original_node_count"] == 6  # noqa: PLR2004


def test_optimize_topic_thresholds_returns_best(graph_with_embeddings_file):
    report = optimize_topic_thresholds(
        graph_with_embeddings_file,
        search="grid",
        trials=8,
        parent_coherence_min=0.1,
        parent_coherence_max=0.3,
        sibling_coherence_lower_min=0.05,
        sibling_coherence_lower_max=0.25,
        sibling_coherence_upper_min=0.5,
        sibling_coherence_upper_max=0.8,
        seed=7,
        top_k=3,
    )

    assert report["best"] is not None
    assert len(report["all_trials"]) > 0
    assert len(report["top_trials"]) <= 3  # noqa: PLR2004
    assert "thresholds" in report["best"]
    assert "parent_coherence" in report["best"]["thresholds"]
    assert "sibling_coherence_lower" in report["best"]["thresholds"]
    assert "sibling_coherence_upper" in report["best"]["thresholds"]
    assert "objective" in report["best"]


def test_derive_topic_threshold_optimization_report_path(graph_with_embeddings_file):
    output_path = derive_topic_threshold_optimization_report_path(graph_with_embeddings_file)
    assert output_path.endswith("_threshold_optimization.json")


def test_topic_optimize_thresholds_cli_writes_report(graph_with_embeddings_file, tmp_path):
    runner = CliRunner()
    report_path = tmp_path / "threshold_optimization.json"

    result = runner.invoke(
        cli,
        [
            "topic",
            "optimize-thresholds",
            graph_with_embeddings_file,
            "--search",
            "grid",
            "--trials",
            "8",
            "--output-report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert "Threshold optimization completed" in result.output
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["best"] is not None
    assert len(report["all_trials"]) > 0
