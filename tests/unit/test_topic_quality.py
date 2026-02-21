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
    """Create a temporary graph JSON file with precomputed embeddings."""
    content = {
        "nodes": {
            "0": {
                "id": 0,
                "topic": "SEO Root",
                "children": [1, 2],
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
                "metadata": {"uuid": "uuid-2", "embedding": [-0.8, 0.2]},
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
                "metadata": {"uuid": "uuid-4", "embedding": [0.6, 0.3]},
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
    """Scoring should flag drift nodes and include descendant removal estimate."""
    report = score_topic_graph(graph_with_embeddings_file)
    summary = report["summary"]

    assert summary["original_node_count"] == 5  # noqa: PLR2004
    assert summary["flagged_node_count"] >= 1
    assert summary["removed_node_count"] == 2  # noqa: PLR2004  # node 2 and descendant 3
    assert "2" in report["removed_node_ids"]
    assert "3" in report["removed_node_ids"]

    flagged = {item["node_id"]: item for item in report["flagged_nodes"]}
    assert "2" in flagged
    assert "DEPTH1_LOW_GTD" in flagged["2"]["reasons"]
    assert "GTD_NEGATIVE" in flagged["2"]["reasons"]


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
    assert report["summary"]["original_node_count"] == 5  # noqa: PLR2004


def test_optimize_topic_thresholds_returns_best(graph_with_embeddings_file):
    report = optimize_topic_thresholds(
        graph_with_embeddings_file,
        search="grid",
        trials=8,
        depth1_min=0.1,
        depth1_max=0.3,
        gtd_neg_min=-0.1,
        gtd_neg_max=0.0,
        ltd_min=0.1,
        ltd_max=0.3,
        seed=7,
        top_k=3,
    )

    assert report["best"] is not None
    assert len(report["all_trials"]) > 0
    assert len(report["top_trials"]) <= 3  # noqa: PLR2004
    assert "thresholds" in report["best"]
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
