"""Topic graph quality scoring and threshold optimization utilities."""

from __future__ import annotations

import json
import random

from collections import deque
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import ConfigurationError
from .graph import Graph

_EPSILON = 1e-12


def _safe_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity with zero-vector protection."""
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a <= _EPSILON or norm_b <= _EPSILON:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _compute_depths(graph: Graph) -> dict[int, int]:
    """Compute shortest depth for each node from root."""
    depths: dict[int, int] = {graph.root.id: 0}
    queue: deque[int] = deque([graph.root.id])

    while queue:
        node_id = queue.popleft()
        node = graph.nodes[node_id]
        current_depth = depths[node_id]

        for child in node.children:
            child_depth = current_depth + 1
            known_depth = depths.get(child.id)
            if known_depth is None or child_depth < known_depth:
                depths[child.id] = child_depth
                queue.append(child.id)

    return depths


def _extract_embedding(raw_value: Any) -> np.ndarray | None:
    """Convert raw embedding payload to a numpy vector."""
    if raw_value is None:
        return None
    if isinstance(raw_value, np.ndarray):
        return raw_value.astype(float)
    if isinstance(raw_value, list):
        try:
            return np.asarray(raw_value, dtype=float)
        except (TypeError, ValueError):
            return None
    return None


def _build_embedding_map(graph: Graph, embedding_key: str) -> dict[int, np.ndarray]:
    """Read embeddings from node metadata."""
    embeddings: dict[int, np.ndarray] = {}
    for node_id, node in graph.nodes.items():
        emb = _extract_embedding(node.metadata.get(embedding_key))
        if emb is not None:
            embeddings[node_id] = emb
    return embeddings


def _fill_missing_embeddings(
    graph: Graph,
    embeddings: dict[int, np.ndarray],
    embedding_key: str,
    embedding_model: str,
) -> dict[int, np.ndarray]:
    """Generate embeddings for nodes that do not have one yet."""
    missing_ids = [node_id for node_id in graph.nodes if node_id not in embeddings]
    if not missing_ids:
        return embeddings

    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via CLI behavior
        raise ConfigurationError(
            "Missing embeddings and sentence-transformers is not installed. "
            "Install with: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(embedding_model)
    texts = [graph.nodes[node_id].topic for node_id in missing_ids]
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    for node_id, vector in zip(missing_ids, vectors, strict=False):
        arr = np.asarray(vector, dtype=float)
        graph.nodes[node_id].metadata[embedding_key] = arr.tolist()
        embeddings[node_id] = arr

    return embeddings


def _collect_descendants(graph: Graph, start_id: int) -> set[int]:
    """Collect all descendants including the start node."""
    seen: set[int] = set()
    stack: list[int] = [start_id]

    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        for child in graph.nodes[node_id].children:
            if child.id not in seen:
                stack.append(child.id)

    return seen


def _collect_descendants_with_cache(
    graph: Graph,
    start_id: int,
    cache: dict[int, set[int]],
) -> set[int]:
    """Collect descendants with memoization for repeated threshold evaluations."""
    cached = cache.get(start_id)
    if cached is not None:
        return cached

    result = _collect_descendants(graph, start_id)
    cache[start_id] = result
    return result


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    """Compute summary statistics for a numeric vector."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "25%": None,
            "median": None,
            "75%": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "25%": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "75%": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def derive_topic_score_report_path(graph_path: str) -> str:
    """Derive default report output path from input graph path."""
    path = Path(graph_path)
    return str(path.with_stem(f"{path.stem}_score_report"))


def derive_topic_threshold_optimization_report_path(graph_path: str) -> str:
    """Derive default threshold optimization report path from input graph path."""
    path = Path(graph_path)
    return str(path.with_stem(f"{path.stem}_threshold_optimization"))


def _build_topic_quality_context(
    graph_path: str,
    *,
    embedding_key: str = "embedding",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Load graph and precompute per-node GTD/LTD metrics once."""
    graph = Graph.load(graph_path)
    depths = _compute_depths(graph)
    max_depth = max(depths.values(), default=0)

    embeddings = _build_embedding_map(graph, embedding_key=embedding_key)
    embeddings = _fill_missing_embeddings(
        graph,
        embeddings,
        embedding_key=embedding_key,
        embedding_model=embedding_model,
    )

    root_embedding = embeddings.get(graph.root.id)
    if root_embedding is None:
        raise ConfigurationError(
            f"Root node is missing embedding under metadata['{embedding_key}'] and could not be generated."
        )

    gtd_values: list[float] = []
    ltd_values: list[float] = []
    gtd_by_id: dict[int, float | None] = {}
    ltd_by_id: dict[int, float | None] = {}

    for node_id, node in graph.nodes.items():
        node_embedding = embeddings.get(node_id)
        if node_embedding is None:
            gtd_val = None
        else:
            gtd_val = _safe_cosine_similarity(node_embedding, root_embedding)
            gtd_values.append(gtd_val)

        parent_sims: list[float] = []
        for parent in node.parents:
            parent_embedding = embeddings.get(parent.id)
            if node_embedding is None or parent_embedding is None:
                continue
            parent_sims.append(_safe_cosine_similarity(node_embedding, parent_embedding))

        ltd_val = max(parent_sims) if parent_sims else None
        if ltd_val is not None:
            ltd_values.append(ltd_val)

        gtd_by_id[node_id] = gtd_val
        ltd_by_id[node_id] = ltd_val

    depth_counts: dict[str, int] = {}
    for depth in depths.values():
        depth_counts[str(depth)] = depth_counts.get(str(depth), 0) + 1

    metrics_per_node = {
        str(node_id): {"gtd": gtd_by_id[node_id], "ltd": ltd_by_id[node_id]}
        for node_id in graph.nodes
    }

    return {
        "graph": graph,
        "depths": depths,
        "max_depth": max_depth,
        "gtd_by_id": gtd_by_id,
        "ltd_by_id": ltd_by_id,
        "metrics_per_node": metrics_per_node,
        "gtd_stats": _summarize(gtd_values),
        "ltd_stats": _summarize(ltd_values),
        "depth_counts": depth_counts,
        "descendants_cache": {},
    }


def _evaluate_thresholds(
    context: dict[str, Any],
    *,
    depth1_gtd: float,
    gtd_neg: float,
    ltd: float,
    include_metrics_per_node: bool = True,
) -> dict[str, Any]:
    """Evaluate a specific threshold combination using precomputed metrics."""
    graph: Graph = context["graph"]
    depths: dict[int, int] = context["depths"]
    max_depth: int = context["max_depth"]
    gtd_by_id: dict[int, float | None] = context["gtd_by_id"]
    ltd_by_id: dict[int, float | None] = context["ltd_by_id"]
    descendants_cache: dict[int, set[int]] = context["descendants_cache"]

    flagged_ids: set[int] = set()
    flagged_nodes: list[dict[str, Any]] = []

    for node_id, node in graph.nodes.items():
        gtd_val = gtd_by_id[node_id]
        ltd_val = ltd_by_id[node_id]
        node_depth = depths.get(node_id, 0)
        reasons: list[str] = []

        if gtd_val is not None and node_depth == 1 and gtd_val < depth1_gtd:
            reasons.append("DEPTH1_LOW_GTD")
        if gtd_val is not None and gtd_val < gtd_neg:
            reasons.append("GTD_NEGATIVE")
        if ltd_val is not None and ltd_val < ltd:
            reasons.append("LOW_LTD")

        if reasons:
            flagged_ids.add(node_id)
            flagged_nodes.append(
                {
                    "node_id": str(node_id),
                    "topic": node.topic,
                    "depth": node_depth,
                    "reasons": reasons,
                    "gtd": gtd_val,
                    "ltd": ltd_val,
                }
            )

    removed_ids: set[int] = set()
    for node_id in flagged_ids:
        removed_ids.update(_collect_descendants_with_cache(graph, node_id, descendants_cache))

    removed_depth_counts: dict[str, int] = {}
    for node_id in removed_ids:
        node_depth = depths.get(node_id, 0)
        key = str(node_depth)
        removed_depth_counts[key] = removed_depth_counts.get(key, 0) + 1

    total_nodes = len(graph.nodes)
    removed_count = len(removed_ids)
    internal_removed = sum(
        count for depth_str, count in removed_depth_counts.items() if int(depth_str) < max_depth
    )
    removed_ratio = (removed_count / total_nodes) if total_nodes else 0.0
    internal_removed_ratio = (internal_removed / total_nodes) if total_nodes else 0.0
    objective = removed_ratio + (1.5 * internal_removed_ratio)

    summary = {
        "original_node_count": total_nodes,
        "flagged_node_count": len(flagged_ids),
        "removed_node_count": removed_count,
        "remaining_node_count": total_nodes - removed_count,
        "gtd_stats": context["gtd_stats"],
        "ltd_stats": context["ltd_stats"],
        "depth_counts": context["depth_counts"],
        "removed_depth_counts": removed_depth_counts,
        "thresholds": {
            "depth1_gtd": depth1_gtd,
            "gtd_neg": gtd_neg,
            "ltd": ltd,
        },
        "objective": objective,
        "removed_ratio": removed_ratio,
        "internal_removed_ratio": internal_removed_ratio,
        "internal_removed_count": internal_removed,
    }

    result: dict[str, Any] = {
        "summary": summary,
        "flagged_nodes": flagged_nodes,
        "removed_node_ids": sorted(str(node_id) for node_id in removed_ids),
    }
    if include_metrics_per_node:
        result["metrics_per_node"] = context["metrics_per_node"]
    return result


def score_topic_graph(
    graph_path: str,
    *,
    depth1_gtd: float = 0.25,
    gtd_neg: float = 0.0,
    ltd: float = 0.25,
    embedding_key: str = "embedding",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Score a topic graph using GTD/LTD metrics and pruning thresholds."""
    context = _build_topic_quality_context(
        graph_path,
        embedding_key=embedding_key,
        embedding_model=embedding_model,
    )
    return _evaluate_thresholds(
        context,
        depth1_gtd=depth1_gtd,
        gtd_neg=gtd_neg,
        ltd=ltd,
        include_metrics_per_node=True,
    )


def _generate_threshold_candidates(
    *,
    search: str,
    trials: int,
    depth1_min: float,
    depth1_max: float,
    gtd_neg_min: float,
    gtd_neg_max: float,
    ltd_min: float,
    ltd_max: float,
    seed: int,
) -> list[dict[str, float]]:
    """Generate threshold candidates for optimization."""
    if trials <= 0:
        raise ValueError("trials must be greater than zero")

    if search == "random":
        rng = random.Random(seed)
        return [
            {
                "depth1_gtd": rng.uniform(depth1_min, depth1_max),
                "gtd_neg": rng.uniform(gtd_neg_min, gtd_neg_max),
                "ltd": rng.uniform(ltd_min, ltd_max),
            }
            for _ in range(trials)
        ]

    per_dim = max(2, round(trials ** (1 / 3)))
    depth1_values = np.linspace(depth1_min, depth1_max, per_dim).tolist()
    gtd_neg_values = np.linspace(gtd_neg_min, gtd_neg_max, per_dim).tolist()
    ltd_values = np.linspace(ltd_min, ltd_max, per_dim).tolist()
    combos = [
        {"depth1_gtd": d1, "gtd_neg": gn, "ltd": l}
        for d1, gn, l in product(depth1_values, gtd_neg_values, ltd_values)
    ]
    return combos[:trials]


def optimize_topic_thresholds(
    graph_path: str,
    *,
    search: str = "random",
    trials: int = 40,
    depth1_min: float = 0.10,
    depth1_max: float = 0.50,
    gtd_neg_min: float = -0.10,
    gtd_neg_max: float = 0.10,
    ltd_min: float = 0.10,
    ltd_max: float = 0.50,
    seed: int = 42,
    embedding_key: str = "embedding",
    embedding_model: str = "all-MiniLM-L6-v2",
    max_removed_ratio: float | None = None,
    max_internal_removed: int | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Optimize topic quality thresholds over one graph."""
    if search not in {"random", "grid"}:
        raise ValueError("search must be one of: random, grid")

    context = _build_topic_quality_context(
        graph_path,
        embedding_key=embedding_key,
        embedding_model=embedding_model,
    )

    candidates = _generate_threshold_candidates(
        search=search,
        trials=trials,
        depth1_min=depth1_min,
        depth1_max=depth1_max,
        gtd_neg_min=gtd_neg_min,
        gtd_neg_max=gtd_neg_max,
        ltd_min=ltd_min,
        ltd_max=ltd_max,
        seed=seed,
    )

    baseline = {"depth1_gtd": 0.25, "gtd_neg": 0.0, "ltd": 0.25}
    if baseline not in candidates:
        candidates.insert(0, baseline)
    candidates = candidates[: max(trials, 1)]

    evaluations: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        report = _evaluate_thresholds(
            context,
            depth1_gtd=float(candidate["depth1_gtd"]),
            gtd_neg=float(candidate["gtd_neg"]),
            ltd=float(candidate["ltd"]),
            include_metrics_per_node=False,
        )
        summary = report["summary"]
        removed_ratio = float(summary["removed_ratio"])
        internal_removed = int(summary["internal_removed_count"])
        passes_constraints = True

        if max_removed_ratio is not None and removed_ratio > max_removed_ratio:
            passes_constraints = False
        if max_internal_removed is not None and internal_removed > max_internal_removed:
            passes_constraints = False

        evaluations.append(
            {
                "trial": idx + 1,
                "thresholds": summary["thresholds"],
                "objective": float(summary["objective"]),
                "removed_count": int(summary["removed_node_count"]),
                "remaining_count": int(summary["remaining_node_count"]),
                "removed_ratio": removed_ratio,
                "internal_removed_count": internal_removed,
                "internal_removed_ratio": float(summary["internal_removed_ratio"]),
                "passes_constraints": passes_constraints,
            }
        )

    feasible = [e for e in evaluations if e["passes_constraints"]]
    pool = feasible if feasible else evaluations
    pool_sorted = sorted(pool, key=lambda e: e["objective"])
    best = pool_sorted[0] if pool_sorted else None

    return {
        "search": {
            "strategy": search,
            "trials_requested": trials,
            "trials_executed": len(evaluations),
            "seed": seed,
            "ranges": {
                "depth1_gtd": [depth1_min, depth1_max],
                "gtd_neg": [gtd_neg_min, gtd_neg_max],
                "ltd": [ltd_min, ltd_max],
            },
        },
        "constraints": {
            "max_removed_ratio": max_removed_ratio,
            "max_internal_removed": max_internal_removed,
        },
        "best": best,
        "top_trials": pool_sorted[: max(1, top_k)],
        "all_trials": sorted(evaluations, key=lambda e: e["objective"]),
        "used_fallback_unconstrained": not bool(feasible) and bool(evaluations),
    }


def write_topic_score_report(report: dict[str, Any], output_path: str) -> None:
    """Write score report JSON to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
