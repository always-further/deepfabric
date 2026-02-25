"""Topic graph quality scoring and threshold optimization utilities."""

from __future__ import annotations

import json
import math
import random

from collections import deque
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import ConfigurationError
from .graph import Graph

_EPSILON = 1e-12

DEFAULT_THRESHOLDS = {
    "parent_coherence": 0.25,
    "sibling_coherence_lower": 0.2,
    "sibling_coherence_upper": 0.68,
}


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


def _compute_sibling_coherence_by_id(
    graph: Graph,
    embeddings: dict[int, np.ndarray],
) -> dict[int, float | None]:
    """Compute per-node sibling coherence: mean cosine similarity to siblings.

    Siblings are children of the same parent(s), excluding the node itself.
    Returns None for the root node, only-children (no siblings exist), and
    nodes with missing embeddings. Nodes with None sibling coherence are
    skipped by the sibling coherence pruning steps (3 and 4).
    """
    sibling_coherence_by_id: dict[int, float | None] = {}

    for node_id, node in graph.nodes.items():
        node_embedding = embeddings.get(node_id)
        if node_embedding is None:
            sibling_coherence_by_id[node_id] = None
            continue

        sibling_ids: set[int] = set()
        for parent in node.parents:
            for child in parent.children:
                if child.id != node_id:
                    sibling_ids.add(child.id)

        if not sibling_ids:
            sibling_coherence_by_id[node_id] = None
            continue

        sims: list[float] = []
        for sib_id in sibling_ids:
            sib_embedding = embeddings.get(sib_id)
            if sib_embedding is not None:
                sims.append(_safe_cosine_similarity(node_embedding, sib_embedding))

        if not sims:
            sibling_coherence_by_id[node_id] = None
        else:
            sibling_coherence_by_id[node_id] = float(np.mean(sims))

    return sibling_coherence_by_id


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
    """Load graph and precompute per-node coherence metrics once."""
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

    global_coherence_values: list[float] = []
    parent_coherence_values: list[float] = []
    global_coherence_by_id: dict[int, float | None] = {}
    parent_coherence_by_id: dict[int, float | None] = {}

    for node_id, node in graph.nodes.items():
        node_embedding = embeddings.get(node_id)
        if node_embedding is None:
            gc_val = None
        else:
            gc_val = _safe_cosine_similarity(node_embedding, root_embedding)
            global_coherence_values.append(gc_val)

        parent_sims: list[float] = []
        for parent in node.parents:
            parent_embedding = embeddings.get(parent.id)
            if node_embedding is None or parent_embedding is None:
                continue
            parent_sims.append(_safe_cosine_similarity(node_embedding, parent_embedding))

        pc_val = max(parent_sims) if parent_sims else None
        if pc_val is not None:
            parent_coherence_values.append(pc_val)

        global_coherence_by_id[node_id] = gc_val
        parent_coherence_by_id[node_id] = pc_val

    sibling_coherence_by_id = _compute_sibling_coherence_by_id(graph, embeddings)
    sibling_coherence_values = [v for v in sibling_coherence_by_id.values() if v is not None]

    depth_counts: dict[str, int] = {}
    for depth in depths.values():
        depth_counts[str(depth)] = depth_counts.get(str(depth), 0) + 1

    metrics_per_node = {
        str(node_id): {
            "global_coherence": global_coherence_by_id[node_id],
            "parent_coherence": parent_coherence_by_id[node_id],
            "sibling_coherence": sibling_coherence_by_id[node_id],
        }
        for node_id in graph.nodes
    }

    return {
        "graph": graph,
        "depths": depths,
        "max_depth": max_depth,
        "global_coherence_by_id": global_coherence_by_id,
        "parent_coherence_by_id": parent_coherence_by_id,
        "sibling_coherence_by_id": sibling_coherence_by_id,
        "metrics_per_node": metrics_per_node,
        "global_coherence_stats": _summarize(global_coherence_values),
        "parent_coherence_stats": _summarize(parent_coherence_values),
        "sibling_coherence_stats": _summarize(sibling_coherence_values),
        "depth_counts": depth_counts,
        "descendants_cache": {},
    }


def _evaluate_thresholds(
    context: dict[str, Any],
    *,
    parent_coherence: float,
    sibling_coherence_lower: float,
    sibling_coherence_upper: float,
    include_metrics_per_node: bool = True,
) -> dict[str, Any]:
    """Evaluate a threshold combination using the 4-step cascading pipeline."""
    graph: Graph = context["graph"]
    depths: dict[int, int] = context["depths"]
    max_depth: int = context["max_depth"]
    gc_by_id: dict[int, float | None] = context["global_coherence_by_id"]
    pc_by_id: dict[int, float | None] = context["parent_coherence_by_id"]
    sc_by_id: dict[int, float | None] = context["sibling_coherence_by_id"]
    descendants_cache: dict[int, set[int]] = context["descendants_cache"]

    all_ids = set(graph.nodes.keys())
    flagged_nodes: list[dict[str, Any]] = []
    step_removals = {
        "step1_negative_global_coherence": 0,
        "step2_low_parent_coherence": 0,
        "step3_low_sibling_coherence": 0,
        "step4_high_sibling_coherence": 0,
    }

    # Step 1: global_coherence < 0 (hardcoded gate)
    step1_flagged: set[int] = set()
    for node_id in all_ids:
        gc_val = gc_by_id[node_id]
        if gc_val is not None and gc_val < 0:
            step1_flagged.add(node_id)

    step1_removed: set[int] = set()
    for node_id in step1_flagged:
        step1_removed.update(
            _collect_descendants_with_cache(graph, node_id, descendants_cache) & all_ids
        )
    step_removals["step1_negative_global_coherence"] = len(step1_removed)
    surviving = all_ids - step1_removed

    # Step 2: parent_coherence < threshold
    step2_flagged: set[int] = set()
    for node_id in surviving:
        pc_val = pc_by_id[node_id]
        if pc_val is not None and pc_val < parent_coherence:
            step2_flagged.add(node_id)

    step2_removed: set[int] = set()
    for node_id in step2_flagged:
        step2_removed.update(
            _collect_descendants_with_cache(graph, node_id, descendants_cache) & surviving
        )
    step_removals["step2_low_parent_coherence"] = len(step2_removed)
    surviving = surviving - step2_removed

    # Step 3: sibling_coherence < lower threshold (outliers)
    step3_flagged: set[int] = set()
    for node_id in surviving:
        sc_val = sc_by_id[node_id]
        if sc_val is not None and sc_val < sibling_coherence_lower:
            step3_flagged.add(node_id)

    step3_removed: set[int] = set()
    for node_id in step3_flagged:
        step3_removed.update(
            _collect_descendants_with_cache(graph, node_id, descendants_cache) & surviving
        )
    step_removals["step3_low_sibling_coherence"] = len(step3_removed)
    surviving = surviving - step3_removed

    # Step 4: sibling_coherence > upper threshold (repetitive)
    step4_flagged: set[int] = set()
    for node_id in surviving:
        sc_val = sc_by_id[node_id]
        if sc_val is not None and sc_val > sibling_coherence_upper:
            step4_flagged.add(node_id)

    step4_removed: set[int] = set()
    for node_id in step4_flagged:
        step4_removed.update(
            _collect_descendants_with_cache(graph, node_id, descendants_cache) & surviving
        )
    step_removals["step4_high_sibling_coherence"] = len(step4_removed)

    # Build flagged nodes list (directly flagged at any step)
    all_flagged = step1_flagged | step2_flagged | step3_flagged | step4_flagged
    for node_id in sorted(all_flagged):
        node = graph.nodes[node_id]
        node_depth = depths.get(node_id, 0)
        reasons: list[str] = []
        if node_id in step1_flagged:
            reasons.append("NEGATIVE_GLOBAL_COHERENCE")
        if node_id in step2_flagged:
            reasons.append("LOW_PARENT_COHERENCE")
        if node_id in step3_flagged:
            reasons.append("LOW_SIBLING_COHERENCE")
        if node_id in step4_flagged:
            reasons.append("HIGH_SIBLING_COHERENCE")
        flagged_nodes.append(
            {
                "node_id": str(node_id),
                "topic": node.topic,
                "depth": node_depth,
                "reasons": reasons,
                "global_coherence": gc_by_id[node_id],
                "parent_coherence": pc_by_id[node_id],
                "sibling_coherence": sc_by_id[node_id],
            }
        )

    removed_ids = step1_removed | step2_removed | step3_removed | step4_removed

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
        "flagged_node_count": len(all_flagged),
        "removed_node_count": removed_count,
        "remaining_node_count": total_nodes - removed_count,
        "global_coherence_stats": context["global_coherence_stats"],
        "parent_coherence_stats": context["parent_coherence_stats"],
        "sibling_coherence_stats": context["sibling_coherence_stats"],
        "depth_counts": context["depth_counts"],
        "removed_depth_counts": removed_depth_counts,
        "thresholds": {
            "parent_coherence": parent_coherence,
            "sibling_coherence_lower": sibling_coherence_lower,
            "sibling_coherence_upper": sibling_coherence_upper,
        },
        "step_removals": step_removals,
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
    parent_coherence: float = DEFAULT_THRESHOLDS["parent_coherence"],
    sibling_coherence_lower: float = DEFAULT_THRESHOLDS["sibling_coherence_lower"],
    sibling_coherence_upper: float = DEFAULT_THRESHOLDS["sibling_coherence_upper"],
    embedding_key: str = "embedding",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Score a topic graph using coherence metrics and pruning thresholds."""
    context = _build_topic_quality_context(
        graph_path,
        embedding_key=embedding_key,
        embedding_model=embedding_model,
    )
    return _evaluate_thresholds(
        context,
        parent_coherence=parent_coherence,
        sibling_coherence_lower=sibling_coherence_lower,
        sibling_coherence_upper=sibling_coherence_upper,
        include_metrics_per_node=True,
    )


def _generate_threshold_candidates(
    *,
    search: str,
    trials: int,
    parent_coherence_min: float,
    parent_coherence_max: float,
    sibling_coherence_lower_min: float,
    sibling_coherence_lower_max: float,
    sibling_coherence_upper_min: float,
    sibling_coherence_upper_max: float,
    seed: int,
) -> list[dict[str, float]]:
    """Generate threshold candidates for optimization."""
    if trials <= 0:
        raise ValueError("trials must be greater than zero")

    if search == "random":
        rng = random.Random(seed)  # noqa: S311
        return [
            {
                "parent_coherence": rng.uniform(parent_coherence_min, parent_coherence_max),
                "sibling_coherence_lower": rng.uniform(
                    sibling_coherence_lower_min, sibling_coherence_lower_max
                ),
                "sibling_coherence_upper": rng.uniform(
                    sibling_coherence_upper_min, sibling_coherence_upper_max
                ),
            }
            for _ in range(trials)
        ]

    per_dim = max(2, math.ceil(trials ** (1 / 3)))
    pc_values = np.linspace(parent_coherence_min, parent_coherence_max, per_dim).tolist()
    scl_values = np.linspace(
        sibling_coherence_lower_min, sibling_coherence_lower_max, per_dim
    ).tolist()
    scu_values = np.linspace(
        sibling_coherence_upper_min, sibling_coherence_upper_max, per_dim
    ).tolist()
    combos = [
        {
            "parent_coherence": pc,
            "sibling_coherence_lower": scl,
            "sibling_coherence_upper": scu,
        }
        for pc, scl, scu in product(pc_values, scl_values, scu_values)
    ]
    return combos[:trials]


def optimize_topic_thresholds(
    graph_path: str,
    *,
    search: str = "random",
    trials: int = 40,
    parent_coherence_min: float = 0.10,
    parent_coherence_max: float = 0.50,
    sibling_coherence_lower_min: float = 0.05,
    sibling_coherence_lower_max: float = 0.40,
    sibling_coherence_upper_min: float = 0.50,
    sibling_coherence_upper_max: float = 0.85,
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
        parent_coherence_min=parent_coherence_min,
        parent_coherence_max=parent_coherence_max,
        sibling_coherence_lower_min=sibling_coherence_lower_min,
        sibling_coherence_lower_max=sibling_coherence_lower_max,
        sibling_coherence_upper_min=sibling_coherence_upper_min,
        sibling_coherence_upper_max=sibling_coherence_upper_max,
        seed=seed,
    )

    baseline = {
        "parent_coherence": DEFAULT_THRESHOLDS["parent_coherence"],
        "sibling_coherence_lower": DEFAULT_THRESHOLDS["sibling_coherence_lower"],
        "sibling_coherence_upper": DEFAULT_THRESHOLDS["sibling_coherence_upper"],
    }
    if baseline not in candidates:
        candidates.insert(0, baseline)
    candidates = candidates[:trials]

    evaluations: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        report = _evaluate_thresholds(
            context,
            parent_coherence=float(candidate["parent_coherence"]),
            sibling_coherence_lower=float(candidate["sibling_coherence_lower"]),
            sibling_coherence_upper=float(candidate["sibling_coherence_upper"]),
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
                "parent_coherence": [parent_coherence_min, parent_coherence_max],
                "sibling_coherence_lower": [
                    sibling_coherence_lower_min,
                    sibling_coherence_lower_max,
                ],
                "sibling_coherence_upper": [
                    sibling_coherence_upper_min,
                    sibling_coherence_upper_max,
                ],
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
