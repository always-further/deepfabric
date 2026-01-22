#!/usr/bin/env python3
"""Look up topic graph and dataset entries by UUID.

Usage:
    python scripts/lookup_by_uuid.py <uuid> --topics <topics.json> --dataset <dataset.jsonl>

Example:
    python scripts/lookup_by_uuid.py abc123-def456 --topics topics.json --dataset dataset.jsonl
    python scripts/lookup_by_uuid.py abc123-def456 --topics topics.json
    python scripts/lookup_by_uuid.py abc123-def456 --dataset dataset.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def find_in_topic_graph(graph_path: str, uuid: str) -> dict | None:
    """Find a node in the topic graph by UUID."""
    if not Path(graph_path).exists():
        return None

    with open(graph_path) as f:
        graph_data = json.load(f)

    # Search nodes for matching UUID
    # Nodes dict may have integer or string keys
    nodes = graph_data.get("nodes", {})
    for node_id, node in nodes.items():
        metadata = node.get("metadata", {})
        if metadata.get("uuid") == uuid:
            return {
                "node_id": node_id,
                "topic": node.get("topic"),
                "metadata": metadata,
                "children": node.get("children", []),
            }

    return None


def find_in_dataset(dataset_path: str, uuid: str) -> list[dict]:
    """Find samples in the dataset by topic_id (UUID).

    Searches for topic_id in both:
    - Top-level: sample["topic_id"]
    - Nested in metadata: sample["metadata"]["topic_id"]
    """
    if not Path(dataset_path).exists():
        return []

    matches = []
    with open(dataset_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                # Check both top-level and nested in metadata
                topic_id = sample.get("topic_id") or sample.get("metadata", {}).get("topic_id")
                if topic_id == uuid:
                    matches.append({"line": line_num, "sample": sample})
            except json.JSONDecodeError:
                continue

    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Look up topic graph and dataset entries by UUID"
    )
    parser.add_argument("uuid", help="UUID to search for")
    parser.add_argument("--topics", "-t", help="Path to topic graph JSON file")
    parser.add_argument("--dataset", "-d", help="Path to dataset JSONL file")
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of formatted text"
    )
    args = parser.parse_args()

    # Validate that at least one file is specified
    if not args.topics and not args.dataset:
        print(
            "Error: At least one of --topics or --dataset must be specified",
            file=sys.stderr,
        )
        sys.exit(1)

    results = {
        "uuid": args.uuid,
        "topic_graph": None,
        "dataset_samples": [],
    }

    # Search topic graph
    if args.topics:
        topic_result = find_in_topic_graph(args.topics, args.uuid)
        results["topic_graph"] = topic_result

    # Search dataset
    if args.dataset:
        dataset_results = find_in_dataset(args.dataset, args.uuid)
        results["dataset_samples"] = dataset_results

    # Output
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"=== Lookup for UUID: {args.uuid} ===\n")

        # Topic graph
        if args.topics:
            print("--- Topic Graph ---")
            if not Path(args.topics).exists():
                print(f"  Topic file not found: {args.topics}")
            elif results["topic_graph"]:
                node = results["topic_graph"]
                print(f"  Node ID: {node['node_id']}")
                print(f"  Topic: {node['topic']}")
                if node["children"]:
                    children_str = ", ".join(str(c) for c in node["children"][:10])
                    if len(node["children"]) > 10:
                        children_str += f"... ({len(node['children'])} total)"
                    print(f"  Children: {children_str}")
                print(f"  Metadata: {json.dumps(node['metadata'], indent=4)}")
            else:
                print("  Not found in topic graph")

        # Dataset
        if args.dataset:
            print("\n--- Dataset ---" if args.topics else "--- Dataset ---")
            if not Path(args.dataset).exists():
                print(f"  Dataset file not found: {args.dataset}")
            elif results["dataset_samples"]:
                print(f"  Found {len(results['dataset_samples'])} sample(s):")
                for match in results["dataset_samples"]:
                    print(f"\n  Line {match['line']}:")
                    sample = match["sample"]
                    # Show a summary
                    if "messages" in sample:
                        print(f"    Messages: {len(sample['messages'])} message(s)")
                        for i, msg in enumerate(sample["messages"][:3]):
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                content = content[:100]
                            print(f"      [{i}] {role}: {content}...")
                    if "metadata" in sample:
                        print(f"    Metadata: {json.dumps(sample['metadata'], indent=6)}")
            else:
                print("  Not found in dataset")


if __name__ == "__main__":
    main()
