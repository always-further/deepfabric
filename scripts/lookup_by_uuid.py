#!/usr/bin/env python3
"""Look up topic graph and dataset entries by UUID.

Usage:
    python scripts/lookup_by_uuid.py <uuid> <config.yaml>

Example:
    python scripts/lookup_by_uuid.py abc123-def456 config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load and parse YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_in_topic_graph(graph_path: str, uuid: str) -> dict | None:
    """Find a node in the topic graph by UUID."""
    if not Path(graph_path).exists():
        return None

    with open(graph_path) as f:
        graph_data = json.load(f)

    # Search nodes for matching UUID
    nodes = graph_data.get("nodes", {})
    for node_id, node in nodes.items():
        metadata = node.get("metadata", {})
        if metadata.get("uuid") == uuid:
            return {
                "node_id": node_id,
                "topic": node.get("topic"),
                "metadata": metadata,
                "children": [c.get("topic") for c in node.get("children", [])],
            }

    return None


def find_in_dataset(dataset_path: str, uuid: str) -> list[dict]:
    """Find samples in the dataset by topic_id (UUID)."""
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
                # Check topic_id field
                if sample.get("topic_id") == uuid:
                    matches.append({"line": line_num, "sample": sample})
            except json.JSONDecodeError:
                continue

    return matches


def find_in_checkpoint(checkpoint_dir: str, output_name: str, uuid: str) -> dict:
    """Find UUID references in checkpoint files."""
    checkpoint_dir = Path(checkpoint_dir)
    stem = Path(output_name).stem

    results = {
        "in_processed_ids": False,
        "samples": [],
        "failures": [],
    }

    # Check metadata
    metadata_path = checkpoint_dir / f"{stem}.checkpoint.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        if uuid in metadata.get("processed_ids", []):
            results["in_processed_ids"] = True

    # Check samples file
    samples_path = checkpoint_dir / f"{stem}.checkpoint.jsonl"
    if samples_path.exists():
        with open(samples_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    if sample.get("topic_id") == uuid:
                        results["samples"].append({"line": line_num, "sample": sample})
                except json.JSONDecodeError:
                    continue

    # Check failures file
    failures_path = checkpoint_dir / f"{stem}.checkpoint.failures.jsonl"
    if failures_path.exists():
        with open(failures_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    failure = json.loads(line)
                    if failure.get("topic_id") == uuid:
                        results["failures"].append({"line": line_num, "failure": failure})
                except json.JSONDecodeError:
                    continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Look up topic graph and dataset entries by UUID"
    )
    parser.add_argument("uuid", help="UUID to search for")
    parser.add_argument("config", help="Path to config YAML file")
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of formatted text"
    )
    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Get paths from config
    topics_path = config.get("topics", {}).get("save_as")
    output_config = config.get("output", {})
    dataset_path = output_config.get("save_as")
    checkpoint_config = output_config.get("checkpoint", {})
    checkpoint_dir = checkpoint_config.get("path", ".checkpoints") if checkpoint_config else None

    results = {
        "uuid": args.uuid,
        "topic_graph": None,
        "dataset_samples": [],
        "checkpoint": None,
    }

    # Search topic graph
    if topics_path:
        topic_result = find_in_topic_graph(topics_path, args.uuid)
        results["topic_graph"] = topic_result

    # Search dataset
    if dataset_path:
        dataset_results = find_in_dataset(dataset_path, args.uuid)
        results["dataset_samples"] = dataset_results

    # Search checkpoint
    if checkpoint_dir and dataset_path:
        checkpoint_results = find_in_checkpoint(checkpoint_dir, dataset_path, args.uuid)
        if (
            checkpoint_results["in_processed_ids"]
            or checkpoint_results["samples"]
            or checkpoint_results["failures"]
        ):
            results["checkpoint"] = checkpoint_results

    # Output
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"=== Lookup for UUID: {args.uuid} ===\n")

        # Topic graph
        print("--- Topic Graph ---")
        if not topics_path:
            print("  No topics.save_as configured")
        elif not Path(topics_path).exists():
            print(f"  Topic file not found: {topics_path}")
        elif results["topic_graph"]:
            node = results["topic_graph"]
            print(f"  Node ID: {node['node_id']}")
            print(f"  Topic: {node['topic']}")
            if node["children"]:
                print(f"  Children: {', '.join(node['children'])}")
            print(f"  Metadata: {json.dumps(node['metadata'], indent=4)}")
        else:
            print("  Not found in topic graph")

        # Dataset
        print("\n--- Dataset ---")
        if not dataset_path:
            print("  No output.save_as configured")
        elif not Path(dataset_path).exists():
            print(f"  Dataset file not found: {dataset_path}")
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
                        content = msg.get("content", "")[:100]
                        print(f"      [{i}] {role}: {content}...")
                else:
                    print(f"    {json.dumps(sample, indent=4)[:500]}...")
        else:
            print("  Not found in dataset")

        # Checkpoint
        print("\n--- Checkpoint ---")
        if not checkpoint_dir:
            print("  No checkpoint configured")
        elif results["checkpoint"]:
            cp = results["checkpoint"]
            print(f"  In processed_ids: {cp['in_processed_ids']}")
            if cp["samples"]:
                print(f"  Checkpoint samples: {len(cp['samples'])}")
            if cp["failures"]:
                print(f"  Checkpoint failures: {len(cp['failures'])}")
                for fail in cp["failures"]:
                    print(f"    Line {fail['line']}: {fail['failure'].get('error', 'unknown error')}")
        else:
            print("  Not found in checkpoint files")


if __name__ == "__main__":
    main()
