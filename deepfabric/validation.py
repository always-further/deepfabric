import time

from .exceptions import ConfigurationError
from .tui import get_tui


def calculate_expected_paths(mode: str, depth: int, degree: int) -> int:
    """
    Calculate expected number of paths for tree/graph generation.

    Args:
        mode: Generation mode ('tree' or 'graph')
        depth: Depth of the tree/graph
        degree: Branching factor

    Returns:
        Expected number of paths
    """
    if mode == "tree":
        # Tree paths = degree^depth (exact - each leaf is a unique path)
        return degree**depth
    # mode == "graph"
    # Graph paths vary widely due to cross-connections
    # Can range from degree^depth * 0.5 to degree^depth * 2+
    # Use base estimate as rough middle ground, but warn it's approximate
    return degree**depth


def validate_path_requirements(
    mode: str,
    depth: int,
    degree: int,
    num_steps: int | str,
    batch_size: int,
    loading_existing: bool = False,
) -> None:
    """
    Validate that the topic generation parameters will produce enough paths.

    Args:
        mode: Generation mode ('tree' or 'graph')
        depth: Depth of the tree/graph
        degree: Branching factor
        num_steps: Number of generation steps, or "auto"/percentage string
        batch_size: Batch size for generation
        loading_existing: Whether loading existing topic model from file

    Raises:
        ConfigurationError: If validation fails
    """
    if loading_existing:
        # Can't validate existing files without loading them
        return

    # Skip validation for dynamic values - resolved later with actual topic count
    if isinstance(num_steps, str):
        return

    expected_paths = calculate_expected_paths(mode, depth, degree)
    required_samples = num_steps * batch_size

    if required_samples > expected_paths:
        # Alternative: provide exact combinations that use all paths
        optimal_combinations = []
        for test_steps in range(1, expected_paths + 1):
            test_batch = expected_paths // test_steps
            if test_steps * test_batch <= expected_paths and test_batch > 0:
                optimal_combinations.append((test_steps, test_batch))

        # Sort by preference (fewer steps first, then larger batches)
        optimal_combinations.sort(key=lambda x: (x[0], -x[1]))

        tui = get_tui()
        tui.error(" Path validation failed - stopping before topic generation")

        # Build recommendations - focus on optimal combinations rather than misleading individual params
        recommendations = []

        if optimal_combinations:
            recommendations.append(
                f"  • Use one of these combinations to utilize the {expected_paths} paths:"
            )
            for steps, batch in optimal_combinations[:3]:  # Show top 3
                total_samples = steps * batch
                recommendations.append(
                    f"    --num-samples {steps} --batch-size {batch}  (generates {total_samples} samples)"
                )

        recommendations.extend(
            [
                f"  • Or increase --depth (currently {depth}) or --degree (currently {degree})",
            ]
        )

        estimation_note = ""
        if mode == "graph":
            estimation_note = " (estimated - graphs vary due to cross-connections)"

        error_msg = (
            f"Insufficient expected paths for dataset generation:\n"
            f"  • Expected {mode} paths: ~{expected_paths}{estimation_note} (depth={depth}, degree={degree})\n"
            f"  • Requested samples: {required_samples} ({num_steps} steps × {batch_size} batch size)\n"
            f"  • Shortfall: ~{required_samples - expected_paths} samples\n\n"
            f"Recommendations:\n" + "\n".join(recommendations)
        )

        if mode == "graph":
            error_msg += f"\n\nNote: Graph path counts are estimates. The actual graph may produce {expected_paths // 2}-{expected_paths * 2} paths due to cross-connections."

        raise ConfigurationError(error_msg)


def show_validation_success(
    mode: str,
    depth: int,
    degree: int,
    num_steps: int | str,
    batch_size: int,
    loading_existing: bool = False,
) -> None:
    """
    Show validation success message.

    Args:
        mode: Generation mode ('tree' or 'graph')
        depth: Depth of the tree/graph
        degree: Branching factor
        num_steps: Number of generation steps, or "auto"/percentage string
        batch_size: Batch size for generation
        loading_existing: Whether loading existing topic model from file
    """
    if loading_existing:
        return

    expected_paths = calculate_expected_paths(mode, depth, degree)
    tui = get_tui()

    # Handle dynamic num_samples (auto or percentage)
    if isinstance(num_steps, str):
        tui.success("Path Validation Passed")
        tui.info(f"  Expected {mode} paths: ~{expected_paths} (depth={depth}, degree={degree})")
        if num_steps == "auto":
            tui.info(f"  Requested samples: auto (will use all ~{expected_paths} paths)")
        else:
            # Percentage string like "50%"
            pct = float(num_steps[:-1])
            estimated_samples = max(1, int(expected_paths * pct / 100))
            tui.info(
                f"  Requested samples: {num_steps} (~{estimated_samples} of {expected_paths} paths)"
            )
        if mode == "graph":
            tui.info("  Note: Graph paths may vary due to cross-connections")
        print()  # Extra space before topic generation
        time.sleep(0.5)  # Brief pause to allow user to see the information
        return

    total_samples = num_steps * batch_size

    tui.success("Path Validation Passed")
    tui.info(f"  Expected {mode} paths: ~{expected_paths} (depth={depth}, degree={degree})")
    tui.info(f"  Requested samples: {total_samples} ({num_steps} steps x {batch_size} batch size)")
    tui.info(f"  Path utilization: ~{min(100, (total_samples / expected_paths) * 100):.1f}%")

    if mode == "graph":
        tui.info("  Note: Graph paths may vary due to cross-connections")
    print()  # Extra space before topic generation
    time.sleep(0.5)  # Brief pause to allow user to see the information
