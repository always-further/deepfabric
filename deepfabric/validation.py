import time

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
    mode: str,  # noqa: ARG001 - kept for API compatibility
    depth: int,  # noqa: ARG001 - kept for API compatibility
    degree: int,  # noqa: ARG001 - kept for API compatibility
    num_samples: int | str,  # noqa: ARG001 - kept for API compatibility
    batch_size: int,  # noqa: ARG001 - kept for API compatibility
    loading_existing: bool = False,  # noqa: ARG001 - kept for API compatibility
) -> None:
    """
    Validate topic generation parameters (informational only, no longer errors).

    When num_samples exceeds available paths, topics will cycle for even coverage.

    Args:
        mode: Generation mode ('tree' or 'graph')
        depth: Depth of the tree/graph
        degree: Branching factor
        num_samples: Total samples to generate, or "auto"/percentage string
        batch_size: Batch size for generation (not used in validation, kept for API compat)
        loading_existing: Whether loading existing topic model from file
    """
    # No validation needed - generator handles all cases:
    # - num_samples < paths: random subset
    # - num_samples == paths: all paths used once
    # - num_samples > paths: topics cycle for even coverage
    pass


def show_validation_success(
    mode: str,
    depth: int,
    degree: int,
    num_samples: int | str,
    batch_size: int,  # noqa: ARG001 - kept for API compatibility
    loading_existing: bool = False,
) -> None:
    """
    Show validation success message.

    Args:
        mode: Generation mode ('tree' or 'graph')
        depth: Depth of the tree/graph
        degree: Branching factor
        num_samples: Total samples to generate, or "auto"/percentage string
        batch_size: Batch size for generation (not used in display, kept for API compat)
        loading_existing: Whether loading existing topic model from file
    """
    if loading_existing:
        return

    expected_paths = calculate_expected_paths(mode, depth, degree)
    tui = get_tui()

    # Handle dynamic num_samples (auto or percentage)
    if isinstance(num_samples, str):
        tui.success("Path Validation Passed")
        tui.info(f"  Expected {mode} paths: ~{expected_paths} (depth={depth}, degree={degree})")
        if num_samples == "auto":
            tui.info(f"  Requested samples: auto (will use all ~{expected_paths} paths)")
        else:
            # Percentage string like "50%"
            pct = float(num_samples[:-1])
            estimated_samples = max(1, int(expected_paths * pct / 100))
            tui.info(
                f"  Requested samples: {num_samples} (~{estimated_samples} of {expected_paths} paths)"
            )
        if mode == "graph":
            tui.info("  Note: Graph paths may vary due to cross-connections")
        print()  # Extra space before topic generation
        time.sleep(0.5)  # Brief pause to allow user to see the information
        return

    tui.success("Path Validation Passed")
    tui.info(f"  Expected {mode} paths: ~{expected_paths} (depth={depth}, degree={degree})")
    tui.info(f"  Requested samples: {num_samples}")

    if num_samples > expected_paths:
        cycles = (num_samples + expected_paths - 1) // expected_paths  # ceil division
        tui.info(f"  Topic cycling: ~{cycles}x passes through topics")
    else:
        tui.info(f"  Path utilization: ~{(num_samples / expected_paths) * 100:.1f}%")

    if mode == "graph":
        tui.info("  Note: Graph paths may vary due to cross-connections")
    print()  # Extra space before topic generation
    time.sleep(0.5)  # Brief pause to allow user to see the information
