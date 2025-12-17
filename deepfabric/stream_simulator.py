"""Buffered stream simulation for TUI preview.

This module provides a fire-and-forget streaming simulation that emits chunks
to the TUI preview without impacting generation performance. The simulation
runs in background while generation continues immediately.
"""

import asyncio

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .constants import STREAM_SIM_CHUNK_DELAY_MS, STREAM_SIM_CHUNK_SIZE

if TYPE_CHECKING:
    from .progress import ProgressReporter

# Track current stream task to prevent interleaving (mutable container to avoid global statement)
_stream_state: dict[str, asyncio.Task | None] = {"current_task": None}


class StreamSimulatorConfig(BaseModel):
    """Configuration for buffered stream simulation."""

    chunk_size: int = Field(default=STREAM_SIM_CHUNK_SIZE, ge=1, le=100)
    chunk_delay_ms: float = Field(default=STREAM_SIM_CHUNK_DELAY_MS, ge=0.0, le=100.0)
    enabled: bool = Field(default=True)


def simulate_stream(
    progress_reporter: "ProgressReporter | None",
    content: str,
    source: str,
    config: StreamSimulatorConfig | None = None,
    **metadata,
) -> asyncio.Task | None:
    """Fire-and-forget stream simulation (non-blocking).

    Starts simulation in background and returns immediately. This is the
    primary interface for stream simulation throughout the codebase.

    Cancels any in-flight stream task before starting a new one to prevent
    interleaved chunks from multiple generations appearing scrambled in the TUI.

    Args:
        progress_reporter: ProgressReporter instance or None
        content: Text to simulate streaming
        source: Source identifier for TUI routing
        config: Optional StreamSimulatorConfig override
        **metadata: Additional metadata passed to emit_chunk

    Returns:
        Task if started, None if no-op (no reporter or disabled)
    """
    _config = config or StreamSimulatorConfig()
    if not progress_reporter or not _config.enabled:
        return None

    # Cancel any in-flight stream to prevent interleaving
    current_task = _stream_state["current_task"]
    if current_task is not None and not current_task.done():
        current_task.cancel()

    async def _simulate_impl() -> None:
        """Internal implementation of chunk emission."""
        if not content:
            return

        delay = _config.chunk_delay_ms / 1000.0
        chunk_size = _config.chunk_size

        try:
            for i in range(0, len(content), chunk_size):
                chunk = content[i : i + chunk_size]
                progress_reporter.emit_chunk(source, chunk, **metadata)
                if delay > 0 and i + chunk_size < len(content):
                    await asyncio.sleep(delay)
        except asyncio.CancelledError:
            # Gracefully handle cancellation
            pass

    _stream_state["current_task"] = asyncio.create_task(_simulate_impl())
    return _stream_state["current_task"]
