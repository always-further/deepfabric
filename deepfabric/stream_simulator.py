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


# Track current stream task to prevent interleaving
class _StreamState:
    current_task: asyncio.Task | None = None


_stream_state = _StreamState()


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
    if _stream_state.current_task is not None and not _stream_state.current_task.done():
        _stream_state.current_task.cancel()

    async def _simulate_impl() -> None:
        """Internal implementation of chunk emission."""
        if not content:
            return

        delay = _config.chunk_delay_ms / 1000.0
        chunk_size = _config.chunk_size

        try:
            for i in range(0, len(content), chunk_size):
                # Await before emitting to ensure cancellation is processed
                # and event loop is not blocked when delay is 0
                if i > 0:
                    await asyncio.sleep(delay)
                chunk = content[i : i + chunk_size]
                progress_reporter.emit_chunk(source, chunk, **metadata)
        except asyncio.CancelledError:
            # Gracefully handle cancellation
            pass

    _stream_state.current_task = asyncio.create_task(_simulate_impl())
    return _stream_state.current_task
